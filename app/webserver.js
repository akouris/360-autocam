'use strict'
// LAN web UI for 360 AutoCam: browse this Mac's folders / mounted shares or
// upload files, queue renders, watch progress (Server-Sent Events), download
// results. Node built-ins only — nothing extra to bundle.
const http   = require('http')
const fs     = require('fs')
const path   = require('path')
const crypto = require('crypto')

const WEB_DIR    = path.join(__dirname, 'web')
const VIDEO_EXT  = new Set(['.insv', '.mp4', '.mov'])
const AUDIO_EXT  = new Set(['.mp3', '.wav', '.m4a', '.aac'])
const STATIC     = { '/': 'index.html', '/app.js': 'app.js', '/style.css': 'style.css' }
const MIME       = { '.html': 'text/html; charset=utf-8', '.js': 'text/javascript; charset=utf-8',
                     '.css': 'text/css; charset=utf-8' }
// headroom an upload must leave on the disk for the job's own working files
const WORK_HEADROOM = 10 * 1024 ** 3

function createWebServer(opts) {
  const { port, pin, uploadsDir, listJobs, enqueue, cancel, outputOf, busy } = opts
  const roots = opts.roots.filter(r => fs.existsSync(r)).map(r => fs.realpathSync(r))
  fs.mkdirSync(uploadsDir, { recursive: true })
  const uploadsReal = fs.realpathSync(uploadsDir)
  const clients = new Set()

  // ── helpers ────────────────────────────────────────────────────────────────
  const json = (res, code, body) => {
    res.writeHead(code, { 'Content-Type': 'application/json', 'Cache-Control': 'no-store' })
    res.end(JSON.stringify(body))
  }
  const fail = (res, code, error) => json(res, code, { error })

  function readJson(req, limit = 64 * 1024) {
    return new Promise((resolve, reject) => {
      let buf = ''
      req.setEncoding('utf8')
      req.on('data', (c) => { buf += c; if (buf.length > limit) { reject(new Error('body too large')); req.destroy() } })
      req.on('end', () => { try { resolve(JSON.parse(buf || '{}')) } catch { reject(new Error('invalid JSON')) } })
      req.on('error', reject)
    })
  }

  function cookie(req, name) {
    const m = (req.headers.cookie || '').match(new RegExp(`(?:^|;\\s*)${name}=([^;]*)`))
    return m ? decodeURIComponent(m[1]) : ''
  }
  function authed(req) {
    if (!pin) return true
    const a = Buffer.from(cookie(req, 'autocam_pin')), b = Buffer.from(pin)
    return a.length === b.length && crypto.timingSafeEqual(a, b)
  }

  // A path is reachable only if its REAL location is inside a browse root or the
  // uploads folder (symlinks such as /Volumes/Macintosh HD → / are rejected).
  function allowed(p, extraRoots = []) {
    let real
    try { real = fs.realpathSync(p) } catch { return null }
    for (const r of [...roots, ...extraRoots]) {
      if (real === r || real.startsWith(r.endsWith(path.sep) ? r : r + path.sep)) return real
    }
    return null
  }

  function kindOf(name) {
    const ext = path.extname(name).toLowerCase()
    return VIDEO_EXT.has(ext) ? 'video' : AUDIO_EXT.has(ext) ? 'audio' : null
  }

  function freeBytes(dir) {
    try { const s = fs.statfsSync(dir); return s.bavail * s.bsize } catch { return null }
  }

  // ── routes ─────────────────────────────────────────────────────────────────
  function browse(req, res, url) {
    const p = url.searchParams.get('path')
    if (!p) {
      return json(res, 200, { path: null, parent: null,
        entries: roots.map(r => ({ name: r === '/Volumes' ? 'Network & external drives' : `Home (${path.basename(r)})`,
                                   path: r, type: 'dir' })) })
    }
    const real = allowed(p)
    if (!real) return fail(res, 403, 'That folder is outside the shared locations')
    let names
    try { names = fs.readdirSync(real) } catch (e) { return fail(res, 404, `Cannot open folder: ${e.code || e.message}`) }
    const entries = []
    for (const name of names) {
      if (name.startsWith('.')) continue
      const full = path.join(real, name)
      let st
      try { st = fs.statSync(full) } catch { continue }
      if (st.isDirectory()) {
        if (allowed(full)) entries.push({ name, path: full, type: 'dir' })
      } else if (st.isFile()) {
        const kind = kindOf(name)
        if (kind) entries.push({ name, path: full, type: kind, size: st.size, mtime: st.mtimeMs })
      }
    }
    entries.sort((a, b) => (a.type !== 'dir') - (b.type !== 'dir') ||
                           a.name.localeCompare(b.name, undefined, { numeric: true }))
    const isRoot = roots.includes(real)
    json(res, 200, { path: real, parent: isRoot ? '' : path.dirname(real), entries })
  }

  // Streams the request body straight to disk: multi-GB uploads never sit in
  // memory. Body = raw file bytes; name/batch come in the query string.
  function upload(req, res, url) {
    // on a refusal, drop the connection so the browser stops sending gigabytes
    const fail = (res, code, error) => { res.setHeader('Connection', 'close'); json(res, code, { error }) }
    const name  = path.basename(url.searchParams.get('name') || '').replace(/[^\w.\- ()]/g, '_')
    const batch = url.searchParams.get('batch') || ''
    if (!/^[a-z0-9]{8,32}$/.test(batch)) return fail(res, 400, 'bad batch id')
    if (!name || !kindOf(name)) return fail(res, 400, 'Only video (.insv .mp4 .mov) or audio (.mp3 .wav .m4a .aac) files')
    const len = parseInt(req.headers['content-length'] || '', 10)
    if (!(len > 0)) return fail(res, 411, 'missing file size')
    if (busy()) return fail(res, 409, 'Busy: another render is running. Try again when it has finished.')

    const dir = path.join(uploadsDir, batch)
    fs.mkdirSync(dir, { recursive: true })
    const free = freeBytes(dir)
    if (free !== null && free < len + WORK_HEADROOM) {
      return fail(res, 507, `Not enough disk space on the render Mac: need ${gb(len + WORK_HEADROOM)}, ${gb(free)} free`)
    }
    const dest = path.join(dir, name)
    const part = dest + '.part'
    const out  = fs.createWriteStream(part)
    let received = 0, finished = false
    const abort = () => {
      if (finished) return
      finished = true
      out.destroy()
      fs.rm(part, { force: true }, () => {})
    }
    req.on('data', (c) => { received += c.length })
    req.on('aborted', abort)
    res.on('close', () => { if (!finished) abort() })
    out.on('error', (e) => { abort(); if (!res.headersSent) fail(res, 500, `write failed: ${e.message}`) })
    out.on('finish', () => {
      if (finished) return
      finished = true
      if (received !== len) { fs.rm(part, { force: true }, () => {}); return fail(res, 400, 'upload incomplete') }
      fs.renameSync(part, dest)
      json(res, 200, { path: dest, name, size: received })
    })
    req.pipe(out)
  }

  async function createJob(req, res) {
    let body
    try { body = await readJson(req) } catch (e) { return fail(res, 400, e.message) }
    const video = body.videoPath && allowed(body.videoPath, [uploadsReal])
    if (!video || kindOf(video) !== 'video') return fail(res, 400, 'Pick a video file from a shared location or upload one')
    let audio = null
    if (body.audioPath) {
      audio = allowed(body.audioPath, [uploadsReal])
      if (!audio || kindOf(audio) !== 'audio') return fail(res, 400, 'Invalid external audio file')
    }
    // uploaded sources are removed after a successful render
    const uploadDir = video.startsWith(uploadsReal + path.sep) ? path.dirname(video) : null
    try {
      json(res, 200, enqueue({ videoPath: video, audioPath: audio, videoMode: body.videoMode,
                               previewMode: !!body.previewMode, uploadDir }))
    } catch (e) {
      // refused (busy): an upload made for this job is useless now
      if (uploadDir) fs.rm(uploadDir, { recursive: true, force: true }, () => {})
      fail(res, 409, e.message)
    }
  }

  function download(req, res, id) {
    const file = outputOf(id)
    if (!file) return fail(res, 404, 'No finished video for this job')
    const size = fs.statSync(file).size
    const headers = {
      'Content-Type': 'video/mp4', 'Accept-Ranges': 'bytes',
      'Content-Disposition': `attachment; filename*=UTF-8''${encodeURIComponent(path.basename(file))}`,
    }
    const m = (req.headers.range || '').match(/^bytes=(\d*)-(\d*)$/)
    if (m && (m[1] || m[2])) {
      let start = m[1] ? parseInt(m[1], 10) : size - parseInt(m[2], 10)
      let end   = m[1] && m[2] ? parseInt(m[2], 10) : size - 1
      if (start < 0 || start >= size || end < start) {
        res.writeHead(416, { 'Content-Range': `bytes */${size}` }); return res.end()
      }
      end = Math.min(end, size - 1)
      res.writeHead(206, { ...headers, 'Content-Range': `bytes ${start}-${end}/${size}`, 'Content-Length': end - start + 1 })
      return fs.createReadStream(file, { start, end }).pipe(res)
    }
    res.writeHead(200, { ...headers, 'Content-Length': size })
    fs.createReadStream(file).pipe(res)
  }

  function events(req, res) {
    res.writeHead(200, { 'Content-Type': 'text/event-stream', 'Cache-Control': 'no-store', Connection: 'keep-alive' })
    res.write(`event: jobs\ndata: ${JSON.stringify(listJobs())}\n\n`)
    clients.add(res)
    req.on('close', () => clients.delete(res))
  }

  function serveStatic(res, file) {
    fs.readFile(path.join(WEB_DIR, file), (err, data) => {
      if (err) return fail(res, 404, 'not found')
      res.writeHead(200, { 'Content-Type': MIME[path.extname(file)], 'Cache-Control': 'no-cache' })
      res.end(data)
    })
  }

  // ── dispatcher ─────────────────────────────────────────────────────────────
  const server = http.createServer(async (req, res) => {
    let url
    try { url = new URL(req.url, 'http://x') } catch { return fail(res, 400, 'bad url') }
    const p = url.pathname, m = req.method
    try {
      if (m === 'GET' && STATIC[p]) return serveStatic(res, STATIC[p])
      if (m === 'GET' && p === '/api/info') {
        return json(res, 200, { pinRequired: !!pin, authed: authed(req), busy: busy(),
                                uploadFree: freeBytes(uploadsDir), uploadHeadroom: WORK_HEADROOM })
      }
      if (m === 'POST' && p === '/api/login') {
        const body = await readJson(req)
        if (!pin || String(body.pin || '') === pin) {
          res.setHeader('Set-Cookie', `autocam_pin=${encodeURIComponent(pin)}; Path=/; HttpOnly; SameSite=Strict; Max-Age=31536000`)
          return json(res, 200, { ok: true })
        }
        return fail(res, 401, 'Wrong PIN')
      }
      if (!p.startsWith('/api/')) return fail(res, 404, 'not found')
      if (!authed(req)) return fail(res, 401, 'PIN required')

      if (m === 'GET'  && p === '/api/browse') return browse(req, res, url)
      if (m === 'PUT'  && p === '/api/upload') return upload(req, res, url)
      if (m === 'GET'  && p === '/api/events') return events(req, res)
      if (m === 'GET'  && p === '/api/jobs')   return json(res, 200, listJobs())
      if (m === 'POST' && p === '/api/jobs')   return createJob(req, res)
      let jm
      if (m === 'POST' && (jm = p.match(/^\/api\/jobs\/(\w+)\/cancel$/))) return json(res, 200, { ok: cancel(jm[1]) })
      if (m === 'GET'  && (jm = p.match(/^\/api\/jobs\/(\w+)\/file$/)))   return download(req, res, jm[1])
      fail(res, 404, 'not found')
    } catch (e) {
      if (!res.headersSent) fail(res, 500, e.message)
      else res.destroy()
    }
  })
  // uploads/downloads of multi-GB files take far longer than Node's 5-minute default
  server.requestTimeout = 0
  server.timeout = 0
  server.keepAliveTimeout = 65000

  const heartbeat = setInterval(() => { for (const c of clients) c.write(': ping\n\n') }, 25000)

  return new Promise((resolve, reject) => {
    server.once('error', reject)
    server.listen(port, '0.0.0.0', () => {
      server.removeListener('error', reject)
      resolve({
        port,
        publish(event, data) {
          const msg = `event: ${event}\ndata: ${JSON.stringify(data)}\n\n`
          for (const c of clients) c.write(msg)
        },
        close() {
          clearInterval(heartbeat)
          for (const c of clients) c.end()
          clients.clear()
          return new Promise(r => { server.close(() => r()); server.closeAllConnections?.() })
        },
      })
    })
  })
}

function gb(n) { return `${(n / 1024 ** 3).toFixed(1)} GB` }

module.exports = { createWebServer }
