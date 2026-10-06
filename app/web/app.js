'use strict'

const $ = (id) => document.getElementById(id)
const h = (tag, cls, text) => {
  const e = document.createElement(tag)
  if (cls) e.className = cls
  if (text != null) e.textContent = text
  return e
}

let tab = 'browse'
let videoMode = 'fisheye'
let picked = { video: null, audio: null }   // browse selections: { path, name }
let jobs = []
let busyUploading = false
let info = {}

// ── API ───────────────────────────────────────────────────────────────────────
async function api(method, url, body) {
  const r = await fetch(url, {
    method, headers: body ? { 'Content-Type': 'application/json' } : {},
    body: body ? JSON.stringify(body) : undefined,
  })
  const data = await r.json().catch(() => ({}))
  if (r.status === 401 && url !== '/api/login') { showLogin(); throw new Error('PIN required') }
  if (!r.ok) throw new Error(data.error || `HTTP ${r.status}`)
  return data
}

function msg(text, kind) {
  $('msg').textContent = text || ''
  $('msg').className = `msg ${kind || ''}`
}

const fmtSize = (n) => n >= 1024 ** 3 ? `${(n / 1024 ** 3).toFixed(1)} GB`
                     : n >= 1024 ** 2 ? `${Math.round(n / 1024 ** 2)} MB` : `${Math.max(1, Math.round(n / 1024))} KB`
const fmtDur = (ms) => {
  const s = Math.round(ms / 1000), m = Math.floor(s / 60)
  return m >= 60 ? `${Math.floor(m / 60)}h ${m % 60}m` : m ? `${m}m ${s % 60}s` : `${s}s`
}

// ── Login ─────────────────────────────────────────────────────────────────────
function showLogin() {
  $('main').classList.add('hidden')
  $('login').classList.remove('hidden')
  $('pin').focus()
}
$('login-form').addEventListener('submit', async (e) => {
  e.preventDefault()
  try {
    await api('POST', '/api/login', { pin: $('pin').value })
    $('login').classList.add('hidden')
    start()
  } catch (err) {
    $('login-msg').textContent = err.message
    $('login-msg').className = 'msg err'
  }
})

// ── Tabs / options ────────────────────────────────────────────────────────────
document.querySelectorAll('.tab').forEach(b => b.addEventListener('click', () => {
  if (busyUploading) return
  tab = b.dataset.tab
  document.querySelectorAll('.tab').forEach(x => x.classList.toggle('active', x === b))
  $('tab-browse').classList.toggle('hidden', tab !== 'browse')
  $('tab-upload').classList.toggle('hidden', tab !== 'upload')
  updateSelection()
}))

document.querySelectorAll('#mode button').forEach(b => b.addEventListener('click', () => {
  videoMode = b.dataset.mode
  document.querySelectorAll('#mode button').forEach(x => x.classList.toggle('active', x === b))
}))

$('up-video').addEventListener('change', updateSelection)
$('up-audio').addEventListener('change', updateSelection)
$('clear-audio').addEventListener('click', () => {
  if (tab === 'browse') { picked.audio = null; markSelected() }
  else $('up-audio').value = ''
  updateSelection()
})

function updateSelection() {
  let v, a
  if (tab === 'browse') {
    v = picked.video?.name
    a = picked.audio?.name
  } else {
    const fv = $('up-video').files[0], fa = $('up-audio').files[0]
    v = fv && `${fv.name} (${fmtSize(fv.size)})`
    a = fa && `${fa.name} (${fmtSize(fa.size)})`
  }
  $('sel-video').textContent = v || 'none selected'
  $('sel-audio').textContent = a || 'none'
  $('clear-audio').classList.toggle('hidden', !a)
  $('go').disabled = !v || busyUploading
  $('go').textContent = tab === 'upload' ? 'Upload & render' : 'Render'
}

// ── Browse ────────────────────────────────────────────────────────────────────
let currentDir = ''

async function browse(dir) {
  const list = $('listing')
  let data
  try { data = await api('GET', `/api/browse?path=${encodeURIComponent(dir || '')}`) }
  catch (e) { list.innerHTML = ''; list.appendChild(h('li', 'empty', e.message)); return }
  currentDir = data.path || ''

  // breadcrumbs: Locations › …path below the shared root
  const crumbs = $('crumbs')
  crumbs.innerHTML = ''
  const link = (label, p) => { const a = h('a', null, label); a.onclick = () => browse(p); return a }
  crumbs.appendChild(link('Locations', ''))
  if (data.path) {
    // walk up the path string, stopping at the shared root
    const parts = []
    let p = data.path
    while (p && p !== '/') { parts.unshift(p); if (p === rootOf(data.path)) break; p = p.substring(0, p.lastIndexOf('/')) || '/' }
    for (const full of parts) {
      crumbs.appendChild(h('span', null, '›'))
      crumbs.appendChild(link(full === rootOf(data.path) ? rootLabel(full) : full.split('/').pop(), full))
    }
  }

  list.innerHTML = ''
  if (data.path && data.parent !== null) {
    const up = h('li')
    up.append(h('span', 'ico', '↩'), h('span', 'nm', '..'))
    up.onclick = () => browse(data.parent)
    list.appendChild(up)
  }
  if (!data.entries.length) list.appendChild(h('li', 'empty', 'No folders or video/audio files here'))
  for (const e of data.entries) {
    if (!data.path) knownRoots.set(e.path, e.name)
    const li = h('li')
    li.dataset.path = e.path
    li.append(
      h('span', 'ico', e.type === 'dir' ? '📁' : e.type === 'video' ? '🎬' : '🎙'),
      h('span', 'nm', e.name),
      h('span', 'meta', e.type === 'dir' ? '' : `${fmtSize(e.size)} · ${new Date(e.mtime).toLocaleDateString()}`),
    )
    li.onclick = () => {
      if (e.type === 'dir') return browse(e.path)
      picked[e.type] = { path: e.path, name: e.name }
      markSelected()
      updateSelection()
    }
    list.appendChild(li)
  }
  markSelected()
}

const knownRoots = new Map()
const rootOf = (p) => [...knownRoots.keys()].find(r => p === r || p.startsWith(r + '/')) || null
const rootLabel = (r) => knownRoots.get(r) || r

function markSelected() {
  for (const li of $('listing').children) {
    const p = li.dataset.path
    li.classList.toggle('sel', !!p && (p === picked.video?.path || p === picked.audio?.path))
  }
}

// ── Upload ────────────────────────────────────────────────────────────────────
function uploadFile(file, batch, label) {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest()
    xhr.open('PUT', `/api/upload?batch=${batch}&name=${encodeURIComponent(file.name)}`)
    const t0 = Date.now()
    xhr.upload.onprogress = (e) => {
      if (!e.lengthComputable) return
      const pct = e.loaded / e.total * 100
      const rate = e.loaded / Math.max(1, (Date.now() - t0) / 1000)
      const eta = rate > 0 ? (e.total - e.loaded) / rate * 1000 : 0
      $('up-fill').style.width = `${pct}%`
      $('up-label').textContent = `${label}: ${fmtSize(e.loaded)} / ${fmtSize(e.total)} · ` +
        `${fmtSize(rate)}/s · ${eta > 0 ? fmtDur(eta) + ' left' : ''}`
    }
    xhr.onload = () => {
      let data = {}
      try { data = JSON.parse(xhr.responseText) } catch {}
      xhr.status === 200 ? resolve(data) : reject(new Error(data.error || `upload failed (HTTP ${xhr.status})`))
    }
    xhr.onerror = () => reject(new Error('upload interrupted (network error)'))
    xhr.send(file)
  })
}

// warn before closing the tab mid-upload
window.addEventListener('beforeunload', (e) => { if (busyUploading) { e.preventDefault(); e.returnValue = '' } })

// ── Submit ────────────────────────────────────────────────────────────────────
$('go').addEventListener('click', async () => {
  msg('')
  const opts = { videoMode, previewMode: $('preview').checked }
  try {
    if (tab === 'browse') {
      if (!picked.video) return
      const r = await api('POST', '/api/jobs', { ...opts, videoPath: picked.video.path, audioPath: picked.audio?.path })
      msg(r.ahead ? `Queued — ${r.ahead} job(s) ahead` : 'Started', 'ok')
      return
    }
    const fv = $('up-video').files[0], fa = $('up-audio').files[0]
    if (!fv) return
    info = await api('GET', '/api/info')
    const need = fv.size + (fa?.size || 0) + (info.uploadHeadroom || 0)
    if (info.uploadFree != null && info.uploadFree < need) {
      throw new Error(`Not enough space on the render Mac (needs ${fmtSize(need)}, ${fmtSize(info.uploadFree)} free)`)
    }
    busyUploading = true
    updateSelection()
    $('up-progress').classList.remove('hidden')
    const batch = Math.random().toString(36).slice(2, 12) + Date.now().toString(36)
    const v = await uploadFile(fv, batch, 'Video')
    const a = fa ? await uploadFile(fa, batch, 'Mic audio') : null
    $('up-label').textContent = 'Upload complete'
    const r = await api('POST', '/api/jobs', { ...opts, videoPath: v.path, audioPath: a?.path })
    msg(r.ahead ? `Uploaded — queued with ${r.ahead} job(s) ahead` : 'Uploaded — render started', 'ok')
    $('up-video').value = ''
    $('up-audio').value = ''
  } catch (e) {
    msg(e.message, 'err')
  } finally {
    busyUploading = false
    updateSelection()
  }
})

// ── Jobs (live via Server-Sent Events) ────────────────────────────────────────
function renderJobs() {
  const ul = $('jobs')
  ul.innerHTML = ''
  const running = jobs.find(j => j.status === 'running')
  $('busy').textContent = running ? `Rendering: ${running.name}` : 'Idle'
  $('busy').classList.toggle('on', !!running)
  if (!jobs.length) { ul.appendChild(h('li', 'empty', 'No jobs yet')); return }

  for (const j of [...jobs].reverse()) {
    const li = h('li', 'job')
    li.id = `job-${j.id}`
    const head = h('div', 'job-head')
    head.append(h('span', 'job-name', j.name), h('span', `badge ${j.status}`, j.status))
    if (j.status === 'queued' || j.status === 'running') {
      const b = h('button', 'btn small', j.status === 'running' ? 'Stop' : 'Cancel')
      b.onclick = () => api('POST', `/api/jobs/${j.id}/cancel`).catch(e => msg(e.message, 'err'))
      head.appendChild(b)
    }
    if (j.status === 'done' && j.outputName) {
      const a = h('a', 'btn small primary', 'Download')
      a.href = `/api/jobs/${j.id}/file`
      a.style.marginLeft = '0'
      head.appendChild(a)
    }
    li.appendChild(head)

    const bits = [j.source === 'web' ? 'from web' : 'from desktop app', j.videoMode === 'equirect' ? 'YouTube 360°' : 'Insta360']
    if (j.audioName) bits.push(`mic: ${j.audioName}`)
    if (j.previewMode) bits.push('60s preview')
    if (j.startedAt) bits.push(j.finishedAt ? `took ${fmtDur(j.finishedAt - j.startedAt)}` : `running ${fmtDur(Date.now() - j.startedAt)}`)
    if (j.outputName) bits.push(j.outputName)
    li.appendChild(h('div', 'job-sub', bits.join(' · ')))

    if (j.steps?.length) {
      const bar = h('div', 'steps')
      for (const s of j.steps) {
        const st = j.stepStates?.[s.id] || {}
        const seg = h('div', `s ${st.status || ''}`)
        seg.title = s.label
        const fill = h('div')
        fill.style.width = `${st.status === 'running' ? st.pct || 0 : 0}%`
        seg.appendChild(fill)
        bar.appendChild(seg)
      }
      li.appendChild(bar)
      const now = j.steps.filter(s => j.stepStates?.[s.id]?.status === 'running')
      if (j.status === 'running' && now.length) {
        li.appendChild(h('div', 'step-now', now.map(s => `${s.label} ${j.stepStates[s.id].pct || 0}%`).join(' · ')))
      }
    }
    if (j.error) li.appendChild(h('div', 'job-err', j.error))
    if (j.status === 'running') {
      const d = h('details')
      d.open = !!openLogs[j.id]
      d.ontoggle = () => { openLogs[j.id] = d.open }
      d.append(h('summary', null, 'Log'), h('pre', null, (j.log || []).join('\n')))
      li.appendChild(d)
    }
    ul.appendChild(li)
  }
  for (const pre of ul.querySelectorAll('pre')) pre.scrollTop = pre.scrollHeight
}
const openLogs = {}

function connectEvents() {
  const es = new EventSource('/api/events')
  es.addEventListener('jobs', (e) => {
    const prev = Object.fromEntries(jobs.map(j => [j.id, j.log]))
    jobs = JSON.parse(e.data)
    for (const j of jobs) if (!j.log && prev[j.id]) j.log = prev[j.id]
    renderJobs()
  })
  es.addEventListener('step', (e) => {
    const { jobId, id, status, pct } = JSON.parse(e.data)
    const j = jobs.find(x => x.id === jobId)
    if (!j) return
    j.stepStates = j.stepStates || {}
    j.stepStates[id] = { status, pct: pct ?? j.stepStates[id]?.pct ?? 0 }
    renderJobs()
  })
  es.addEventListener('log', (e) => {
    const { jobId, line } = JSON.parse(e.data)
    const j = jobs.find(x => x.id === jobId)
    if (!j) return
    j.log = [...(j.log || []), line].slice(-300)
    const pre = document.querySelector(`#job-${jobId} pre`)
    if (pre) { pre.textContent = j.log.join('\n'); pre.scrollTop = pre.scrollHeight }
  })
  // EventSource reconnects by itself; a 401 (PIN changed) surfaces via fetch calls
}

// refresh "running for …" once a minute
setInterval(() => { if (jobs.some(j => j.status === 'running')) renderJobs() }, 60000)

// ── Init ──────────────────────────────────────────────────────────────────────
function start() {
  $('main').classList.remove('hidden')
  browse('')
  connectEvents()
  updateSelection()
}

;(async () => {
  try { info = await api('GET', '/api/info') } catch (e) { msg(e.message, 'err') }
  if (info.pinRequired && !info.authed) showLogin()
  else start()
})()
