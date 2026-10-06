'use strict'
const { app, BrowserWindow, ipcMain, dialog, shell, powerSaveBlocker } = require('electron')
const { execSync, spawn } = require('child_process')
const path = require('path')
const fs   = require('fs')
const os   = require('os')
const crypto = require('crypto')
const { createWebServer } = require('./webserver')

// ── Paths ─────────────────────────────────────────────────────────────────────
const IS_PKG     = app.isPackaged
const SCRIPTS_DIR = IS_PKG
  ? path.join(process.resourcesPath, 'scripts')
  : path.resolve(__dirname, '..')

const APP_DATA   = app.getPath('userData')
const MODELS_DIR = path.join(APP_DATA, 'models')
const SETTINGS   = path.join(APP_DATA, 'settings.json')
const REQS       = IS_PKG
  ? path.join(process.resourcesPath, 'requirements.txt')
  : path.join(__dirname, 'resources', 'requirements.txt')

// Self-contained runtime (built by build/bundle.sh): Python + packages, ffmpeg,
// and all models ship inside the app, so it runs on a Mac with nothing installed.
// Packaged: Contents/Resources/{python,bin,models}. Dev: app/bundle if built.
const BUNDLE_DIR = IS_PKG ? process.resourcesPath : path.join(__dirname, 'bundle')
const BUNDLED = {
  python: path.join(BUNDLE_DIR, 'python', 'bin', 'python3'),
  bin:    path.join(BUNDLE_DIR, 'bin'),
  ffmpeg: path.join(BUNDLE_DIR, 'bin', 'ffmpeg'),
  landmarker: path.join(BUNDLE_DIR, 'models', 'face_landmarker.task'),
  pyannote:   path.join(BUNDLE_DIR, 'models', 'pyannote'),
}
const HAS_BUNDLE = fs.existsSync(BUNDLED.python) && fs.existsSync(BUNDLED.ffmpeg)

// Without a bundle (dev): the project venv, else an app-managed venv in userData.
const PROJECT_VENV = IS_PKG ? null : path.resolve(__dirname, '..', 'venv')
function findReadyVenv() {
  if (HAS_BUNDLE) return path.join(BUNDLE_DIR, 'python')
  if (PROJECT_VENV && fs.existsSync(path.join(PROJECT_VENV, 'bin', 'python3'))) {
    return PROJECT_VENV
  }
  return path.join(APP_DATA, 'venv')
}

// VENV_DIR is mutable so we can switch after setup:install creates a new one
let VENV_DIR   = findReadyVenv()
let PYTHON_BIN = path.join(VENV_DIR, 'bin', 'python3')
let PIP_BIN    = path.join(VENV_DIR, 'bin', 'pip')

function refreshVenvPaths(dir) {
  VENV_DIR   = dir
  PYTHON_BIN = path.join(dir, 'bin', 'python3')
  PIP_BIN    = path.join(dir, 'bin', 'pip')
}

// Working files live on the local disk (never next to the source video, which
// may sit on a slow or read-only network share) and are deleted after a
// successful run. Only the finished video lands next to the source.
const WORK_ROOT   = path.join(app.getPath('cache'), '360 AutoCam', 'jobs')
const OUTPUTS_DIR = path.join(app.getPath('videos'), '360 AutoCam')    // fallback when the source folder is read-only
const UPLOADS_DIR = path.join(OUTPUTS_DIR, 'Uploads')                   // web uploads (deleted after a successful run)
const STALE_DAYS  = 7

// ── Settings ──────────────────────────────────────────────────────────────────
const DEFAULT_SETTINGS = {
  hfToken: '',
  keepWorkFiles: false,
  web: { enabled: true, port: 8360, pin: '' },
}
function loadSettings() {
  let s = {}
  try { s = JSON.parse(fs.readFileSync(SETTINGS, 'utf8')) } catch {}
  return { ...DEFAULT_SETTINGS, ...s, web: { ...DEFAULT_SETTINGS.web, ...(s.web || {}) } }
}
// The web UI always needs a PIN; one is generated if none is set.
function ensurePin() {
  const s = loadSettings()
  if (s.web.pin) return s
  return saveSettings({ web: { pin: String(crypto.randomInt(100000, 1000000)) } })
}
function saveSettings(patch) {
  const cur = loadSettings()
  const next = { ...cur, ...patch, web: { ...cur.web, ...(patch.web || {}) } }
  fs.mkdirSync(APP_DATA, { recursive: true })
  fs.writeFileSync(SETTINGS, JSON.stringify(next, null, 2))
  return next
}

// ── Window ────────────────────────────────────────────────────────────────────
let win

function createWindow() {
  const iconPath = path.join(IS_PKG ? process.resourcesPath : __dirname, 'resources', 'icon.icns')
  win = new BrowserWindow({
    width: 980, height: 740,
    minWidth: 780, minHeight: 600,
    titleBarStyle: 'hiddenInset',
    trafficLightPosition: { x: 16, y: 16 },
    backgroundColor: '#161616',
    icon: fs.existsSync(iconPath) ? iconPath : undefined,
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false,
    },
  })
  win.on('closed', () => { win = null })
  win.loadFile(path.join(__dirname, 'renderer', 'index.html'))
}

app.whenReady().then(() => {
  pruneStale()
  createWindow()
  startWeb()
})
app.on('activate', () => { if (!win) createWindow() })
// While the web UI is serving the LAN, closing the window keeps the app (and
// any running job) alive; it quits from the Dock / ⌘Q as usual.
app.on('window-all-closed', () => { if (!web) app.quit() })
app.on('before-quit', () => {
  for (const j of jobs) if (j.status === 'running') abortJob(j)
})

// Push an event to the desktop window (if open) and every web client.
function toWindow(ch, data) {
  if (win && !win.isDestroyed()) win.webContents.send(ch, data)
}

// ── Hardware info ─────────────────────────────────────────────────────────────
ipcMain.handle('hardware:info', () => {
  const isARM = process.arch === 'arm64'
  let cpu   = isARM ? 'Apple Silicon' : 'Intel'
  let cores = os.cpus().length
  let ramGB = Math.round(os.totalmem() / 1024 ** 3)
  try {
    if (!isARM) cpu = execSync('sysctl -n machdep.cpu.brand_string').toString().trim()
    cores = parseInt(execSync('sysctl -n hw.physicalcpu').toString())
    ramGB = Math.round(parseInt(execSync('sysctl -n hw.memsize').toString()) / 1024 ** 3)
  } catch (_) {}
  const perfEst = isARM
    ? `~${ramGB >= 32 ? 30 : ramGB >= 16 ? 40 : 50} min / 30-min video`
    : '~60+ min / 30-min video (Intel)'
  return { cpu, cores, ramGB, isARM, hasVideoToolbox: true, perfEst }
})

// ── First-time setup ──────────────────────────────────────────────────────────
// Self-check of the bundled runtime: fail at launch with a clear message rather
// than halfway through a 30-minute job.
function bundleSelfCheck() {
  const problems = []
  const env = { HOME: os.homedir(), PATH: `${BUNDLED.bin}:/usr/bin:/bin`,
                PYANNOTE_CACHE: BUNDLED.pyannote, HF_HUB_OFFLINE: '1', PYTHONWARNINGS: 'ignore',
                PYTHONDONTWRITEBYTECODE: '1' }
  try {
    execSync(`"${BUNDLED.python}" -c "import torch, pyannote.audio, mediapipe, cv2, scipy, numpy"`,
      { timeout: 60000, env, stdio: 'pipe' })
  } catch (e) { problems.push('bundled Python packages failed to load') }
  try {
    const enc = execSync(`"${BUNDLED.ffmpeg}" -hide_banner -encoders`, { timeout: 20000, stdio: 'pipe' }).toString()
    if (!enc.includes('hevc_videotoolbox')) problems.push('bundled ffmpeg lacks VideoToolbox HEVC')
  } catch (e) { problems.push('bundled ffmpeg does not run') }
  if (!fs.existsSync(BUNDLED.landmarker)) problems.push('face model missing')
  for (const m of ['speaker-diarization-3.1', 'segmentation-3.0', 'wespeaker-voxceleb-resnet34-LM']) {
    if (!fs.existsSync(path.join(BUNDLED.pyannote, `models--pyannote--${m}`))) problems.push(`model ${m} missing`)
  }
  if (process.arch !== 'arm64') problems.push('this build needs an Apple Silicon Mac (M1 or newer)')
  return problems
}

ipcMain.handle('setup:check', () => {
  if (HAS_BUNDLE) {
    const problems = bundleSelfCheck()
    return problems.length
      ? { ready: false, bundled: true, reason: problems.join('; ') }
      : { ready: true, bundled: true }
  }
  // Re-evaluate in case the project venv appeared since app start
  const bestVenv = findReadyVenv()
  if (bestVenv !== VENV_DIR) refreshVenvPaths(bestVenv)

  if (!fs.existsSync(PYTHON_BIN)) return { ready: false, reason: 'venv not created' }
  try {
    execSync(`"${PYTHON_BIN}" -c "import pyannote.audio, mediapipe, cv2, scipy, torch"`,
      { timeout: 20000 })
    return { ready: true }
  } catch (_) {
    return { ready: false, reason: 'packages missing' }
  }
})

ipcMain.handle('setup:install', (event) => new Promise((resolve) => {
  const send = (msg) => event.sender.send('setup:log', String(msg).trim())

  const candidates = [
    '/opt/homebrew/bin/python3',
    '/usr/local/bin/python3',
    '/usr/bin/python3',
  ]
  let sysPython = null
  for (const p of candidates) { if (fs.existsSync(p)) { sysPython = p; break } }
  if (!sysPython) {
    try { sysPython = execSync('which python3').toString().trim() } catch (_) {}
  }
  if (!sysPython) {
    return resolve({ success: false, error: 'Python 3 not found. Install via: brew install python3' })
  }
  send(`Found Python: ${sysPython}`)

  const targetVenv = path.join(APP_DATA, 'venv')
  send('Creating virtual environment…')
  try { execSync(`"${sysPython}" -m venv "${targetVenv}"`) }
  catch (e) { return resolve({ success: false, error: `venv failed: ${e.message}` }) }
  refreshVenvPaths(targetVenv)

  send('Installing packages — this takes 5–15 minutes on first run…')
  const venvBin = path.join(VENV_DIR, 'bin')
  const pip = spawn(PIP_BIN, ['install', '-r', REQS], {
    env: { ...process.env, PATH: `${venvBin}:${process.env.PATH}` },
  })
  const flush = (d) => d.toString().split('\n').filter(l => l.trim()).forEach(send)
  pip.stdout.on('data', flush)
  pip.stderr.on('data', flush)
  pip.on('close', (code) => {
    if (code !== 0) return resolve({ success: false, error: `pip exited ${code}` })

    send('Downloading MediaPipe face landmark model (3.6 MB)…')
    fs.mkdirSync(MODELS_DIR, { recursive: true })
    const modelPath = path.join(MODELS_DIR, 'face_landmarker.task')
    if (fs.existsSync(modelPath)) {
      send('Model already present.')
      return resolve({ success: true })
    }
    const dlScript = [
      'import urllib.request',
      `url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"`,
      `out = ${JSON.stringify(modelPath)}`,
      'def prog(b,bs,ts): pct=min(100,int(b*bs*100/ts)) if ts>0 else 0; print(f"  {pct}%",flush=True)',
      'urllib.request.urlretrieve(url, out, prog)',
      'print("Model downloaded.")',
    ].join('\n')
    const dl = spawn(PYTHON_BIN, ['-c', dlScript])
    dl.stdout.on('data', flush)
    dl.stderr.on('data', flush)
    dl.on('close', (c) => resolve(c === 0
      ? { success: true }
      : { success: false, error: 'Failed to download face_landmarker model' }))
  })
}))

ipcMain.handle('settings:load', () => loadSettings())
ipcMain.handle('settings:save', (_, patch) => {
  const before = loadSettings().web
  const s = saveSettings(patch)
  if (JSON.stringify(before) !== JSON.stringify(s.web)) startWeb()
  return s
})

// ── File dialogs ──────────────────────────────────────────────────────────────
ipcMain.handle('dialog:video', async () => {
  const r = await dialog.showOpenDialog(win, {
    title: 'Select 360° Video',
    filters: [{ name: 'Video Files', extensions: ['insv', 'mp4', 'mov'] }],
    properties: ['openFile'],
  })
  return r.canceled ? null : r.filePaths[0]
})

ipcMain.handle('dialog:audio', async () => {
  const r = await dialog.showOpenDialog(win, {
    title: 'Select External Mic Recording (optional)',
    filters: [{ name: 'Audio Files', extensions: ['mp3', 'wav', 'm4a', 'aac'] }],
    properties: ['openFile'],
  })
  return r.canceled ? null : r.filePaths[0]
})

ipcMain.on('shell:open', (_, p) => shell.showItemInFolder(p))
ipcMain.on('shell:url',  (_, u) => { if (/^https?:\/\//.test(u)) shell.openExternal(u) })

// ── Job directories ───────────────────────────────────────────────────────────
// One work dir per source file (keyed by its absolute path), so an interrupted
// job resumes where it stopped.
function jobDirFor(videoPath) {
  const base = path.basename(videoPath, path.extname(videoPath))
  const key  = crypto.createHash('sha1').update(path.resolve(videoPath)).digest('hex').slice(0, 8)
  return path.join(WORK_ROOT, `${base}_${key}`)
}

function hasFrames(jobDir) {
  for (const sub of ['front', 'back', 'equirect']) {
    const d = path.join(jobDir, 'frames', sub)
    try { if (fs.readdirSync(d).some(f => f.endsWith('.jpg'))) return true } catch (_) {}
  }
  return false
}

function completedSteps(jobDir) {
  const d = (f) => fs.existsSync(path.join(jobDir, f))
  return {
    audio:    d('data/audio.wav'),
    sync:     d('data/external_audio_aligned.wav'),
    extract:  hasFrames(jobDir) && d('frames/meta.json'),
    // geometry-v2 outputs: older job dirs lack these, so those steps re-run
    detect:   d('data/speaker_positions.json') && d('data/face_tracks.json'),
    presence: d('data/presence.json'),
    diarize:  d('data/diarization.json'),
    automap:  d('data/speaker_map.json') && d('data/speaker_probs.json'),
    render:   d('output/final.mp4'),
  }
}

function clearJobDir(videoPath) {
  const jobDir = jobDirFor(videoPath)
  if (fs.existsSync(jobDir)) fs.rmSync(jobDir, { recursive: true, force: true })
}

// Work dirs and uploads untouched for a week belong to abandoned jobs.
function pruneStale() {
  const cutoff = Date.now() - STALE_DAYS * 86400e3
  for (const root of [WORK_ROOT, UPLOADS_DIR]) {
    let names = []
    try { names = fs.readdirSync(root) } catch { continue }
    for (const n of names) {
      const p = path.join(root, n)
      try { if (fs.statSync(p).mtimeMs < cutoff) fs.rmSync(p, { recursive: true, force: true }) } catch {}
    }
  }
}

function isWritableDir(dir) {
  try { fs.accessSync(dir, fs.constants.W_OK); return true } catch { return false }
}

// Where the finished video goes: next to the source if that folder is writable
// (uploads always go to OUTPUTS_DIR). Full renders never overwrite an existing
// file; previews replace the previous preview.
function finalOutputPath(job) {
  const base = path.basename(job.videoPath, path.extname(job.videoPath))
  const srcDir = path.dirname(job.videoPath)
  const dir = !job.uploadDir && isWritableDir(srcDir) ? srcDir : OUTPUTS_DIR
  fs.mkdirSync(dir, { recursive: true })
  const stem = `${base}_autocam${job.previewMode ? '_preview' : ''}`
  let out = path.join(dir, `${stem}.mp4`)
  for (let n = 2; !job.previewMode && fs.existsSync(out); n++) out = path.join(dir, `${stem} (${n}).mp4`)
  return out
}

async function moveFile(src, dst) {
  try { await fs.promises.rename(src, dst) }
  catch (e) {
    if (e.code !== 'EXDEV') throw e
    await fs.promises.copyFile(src, dst)   // different volume (e.g. network share)
    await fs.promises.unlink(src)
  }
}

ipcMain.handle('job:scan', (_, videoPath) => {
  const jobDir = jobDirFor(videoPath)
  if (!fs.existsSync(jobDir)) return { jobDir: null, completed: {} }
  return { jobDir, completed: completedSteps(jobDir) }
})

ipcMain.handle('job:clear', (_, videoPath) => clearJobDir(videoPath))

// ── Jobs ──────────────────────────────────────────────────────────────────────
// Shared by the desktop window and the LAN web UI. Exactly one render at a
// time (see enqueueJob): a render already saturates the machine.
const jobs = []
let jobSeq = 0
let sleepBlock = null
const LOG_TAIL = 300

function publicJob(j) {
  return {
    id: j.id, name: path.basename(j.videoPath), audioName: j.audioPath ? path.basename(j.audioPath) : null,
    source: j.source, status: j.status, previewMode: j.previewMode, videoMode: j.videoMode,
    steps: j.steps, stepStates: j.stepStates, error: j.error,
    outputName: j.outputPath ? path.basename(j.outputPath) : null,
    outputPath: j.source === 'desktop' ? j.outputPath : undefined,
    createdAt: j.createdAt, startedAt: j.startedAt, finishedAt: j.finishedAt,
  }
}

function publishJobs() {
  const list = jobs.map(publicJob)
  toWindow('jobs:update', list)
  if (web) web.publish('jobs', list)
}

function enqueueJob({ videoPath, audioPath, videoMode, previewMode, source, uploadDir }) {
  if (!videoPath || !fs.existsSync(videoPath)) throw new Error(`Video not found: ${videoPath}`)
  if (audioPath && !fs.existsSync(audioPath)) throw new Error(`Audio not found: ${audioPath}`)
  // ONE render at a time, no queue: while a job runs, every new request (web
  // or this window) is refused, so users can't stack renders on the machine
  const active = jobs.find(j => j.status === 'queued' || j.status === 'running')
  if (active) {
    const mins = Math.round((Date.now() - (active.startedAt || active.createdAt)) / 60000)
    throw new Error(`Busy: already rendering ${path.basename(active.videoPath)}` +
      ` (started ${mins} min ago from ${active.source === 'web' ? 'the web' : 'the desktop app'}).` +
      ' Try again when it has finished.')
  }
  const job = {
    id: `j${++jobSeq}`, videoPath, audioPath: audioPath || null,
    videoMode: videoMode === 'equirect' ? 'equirect' : 'fisheye', previewMode: !!previewMode,
    source: source || 'desktop', uploadDir: uploadDir || null,
    status: 'queued', steps: [], stepStates: {}, log: [], error: null, outputPath: null,
    createdAt: Date.now(), startedAt: null, finishedAt: null, aborted: false, procs: new Set(),
  }
  jobs.push(job)
  // keep the finished history short
  const finished = jobs.filter(j => !['queued', 'running'].includes(j.status))
  for (const j of finished.slice(0, Math.max(0, finished.length - 30))) jobs.splice(jobs.indexOf(j), 1)
  publishJobs()
  setImmediate(pump)
  return { jobId: job.id, ahead: jobs.filter(j => j !== job && (j.status === 'queued' || j.status === 'running')).length }
}

function abortJob(job) {
  job.aborted = true
  for (const proc of job.procs) {
    try { process.kill(-proc.pid, 'SIGTERM') } catch (_) { try { proc.kill('SIGTERM') } catch (_) {} }
  }
  job.procs.clear()
}

function cancelJob(id) {
  const job = jobs.find(j => j.id === id)
  if (!job) return false
  if (job.status === 'queued') {
    job.status = 'stopped'
    job.finishedAt = Date.now()
    publishJobs()
    return true
  }
  if (job.status === 'running') { abortJob(job); return true }
  return false
}

async function pump() {
  if (jobs.some(j => j.status === 'running')) return
  const job = jobs.find(j => j.status === 'queued')
  if (!job) {
    if (sleepBlock !== null) { powerSaveBlocker.stop(sleepBlock); sleepBlock = null }
    return
  }
  // a render takes ~30+ min: keep the Mac from sleeping mid-job
  if (sleepBlock === null) sleepBlock = powerSaveBlocker.start('prevent-app-suspension')
  job.status = 'running'
  job.startedAt = Date.now()
  publishJobs()
  toWindow('pipeline:started', publicJob(job))
  try {
    await runJob(job)
    if (job.aborted) {
      job.status = 'stopped'
      toWindow('pipeline:aborted', { jobId: job.id })
    } else {
      job.status = 'done'
      toWindow('pipeline:done', { jobId: job.id, outputPath: job.outputPath })
    }
  } catch (err) {
    if (job.aborted) {
      job.status = 'stopped'
      toWindow('pipeline:aborted', { jobId: job.id })
    } else {
      job.status = 'error'
      job.error = err.message
      toWindow('pipeline:error', err.message)
    }
  }
  for (const st of Object.values(job.stepStates)) if (st.status === 'running') st.status = 'stopped'
  job.finishedAt = Date.now()
  publishJobs()
  pump()
}

ipcMain.handle('pipeline:run', (_, args) => {
  try { return enqueueJob({ ...args, source: 'desktop' }) }
  catch (e) { return { error: e.message } }
})
ipcMain.on('pipeline:stop', () => {
  const job = jobs.find(j => j.status === 'running')
  if (job) abortJob(job)
})
ipcMain.handle('jobs:list', () => jobs.map(publicJob))
ipcMain.handle('jobs:cancel', (_, id) => cancelJob(id))

// ── Pipeline ──────────────────────────────────────────────────────────────────
function spawnScript(job, script, args, opts, onLine) {
  return new Promise((resolve, reject) => {
    // own process group, so Stop also reaches the ffmpeg workers a script starts.
    // nice 10: everything else on the Mac (live streaming, audio) wins the CPU;
    // the pipeline only uses what is left. ffmpeg children inherit it.
    const proc = spawn('/usr/bin/nice', ['-n', '10', PYTHON_BIN, path.join(SCRIPTS_DIR, script), ...args],
                       { ...opts, detached: true })
    job.procs.add(proc)
    const flush = (d) => d.toString().split(/\r?\n|\r/)
      .filter(l => l.trim())
      .forEach(l => onLine(l.trim()))
    proc.stdout.on('data', flush)
    proc.stderr.on('data', flush)
    proc.on('close', (code) => {
      job.procs.delete(proc)
      if (job.aborted || code === 0) resolve()
      else reject(new Error(`${script} exited with code ${code}`))
    })
    proc.on('error', (err) => { job.procs.delete(proc); reject(err) })
  })
}

async function runJob(job) {
  const { videoPath, audioPath, videoMode, previewMode } = job
  const log = (line) => {
    job.log.push(line)
    if (job.log.length > LOG_TAIL) job.log.shift()
    toWindow('pipeline:log', line)
    if (web) web.publish('log', { jobId: job.id, line })
  }
  const setStep = (id, status, pct) => {
    job.stepStates[id] = { status, pct: pct ?? job.stepStates[id]?.pct ?? 0 }
    toWindow('pipeline:step', { id, status, pct })
    if (web) web.publish('step', { jobId: job.id, id, status, pct })
  }

  const jobDir = jobDirFor(videoPath)
  for (const d of ['data', 'frames', 'output']) {
    fs.mkdirSync(path.join(jobDir, d), { recursive: true })
  }
  log(`Working folder: ${jobDir}`)

  const venvBin = path.join(VENV_DIR, 'bin')
  // Resolve face_landmarker model: bundle, userData/models, then project data/
  const modelCandidates = [
    BUNDLED.landmarker,
    path.join(MODELS_DIR, 'face_landmarker.task'),
    path.join(SCRIPTS_DIR, 'data', 'face_landmarker.task'),
  ]
  const modelPath = modelCandidates.find(fs.existsSync) || modelCandidates[0]
  // Strip HF_TOKEN from inherited env to ensure only the settings value is used
  const { HF_TOKEN: _hft, ...inheritedEnv } = process.env
  const env = {
    ...inheritedEnv,
    HF_TOKEN:              loadSettings().hfToken || '',
    VIDEO_MODE:            videoMode || 'fisheye',
    FACE_LANDMARKER_MODEL: modelPath,
    PYTHONWARNINGS:        'ignore',
    PYTHONUNBUFFERED:      '1',        // force immediate stdout/stderr flush (required for progress bars)
    PYTHONDONTWRITEBYTECODE: '1',      // never write __pycache__ into the (signed) app bundle
    GLOG_minloglevel:      '2',        // suppress MediaPipe C++ INFO/WARNING noise
    // bundled: only the app's own python/ffmpeg + system dirs — never Homebrew
    PATH: HAS_BUNDLE
      ? `${BUNDLED.bin}:${venvBin}:/usr/bin:/bin:/usr/sbin:/sbin`
      : `${venvBin}:/opt/homebrew/bin:/usr/local/bin:${process.env.PATH}`,
    // bundled pyannote weights → no Hugging Face token or network needed
    ...(HAS_BUNDLE ? { PYANNOTE_CACHE: BUNDLED.pyannote, HF_HUB_OFFLINE: '1' } : {}),
  }
  const opts = { cwd: jobDir, env }

  const hasAudio = !!(audioPath)
  const renderOut = previewMode ? 'output/preview.mp4' : 'output/final.mp4'
  // AUDIO-FIRST ORDER: extract camera audio and sync the external mic before any
  // frame work, so an audio problem fails in seconds instead of after minutes of
  // frame extraction + face detection.
  const steps = [
    { id: 'audio',     label: 'Extract Audio',    script: '0_extract_audio.py',
      args: [videoPath] },
    hasAudio ? { id: 'sync', label: 'Sync Audio', script: '5_sync_audio.py',
      args: [videoPath, audioPath] } : null,
    { id: 'extract',   label: 'Extract Frames',   script: '1_extract_frames.py',
      args: [videoPath, '2'] },
    { id: 'detect',    label: 'Detect Speakers',  script: '2_detect_speakers.py',
      args: [] },
    { id: 'presence',  label: 'Detect Presence',  script: '2b_detect_presence.py',
      args: [] },
    // Diarization + mapping ALWAYS use the camera audio (synced to the video,
    // cleaner than a studio feed). The external studio mic is used ONLY as the
    // final audio overlay in render — never for analysis.
    { id: 'diarize',   label: 'Diarize',          script: '3_diarize.py',
      args: [videoPath, ...(previewMode ? ['--end-time', '60'] : [])] },
    { id: 'automap',   label: 'Map Speakers',     script: '3_automap.py',
      args: [videoPath, ...(previewMode ? ['--end-time', '60'] : [])] },
    { id: 'render',    label: previewMode ? 'Render (60s preview)' : 'Render',
      script: '4_render.py',
      args: [
        videoPath, renderOut, ...(previewMode ? ['--end-time', '60'] : []),
        // Overlay the synced studio mic as the final audio track (better sound).
        ...(hasAudio ? ['--external-audio', 'data/external_audio_aligned.wav'] : []),
      ] },
  ].filter(Boolean)

  job.steps = steps.map(s => ({ id: s.id, label: s.label }))
  for (const s of steps) job.stepStates[s.id] = { status: 'pending', pct: 0 }
  toWindow('pipeline:steps', job.steps)
  publishJobs()

  const done = completedSteps(jobDir)
  if (previewMode) done.render  = false  // always re-render in preview mode
  if (previewMode) done.automap = false  // preview map must not reuse full-video map
  // Cascade: if a prerequisite is not done, downstream steps must re-run.
  // Camera audio drives diarization+mapping; the synced studio mic only feeds the
  // render's final audio overlay.
  if (!done.audio)    done.sync    = false   // sync needs the camera audio reference
  if (!done.audio)    done.diarize = false   // diarize uses the camera audio
  if (!done.extract)  done.detect  = false   // seats + face tracks come from the frames
  if (!done.diarize)  done.automap = false
  if (!done.detect)   done.automap = false
  if (!done.detect)   done.presence = false   // presence is built from the face tracks
  if (!done.presence) done.automap = false    // mapping gates speakers on presence
  if (!done.automap)  done.render  = false
  if (hasAudio && !done.sync) done.render = false  // render overlays the synced mic
  if (!done.presence) done.render  = false    // render consumes presence to gate empty chairs

  // ── Step runner ──────────────────────────────────────────────────────────────
  async function runStep(step) {
    if (!step || job.aborted) return
    if (done[step.id]) {
      log(`Skipping ${step.id} — output already exists`)
      setStep(step.id, 'done', 100)
      return
    }
    setStep(step.id, 'running', 0)
    try { await spawnScript(job, step.script, step.args, opts, (line) => {
      if (/^[WIE]\d{4}\s/.test(line)) return  // MediaPipe C++ noise

      const progM = line.match(/^PROGRESS\s+(\d+)\/(\d+)/)
      if (progM) {
        setStep(step.id, 'running', Math.round(parseInt(progM[1]) / parseInt(progM[2]) * 100))
        return
      }
      const tqdmM = line.match(/^\s*(\d{1,3})%\|/)
      if (tqdmM) {
        setStep(step.id, 'running', parseInt(tqdmM[1]))
        return
      }
      if (step.id === 'render') {
        const segM = line.match(/^\[(\d+)\/(\d+)\]/)
        if (segM) {
          setStep(step.id, 'running', Math.round(parseInt(segM[1]) / parseInt(segM[2]) * 100))
          return
        }
      }
      log(line)
    }) } catch (err) {
      setStep(step.id, 'error')
      throw err
    }
    if (job.aborted) return  // killed steps must not be marked done
    setStep(step.id, 'done', 100)
  }

  // ── Execution ────────────────────────────────────────────────────────────────
  // Steps are organised into phases; detect + diarize run in parallel after extract.
  const stepMap = Object.fromEntries(steps.map(s => [s.id, s]))

  // Phase 0 — camera audio first: an audio problem fails in seconds
  await runStep(stepMap['audio'])

  // Phase 1 — extract frames (serial prerequisite for everything visual)
  await runStep(stepMap['extract'])

  // Phase 2 — detect → presence, in parallel with (optional sync then) diarize
  if (!job.aborted) {
    const detectChain  = async () => {
      await runStep(stepMap['detect'])
      await runStep(stepMap['presence'])
    }
    const diarizeChain = async () => {
      await runStep(stepMap['sync'])
      await runStep(stepMap['diarize'])
    }
    const results = await Promise.allSettled([detectChain(), diarizeChain()])
    // Surface ALL failures (both detect and diarize can fail simultaneously)
    const failed = results.filter(r => r.status === 'rejected')
    if (failed.length) throw new Error(failed.map(r => r.reason?.message ?? String(r.reason)).join(' | '))
  }

  // Phase 3 — automap then render (serial)
  await runStep(stepMap['automap'])
  await runStep(stepMap['render'])
  if (job.aborted) return

  // ── Deliver + clean up ───────────────────────────────────────────────────────
  const rendered = path.join(jobDir, renderOut)
  if (!fs.existsSync(rendered)) throw new Error(`render finished but ${renderOut} is missing`)
  job.outputPath = finalOutputPath(job)
  await moveFile(rendered, job.outputPath)
  log(`Saved ${job.outputPath}`)

  if (previewMode) {
    // the analysis is reused by the full run, so a preview keeps its work folder
    log('Preview: working files kept so the full render can reuse the analysis')
  } else if (loadSettings().keepWorkFiles) {
    log(`Working files kept (setting): ${jobDir}`)
  } else {
    await fs.promises.rm(jobDir, { recursive: true, force: true })
    if (job.uploadDir) await fs.promises.rm(job.uploadDir, { recursive: true, force: true })
    log('Working files removed')
  }
}

// ── LAN web UI ────────────────────────────────────────────────────────────────
let web = null

function lanAddresses(port) {
  const urls = []
  try {
    const host = execSync('scutil --get LocalHostName', { stdio: 'pipe' }).toString().trim()
    if (host) urls.push(`http://${host}.local:${port}`)
  } catch {}
  for (const addrs of Object.values(os.networkInterfaces())) {
    for (const a of addrs || []) {
      if (a.family === 'IPv4' && !a.internal) urls.push(`http://${a.address}:${port}`)
    }
  }
  return urls
}

function webStatus() {
  const s = loadSettings().web
  if (!s.enabled) return { enabled: false }
  if (!web) return { enabled: true, error: webError }
  return { enabled: true, port: web.port, urls: lanAddresses(web.port), pin: s.pin }
}

let webError = null
async function startWeb() {
  if (web) { await web.close(); web = null }
  webError = null
  const s = ensurePin().web
  if (s.enabled) {
    try {
      web = await createWebServer({
        port: s.port, pin: s.pin,
        roots: [os.homedir(), '/Volumes'],
        uploadsDir: UPLOADS_DIR,
        listJobs: () => jobs.map(j => ({ ...publicJob(j), log: j.status === 'running' ? j.log : undefined })),
        enqueue: (args) => enqueueJob({ ...args, source: 'web' }),
        cancel: cancelJob,
        outputOf: (id) => {
          const j = jobs.find(x => x.id === id)
          return j && j.status === 'done' && j.outputPath && fs.existsSync(j.outputPath) ? j.outputPath : null
        },
        busy: () => jobs.some(j => j.status === 'running'),
      })
    } catch (e) {
      webError = e.code === 'EADDRINUSE' ? `port ${s.port} is already in use` : e.message
    }
  }
  toWindow('web:status', webStatus())
}
ipcMain.handle('web:status', () => webStatus())
