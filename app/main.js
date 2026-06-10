'use strict'
const { app, BrowserWindow, ipcMain, dialog, shell } = require('electron')
const { execSync, spawn } = require('child_process')
const path = require('path')
const fs   = require('fs')
const os   = require('os')

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

// Prefer the project venv when running in dev (already has all packages).
// Fall back to the app-managed venv in userData for packaged distribution.
const PROJECT_VENV = IS_PKG ? null : path.resolve(__dirname, '..', 'venv')
function findReadyVenv() {
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

// ── Window ────────────────────────────────────────────────────────────────────
let win

app.whenReady().then(() => {
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
  win.loadFile(path.join(__dirname, 'renderer', 'index.html'))
})
app.on('window-all-closed', () => app.quit())

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

// ── Settings ──────────────────────────────────────────────────────────────────
ipcMain.handle('settings:load', () => {
  try { return JSON.parse(fs.readFileSync(SETTINGS, 'utf8')) } catch { return {} }
})
ipcMain.handle('settings:save', (_, s) => {
  fs.mkdirSync(APP_DATA, { recursive: true })
  fs.writeFileSync(SETTINGS, JSON.stringify(s, null, 2))
  return true
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

// ── First-time setup ──────────────────────────────────────────────────────────
ipcMain.handle('setup:check', () => {
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

// ── Pipeline ──────────────────────────────────────────────────────────────────
const currentProcesses = new Set()  // all live child processes (parallel steps possible)
let pipelineAborted = false

ipcMain.on('pipeline:stop', () => {
  pipelineAborted = true
  for (const proc of currentProcesses) {
    try { proc.kill('SIGTERM') } catch (_) {}
  }
  currentProcesses.clear()
  if (win) win.webContents.send('pipeline:aborted')
})

ipcMain.on('shell:open', (_, p) => shell.showItemInFolder(p))

// ── Job scanning (resume support) ─────────────────────────────────────────────
function jobDirFor(videoPath) {
  const base = path.basename(videoPath, path.extname(videoPath))
  return path.join(path.dirname(videoPath), `autocam_${base}`)
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
    extract:  hasFrames(jobDir),
    detect:   d('data/speaker_positions.json'),
    presence: d('data/presence.json'),
    diarize:  d('data/diarization.json'),
    automap:  d('data/speaker_map.json'),
    render:   d('output/final.mp4'),
  }
}

ipcMain.handle('job:scan', (_, videoPath) => {
  const jobDir = jobDirFor(videoPath)
  if (!fs.existsSync(jobDir)) return { jobDir: null, completed: {} }
  return { jobDir, completed: completedSteps(jobDir) }
})

ipcMain.handle('job:clear', (_, videoPath) => {
  const jobDir = jobDirFor(videoPath)
  if (!fs.existsSync(jobDir)) return
  for (const sub of ['data', 'output']) {
    const d = path.join(jobDir, sub)
    if (fs.existsSync(d)) fs.rmSync(d, { recursive: true, force: true })
    fs.mkdirSync(d, { recursive: true })
  }
  // Recreate frames dir with per-lens subdirs so detect step never skips silently
  const framesDir = path.join(jobDir, 'frames')
  if (fs.existsSync(framesDir)) fs.rmSync(framesDir, { recursive: true, force: true })
  for (const sub of ['front', 'back', 'equirect']) {
    fs.mkdirSync(path.join(framesDir, sub), { recursive: true })
  }
})

ipcMain.handle('pipeline:run',  (event, args) => runPipeline(event, args))

function spawnScript(script, args, opts, onLine) {
  return new Promise((resolve, reject) => {
    const proc = spawn(PYTHON_BIN, [path.join(SCRIPTS_DIR, script), ...args], opts)
    currentProcesses.add(proc)
    const flush = (d) => d.toString().split(/\r?\n|\r/)
      .filter(l => l.trim())
      .forEach(l => onLine(l.trim()))
    proc.stdout.on('data', flush)
    proc.stderr.on('data', flush)
    proc.on('close', (code) => {
      currentProcesses.delete(proc)
      if (pipelineAborted || code === 0) resolve()
      else reject(new Error(`${script} exited with code ${code}`))
    })
    proc.on('error', (err) => { currentProcesses.delete(proc); reject(err) })
  })
}

async function runPipeline(event, { videoPath, audioPath, hfToken, videoMode, previewMode }) {
  const send = (ch, d) => event.sender.send(ch, d)
  pipelineAborted = false

  // Create job directory alongside the video
  const ext  = path.extname(videoPath)
  const base = path.basename(videoPath, ext)
  const jobDir = path.join(path.dirname(videoPath), `autocam_${base}`)
  for (const d of ['data', 'frames', 'output']) {
    fs.mkdirSync(path.join(jobDir, d), { recursive: true })
  }

  const venvBin = path.join(VENV_DIR, 'bin')
  // Resolve face_landmarker model: userData/models, then project data/, then SCRIPTS_DIR/data/
  const modelCandidates = [
    path.join(MODELS_DIR, 'face_landmarker.task'),
    path.join(SCRIPTS_DIR, 'data', 'face_landmarker.task'),
  ]
  const modelPath = modelCandidates.find(fs.existsSync) || modelCandidates[0]
  // Strip HF_TOKEN from inherited env to ensure only the settings value is used
  const { HF_TOKEN: _hft, ...inheritedEnv } = process.env
  const env = {
    ...inheritedEnv,
    HF_TOKEN:              hfToken   || '',
    VIDEO_MODE:            videoMode || 'fisheye',
    FACE_LANDMARKER_MODEL: modelPath,
    PYTHONWARNINGS:        'ignore',
    PYTHONUNBUFFERED:      '1',        // force immediate stdout/stderr flush (required for progress bars)
    GLOG_minloglevel:      '2',        // suppress MediaPipe C++ INFO/WARNING noise
    PATH: `${venvBin}:/opt/homebrew/bin:/usr/local/bin:${process.env.PATH}`,
  }
  const opts = { cwd: jobDir, env }

  const hasAudio = !!(audioPath)
  // AUDIO-FIRST ORDER: extract camera audio and sync the external mic before any
  // frame work, so an audio problem fails in seconds instead of after minutes of
  // frame extraction + face detection.
  const steps = [
    { id: 'audio',     label: 'Extract Audio',    script: '0_extract_audio.py',
      args: [videoPath] },
    hasAudio ? { id: 'sync', label: 'Sync Audio', script: '5_sync_audio.py',
      args: [videoPath, audioPath] } : null,
    { id: 'extract',   label: 'Extract Frames',   script: '1_extract_frames.py',
      args: [videoPath, '10'] },
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
        ...(previewMode ? [videoPath, 'output/preview.mp4', '--end-time', '60'] : [videoPath]),
        // Overlay the synced studio mic as the final audio track (better sound).
        ...(hasAudio ? ['--external-audio', 'data/external_audio_aligned.wav'] : []),
      ] },
  ].filter(Boolean)

  const allStepDefs = steps.map(s => ({ id: s.id, label: s.label }))
  send('pipeline:steps', allStepDefs)

  const done = completedSteps(jobDir)
  if (previewMode) done.render  = false  // always re-render in preview mode
  if (previewMode) done.automap = false  // preview map must not reuse full-video map
  // Cascade: if a prerequisite is not done, downstream steps must re-run.
  // Camera audio drives diarization+mapping; the synced studio mic only feeds the
  // render's final audio overlay.
  if (!done.audio)    done.sync    = false   // sync needs the camera audio reference
  if (!done.audio)    done.diarize = false   // diarize uses the camera audio
  if (!done.diarize)  done.automap = false
  if (!done.detect)   done.automap = false
  if (!done.detect)   done.presence = false   // presence is built from detected positions
  if (!done.automap)  done.render  = false
  if (hasAudio && !done.sync) done.render = false  // render overlays the synced mic
  if (!done.presence) done.render  = false    // render consumes presence to gate empty chairs

  // ── Step runner ──────────────────────────────────────────────────────────────
  async function runStep(step) {
    if (pipelineAborted) return
    if (done[step.id]) {
      send('pipeline:log', `Skipping ${step.id} — output already exists`)
      send('pipeline:step', { id: step.id, status: 'done', pct: 100 })
      return
    }
    send('pipeline:step', { id: step.id, status: 'running', pct: 0 })
    try { await spawnScript(step.script, step.args, opts, (line) => {
      if (/^[WIE]\d{4}\s/.test(line)) return  // MediaPipe C++ noise

      const progM = line.match(/^PROGRESS\s+(\d+)\/(\d+)/)
      if (progM) {
        send('pipeline:step', { id: step.id, status: 'running',
          pct: Math.round(parseInt(progM[1]) / parseInt(progM[2]) * 100) })
        return
      }
      const tqdmM = line.match(/^\s*(\d{1,3})%\|/)
      if (tqdmM) {
        send('pipeline:step', { id: step.id, status: 'running', pct: parseInt(tqdmM[1]) })
        return
      }
      if (step.id === 'render') {
        const segM = line.match(/^\[(\d+)\/(\d+)\]/)
        if (segM) {
          send('pipeline:step', { id: step.id, status: 'running',
            pct: Math.round(parseInt(segM[1]) / parseInt(segM[2]) * 100) })
          return
        }
      }
      send('pipeline:log', line)
    }) } catch (err) {
      send('pipeline:step', { id: step.id, status: 'error' })
      throw err
    }
    if (pipelineAborted) return  // killed steps must not be marked done
    send('pipeline:step', { id: step.id, status: 'done', pct: 100 })
  }

  // ── Execution ────────────────────────────────────────────────────────────────
  // Steps are organised into phases; detect + diarize run in parallel after extract.
  const stepMap = Object.fromEntries(steps.map(s => [s.id, s]))

  try {
    // Phase 1 — extract frames (serial prerequisite for everything)
    await runStep(stepMap['extract'])

    // Phase 2 — detect + (optional sync then) diarize in parallel
    if (!pipelineAborted) {
      const detectChain  = () => runStep(stepMap['detect'])
      const diarizeChain = async () => {
        if (stepMap['sync']) await runStep(stepMap['sync'])
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
  } catch (err) {
    if (!pipelineAborted) {
      send('pipeline:error', err.message)
    }
    return
  }

  if (!pipelineAborted) {
    const out = path.join(jobDir, 'output', previewMode ? 'preview.mp4' : 'final.mp4')
    send('pipeline:done', { outputPath: fs.existsSync(out) ? out : jobDir })
  }
}
