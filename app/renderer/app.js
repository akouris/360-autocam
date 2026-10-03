'use strict'

// ── State ─────────────────────────────────────────────────────────────────────
let videoPath        = null
let audioPath        = null
let outputPath       = null
let running          = false
let logExpanded      = false
let videoMode        = 'fisheye'
let previewMode      = false
let resumeTimer      = null
let resumeCountdown  = 60
let pendingStates    = null   // step states waiting for resume decision

// Step definitions (ordered) — matches main.js dynamically, but we pre-populate
const STEP_DEFS = [
  { id: 'extract',  label: 'Extract Frames',   est: '~1 min'  },
  { id: 'detect',   label: 'Detect Speakers',  est: '~30 sec' },
  { id: 'sync',     label: 'Sync Audio',        est: '~2 min', optional: true },
  { id: 'diarize',  label: 'Diarize',           est: '~12 min' },
  { id: 'automap',  label: 'Map Speakers',      est: '~5 min'  },
  { id: 'render',   label: 'Render',            est: '~20 min' },
]

// ── DOM helpers ───────────────────────────────────────────────────────────────
const $  = (id) => document.getElementById(id)
const el = (tag, cls, text) => {
  const e = document.createElement(tag)
  if (cls)  e.className = cls
  if (text) e.textContent = text
  return e
}

// ── Mode toggle ───────────────────────────────────────────────────────────────
document.querySelectorAll('.mode-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    if (running) return
    videoMode = btn.dataset.mode
    document.querySelectorAll('.mode-btn').forEach(b => b.classList.remove('active'))
    btn.classList.add('active')
    const hint = $('dz-video-hint')
    if (hint) hint.textContent = videoMode === 'equirect'
      ? 'Click or drop .mp4 (YouTube 360°)'
      : 'Click or drop .insv / .mp4'
  })
})

// ── Hardware info ─────────────────────────────────────────────────────────────
async function loadHardware() {
  const hw = await window.api.hardware()
  $('hw-cpu').textContent = `${hw.cpu} · ${hw.cores}C`
  $('hw-ram').textContent = `${hw.ramGB} GB RAM`
  if (!hw.hasVideoToolbox) $('hw-vt').textContent = 'No VideoToolbox'
  $('hw-est').textContent = hw.perfEst
}

// ── Settings ──────────────────────────────────────────────────────────────────
async function loadSettings() {
  const s = await window.api.loadSettings()
  if (s.hfToken) $('hf-token').value = s.hfToken
}

$('hf-token').addEventListener('blur', () => {
  window.api.saveSettings({ hfToken: $('hf-token').value })
})

// ── File drop zones ───────────────────────────────────────────────────────────
function setupDropZone(dzId, type) {
  const dz = $(dzId)

  dz.addEventListener('click', async () => {
    if (running) return
    const p = type === 'video'
      ? await window.api.selectVideo()
      : await window.api.selectAudio()
    if (p) setFile(type, p)
  })

  dz.addEventListener('dragover', (e) => {
    e.preventDefault()
    dz.classList.add('drag-over')
  })
  dz.addEventListener('dragleave', () => dz.classList.remove('drag-over'))
  dz.addEventListener('drop', (e) => {
    e.preventDefault()
    dz.classList.remove('drag-over')
    const f = e.dataTransfer.files[0]
    if (f) setFile(type, f.path)
  })
}

async function setFile(type, p) {
  if (type === 'video') dismissResumeOverlay()  // cancel any in-progress timer from a prior video

  const name = p.split('/').pop()
  const dzId = type === 'video' ? 'dz-video' : 'dz-audio'
  const dz   = $(dzId)

  if (type === 'video') videoPath = p
  else                  audioPath = p

  dz.innerHTML = ''
  dz.classList.add('filled')

  const icon = el('span', 'dz-icon', type === 'video' ? '📹' : '🎙')
  const fname = el('span', 'dz-filename', name)
  const clear = el('span', 'dz-clear', '✕ Remove')
  clear.addEventListener('click', (e) => {
    e.stopPropagation()
    clearFile(type)
  })

  dz.appendChild(icon)
  dz.appendChild(fname)
  dz.appendChild(clear)

  updateRunButton()
  if (type === 'audio') renderSteps(lastStepStates)

  // Scan for existing job directory when video is selected
  if (type === 'video') {
    lastStepStates = {}
    renderSteps({})
    const scan = await window.api.scanJob(p)
    if (scan.jobDir) {
      const states = {}
      for (const [id, isDone] of Object.entries(scan.completed)) {
        if (isDone) states[id] = { status: 'done', pct: 100 }
      }
      const defs = STEP_DEFS.filter(s => !s.optional || audioPath)
      const anyDone = defs.some(s => states[s.id]?.status === 'done')
      if (anyDone) {
        showResumeOverlay(states)
      } else {
        lastStepStates = states
        renderSteps(states)
        $('btn-run').textContent = '▶ Run Pipeline'
      }
    } else {
      $('btn-run').textContent = '▶ Run Pipeline'
    }
  }
}

// ── Resume confirmation overlay ───────────────────────────────────────────────
function showResumeOverlay(states) {
  pendingStates = states
  const defs = STEP_DEFS.filter(s => !s.optional || audioPath)
  const firstPending = defs.find(s => states[s.id]?.status !== 'done')
  const resumeFrom   = firstPending ? firstPending.label : 'beginning'
  $('resume-desc').textContent =
    `Previous results detected. Will resume from "${resumeFrom}", or start all steps from scratch.`

  resumeCountdown = 60
  $('resume-countdown').textContent = resumeCountdown
  $('resume-bar-fill').style.width = '100%'
  $('resume-overlay').classList.remove('hidden')

  resumeTimer = setInterval(() => {
    resumeCountdown--
    $('resume-countdown').textContent = resumeCountdown
    $('resume-bar-fill').style.width = `${(resumeCountdown / 60) * 100}%`
    if (resumeCountdown <= 0) confirmResume()
  }, 1000)
}

function dismissResumeOverlay() {
  if (resumeTimer) { clearInterval(resumeTimer); resumeTimer = null }
  $('resume-overlay').classList.add('hidden')
}

function confirmResume() {
  dismissResumeOverlay()
  const states = pendingStates || {}
  pendingStates = null
  lastStepStates = states
  renderSteps(states)
  applyResumeLabel(states)
}

async function confirmFresh() {
  dismissResumeOverlay()
  pendingStates = null
  if (videoPath) await window.api.clearJob(videoPath)
  lastStepStates = {}
  renderSteps({})
  $('btn-run').textContent = '▶ Run Pipeline'
}

$('btn-resume-now').addEventListener('click', confirmResume)
$('btn-resume-fresh').addEventListener('click', confirmFresh)

function applyResumeLabel(states) {
  const defs = STEP_DEFS.filter(s => !s.optional || audioPath)
  const allDone = defs.every(s => states[s.id]?.status === 'done')
  if (allDone) {
    $('btn-run').textContent = '↺ Re-run Pipeline'
    return
  }
  const firstPending = defs.find(s => states[s.id]?.status !== 'done')
  if (firstPending) {
    $('btn-run').textContent = `▶ Resume from ${firstPending.label}`
  }
}

function clearFile(type) {
  if (type === 'video') {
    videoPath = null
    lastStepStates = {}
    renderSteps({})
    $('btn-run').textContent = '▶ Run Pipeline'
    const dz = $('dz-video')
    dz.classList.remove('filled')
    dz.innerHTML = '<span class="dz-icon">📹</span><span class="dz-hint" id="dz-video-hint">Click or drop .insv / .mp4</span>'
  } else {
    audioPath = null
    const dz = $('dz-audio')
    dz.classList.remove('filled')
    dz.innerHTML = '<span class="dz-icon">🎙</span><span class="dz-hint">Click or drop .mp3 / .wav</span>'
  }
  updateRunButton()
  renderSteps(lastStepStates)
}

setupDropZone('dz-video', 'video')
setupDropZone('dz-audio', 'audio')

// ── Steps rendering ───────────────────────────────────────────────────────────
let lastStepStates = {}
const stepStartTimes = {}   // stepId -> Date.now() when it started running
let elapsedTimer = null

function fmtElapsed(ms) {
  const s = Math.floor(ms / 1000)
  const m = Math.floor(s / 60)
  return m > 0 ? `${m}m ${s % 60}s` : `${s}s`
}

function updateProgressStrip() {
  const strip = $('progress-strip')
  const fill  = $('progress-bar-fill')
  const label = $('progress-label')

  // Find the currently running step; when detect+diarize run in parallel, prefer the later one
  const allRunning = STEP_DEFS.filter(s => lastStepStates[s.id]?.status === 'running')
  const runningDef = allRunning[allRunning.length - 1]
  if (!runningDef) {
    strip.classList.add('hidden')
    return
  }
  strip.classList.remove('hidden')
  const pct     = lastStepStates[runningDef.id]?.pct ?? 0
  const elapsed = stepStartTimes[runningDef.id] ? fmtElapsed(Date.now() - stepStartTimes[runningDef.id]) : '0s'

  fill.style.width = pct > 0 ? `${pct}%` : '0%'
  fill.classList.toggle('indeterminate', pct === 0)

  label.innerHTML =
    `<span class="prog-step">${runningDef.label}</span>` +
    (pct > 0 ? `<span class="prog-pct">${pct}%</span>` : `<span class="prog-pct">—</span>`) +
    `<span class="prog-time" id="strip-elapsed">${elapsed}</span>`
}

function startElapsedTicker() {
  if (elapsedTimer) return
  elapsedTimer = setInterval(() => {
    // Update strip elapsed
    const el = $('strip-elapsed')
    const runningId = Object.keys(stepStartTimes)[0]
    if (el && runningId) el.textContent = fmtElapsed(Date.now() - stepStartTimes[runningId])
  }, 1000)
}

function stopElapsedTicker() {
  if (elapsedTimer) { clearInterval(elapsedTimer); elapsedTimer = null }
  $('progress-strip').classList.add('hidden')
}

function renderSteps(states) {
  lastStepStates = states
  const list  = $('steps-list')
  const defs  = STEP_DEFS.filter(s => !s.optional || audioPath)
  list.innerHTML = ''

  let anyRunning = false

  for (const def of defs) {
    const state  = states[def.id] || {}
    const status = state.status || 'pending'
    const pct    = state.pct ?? 0

    if (status === 'running') {
      if (!stepStartTimes[def.id]) stepStartTimes[def.id] = Date.now()
      anyRunning = true
    } else {
      delete stepStartTimes[def.id]
    }

    const stepEl = el('div', `step ${status}`)
    stepEl.id = `step-${def.id}`

    // Icon
    const icon = el('div', `step-icon ${status}`)
    icon.textContent = status === 'done'    ? '✓'
                     : status === 'error'   ? '✕'
                     : status === 'running' ? '◉'
                     : status === 'skipped' ? '—'
                     :                        '○'
    stepEl.appendChild(icon)

    // Body
    const body  = el('div', 'step-body')
    const label = el('div', 'step-label', def.label)
    body.appendChild(label)

    // Progress bar (visible when running)
    const bar = el('div', `step-bar${status === 'running' && pct === 0 ? ' indeterminate' : ''}`)
    const fill = el('div', 'step-bar-fill')
    fill.style.width = `${pct}%`
    bar.appendChild(fill)
    body.appendChild(bar)


    stepEl.appendChild(body)

    // Estimate (pending only)
    const est = el('div', 'step-est', status === 'pending' ? def.est : '')
    stepEl.appendChild(est)

    list.appendChild(stepEl)
  }

  updateProgressStrip()
  if (anyRunning) startElapsedTicker()
  else stopElapsedTicker()
}

// Initial render
renderSteps({})

// ── Pipeline steps from main process ─────────────────────────────────────────
window.api.onPipelineSteps((steps) => {
  const states = {}
  steps.forEach(s => { states[s.id] = { status: 'pending' } })
  renderSteps(states)
})

window.api.onPipelineStep(({ id, status, pct }) => {
  lastStepStates[id] = { status, pct: pct ?? lastStepStates[id]?.pct ?? 0 }
  renderSteps(lastStepStates)
})

// ── Log output ───────────────────────────────────────────────────────────────
window.api.onPipelineLog((line) => appendLog(line))

function appendLog(line) {
  const scroll = $('log-scroll')
  const p = el('div', null, line)
  scroll.appendChild(p)
  scroll.scrollTop = scroll.scrollHeight
}

// Copy log
$('btn-copy-log').addEventListener('click', (e) => {
  e.stopPropagation()
  const text = $('log-scroll').innerText
  navigator.clipboard.writeText(text).then(() => {
    $('btn-copy-log').textContent = '✓ Copied'
    setTimeout(() => { $('btn-copy-log').textContent = '⎘ Copy' }, 2000)
  })
})

// Log collapse/expand
$('log-header').addEventListener('click', () => {
  logExpanded = !logExpanded
  const logArea = $('log-area')
  const toggle  = $('log-toggle')
  if (logExpanded) {
    logArea.classList.remove('collapsed')
    toggle.textContent = '▴ collapse'
  } else {
    logArea.classList.add('collapsed')
    toggle.textContent = '▾ expand'
  }
})

// ── Buttons ───────────────────────────────────────────────────────────────────
function updateRunButton() {
  $('btn-run').disabled = !videoPath || running
}

$('preview-mode').addEventListener('change', () => { previewMode = $('preview-mode').checked })

$('btn-run').addEventListener('click', startPipeline)
$('btn-stop').addEventListener('click', () => {
  window.api.stopPipeline()
  setStatus('Stopping…')
})
$('btn-open').addEventListener('click', () => {
  if (outputPath) window.api.openOutput(outputPath)
})

// ── Pipeline control ──────────────────────────────────────────────────────────
async function startPipeline() {
  if (!videoPath) return

  const hfToken = $('hf-token').value.trim()
  if (!hfToken && !bundledRuntime) {
    setStatus('⚠ Enter your Hugging Face token first')
    return
  }

  // Save token on run
  window.api.saveSettings({ hfToken })

  running = true
  outputPath = null
  $('btn-run').disabled  = true
  $('btn-stop').disabled = false
  $('btn-open').classList.remove('visible')
  $('log-scroll').innerHTML = ''
  lastStepStates = {}
  Object.keys(stepStartTimes).forEach(k => delete stepStartTimes[k])
  stopElapsedTicker()
  renderSteps({})
  setStatus('Running…')

  // Auto-expand log
  if (!logExpanded) $('log-header').click()

  const result = await window.api.runPipeline({ videoPath, audioPath, hfToken, videoMode, previewMode })
  // pipeline:done / pipeline:error / pipeline:aborted events handle completion
}

window.api.onPipelineDone(({ outputPath: out }) => {
  running    = false
  outputPath = out
  $('btn-run').disabled  = false
  $('btn-stop').disabled = true
  $('btn-run').textContent = '↺ Re-run Pipeline'
  $('btn-open').classList.add('visible')
  setStatus('✓ Done')
})

window.api.onPipelineError((msg) => {
  running = false
  $('btn-run').disabled  = false
  $('btn-stop').disabled = true
  setStatus(`✕ Error: ${msg}`)
  appendLog(`ERROR: ${msg}`)
})

window.api.onPipelineAborted(() => {
  running = false
  $('btn-run').disabled  = false
  $('btn-stop').disabled = true
  setStatus('Stopped')
})

function setStatus(msg) {
  $('status-text').textContent = msg
}

// ── First-time setup ──────────────────────────────────────────────────────────
let bundledRuntime = false   // self-contained app: models ship inside, no token needed

async function checkSetup() {
  const res = await window.api.checkSetup()
  bundledRuntime = !!res.bundled
  if (bundledRuntime) $('token-section').classList.add('hidden')
  if (res.ready) return
  showSetupOverlay()
  if (bundledRuntime) {
    // nothing to install — the app itself is incomplete or on the wrong Mac
    $('setup-card').querySelector('h2').textContent = 'This copy of 360 AutoCam can\'t run'
    $('setup-card').querySelector('p').textContent = res.reason
    $('btn-setup-start').classList.add('hidden')
    $('setup-progress').classList.add('hidden')
  }
}

function showSetupOverlay() {
  $('setup-overlay').classList.remove('hidden')
}

const setupLog = $('setup-log')
window.api.onSetupLog((msg) => {
  const p = el('div', null, msg)
  setupLog.appendChild(p)
  setupLog.scrollTop = setupLog.scrollHeight
})

$('btn-setup-start').addEventListener('click', async () => {
  $('btn-setup-start').disabled = true
  $('btn-setup-start').textContent = 'Installing…'
  setupLog.innerHTML = ''

  const res = await window.api.runSetup()
  if (res.success) {
    $('setup-overlay').classList.add('hidden')
  } else {
    $('btn-setup-start').disabled = false
    $('btn-setup-start').textContent = 'Retry'
    const err = el('div', null, `Error: ${res.error}`)
    err.style.color = 'var(--red)'
    setupLog.appendChild(err)
  }
})

// ── Init ──────────────────────────────────────────────────────────────────────
;(async () => {
  await Promise.all([loadHardware(), loadSettings()])
  updateRunButton()
  await checkSetup()
})()
