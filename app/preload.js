'use strict'
const { contextBridge, ipcRenderer } = require('electron')

const on = (channel, fn) => {
  const sub = (_, ...args) => fn(...args)
  ipcRenderer.on(channel, sub)
  return () => ipcRenderer.removeListener(channel, sub)
}

contextBridge.exposeInMainWorld('api', {
  // Info
  hardware:     ()    => ipcRenderer.invoke('hardware:info'),
  loadSettings: ()    => ipcRenderer.invoke('settings:load'),
  saveSettings: (s)   => ipcRenderer.invoke('settings:save', s),

  // File pickers
  selectVideo: () => ipcRenderer.invoke('dialog:video'),
  selectAudio: () => ipcRenderer.invoke('dialog:audio'),

  // First-time setup
  checkSetup:  () => ipcRenderer.invoke('setup:check'),
  runSetup:    () => ipcRenderer.invoke('setup:install'),

  // Job resume
  scanJob:      (videoPath) => ipcRenderer.invoke('job:scan', videoPath),
  clearJob:     (videoPath) => ipcRenderer.invoke('job:clear', videoPath),

  // Pipeline
  runPipeline:  (args) => ipcRenderer.invoke('pipeline:run', args),
  stopPipeline: ()     => ipcRenderer.send('pipeline:stop'),
  openOutput:   (p)    => ipcRenderer.send('shell:open', p),

  // Event subscriptions (return unsubscribe fn)
  onSetupLog:       (fn) => on('setup:log',       fn),
  onPipelineSteps:  (fn) => on('pipeline:steps',  fn),
  onPipelineStep:   (fn) => on('pipeline:step',   fn),
  onPipelineLog:    (fn) => on('pipeline:log',    fn),
  onPipelineDone:   (fn) => on('pipeline:done',   fn),
  onPipelineError:  (fn) => on('pipeline:error',  fn),
  onPipelineAborted:(fn) => on('pipeline:aborted', fn),
})
