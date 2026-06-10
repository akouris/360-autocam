# 360-autocam

Turns a 360° Insta360 video of a panel discussion, podcast, or roundtable into a broadcast-ready 9:16 vertical video that automatically cuts between speakers.

Place the camera on the table. Everyone sits around it. The pipeline figures out who is where, who is speaking, and produces a final video that stays on the active speaker while cutting to split-screens and cycling through reactions — no manual editing required.

## Demo

[![360-autocam demo — watch on YouTube](https://i.ytimg.com/vi/F0w0of0BS8k/hqdefault.jpg)](https://www.youtube.com/watch?v=F0w0of0BS8k)

▶ **[Watch the demo](https://www.youtube.com/watch?v=F0w0of0BS8k)** — a 360° recording auto-cut into a 9:16 vertical with speaker switching.

---

## What you get

- **Presence-driven carousel** — the output continuously cycles between single shots and split-screens of whoever is **physically present**, driven by per-frame face detection. It never freezes on a static shot and never shows an empty chair for someone who has left the room — this holds even if diarization or speaker mapping is imperfect.
- **Split-screen first** — the default shot shows the speaker + another present participant; brief solo intros mark genuine speaker changes
- **Automatic speaker mapping** — lip-sync correlation (MediaPipe + audio RMS) matches diarization IDs to camera positions; no manual labelling
- **External mic support** — feed in a console mix or direct mic recording; the pipeline auto-syncs it to the camera and lays it over the final video as the high-quality audio track
- **Smooth, in-sync audio** — one continuous audio track encoded once (no per-cut glitches), with cut boundaries frame-aligned so audio and video never drift apart
- **Hardware-accelerated encoding** — VideoToolbox HEVC on Apple Silicon

---

## Two ways to run

### Option A — Electron app (macOS)

A point-and-click GUI. Drop in the video, optionally add a mic recording, enter your HF token, click Run.

```bash
cd app
npm install
npm start
```

On first launch the app detects your Python environment and walks through dependency installation if needed. Output lands in `autocam_<videoname>/` next to your input file.

To build a distributable `.app` / DMG:

```bash
npm run build   # → dist/360 AutoCam-1.0.0-arm64.dmg
```

### Option B — Command line

```bash
# Basic
./run_all.sh /path/to/video.insv

# With external microphone recording
./run_all.sh /path/to/video.insv --external-audio /path/to/console_mix.mp3
```

Output: `output/final.mp4`

---

## Requirements

### System

- **macOS** (Apple Silicon recommended; Intel works but is slower)
- **FFmpeg** — `brew install ffmpeg`
- **Python 3.9+** — `brew install python3` or system Python
- **Node.js 18+** — only needed for the Electron app

### Hugging Face

> **Audio-first:** the pipeline extracts the camera audio and syncs any external
> mic **before** the expensive frame work, so an audio problem (no audio track,
> wrong/non-overlapping mic file) fails in seconds instead of after minutes.

The diarization model requires a free Hugging Face account and accepted terms:

1. Create an account at [huggingface.co](https://huggingface.co)
2. Accept the terms for [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
3. Accept the terms for [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)
4. Create a token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)

For the CLI, save the token to `.env`:

```
HF_TOKEN=hf_xxxxxxxxxxxx
```

For the Electron app, enter it in the HF Token field — it's saved automatically.

---

## CLI setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install "numpy==1.26.4"
pip install pyannote.audio==3.4.0 "huggingface_hub==0.23.4" torch torchaudio mediapipe opencv-python scipy
```

Download the MediaPipe face landmark model (3.6 MB):

```bash
mkdir -p data
curl -L -o data/face_landmarker.task \
  "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
```

> **numpy must be 1.26.4** — pyannote uses `np.NaN` which was removed in numpy 2.0.

---

## Pipeline

Steps run in this order (audio first, then frames, then analysis):

| Step | Script | Output | Time (30-min video, M1) |
|------|--------|--------|------------------------|
| 0 | `0_extract_audio.py` | `data/audio.wav` — camera audio (16 kHz mono) | ~30 sec |
| 0b *(optional)* | `5_sync_audio.py` | `data/external_audio_aligned.wav` | ~1 min |
| 1 | `1_extract_frames.py` | `frames/` — one frame every 10s | ~1 min |
| 2 | `2_detect_speakers.py` | `data/speaker_positions.json` — yaw/pitch/fov per speaker | ~30 sec |
| 2b | `2b_detect_presence.py` | `data/presence.json` — per-time face presence per seat | ~1 min |
| 3 | `3_diarize.py` | `data/diarization.json` + `data/overlaps.json` | ~12 min |
| 3b | `3_automap.py` | `data/speaker_map.json` — diarization ID → camera position | ~5 min |
| 4 | `4_render.py` | `output/final.mp4` | ~13 min |

**Total: ~30–35 min for a 30-minute video on Apple M1.**

> **Audio roles:** all *analysis* (diarization + speaker mapping) runs on the
> **camera audio**, which is intrinsically in sync with the video and cleaner for
> this purpose. An external studio mic, if provided, is used **only** as the final
> audio overlay in the render — never for analysis.

### Presence detection (step 2b) and the carousel

`2b_detect_presence.py` scans every extracted frame, takes a perspective crop at each known seat, and records whether a face is visible there — producing a smoothed, per-time presence track per seat (`data/presence.json`). The renderer uses this to drive the carousel: the secondary slot only ever shows a seat whose occupant is currently present, and during stretches with no clearly-mapped speaker it rotates the primary through the present seats. This guarantees the output never holds a static single shot and never shows an empty chair — independent of diarization/mapping quality.

### How speaker mapping works

`3_automap.py` samples short segments for each diarized speaker, extracts camera frames at each seat, measures lip aperture with MediaPipe FaceLandmarker, and correlates that against the audio RMS envelope. Broadcast/background audio (studio monitors, music) is identified by lip-sync, not duration: a "speaker" whose audio doesn't move *any* on-camera mouth (correlation below threshold) is excluded as broadcast. The busiest real speakers seed the seats 1-to-1 (Hungarian), then any extra diarization labels (pyannote often over-segments one person into several) attach to their best-matching seat. Diarization's speaker count is also capped using the number of detected faces, to bound over-segmentation on noisy audio.

> **Note:** lip-sync correlation can be weak on far-field / studio footage. If seats
> look mis-assigned, override `data/speaker_map.json` manually (see below). The
> presence-driven carousel still produces a watchable result regardless.

### External audio sync

`5_sync_audio.py` takes any audio file (MP3, WAV, M4A, AAC) from a mixing console or dedicated mic. The recordings can differ in length and start time — the script finds the matching window via FFT cross-correlation at 8 kHz, reports a **match prominence** (main correlation peak ÷ next-best peak; a genuine overlap scores ≫1, an unrelated/wrong file ≈1) and refuses with a clear message if they don't actually share content. It then measures clock drift across the overlap and applies a gentle time-stretch if needed. The aligned track is laid over the final render as the audio; analysis still uses the camera audio.

---

## Manual overrides

If `3_automap.py` gets the assignment wrong (weak lip-sync, far-field/studio audio, or a very quiet speaker), edit `data/speaker_map.json` directly before running the render:

```json
{
  "SPEAKER_00": "front_0",
  "SPEAKER_01": "front_1",
  "SPEAKER_02": "back_0"
}
```

Keys are pyannote speaker IDs; values are keys from `data/speaker_positions.json`. Several diarization IDs may map to the same seat (that's expected when one person is over-segmented into multiple IDs).

---

## Audio & A/V sync

The render cuts only the **video** per shot; the audio is one continuous track laid over the finished video in a single encode. This avoids the per-cut clicks/jumps you get from concatenating hundreds of separately-encoded audio slices. Every cut boundary is snapped to the video frame grid so each segment is an exact whole number of frames — segment durations sum exactly, so audio and video stay locked (verified: stream durations match within a few milliseconds over 30 minutes).

## Output format

- Resolution: 1080×1920 (9:16 vertical)
- Video: HEVC / H.265 (VideoToolbox hardware encode), 8 Mbps
- Audio: AAC 192k (external mic overlay if provided, else camera audio)
- Container: MP4

---

## Project layout

```
360-autocam/
├── run_all.sh               # CLI entry point
├── 0_extract_audio.py
├── 1_extract_frames.py
├── 2_detect_speakers.py
├── 2b_detect_presence.py
├── 3_diarize.py
├── 3_automap.py
├── 4_render.py
├── 5_sync_audio.py
├── data/
│   ├── audio.wav                    # camera audio, extracted by step 0
│   ├── external_audio_aligned.wav   # created by 5_sync_audio.py (optional)
│   ├── speaker_positions.json       # created by step 2
│   ├── presence.json                # created by step 2b
│   ├── diarization.json             # created by step 3
│   ├── overlaps.json                # created by step 3
│   ├── speaker_map.json             # created by step 3b, editable
│   └── face_landmarker.task         # download separately (see setup)
├── frames/                          # created by step 1
├── output/                          # final.mp4 lands here
└── app/                             # Electron macOS app
    ├── main.js
    ├── preload.js
    ├── renderer/
    └── resources/
```

---

## Limitations & notes

- **Speaker mapping depends on lip-sync**, which can be weak on far-field or
  acoustically busy footage (e.g. a radio studio with monitors and music). When
  it's uncertain, seats may be mis-assigned — override `data/speaker_map.json`
  manually. The presence-driven carousel still produces a watchable result either
  way (it shows the present people; only *which* speaker is emphasised may be off).
- **macOS / Apple Silicon focused.** Encoding uses VideoToolbox; diarization uses
  MPS. Intel Macs and other platforms work but are slower and untested here.
- The external mic only improves the *audio*; if the mic recording doesn't cover
  the whole take, the uncovered span is silence-padded.

## Tested on

- Insta360 X4 (8K, .insv)
- Apple M1 Pro, 32 GB RAM, macOS Sequoia
- Python 3.9, FFmpeg 7.x, Electron 28
