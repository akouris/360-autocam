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
- **Audio + video speaker detection** — every face's mouth is tracked at 12.5 fps and checked against the audio for every diarized segment, so the camera follows who is *visibly* talking, not just a voice label
- **Double / triple shots when unsure** — when it can't tell who is speaking (or two people talk at once) it shows the likely speakers together instead of guessing wrong
- **Accurate framing** — each pane follows the face's real position, is zoomed by face size, and keeps the face in the upper part of the frame; crops come from the stitched 360° view, so people sitting on the lens seam frame correctly
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

In development (`npm start`) the app uses `app/bundle` if you've built it, otherwise the project `venv`.

**Output & cleanup.** The finished video is saved as `<videoname>_autocam.mp4` next to the input (or in `~/Movies/360 AutoCam/` if that folder is read-only; a re-run never overwrites, it adds ` (2)`). Working files (audio, frames, analysis) live in `~/Library/Caches/360 AutoCam/jobs/` and are deleted after a successful run. An interrupted or failed job keeps them, so the next run resumes; leftovers older than 7 days are pruned at launch. A 60s preview keeps them too, so the full render reuses the analysis. *Settings → Keep working files* turns cleanup off for debugging.

### Web UI on the local network

While the app is open it also serves a web UI. The address is shown in the 🌐 chip in the title bar (e.g. `http://macultra.local:8360`); click the chip for settings (on/off, port, access PIN). A PIN is always required: one is generated on first launch and shown in the settings, and you can change it there. From any browser on the LAN you can:

- **Browse this Mac**: pick a video (and optional mic audio) from the render Mac's home folder or any mounted drive/share under `/Volumes`. Nothing is copied, so this is the way to go for big files: mount the NAS share on the render Mac once (Finder → ⌘K) and it shows up here.
- **Upload**: stream a file from your own computer. The file goes straight to disk (`~/Movies/360 AutoCam/Uploads/`), free space is checked first, and the upload is deleted after a successful render. Results go to `~/Movies/360 AutoCam/`.

Only ONE render runs at a time, from the web or the window. While one is running, new renders and uploads are refused with a message saying what is rendering. The page shows live progress and a **Download** button when a job is done. While a job runs, the Mac is kept from sleeping. The pipeline runs at low priority (`nice 10`): when the Mac is otherwise idle it renders at full speed, but other work on it (streaming, recording) gets the CPU first. Closing the window keeps the web UI running; quit with ⌘Q. On first launch, macOS asks whether to allow incoming connections: click Allow. Ad-hoc signed builds may ask again after an update.

### Building the self-contained app (for other Macs)

The packaged app carries everything it needs: standalone Python 3.9 with the exact tested packages (`app/resources/requirements.lock.txt`), static arm64 `ffmpeg`/`ffprobe`, the face model and the pyannote weights. It runs offline on any Apple Silicon Mac (M1 or newer, macOS 12+; 16 GB RAM recommended) with no Homebrew, no Python and no Hugging Face token.

```bash
cd app
npm install
npm run dist    # = ./build/bundle.sh (assemble app/bundle) + electron-builder
# → dist/360 AutoCam-1.0.0-arm64.dmg and dist/360 AutoCam-1.0.0-arm64-mac.zip (~480 MB each)
```

`build/bundle.sh` needs the pyannote weights in `~/.cache/torch/pyannote` (run the pipeline from source once with `HF_TOKEN` set). The app is **ad-hoc signed** (`build/adhoc-sign.js`) — free, no Apple Developer ID. Python bytecode is pre-compiled and sealed, and the app runs Python with `PYTHONDONTWRITEBYTECODE=1`, so using it never breaks its signature.

### Distributing without a Developer ID

macOS only blocks un-notarized apps that carry the "downloaded from the internet" mark (browsers, Mail, Messages, AirDrop add it; `curl`, USB sticks and most network shares don't).

- **Recommended — one-line install.** Put the zip and `app/build/install.sh` somewhere reachable (for Dropbox, use `?dl=1` links). Colleagues paste:
  ```bash
  curl -fsSL "<install.sh link>" | bash -s "<zip link>"
  ```
  It installs to /Applications, verifies the signature and opens the app — no prompts.
- **DMG/zip via a browser link:** the first launch is blocked once. Fix with System Settings → Privacy & Security → **Open Anyway**, or in Terminal:
  `xattr -dr com.apple.quarantine "/Applications/360 AutoCam.app"`

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

Only needed when running from source (the packaged app ships the models). The diarization model requires a free Hugging Face account and accepted terms:

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
| 1 | `1_extract_frames.py` | `frames/` — one frame every 2s; `data/lens.json` — calibrated lens FOV | ~2 min |
| 2 | `2_detect_speakers.py` | `data/speaker_positions.json` (seats) + `data/face_tracks.json` (faces per frame) | ~2 min |
| 2b | `2b_detect_presence.py` | `data/presence.json` — per-time face presence per seat | seconds |
| 3 | `3_diarize.py` | `data/diarization.json` + `data/overlaps.json` | ~12 min |
| 3b | `3_automap.py` | `data/speaker_probs.json` (who speaks, every 0.5s), `speaker_map.json`, `framing_tracks.json`, `presence_fine.json` | ~9 min |
| 4 | `4_render.py` | `output/final.mp4` (+ `data/timeline.json`) | ~15 min |

Times measured on an Apple M4 by scaling a 74s test clip; the slow parts are diarization, mouth tracking and encoding.

> **Audio roles:** all *analysis* (diarization + speaker mapping) runs on the
> **camera audio**, which is intrinsically in sync with the video and cleaner for
> this purpose. An external studio mic, if provided, is used **only** as the final
> audio overlay in the render — never for analysis.

### Geometry (`geometry.py`)

All angle ↔ pixel math lives in one module shared by detection (Python) and rendering (FFmpeg `v360`), verified to agree pixel-for-pixel. Both lenses are stitched (`v360=dfisheye`), so yaw is one global frame: 0° = front lens, 180° = back lens. Insta360 lenses cover ~190–197°, not 180°; step 1 measures the real FOV per video by making the lens seams line up, and every later step uses it.

### Seats, presence and the carousel (steps 2 / 2b)

Step 2 runs MediaPipe all the way round the 360° view on every frame. Seats are peaks in the yaw distribution of faces, so two people sitting close together stay separate. Background people (much smaller faces), static faces (posters, a face on a monitor) and faces seen too rarely are rejected. Presence (who is physically there, over time) comes from these face tracks, tightened by the 12.5 fps mouth-tracking pass in step 3b, so someone who walks off drops out within seconds. The renderer only ever shows present people, and with no clear speaker it rotates through them — it never holds a static single shot and never shows an empty chair, independent of diarization quality.

### How speaker detection works (step 3b)

1. **Mouth tracking** — one pass over the video at 12.5 fps renders a tight crop per seat (faces ~100px) and measures mouth openness (inner-lip gap ÷ face height, plus MediaPipe's `jawOpen`). A face only counts for a seat if it's nearer that seat than any other, so a neighbour leaning in can't hijack it.
2. **Per-segment evidence** — for *every* diarized segment, each seat is scored by how well its mouth movement tracks the audio loudness (±240 ms lag search) and how much the mouth moves compared with everyone at the table.
3. **Label → seat** — each diarization label is mapped by pooling all its segments. Labels that never move any on-camera mouth (music, phone-ins, broadcast audio) are off-camera and ignored.
4. **Speaker probabilities** — every 0.5s, a probability per seat combines the label's seat with the local mouth evidence. When diarization lumps two people under one label, the mouth evidence still picks out who is talking.

### How shots are chosen (step 4)

Three layouts: **single** (one person, full frame), **split** (two stacked halves) and **triple** (the speaker on the top half, two people side by side in the bottom half). Turns are found by Viterbi smoothing with a switch penalty, so a one-second "yeah" doesn't cut the camera (the interjector gets a pane instead). A **confident** turn gets a 2s solo intro, then cycles split → triple → split → single (triples only when 3+ people are present; a scheduled single becomes a split if someone is chiming in). An **uncertain** stretch shows the 2 or 3 likely speakers together (double / triple shot, alternating on long stretches), and **overlapping speech** shows the overlapping people together. The full shot list is written to `data/timeline.json`.

### External audio sync

`5_sync_audio.py` takes any audio file (MP3, WAV, M4A, AAC) from a mixing console or dedicated mic. The recordings can differ in length and start time — the script finds the matching window via FFT cross-correlation at 8 kHz, reports a **match prominence** (main correlation peak ÷ next-best peak; a genuine overlap scores ≫1, an unrelated/wrong file ≈1) and refuses with a clear message if they don't actually share content. It then measures clock drift across the overlap and applies a gentle time-stretch if needed. The aligned track is laid over the final render as the audio; analysis still uses the camera audio.

---

## Manual overrides

If a diarization label is tied to the wrong person, create `data/speaker_map_manual.json` and re-run step 3b (it reuses the cached mouth tracks, so it takes seconds) and then the render:

```json
{
  "SPEAKER_00": "seat_0",
  "SPEAKER_01": "seat_2"
}
```

Keys are pyannote speaker IDs; values are keys from `data/speaker_positions.json` (seats are numbered by yaw). Several IDs may map to the same seat. Labels you leave out are treated as off-camera audio. Per-moment mouth evidence still applies on top of your map.

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
├── geometry.py              # shared 360° geometry, lens calibration, framing
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
│   ├── lens.json                    # calibrated lens FOV, step 1
│   ├── speaker_positions.json       # seats, step 2
│   ├── face_tracks.json             # faces per frame, step 2
│   ├── presence.json                # step 2b
│   ├── diarization.json             # step 3
│   ├── overlaps.json                # step 3
│   ├── mouth_tracks.npz             # 12.5 fps mouth/face tracks (cache), step 3b
│   ├── speaker_probs.json           # who speaks, every 0.5s, step 3b
│   ├── speaker_map.json             # label → seat, step 3b
│   ├── speaker_map_manual.json      # optional hand override
│   ├── timeline.json                # shot list, step 4
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

- **Speaker detection needs visible mouths.** A face turned fully away, or a mic
  covering the mouth, weakens the evidence. When it's unsure, the renderer shows
  the likely speakers together rather than guessing; if a label is consistently
  wrong, use `data/speaker_map_manual.json`.
- **macOS / Apple Silicon focused.** Encoding uses VideoToolbox; diarization uses
  MPS. Intel Macs and other platforms work but are slower and untested here.
- The external mic only improves the *audio*; if the mic recording doesn't cover
  the whole take, the uncovered span is silence-padded.

## Tested on

- Insta360 X4 (8K, .insv)
- Apple M1 Pro, 32 GB RAM, macOS Sequoia
- Python 3.9, FFmpeg 7.x, Electron 28
