#!/bin/bash
# Full pipeline: 360° video → 9:16 auto-cut vertical video
#
# Usage:
#   ./run_all.sh <video_path>
#   ./run_all.sh <video_path> --mode equirect          # YouTube 360° (default: fisheye)
#   ./run_all.sh <video_path> --external-audio <mic.mp3>
#   ./run_all.sh <video_path> --mode equirect --external-audio <mic.mp3>
#
# Requires: HF_TOKEN in .env

set -e

if [ -f .env ]; then
  export $(grep -v '^#' .env | xargs)
fi

VIDEO="$1"
if [ -z "$VIDEO" ]; then
  echo "Usage: $0 <video_path> [--mode fisheye|equirect] [--external-audio <mic.mp3>]"
  exit 1
fi

EXTERNAL_AUDIO=""
export VIDEO_MODE="fisheye"

shift
while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)
      export VIDEO_MODE="$2"
      shift 2
      ;;
    --external-audio)
      EXTERNAL_AUDIO="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
done

echo "Mode: $VIDEO_MODE"
source venv/bin/activate

# AUDIO-FIRST: extract camera audio and sync the external mic before any frame
# work, so audio problems fail fast instead of after minutes of frame extraction.
echo "=== Step 0: Extract camera audio ==="
python3 0_extract_audio.py "$VIDEO"

# The external studio mic is synced now but used ONLY as the final audio overlay
# in render. All analysis (diarization + mapping) runs on the camera audio.
if [ -n "$EXTERNAL_AUDIO" ]; then
  echo "=== Step 0b: Sync external audio (for final overlay only) ==="
  python3 5_sync_audio.py "$VIDEO" "$EXTERNAL_AUDIO"
  RENDER_AUDIO="--external-audio data/external_audio_aligned.wav"
else
  RENDER_AUDIO=""
fi

echo "=== Step 1: Extract frames ==="
python3 1_extract_frames.py "$VIDEO" 2

echo "=== Step 2: Detect speaker positions ==="
python3 2_detect_speakers.py

echo "=== Step 2b: Detect per-time presence ==="
python3 2b_detect_presence.py

# Diarize on CAMERA audio (data/audio.wav from step 0) — never the studio feed.
echo "=== Step 3: Speaker diarization (camera audio) ==="
python3 3_diarize.py "$VIDEO"

echo "=== Step 3b: Track mouths + map speakers (who speaks when) ==="
python3 3_automap.py "$VIDEO"

echo "=== Step 4: Render ==="
python3 4_render.py "$VIDEO" $RENDER_AUDIO

echo "=== Done! Check output/final.mp4 ==="
