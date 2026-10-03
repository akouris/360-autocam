#!/bin/bash
# Assemble everything the packaged app needs so it runs on any Apple Silicon Mac
# with nothing installed (no Homebrew, no Python, no Hugging Face token).
#
#   app/bundle/python   standalone CPython 3.9 + the exact tested packages
#   app/bundle/bin      static arm64 ffmpeg + ffprobe (VideoToolbox, v360)
#   app/bundle/models   face_landmarker.task + pyannote weights (offline cache)
#
# Usage (from app/):  ./build/bundle.sh        then: npm run build
# Re-running is safe; delete app/bundle to start over.
set -euo pipefail

cd "$(dirname "$0")/.."
APP_DIR="$(pwd)"
OUT="$APP_DIR/bundle"
CACHE="$APP_DIR/build/.cache"
mkdir -p "$OUT" "$CACHE"

PY_URL="https://github.com/astral-sh/python-build-standalone/releases/download/20251031/cpython-3.9.25%2B20251031-aarch64-apple-darwin-install_only.tar.gz"
FF_BASE="https://ffmpeg.martin-riedl.de/redirect/latest/macos/arm64/release"
LANDMARKER_URL="https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
PYANNOTE_CACHE_SRC="${PYANNOTE_CACHE:-$HOME/.cache/torch/pyannote}"
PYANNOTE_MODELS="models--pyannote--speaker-diarization-3.1 models--pyannote--segmentation-3.0 models--pyannote--wespeaker-voxceleb-resnet34-LM"

# ── Python ────────────────────────────────────────────────────────────────────
if [ ! -x "$OUT/python/bin/python3" ]; then
  echo "== Python: downloading standalone CPython 3.9 (arm64)"
  curl -fL -o "$CACHE/python.tar.gz" "$PY_URL"
  rm -rf "$OUT/python"
  tar -xzf "$CACHE/python.tar.gz" -C "$OUT"      # → $OUT/python
fi
echo "== Python: installing the locked, tested package set"
"$OUT/python/bin/python3" -m pip install --quiet --disable-pip-version-check --no-warn-script-location \
  --no-cache-dir --no-deps -r "$APP_DIR/resources/requirements.lock.txt"
# --no-deps: reproduce the exact tested set. Some packages declare numpy>=2 but
# the pipeline is verified on numpy 1.26.4 (pyannote needs np.NaN).
# trim: bytecode caches and test suites aren't needed at runtime
find "$OUT/python" -name "__pycache__" -type d -prune -exec rm -rf {} +
find "$OUT/python/lib/python3.9/site-packages" -type d \( -name tests -o -name test \) -prune -exec rm -rf {} + 2>/dev/null || true
rm -rf "$OUT/python/lib/python3.9/test"

# ── FFmpeg ────────────────────────────────────────────────────────────────────
mkdir -p "$OUT/bin"
for tool in ffmpeg ffprobe; do
  if [ ! -x "$OUT/bin/$tool" ]; then
    echo "== FFmpeg: downloading static arm64 $tool"
    curl -fL -o "$CACHE/$tool.zip" "$FF_BASE/$tool.zip"
    unzip -o -q "$CACHE/$tool.zip" -d "$OUT/bin"
    chmod +x "$OUT/bin/$tool"
  fi
done
"$OUT/bin/ffmpeg" -hide_banner -encoders | grep -q hevc_videotoolbox || { echo "ffmpeg lacks hevc_videotoolbox"; exit 1; }
"$OUT/bin/ffmpeg" -hide_banner -filters  | grep -q " v360 "          || { echo "ffmpeg lacks v360"; exit 1; }

# ── Models ────────────────────────────────────────────────────────────────────
mkdir -p "$OUT/models/pyannote"
if [ ! -s "$OUT/models/face_landmarker.task" ]; then
  echo "== Models: face_landmarker.task"
  curl -fL -o "$OUT/models/face_landmarker.task" "$LANDMARKER_URL"
fi
echo "== Models: pyannote weights (from $PYANNOTE_CACHE_SRC)"
for m in $PYANNOTE_MODELS; do
  if [ ! -d "$PYANNOTE_CACHE_SRC/$m" ]; then
    echo "Missing $PYANNOTE_CACHE_SRC/$m — run the pipeline once with HF_TOKEN set so pyannote downloads it."
    exit 1
  fi
  rm -rf "$OUT/models/pyannote/$m"
  cp -RL "$PYANNOTE_CACHE_SRC/$m" "$OUT/models/pyannote/$m"   # -L: real files, no symlinks
done

# ── Smoke test: the bundle alone, with a bare environment ────────────────────
echo "== Smoke test"
env -i HOME="$CACHE/emptyhome" PATH="$OUT/bin:/usr/bin:/bin" \
  PYANNOTE_CACHE="$OUT/models/pyannote" HF_HUB_OFFLINE=1 \
  "$OUT/python/bin/python3" -c "import torch, torchaudio, pyannote.audio, mediapipe, cv2, scipy, numpy; print('python ok:', torch.__version__, numpy.__version__, 'mps', torch.backends.mps.is_available())"

du -sh "$OUT/python" "$OUT/bin" "$OUT/models"
echo "== Bundle ready in $OUT"
