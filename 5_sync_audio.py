#!/usr/bin/env python3
"""
Step 5 (optional): Sync external microphone audio to the camera recording.

Usage:
    python3 5_sync_audio.py <video_path> <external_audio>

The external audio can be any duration — longer, shorter, or starting
before/after the camera recording.  Only the overlapping content matters;
the script finds the matching portion automatically.

Output: data/external_audio_aligned.wav
  - Trimmed / silence-padded to exactly match the video duration
  - Drift-corrected via atempo if the two recording clocks diverge

Pass this file to 4_render.py with --external-audio, or re-run
3_diarize.py with --audio to get cleaner diarization from the mic feed.
"""

import json, os, sys, subprocess, tempfile
import numpy as np

FINGERPRINT_SR = 8000   # Hz — adequate for speech + music fingerprinting
DRIFT_WIN_S    = 60     # seconds of audio used for drift measurement at each end
MAX_DRIFT_MS   = 30     # apply atempo time-stretch if drift exceeds this


def load_mono_8k(path, label="audio"):
    """Decode any audio/video file to mono float32 at FINGERPRINT_SR via FFmpeg.

    Exits with a clear, specific message on any failure (missing file, no audio
    stream, decode error, silent/empty track) instead of a bare traceback."""
    if not os.path.exists(path):
        print(f"Error: {label} file not found: {path}", flush=True)
        sys.exit(1)

    tmp = tempfile.mktemp(suffix=".wav")
    try:
        r = subprocess.run([
            "ffmpeg", "-i", path,
            "-ac", "1", "-ar", str(FINGERPRINT_SR),
            "-acodec", "pcm_s16le", tmp, "-y", "-loglevel", "error",
        ], capture_output=True, text=True)
        if r.returncode != 0 or not os.path.exists(tmp):
            print(f"Error: could not decode {label} audio from {path}.", flush=True)
            if r.stderr.strip():
                print(r.stderr.strip(), flush=True)
            print(f'Hint: check the file is valid with: ffprobe "{path}"', flush=True)
            sys.exit(1)
        from scipy.io import wavfile
        _, data = wavfile.read(tmp)
        data = data.astype(np.float32)
        if data.size == 0:
            print(f"Error: {label} audio has no samples — {path}", flush=True)
            sys.exit(1)
        if float(np.abs(data).max()) < 1.0:
            print(f"Error: {label} audio is silent; cannot align on silence — {path}",
                  flush=True)
            sys.exit(1)
        return data
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def find_lag(ref, query):
    """
    FFT cross-correlation.  Returns (lag_samples, prominence): positive lag means
    query starts that many samples AFTER the start of ref.

    prominence = highest peak / second-highest peak (peaks ≥1s apart). A genuine
    shared-content match produces ONE dominant spike (prominence ≫ 1); unrelated
    recordings produce many peaks of similar height (prominence ≈ 1). This cleanly
    separates real overlaps (≈150×) from noise (≈1×) regardless of signal length.
    """
    from scipy.signal import fftconvolve
    r = ref   / (ref.std()   + 1e-9)
    q = query / (query.std() + 1e-9)
    corr = np.abs(fftconvolve(r, q[::-1], mode="full"))
    peak_i = int(np.argmax(corr))
    peak_v = float(corr[peak_i])
    # second-highest peak, excluding ±1s around the main peak
    guard  = FINGERPRINT_SR
    masked = corr.copy()
    masked[max(0, peak_i - guard): peak_i + guard] = 0.0
    second = float(masked.max())
    prominence = peak_v / (second + 1e-9)
    return peak_i - (len(q) - 1), prominence


def get_duration(path):
    r = subprocess.run([
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "json", path,
    ], capture_output=True, text=True, check=True)
    return float(json.loads(r.stdout)["format"]["duration"])


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 5_sync_audio.py <video_path> <external_audio>")
        sys.exit(1)

    video_path = sys.argv[1]
    ext_path   = sys.argv[2]
    cam_wav    = "data/audio.wav"

    # Camera reference: prefer the pre-extracted audio (step 0); otherwise decode
    # straight from the video so sync never silently depends on step ordering.
    cam_src = cam_wav if os.path.exists(cam_wav) else video_path
    if cam_src == video_path:
        print("data/audio.wav not present — decoding camera audio from the video.",
              flush=True)

    # ── 1. Load both signals at 8kHz ────────────────────────────────────────
    print("Loading camera audio at 8kHz …", flush=True)
    cam = load_mono_8k(cam_src, label="camera")
    print("Loading external audio at 8kHz …", flush=True)
    ext = load_mono_8k(ext_path, label="external")
    print(f"  Camera:   {len(cam)/FINGERPRINT_SR:.1f}s")
    print(f"  External: {len(ext)/FINGERPRINT_SR:.1f}s")

    # ── 2. Find global offset ────────────────────────────────────────────────
    print("\nComputing cross-correlation …", flush=True)
    lag, prominence = find_lag(cam, ext)
    offset_s = lag / FINGERPRINT_SR
    direction = "after" if offset_s > 0 else "before"
    print(f"Offset: {offset_s:+.3f}s  "
          f"(external starts {abs(offset_s):.3f}s {direction} camera)")
    print(f"Match prominence: {prominence:.1f}×  (main peak vs. next-best peak)")
    if prominence < 4.0:
        print("\nError: the external audio does not share content with the camera "
              "recording — no clear alignment peak (prominence "
              f"{prominence:.1f}×, need ≥4×).", flush=True)
        print("This usually means the wrong file was selected, or the mic recording "
              "does not overlap this take in time.", flush=True)
        print("Nothing was written; pick the correct external audio and retry sync.",
              flush=True)
        sys.exit(1)

    # ── 3. Measure clock drift ───────────────────────────────────────────────
    WIN       = int(DRIFT_WIN_S * FINGERPRINT_SR)
    drift_ms  = 0.0
    drift_str = "n/a"

    # Index into each signal where the overlap starts
    cam_ov0 = max(0,  lag)
    ext_ov0 = max(0, -lag)
    overlap  = min(len(cam) - cam_ov0, len(ext) - ext_ov0)

    if overlap > 2 * WIN:
        lag_start, _ = find_lag(cam[cam_ov0 : cam_ov0 + WIN],
                                ext[ext_ov0 : ext_ov0 + WIN])
        lag_end, _   = find_lag(cam[cam_ov0 + overlap - WIN : cam_ov0 + overlap],
                                ext[ext_ov0 + overlap - WIN : ext_ov0 + overlap])
        drift_ms  = (lag_end - lag_start) / FINGERPRINT_SR * 1000
        drift_str = f"{drift_ms:+.1f}ms over {overlap/FINGERPRINT_SR:.0f}s"

    print(f"Drift:  {drift_str}")

    # ── 4. Build FFmpeg command to export aligned audio ──────────────────────
    video_dur = get_duration(video_path)
    out_path  = "data/external_audio_aligned.wav"

    filters = []

    # Pad silence at start if external recording begins after the camera
    if offset_s > 0:
        delay_ms = int(offset_s * 1000)
        filters.append(f"adelay={delay_ms}|{delay_ms}:all=1")

    # Correct clock drift with gentle time-stretch
    if abs(drift_ms) > MAX_DRIFT_MS:
        stretch = 1.0 - (drift_ms / 1000.0) / max(overlap / FINGERPRINT_SR, 1)
        stretch = max(0.5, min(2.0, stretch))
        print(f"Applying time-stretch {stretch:.6f} to correct drift")
        filters.append(f"atempo={stretch:.6f}")

    # Pad short external audio with silence; trim any excess
    filters += [f"apad=whole_dur={video_dur}", f"atrim=end={video_dur}"]

    filter_str = ",".join(filters)

    cmd = ["ffmpeg", "-y", "-loglevel", "error"]

    # Skip the leading portion of external audio that predates the camera
    if offset_s < 0:
        cmd += ["-ss", f"{abs(offset_s):.6f}"]

    cmd += [
        "-i", ext_path,
        "-af", filter_str,
        "-ac", "2", "-ar", "48000",
        "-acodec", "pcm_s16le",
        out_path,
    ]

    print(f"\nExporting aligned audio → {out_path} …")
    subprocess.run(cmd, check=True)
    aligned_dur = get_duration(out_path)
    print(f"Done.  {out_path}  ({aligned_dur:.1f}s / {video_dur:.1f}s video)")

    print("\nNext steps:")
    print("  Cleaner diarization (recommended):")
    print("    python3 3_diarize.py <video> --audio data/external_audio_aligned.wav")
    print("  Or skip re-diarization and go straight to render:")
    print("    python3 4_render.py <video> --external-audio data/external_audio_aligned.wav")
