#!/usr/bin/env python3
"""
Step 1: Extract one frame every N seconds (default 2s) and calibrate the lens.

Fisheye mode (Insta360):   two streams (front + back lens) → frames/front/, frames/back/
Equirect mode (YouTube 360°): single stream                → frames/equirect/

Frames are downscaled to 1920px — plenty for face detection (a seated face is
~100px) and 4× less disk/IO than full 8K.

Also writes:
  frames/meta.json  {"interval": N}              — read by later steps
  data/lens.json    {"fov": <deg>}               — fisheye only: true lens FOV,
                    calibrated so the two lenses' seams line up (see geometry.py)

Mode is read from the VIDEO_MODE environment variable (default: fisheye).
"""

import subprocess, sys, os, re, json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import geometry

FRAME_SIZE = 1920


def get_duration(video_path):
    r = subprocess.run([
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", video_path
    ], capture_output=True, text=True)
    return float(r.stdout.strip())


def run_ffmpeg_progress(cmd, duration_s):
    """Run ffmpeg, parsing -progress stderr lines to emit PROGRESS N/M."""
    proc = subprocess.Popen(cmd, stderr=subprocess.PIPE, text=True, bufsize=1)
    last_pct = -1
    for line in proc.stderr:
        m = re.match(r'out_time_us=(\d+)', line.strip())
        if m:
            elapsed = int(m.group(1)) / 1e6
            pct = min(99, int(elapsed / duration_s * 100))
            if pct != last_pct:
                print(f"PROGRESS {pct}/100", flush=True)
                last_pct = pct
    proc.wait()
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd)
    print("PROGRESS 100/100", flush=True)


def calibrate_lens(n_samples=8):
    """Median seam-alignment FOV over frames spread through the video."""
    import cv2, numpy as np
    files = sorted(f for f in os.listdir("frames/front") if f.endswith(".jpg"))
    files = [f for f in files if os.path.exists(os.path.join("frames/back", f))]
    if not files:
        return geometry.DEFAULT_LENS_FOV
    picks = [files[int(i)] for i in np.linspace(0, len(files) - 1, min(n_samples, len(files)))]
    fovs = []
    for f in picks:
        fr = cv2.imread(os.path.join("frames/front", f))
        bk = cv2.imread(os.path.join("frames/back", f))
        if fr is None or bk is None:
            continue
        fovs.append(geometry.calibrate_fov([(fr, bk)])[0])
    if not fovs:
        return geometry.DEFAULT_LENS_FOV
    fov = float(np.median(fovs))
    print(f"  lens FOV calibration: {['%.1f' % x for x in fovs]} → {fov:.1f}°", flush=True)
    return fov


def extract_frames(video_path: str, interval: float = 2):
    mode = os.environ.get("VIDEO_MODE", "fisheye")
    dur  = get_duration(video_path)

    hw = ["-hwaccel", "videotoolbox"]
    scale = f"scale={FRAME_SIZE}:-2"
    # Pick frames at exactly t = 0, N, 2N, … so frame i ↔ time i·N. (fps=1/N
    # picks the frame at (i+½)·N — every later step would be off by N/2.)
    pick = f"select='isnan(prev_selected_t)+gte(t-prev_selected_t\\,{interval}-0.001)'"

    if mode == "equirect":
        os.makedirs("frames/equirect", exist_ok=True)
        print(f"Extracting equirectangular stream, 1 frame every {interval}s ({dur:.0f}s video)…", flush=True)
        cmd = ["ffmpeg"] + hw + [
            "-i", video_path,
            "-map", "0:v:0",
            "-vf", f"{pick},scale=3840:-2", "-fps_mode", "vfr",
            "-q:v", "3",
            "frames/equirect/frame_%04d.jpg",
            "-progress", "pipe:2", "-y",
        ]
        run_ffmpeg_progress(cmd, dur)
        count = len([f for f in os.listdir("frames/equirect") if f.endswith(".jpg")])
        print(f"  → {count} frames in frames/equirect/", flush=True)
    else:
        for sub in ["front", "back"]:
            os.makedirs(f"frames/{sub}", exist_ok=True)
        print(f"Extracting front + back lenses in one pass, 1 frame every {interval}s ({dur:.0f}s video)…", flush=True)
        # Single FFmpeg pass: both streams extracted simultaneously — halves decode time
        cmd = ["ffmpeg"] + hw + [
            "-i", video_path,
            "-map", "0:v:0", "-vf", f"{pick},{scale}", "-fps_mode", "vfr", "-q:v", "3",
            "frames/front/frame_%04d.jpg",
            "-map", "0:v:1", "-vf", f"{pick},{scale}", "-fps_mode", "vfr", "-q:v", "3",
            "frames/back/frame_%04d.jpg",
            "-progress", "pipe:2", "-y",
        ]
        run_ffmpeg_progress(cmd, dur)
        for lens in ["front", "back"]:
            count = len([f for f in os.listdir(f"frames/{lens}") if f.endswith(".jpg")])
            print(f"  → {count} frames in frames/{lens}/", flush=True)

        os.makedirs("data", exist_ok=True)
        fov = calibrate_lens()
        with open(geometry.LENS_JSON, "w") as f:
            json.dump({"fov": fov}, f)

    with open("frames/meta.json", "w") as f:
        json.dump({"interval": interval, "mode": mode}, f)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 1_extract_frames.py <video_path> [interval_seconds]")
        sys.exit(1)
    extract_frames(sys.argv[1], float(sys.argv[2]) if len(sys.argv) > 2 else 2)
