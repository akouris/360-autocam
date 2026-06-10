#!/usr/bin/env python3
"""
Step 1: Extract one frame every N seconds.

Fisheye mode (Insta360):   two streams (front + back lens) → frames/front/, frames/back/
Equirect mode (YouTube 360°): single stream                → frames/equirect/

Mode is read from the VIDEO_MODE environment variable (default: fisheye).
"""

import subprocess, sys, os, re

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


def extract_frames(video_path: str, interval: int = 10):
    mode = os.environ.get("VIDEO_MODE", "fisheye")
    dur  = get_duration(video_path)

    hw = ["-hwaccel", "videotoolbox"]

    if mode == "equirect":
        os.makedirs("frames/equirect", exist_ok=True)
        print(f"Extracting equirectangular stream, 1 frame every {interval}s ({dur:.0f}s video)…", flush=True)
        cmd = ["ffmpeg"] + hw + [
            "-i", video_path,
            "-map", "0:v:0",
            "-vf", f"fps=1/{interval}",
            "-q:v", "2",
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
            "-map", "0:v:0", "-vf", f"fps=1/{interval}", "-q:v", "2",
            "frames/front/frame_%04d.jpg",
            "-map", "0:v:1", "-vf", f"fps=1/{interval}", "-q:v", "2",
            "frames/back/frame_%04d.jpg",
            "-progress", "pipe:2", "-y",
        ]
        run_ffmpeg_progress(cmd, dur)
        for lens in ["front", "back"]:
            count = len([f for f in os.listdir(f"frames/{lens}") if f.endswith(".jpg")])
            print(f"  → {count} frames in frames/{lens}/", flush=True)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 1_extract_frames.py <video_path> [interval_seconds]")
        sys.exit(1)
    extract_frames(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 10)
