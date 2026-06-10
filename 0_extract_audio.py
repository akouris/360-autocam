#!/usr/bin/env python3
"""
Step 0: Extract the camera's own audio track to data/audio.wav.

Runs FIRST — before any frame extraction or face detection — so that audio
problems (no audio stream, unreadable file, and downstream external-mic sync
failures) surface within seconds instead of after minutes of wasted frame work.

Output: data/audio.wav  (16 kHz mono PCM; reused by sync, diarize and automap)
"""
import os, sys, subprocess


def has_audio_stream(video_path):
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a",
         "-show_entries", "stream=index", "-of", "csv=p=0", video_path],
        capture_output=True, text=True)
    return bool(probe.stdout.strip())


def extract_audio(video_path, audio_path="data/audio.wav"):
    os.makedirs("data", exist_ok=True)

    if not os.path.exists(video_path):
        print(f"Error: video not found: {video_path}", flush=True)
        sys.exit(1)

    if not has_audio_stream(video_path):
        print(f"Error: {video_path} has no audio stream — cannot diarize or sync.",
              flush=True)
        sys.exit(1)

    print(f"Extracting camera audio from {video_path} → {audio_path} …", flush=True)
    r = subprocess.run(
        ["ffmpeg", "-i", video_path, "-vn", "-acodec", "pcm_s16le",
         "-ar", "16000", "-ac", "1", audio_path, "-y", "-loglevel", "error"],
        capture_output=True, text=True)
    if r.returncode != 0:
        print("Error: ffmpeg failed to extract camera audio:", flush=True)
        print(r.stderr.strip(), flush=True)
        sys.exit(1)

    size = os.path.getsize(audio_path) if os.path.exists(audio_path) else 0
    if size < 1000:
        print(f"Error: extracted audio is empty ({size} bytes).", flush=True)
        sys.exit(1)

    print(f"Camera audio saved: {audio_path}  ({size/1e6:.1f} MB)", flush=True)
    return audio_path


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 0_extract_audio.py <video_path>")
        sys.exit(1)
    extract_audio(sys.argv[1])
