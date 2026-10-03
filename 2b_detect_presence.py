#!/usr/bin/env python3
"""
Step 2b: Per-time face PRESENCE for each seat.

Built from data/face_tracks.json (step 2 already scanned every frame), so no
second detection pass is needed. Presence drives the render carousel so we
never show an empty chair for someone who has left, nor hide someone who is
present but quiet.

Presence is intrinsically noisy (a speaker turning their head drops a frame),
so raw per-frame hits are temporally smoothed: a seat counts as present at
time t if its face was seen within ±SMOOTH_PAD seconds.

Output: data/presence.json
  {
    "interval": <seconds per frame>,
    "num_frames": N,
    "positions": { "<seat_key>": [t0, t1, ...] }   # frame times present (smoothed)
  }
"""
import os, sys, json

SMOOTH_PAD = 8.0    # seconds: presence persists ±this around a real sighting


def detect_presence(tracks):
    interval   = float(tracks["interval"])
    num_frames = int(tracks["num_frames"])
    pad = max(1, int(round(SMOOTH_PAD / interval)))
    presence = {}
    for seat, track in tracks["seats"].items():
        hits = [i for i, t in enumerate(track) if t]
        present = set()
        for i in hits:
            present.update(range(max(0, i - pad), min(num_frames, i + pad + 1)))
        presence[seat] = [round(i * interval, 2) for i in sorted(present)]
        print(f"  {seat}: seen in {len(hits)}/{num_frames} frames, "
              f"present (smoothed) {len(present)}/{num_frames}", flush=True)
    return {"interval": interval, "num_frames": num_frames, "positions": presence}


if __name__ == "__main__":
    path = "data/face_tracks.json"
    if not os.path.exists(path):
        print(f"{path} not found — run step 2 (detect speakers) first.")
        sys.exit(1)
    with open(path) as f:
        result = detect_presence(json.load(f))
    print("PROGRESS 1/1", flush=True)
    with open("data/presence.json", "w") as f:
        json.dump(result, f, indent=2)
    print("Presence written to data/presence.json", flush=True)
