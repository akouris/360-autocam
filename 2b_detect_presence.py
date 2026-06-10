#!/usr/bin/env python3
"""
Step 2b: Per-time face PRESENCE for each known speaker position.

Step 2 (2_detect_speakers.py) finds WHERE people sit (cluster positions) but
early-exits and discards timing. This step scans EVERY extracted frame and
records, for each known position, whether that person's face is visible at that
moment. Output drives the render carousel so we never show an empty chair for
someone who has left, nor hide someone who is present but quiet.

Presence is intrinsically noisy (a speaker turning their head drops a frame),
so raw per-frame hits are temporally smoothed: a position counts as present at
time t if a face was detected there within ±SMOOTH_PAD seconds.

Output: data/presence.json
  {
    "interval": <seconds per frame>,
    "num_frames": N,
    "positions": { "<pos_key>": [t0, t1, ...] }   # frame centre-times present (smoothed)
  }
"""
import os, sys, json, math
import numpy as np
import cv2

import importlib.util
_spec = importlib.util.spec_from_file_location(
    "det", os.path.join(os.path.dirname(__file__), "2_detect_speakers.py"))
det = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(det)

INTERVAL          = int(os.environ.get("FRAME_INTERVAL", "10"))  # seconds/frame (matches step 1)
PRESENCE_HFOV     = 75.0    # crop fov for the per-position probe
CENTER_YAW_TOL    = 15.0    # face within this angle of crop centre counts as "this person"
CENTER_PITCH_FRAC = 0.38    # vertical tolerance as fraction of crop height
SMOOTH_PAD        = 15.0    # seconds: presence persists ±this around a real hit


def _local_from_global(global_yaw, lens):
    """Invert global_yaw=(offset+sign*local)%360 back to signed local yaw."""
    offset, sign = (0.0, +1) if lens == "front" else (180.0, -1)
    local = ((global_yaw - offset) * sign) % 360.0
    if local > 180.0:
        local -= 360.0
    return local


def detect_presence(frames_root, positions):
    focal_norm = 0.5 / math.tan(math.radians(PRESENCE_HFOV / 2))
    cx_tol = focal_norm * math.tan(math.radians(CENTER_YAW_TOL))

    # frame list per lens (front+back share indices); equirect uses single dir
    lens_dirs = {}
    for lens in ("front", "back", "equirect"):
        d = os.path.join(frames_root, lens)
        if os.path.isdir(d):
            lens_dirs[lens] = sorted(f for f in os.listdir(d) if f.endswith(".jpg"))
    if not lens_dirs:
        print(f"No frame dirs under {frames_root}", flush=True)
        return {}

    num_frames = max(len(v) for v in lens_dirs.values())
    print(f"Presence scan: {num_frames} frames × {len(positions)} positions "
          f"(interval={INTERVAL}s)…", flush=True)

    # raw[pos] = set of frame indices where a face was detected at that position
    raw = {p: set() for p in positions}
    pos_by_lens = {}
    for p, info in positions.items():
        pos_by_lens.setdefault(info.get("lens", "front"), []).append(p)

    with det._make_landmarker() as lm:
        for idx in range(num_frames):
            print(f"PROGRESS {idx+1}/{num_frames}", flush=True)
            fname = f"frame_{idx+1:04d}.jpg"
            for lens, plist in pos_by_lens.items():
                fp = os.path.join(frames_root, lens, fname)
                img = cv2.imread(fp)
                if img is None:
                    continue
                for p in plist:
                    info  = positions[p]
                    lyaw  = _local_from_global(info["yaw_degrees"], lens)
                    pitch = info.get("pitch_degrees", 0)
                    crop  = det.fisheye_to_flat(img, lyaw, pitch,
                                                h_fov_deg=PRESENCE_HFOV, out_size=det.CROP_SIZE)
                    for (cx, cy) in det.detect_faces_on_flat_crop(crop, lm):
                        if abs(cx - 0.5) <= cx_tol and abs(cy - 0.5) <= CENTER_PITCH_FRAC:
                            raw[p].add(idx)
                            break

    # temporal smoothing: a position is present at frame i if a raw hit lies
    # within ±SMOOTH_PAD seconds (±pad_frames frames)
    pad_frames = int(round(SMOOTH_PAD / INTERVAL))
    presence = {}
    for p in positions:
        hits = raw[p]
        present_times = []
        for i in range(num_frames):
            if any((i + k) in hits for k in range(-pad_frames, pad_frames + 1)):
                present_times.append(round(i * INTERVAL, 2))
        presence[p] = present_times
        spk = ""
        print(f"  {p}: raw {len(hits)}/{num_frames} frames, "
              f"smoothed present {len(present_times)}/{num_frames}", flush=True)

    return {"interval": INTERVAL, "num_frames": num_frames, "positions": presence}


if __name__ == "__main__":
    pos_path = "data/speaker_positions.json"
    if not os.path.exists(pos_path):
        print(f"{pos_path} not found — run step 2 first.")
        sys.exit(1)
    positions = json.load(open(pos_path))
    result = detect_presence("frames", positions)
    if not result:
        sys.exit(1)
    os.makedirs("data", exist_ok=True)
    with open("data/presence.json", "w") as f:
        json.dump(result, f, indent=2)
    print("Presence written to data/presence.json", flush=True)
