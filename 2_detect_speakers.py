#!/usr/bin/env python3
"""
Step 2: Detect faces and cluster by angular position.

Extracts flat perspective crops from the fisheye using the equidistant
projection model (same as FFmpeg v360=fisheye:flat), then runs MediaPipe
FaceLandmarker on those crops. This avoids the fisheye distortion that
makes Haar cascade give wrong positions.

Mode is read from VIDEO_MODE env var (default: fisheye).
"""

import cv2, os, json, math, subprocess, sys, tempfile
import numpy as np
from sklearn.cluster import DBSCAN

FACE_LANDMARKER_MODEL = os.environ.get(
    "FACE_LANDMARKER_MODEL",
    os.path.join(os.path.dirname(__file__), "data", "face_landmarker.task"),
)

# Sectors to scan per lens: yaw values (degrees) for perspective crop centres.
# h_fov=75° per crop; these 5 sectors cover ±97.5° with heavy overlap.
SECTORS    = [-60, -30, 0, 30, 60]
CROP_H_FOV = 75   # degrees
CROP_SIZE  = 640  # pixels (square crop)

# Early-exit: stop scanning frames once clusters have been stable for N consecutive checks
EARLY_EXIT_INTERVAL = 30   # check stability every N frames
EARLY_EXIT_STABLE_N = 3    # require this many consecutive stable checks before stopping

_REMAP_CACHE: dict = {}    # (W, H, yaw, pitch, h_fov, out_size) → (map_x, map_y)


# ── Equidistant fisheye → flat perspective remap ──────────────────────────────

def _build_remap(fisheye_w, fisheye_h, yaw_deg, pitch_deg=0,
                 h_fov_deg=CROP_H_FOV, out_size=CROP_SIZE):
    """
    Build cv2.remap map_x/map_y that extracts a rectilinear view from an
    equidistant fisheye image (the projection that FFmpeg v360=fisheye uses).
    fisheye circle is assumed to be inscribed in the image (radius = min(W,H)/2).
    """
    key = (fisheye_w, fisheye_h, yaw_deg, pitch_deg, h_fov_deg, out_size)
    if key in _REMAP_CACHE:
        return _REMAP_CACHE[key]

    r_max   = min(fisheye_w, fisheye_h) / 2.0   # fisheye circle radius in pixels
    cx, cy  = fisheye_w / 2.0, fisheye_h / 2.0

    yaw   = math.radians(yaw_deg)
    pitch = math.radians(pitch_deg)
    hfov  = math.radians(h_fov_deg)
    vfov  = hfov  # square crop → same vertical fov

    # For each output pixel, compute the 3-D ray direction in the flat image
    # and rotate it by (yaw, pitch).
    out_xs = np.linspace(-hfov / 2, hfov / 2, out_size)
    out_ys = np.linspace(-vfov / 2, vfov / 2, out_size)
    ax, ay = np.meshgrid(out_xs, out_ys)   # horizontal / vertical angle grids

    # Rectilinear ray (z points forward in the lens frame)
    rx =  np.tan(ax)
    ry = -np.tan(ay)
    rz =  np.ones_like(ax)
    norm = np.sqrt(rx**2 + ry**2 + rz**2)
    rx, ry, rz = rx/norm, ry/norm, rz/norm

    # Rotate ray by yaw (around Y axis) then pitch (around X axis)
    # Yaw rotation
    cos_y, sin_y = math.cos(yaw), math.sin(yaw)
    rx2 =  cos_y * rx + sin_y * rz
    ry2 =  ry
    rz2 = -sin_y * rx + cos_y * rz
    # Pitch rotation
    cos_p, sin_p = math.cos(pitch), math.sin(pitch)
    rx3 = rx2
    ry3 = cos_p * ry2 - sin_p * rz2
    rz3 = sin_p * ry2 + cos_p * rz2

    # Equidistant fisheye: theta = angle from optical axis; r = r_max * theta / (pi/2)
    theta = np.arccos(np.clip(rz3, -1, 1))
    # Negate ry3 so that the fisheye y-axis (y-down) maps correctly.
    # Without this, the extracted crop is upside-down.
    phi   = np.arctan2(-ry3, rx3)

    r_px  = (r_max * theta) / (math.pi / 2)
    map_x = (cx + r_px * np.cos(phi)).astype(np.float32)
    map_y = (cy + r_px * np.sin(phi)).astype(np.float32)

    _REMAP_CACHE[key] = (map_x, map_y)
    return map_x, map_y


def fisheye_to_flat(img, yaw_deg, pitch_deg=0, h_fov_deg=CROP_H_FOV, out_size=CROP_SIZE):
    """Extract a flat perspective crop from a fisheye image."""
    H, W = img.shape[:2]
    mx, my = _build_remap(W, H, yaw_deg, pitch_deg, h_fov_deg, out_size)
    return cv2.remap(img, mx, my, cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=0)


# ── MediaPipe face detection ──────────────────────────────────────────────────

def _make_landmarker():
    import mediapipe as mp
    opts = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=FACE_LANDMARKER_MODEL),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_faces=4,
        min_face_detection_confidence=0.3,
        min_face_presence_confidence=0.3,
        min_tracking_confidence=0.3,
    )
    return mp.tasks.vision.FaceLandmarker.create_from_options(opts)


def detect_faces_on_flat_crop(crop_bgr, landmarker):
    """Run MediaPipe on a perspective crop; return list of (cx_norm, cy_norm)."""
    import mediapipe as mp
    rgb    = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect(mp_img)
    faces  = []
    for face in result.face_landmarks:
        xs = [lm.x for lm in face]
        ys = [lm.y for lm in face]
        faces.append((float(np.mean(xs)), float(np.mean(ys))))
    return faces


# ── Per-lens scanning ─────────────────────────────────────────────────────────

def scan_lens(frames_dir, lens_label, yaw_offset, stream, yaw_sign,
              progress_offset=0, progress_total=None):
    """
    For each frame in frames_dir, extract perspective crops at SECTORS yaw values,
    run MediaPipe, and record each detected face as (local_yaw, global_yaw, pitch).

    progress_offset / progress_total: emit PROGRESS relative to a grand total
    across all lenses so the progress bar never goes backward.

    yaw_sign controls the back-lens horizontal mirror:
      front lens:  yaw_sign = +1  (right in scene = positive local_yaw)
      back lens:   yaw_sign = -1  (back faces opposite direction, so mirrored)
    global_yaw = (yaw_offset + yaw_sign * local_yaw) % 360
    """
    frame_files = sorted(f for f in os.listdir(frames_dir) if f.endswith(".jpg"))
    total      = len(frame_files)
    prog_denom = progress_total if progress_total is not None else total
    detections = []

    print(f"Scanning {lens_label} lens: {total} frames × {len(SECTORS)} sectors…", flush=True)

    stable_count = 0
    prev_centers: list | None = None
    focal_norm = 0.5 / math.tan(math.radians(CROP_H_FOV / 2))

    with _make_landmarker() as lm:
        for i, fname in enumerate(frame_files):
            print(f"PROGRESS {progress_offset + i + 1}/{prog_denom}", flush=True)
            img = cv2.imread(os.path.join(frames_dir, fname))
            if img is None:
                continue

            for sector_yaw in SECTORS:
                crop  = fisheye_to_flat(img, sector_yaw)
                faces = detect_faces_on_flat_crop(crop, lm)
                for (cx_norm, cy_norm) in faces:
                    local_yaw   = sector_yaw + math.degrees(math.atan((cx_norm - 0.5) / focal_norm))
                    local_pitch = -math.degrees(math.atan((cy_norm - 0.5) / focal_norm))
                    global_yaw  = (yaw_offset + yaw_sign * local_yaw) % 360.0
                    detections.append({
                        "local_yaw":  local_yaw,
                        "global_yaw": global_yaw,
                        "pitch":      local_pitch,
                        "frame":      i,
                    })

            # Early-exit check: stop once clusters have been stable for several intervals
            if detections and (i + 1) % EARLY_EXIT_INTERVAL == 0:
                yaws = np.array([d["local_yaw"] for d in detections]).reshape(-1, 1)
                db   = DBSCAN(eps=15.0, min_samples=3).fit(yaws)
                curr = sorted(float(np.mean(yaws[db.labels_ == l]))
                              for l in set(db.labels_) if l != -1)
                if (prev_centers is not None
                        and len(curr) == len(prev_centers)
                        and len(curr) > 0
                        and all(abs(a - b) < 5.0 for a, b in zip(curr, prev_centers))):
                    stable_count += 1
                    if stable_count >= EARLY_EXIT_STABLE_N:
                        print(f"  Clusters stable — stopping after {i+1}/{total} frames", flush=True)
                        break
                else:
                    stable_count = 0
                prev_centers = curr

    print(f"  → {len(detections)} raw detections in {lens_label}", flush=True)
    return detections


def cluster_detections(detections, lens_label, yaw_offset, stream, yaw_sign):
    """DBSCAN-cluster detections by local_yaw; return position dict."""
    if not detections:
        return {}

    local_yaws = np.array([d["local_yaw"] for d in detections], dtype=float).reshape(-1, 1)
    db     = DBSCAN(eps=15.0, min_samples=3).fit(local_yaws)
    labels = db.labels_

    clusters = {}
    for label in set(labels):
        if label == -1:
            continue
        mask       = labels == label
        mean_local = float(np.mean(local_yaws[mask]))
        mean_pitch = float(np.mean([d["pitch"] for d, m in zip(detections, mask) if m]))
        count      = int(mask.sum())
        global_yaw = round((yaw_offset + yaw_sign * mean_local) % 360.0, 1)
        pitch      = round(max(-45.0, min(45.0, mean_pitch)), 1)

        key = f"{lens_label}_{label}"
        clusters[key] = {
            "yaw_degrees":     global_yaw,
            "pitch_degrees":   pitch,
            "h_fov":           CROP_H_FOV,
            "lens":            lens_label,
            "stream":          stream,
            "detection_count": count,
        }
        print(f"  cluster {key}: yaw={global_yaw}°  pitch={pitch}°  n={count}", flush=True)

    return clusters


# ── Equirect (YouTube 360°) path ──────────────────────────────────────────────

def detect_equirect(frames_dir):
    """For equirectangular frames: detect faces directly (less distortion)."""
    frame_files = sorted(f for f in os.listdir(frames_dir) if f.endswith(".jpg"))
    total = len(frame_files)
    detections = []

    with _make_landmarker() as lm:
        for i, fname in enumerate(frame_files):
            print(f"PROGRESS {i+1}/{total}", flush=True)
            img = cv2.imread(os.path.join(frames_dir, fname))
            if img is None:
                continue
            import mediapipe as mp
            H, W = img.shape[:2]
            rgb    = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = lm.detect(mp_img)
            for face in result.face_landmarks:
                xs = [lm2.x for lm2 in face]
                ys = [lm2.y for lm2 in face]
                detections.append({
                    "cx": float(np.mean(xs)) * W,
                    "cy": float(np.mean(ys)) * H,
                    "width": W, "height": H,
                })

    from sklearn.cluster import DBSCAN as _DB
    if not detections:
        return {}
    xs      = np.array([d["cx"] for d in detections])
    widths  = np.array([d["width"] for d in detections])
    xs_norm = (xs / widths) * 1000
    db = _DB(eps=80.0, min_samples=3).fit(xs_norm.reshape(-1, 1))
    clusters = {}
    for label in set(db.labels_):
        if label == -1:
            continue
        mask   = db.labels_ == label
        mean_x = float(np.mean(xs[mask]))
        mean_y = float(np.mean([d["cy"] for d, m in zip(detections, mask) if m]))
        mean_w = float(np.mean(widths[mask]))
        mean_h = float(np.mean([d["height"] for d in detections]))
        yaw    = round((mean_x / mean_w) * 360.0 - 180.0, 1)
        pitch  = round((0.5 - mean_y / mean_h) * 180.0, 1)
        clusters[f"equirect_{label}"] = {
            "yaw_degrees": yaw, "pitch_degrees": pitch, "h_fov": 75,
            "lens": "equirect", "stream": 0, "detection_count": int(mask.sum()),
        }
    return clusters


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    mode         = os.environ.get("VIDEO_MODE", "fisheye")
    all_clusters = {}

    if mode == "equirect":
        frames_dir = "frames/equirect"
        if not os.path.isdir(frames_dir) or not os.listdir(frames_dir):
            print(f"No frames found in {frames_dir}. Run step 1 first.")
        else:
            all_clusters = detect_equirect(frames_dir)
    else:
        # yaw_sign: front=+1 (right in scene = +yaw), back=-1 (mirrored since lens faces backward)
        lens_configs = [("front", 0.0, 0, +1), ("back", 180.0, 1, -1)]
        # Pre-count frames across all lenses so PROGRESS never resets between lenses
        def _count_frames(d):
            try: return len([f for f in os.listdir(d) if f.endswith(".jpg")])
            except OSError: return 0
        grand_total  = sum(_count_frames(f"frames/{l}") for l, *_ in lens_configs)
        prog_offset  = 0
        for lens, yaw_offset, stream, yaw_sign in lens_configs:
            frames_dir = f"frames/{lens}"
            if not os.path.isdir(frames_dir) or not os.listdir(frames_dir):
                print(f"No frames in {frames_dir}, skipping.")
                continue
            dets     = scan_lens(frames_dir, lens, yaw_offset, stream, yaw_sign,
                                 progress_offset=prog_offset, progress_total=grand_total)
            prog_offset += _count_frames(frames_dir)
            clusters = cluster_detections(dets, lens, yaw_offset, stream, yaw_sign)
            all_clusters.update(clusters)

    if all_clusters:
        os.makedirs("data", exist_ok=True)
        out_path = "data/speaker_positions.json"
        with open(out_path, "w") as f:
            json.dump(all_clusters, f, indent=2)
        print(f"\nSpeaker positions saved to {out_path}:")
        for name, info in sorted(all_clusters.items()):
            print(f"  {name}: yaw={info['yaw_degrees']}°  pitch={info['pitch_degrees']}°  "
                  f"stream={info['stream']}  ({info['detection_count']} detections)")
    else:
        print("No stable speaker positions found.")
        sys.exit(1)
