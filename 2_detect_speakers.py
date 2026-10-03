#!/usr/bin/env python3
"""
Step 2: Find every face in every extracted frame, then work out the SEATS.

For each frame, rectilinear crops are sampled all the way round the 360° view
(geometry.Panorama — both lenses, seam included) and MediaPipe FaceLandmarker
runs on each. Every face becomes a detection with an exact global yaw/pitch,
its angular size, and a mouth-open value.

Seats are peaks in the circular yaw distribution of detections (people sitting
close together give separate peaks instead of being averaged into the gap
between them). Rejected as seats:
  • background faces — much smaller than the people at the table
  • static faces     — posters, logos, faces on a monitor (no expression change)
  • rare faces       — seen in too few frames

Outputs
  data/speaker_positions.json  {seat_key: {yaw_degrees, pitch_degrees, face_deg, ...}}
  data/face_tracks.json        per-seat face position/size in every frame
                               (drives presence, framing and mouth tracking)

Mode is read from VIDEO_MODE env var (default: fisheye).
"""

import cv2, os, json, math, sys
import numpy as np
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import geometry

FACE_LANDMARKER_MODEL = os.environ.get(
    "FACE_LANDMARKER_MODEL",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "face_landmarker.task"),
)

SECTOR_YAWS  = list(range(-180, 180, 30))   # 12 crops round the full circle
CROP_FOV     = 75.0                          # heavy overlap between neighbours
CROP_SIZE    = 640
N_WORKERS    = max(1, min(4, (os.cpu_count() or 4) - 2))

MIN_FACE_DEG       = 4.5    # faces smaller than this are background people …
MIN_FACE_REL       = 0.45   # … or smaller than this × the typical table face
MIN_SEAT_FRAC      = 0.06   # a seat must be occupied in ≥6% of frames …
MIN_SEAT_FRAMES    = 3      # … and at least this many frames
SEAT_MERGE_DEG     = 7.0    # peaks closer than this are one person
SEAT_GATE_DEG      = 15.0   # max yaw distance from a seat for a detection to count
SAME_PERSON_DEG    = 22.0   # nearby peaks this close may be one person who moved …
CO_OCCUR_FRAC      = 0.2    # … if both are seen in the same frame less often than this
STATIC_JAW_STD     = 0.006  # a face whose mouth never moves …
STATIC_YAW_STD     = 0.35   # … and never moves at all, is a poster / screen

# MediaPipe landmark indices
FOREHEAD, CHIN = 10, 152


def load_frame_meta():
    try:
        with open("frames/meta.json") as f:
            return json.load(f)
    except Exception:
        return {"interval": float(os.environ.get("FRAME_INTERVAL", "10"))}


def frame_files():
    """Sorted frame names present for this mode."""
    mode = geometry.video_mode()
    d = "frames/equirect" if mode == "equirect" else "frames/front"
    if not os.path.isdir(d):
        return []
    return sorted(f for f in os.listdir(d) if f.endswith(".jpg"))


def load_panorama(fname, fov=None):
    if geometry.video_mode() == "equirect":
        img = cv2.imread(os.path.join("frames/equirect", fname))
        return None if img is None else geometry.Panorama(equirect=img)
    fr = cv2.imread(os.path.join("frames/front", fname))
    bk = cv2.imread(os.path.join("frames/back", fname))
    if fr is None or bk is None:
        return None
    return geometry.Panorama(front=fr, back=bk, fov=fov)


# ── MediaPipe ─────────────────────────────────────────────────────────────────

def make_landmarker(num_faces=4, video=False):
    import mediapipe as mp
    opts = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=FACE_LANDMARKER_MODEL),
        running_mode=(mp.tasks.vision.RunningMode.VIDEO if video
                      else mp.tasks.vision.RunningMode.IMAGE),
        num_faces=num_faces,
        output_face_blendshapes=True,
        min_face_detection_confidence=0.35,
        min_face_presence_confidence=0.35,
        min_tracking_confidence=0.35,
    )
    return mp.tasks.vision.FaceLandmarker.create_from_options(opts)


def jaw_open(result, i):
    try:
        for c in result.face_blendshapes[i]:
            if c.category_name == "jawOpen":
                return float(c.score)
    except Exception:
        pass
    return 0.0


def faces_in_view(img_bgr, landmarker, yaw, pitch, h_fov, v_fov):
    """Run FaceLandmarker on a rectilinear view; return faces in GLOBAL angles:
    [{yaw, pitch, size, jaw, edge}] — size = forehead→chin angle in degrees,
    edge = distance of the face from the view centre (0 = centre, 1 = corner)."""
    import mediapipe as mp
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    res = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
    out = []
    for i, lm in enumerate(res.face_landmarks):
        xs = np.array([p.x for p in lm]); ys = np.array([p.y for p in lm])
        u, v = float((xs.min() + xs.max()) / 2), float((ys.min() + ys.max()) / 2)
        gy, gp = geometry.view_point_to_yaw_pitch(u, v, yaw, pitch, h_fov, v_fov)
        ty, tp = geometry.view_point_to_yaw_pitch(lm[FOREHEAD].x, lm[FOREHEAD].y, yaw, pitch, h_fov, v_fov)
        cy_, cp = geometry.view_point_to_yaw_pitch(lm[CHIN].x, lm[CHIN].y, yaw, pitch, h_fov, v_fov)
        size = ang_dist(ty, tp, cy_, cp)
        out.append({"yaw": gy, "pitch": gp, "size": size, "jaw": jaw_open(res, i),
                    "edge": float(max(abs(u - 0.5), abs(v - 0.5)) * 2)})
    return out


def ang_dist(y1, p1, y2, p2):
    a1, b1, a2, b2 = map(math.radians, (y1, p1, y2, p2))
    c = (math.sin(b1) * math.sin(b2) + math.cos(b1) * math.cos(b2) * math.cos(a1 - a2))
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def yaw_diff(a, b):
    return abs(geometry.norm_yaw(a - b))


# ── Per-frame scan (parallel) ─────────────────────────────────────────────────

_LM = None
_FOV = None


def _init_worker(fov):
    global _LM, _FOV
    _LM = make_landmarker()
    _FOV = fov


def scan_frame(args):
    idx, fname = args
    pano = load_panorama(fname, _FOV)
    if pano is None:
        return idx, []
    faces = []
    for sy in SECTOR_YAWS:
        crop = pano.view(sy, 0.0, CROP_FOV, CROP_FOV, CROP_SIZE, CROP_SIZE)
        faces += faces_in_view(crop, _LM, sy, 0.0, CROP_FOV, CROP_FOV)
    # the same face appears in 2–3 overlapping sectors: keep the most central copy
    faces.sort(key=lambda f: f["edge"])
    kept = []
    for f in faces:
        if all(ang_dist(f["yaw"], f["pitch"], k["yaw"], k["pitch"])
               > max(3.0, 0.6 * max(f["size"], k["size"])) for k in kept):
            kept.append(f)
    return idx, kept


# ── Seats ─────────────────────────────────────────────────────────────────────

def circ_median(yaws):
    """Median of angles (robust to wrap at ±180)."""
    yaws = np.asarray(yaws, dtype=float)
    ref = math.degrees(math.atan2(np.sin(np.radians(yaws)).mean(), np.cos(np.radians(yaws)).mean()))
    return geometry.norm_yaw(ref + float(np.median([geometry.norm_yaw(y - ref) for y in yaws])))


def find_seats(dets, num_frames):
    """dets: list of (frame, face). Returns ({key: seat}, assignment list)."""
    if not dets:
        return {}, []
    sizes = np.array([f["size"] for _, f in dets])
    table_face = float(np.percentile(sizes, 75))
    min_size = max(2.0, min(MIN_FACE_DEG, MIN_FACE_REL * table_face))
    fg = [(i, f) for i, f in dets if f["size"] >= min_size]
    print(f"  typical face {table_face:.1f}°, ignoring faces < {min_size:.1f}° "
          f"({len(dets) - len(fg)} background detections)", flush=True)

    # circular yaw histogram, 1° bins, Gaussian-smoothed (σ = 2.5°)
    hist = np.zeros(360)
    for _, f in fg:
        hist[int(round(f["yaw"])) % 360] += 1
    k = np.arange(-10, 11)
    kern = np.exp(-0.5 * (k / 2.5) ** 2)
    sm = np.array([sum(hist[(b + kk) % 360] * w for kk, w in zip(k, kern)) for b in range(360)])

    peaks = [b for b in range(360)
             if sm[b] > 0 and sm[b] >= sm[(b - 1) % 360] and sm[b] > sm[(b + 1) % 360]]
    peaks.sort(key=lambda b: -sm[b])
    centres = []
    for b in peaks:
        if all(yaw_diff(b, c) >= SEAT_MERGE_DEG for c in centres):
            centres.append(float(b))

    min_frames = max(MIN_SEAT_FRAMES, int(MIN_SEAT_FRAC * num_frames))

    def assign(centres):
        out = {c: [] for c in centres}
        for i, f in fg:
            near = sorted(centres, key=lambda c: yaw_diff(f["yaw"], c))
            c = near[0]
            gate = SEAT_GATE_DEG
            if len(near) > 1:
                gate = min(gate, 0.6 * yaw_diff(c, near[1]))
            if yaw_diff(f["yaw"], c) <= gate:
                out[c].append((i, f))
        return out

    # One person who shifts around makes two nearby peaks; two neighbours make two
    # peaks that are seen IN THE SAME FRAMES. Merge nearby peaks that rarely co-occur.
    groups = assign(centres)
    merged_any = True
    while merged_any and len(centres) > 1:
        merged_any = False
        frames_of = {c: {i for i, _ in groups[c]} for c in centres}
        pairs = sorted(((yaw_diff(a, b), a, b) for ai, a in enumerate(centres)
                        for b in centres[ai + 1:] if yaw_diff(a, b) < SAME_PERSON_DEG),
                       key=lambda x: x[0])
        for _, a, b in pairs:
            fa, fb = frames_of[a], frames_of[b]
            if not fa or not fb:
                continue
            if len(fa & fb) < CO_OCCUR_FRAC * min(len(fa), len(fb)):
                keep, drop = (a, b) if len(fa) >= len(fb) else (b, a)
                yaws = [f["yaw"] for _, f in groups[a] + groups[b]]
                centres = [c for c in centres if c not in (a, b)] + [circ_median(yaws) % 360.0]
                groups = assign(centres)
                merged_any = True
                break

    # keep seats occupied often enough; re-assign after dropping weak ones
    centres = [c for c in centres if len({i for i, _ in groups[c]}) >= min_frames]
    groups = assign(centres) if centres else {}

    seats = {}
    for c in sorted(centres, key=geometry.norm_yaw):
        g = groups[c]
        if not g:
            continue
        yaws = [f["yaw"] for _, f in g]
        jaws = [f["jaw"] for _, f in g]
        if np.std(jaws) < STATIC_JAW_STD and np.std([geometry.norm_yaw(y - c) for y in yaws]) < STATIC_YAW_STD:
            print(f"  yaw {geometry.norm_yaw(c):+.0f}°: static face (poster/screen) — not a seat", flush=True)
            continue
        yaw = circ_median(yaws)
        key = f"seat_{len(seats)}"
        seats[key] = {
            "yaw_degrees":     round(yaw, 2),
            "pitch_degrees":   round(float(np.median([f["pitch"] for _, f in g])), 2),
            "face_deg":        round(float(np.median([f["size"] for _, f in g])), 2),
            "lens":            "front" if abs(yaw) <= 90 else "back",
            "frames_present":  len({i for i, _ in g}),
            "detection_count": len(g),
            "geom":            2,
        }
    return seats, fg


def build_tracks(seats, fg, num_frames):
    """Per seat, per frame: [yaw, pitch, size, jaw] of that seat's face or None."""
    tracks = {k: [None] * num_frames for k in seats}
    best = {}
    for i, f in fg:
        near = sorted(seats, key=lambda k: yaw_diff(f["yaw"], seats[k]["yaw_degrees"]))
        if not near:
            continue
        k = near[0]
        gate = SEAT_GATE_DEG
        if len(near) > 1:
            gate = min(gate, 0.6 * yaw_diff(seats[k]["yaw_degrees"], seats[near[1]]["yaw_degrees"]))
        d = yaw_diff(f["yaw"], seats[k]["yaw_degrees"])
        if d > gate:
            continue
        if (k, i) not in best or d < best[(k, i)]:   # two faces near a seat: keep the closer
            best[(k, i)] = d
            tracks[k][i] = [round(f["yaw"], 2), round(f["pitch"], 2),
                            round(f["size"], 2), round(f["jaw"], 3)]
    return tracks


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    files = frame_files()
    if not files:
        print("No frames found. Run step 1 first.")
        sys.exit(1)
    meta = load_frame_meta()
    interval = float(meta.get("interval", 10))
    fov = geometry.lens_fov() if geometry.video_mode() == "fisheye" else None
    N = len(files)
    print(f"Scanning {N} frames × {len(SECTOR_YAWS)} directions "
          f"(interval {interval}s, lens fov {fov}, {N_WORKERS} workers)…", flush=True)

    per_frame = [[] for _ in range(N)]
    done = 0
    with Pool(N_WORKERS, initializer=_init_worker, initargs=(fov,)) as pool:
        for idx, faces in pool.imap_unordered(scan_frame, list(enumerate(files)), chunksize=2):
            per_frame[idx] = faces
            done += 1
            print(f"PROGRESS {done}/{N}", flush=True)

    dets = [(i, f) for i, fl in enumerate(per_frame) for f in fl]
    print(f"  → {len(dets)} face detections", flush=True)

    seats, fg = find_seats(dets, N)
    if not seats:
        print("No stable seats found.")
        sys.exit(1)

    os.makedirs("data", exist_ok=True)
    with open("data/speaker_positions.json", "w") as f:
        json.dump(seats, f, indent=2)
    with open("data/face_tracks.json", "w") as f:
        json.dump({"interval": interval, "num_frames": N,
                   "seats": build_tracks(seats, fg, N)}, f)

    print("\nSeats saved to data/speaker_positions.json:")
    for k, s in seats.items():
        print(f"  {k}: yaw={s['yaw_degrees']:+.1f}°  pitch={s['pitch_degrees']:+.1f}°  "
              f"face={s['face_deg']:.1f}°  present {s['frames_present']}/{N} frames", flush=True)
