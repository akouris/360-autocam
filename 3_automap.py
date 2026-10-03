#!/usr/bin/env python3
"""
Step 3b: Who is speaking, when — audio (diarization) fused with video (mouths).

1. MOUTH TRACKING  (cached → data/mouth_tracks.npz)
   One pass over the video at TRACK_FPS. For every seat a tight crop that
   follows the face (from data/face_tracks.json) is rendered via v360, and
   MediaPipe FaceLandmarker (VIDEO mode) measures per frame:
     mouth  — inner-lip gap ÷ face height (size-independent openness)
     jaw    — jawOpen blendshape
     yaw/pitch/size — precise face position (also used for render framing)
   Faces are ~100px tall in these crops (the old 256px/75° crops gave 30–46px
   faces and 0.4–2px lip movement = noise).

2. PER-SEGMENT EVIDENCE
   For EVERY diarization segment and every seat:
     sync   — best correlation (±240ms lag search) between mouth openness and
              the audio loudness envelope
     motion — how much the mouth moves vs that person's own baseline
   → softmax → p(seat | segment). Absent seats get ~0.

3. LABEL → SEAT
   Each diarization label is mapped by pooling ALL of its segments (duration-
   weighted). Labels whose segments never move any on-camera mouth are
   off-camera audio (music, phone-in, broadcast) and are excluded.

4. SPEAKER PROBABILITIES  → data/speaker_probs.json
   Every STEP seconds: which seat is speaking, as a probability distribution
   (label prior × local mouth evidence), plus an overlap flag (two labels, or
   pyannote OSD). The renderer uses this to pick single / double / triple shots
   — when it's not sure who is talking it shows the candidates together.

Also writes data/speaker_map.json (label → seat) and data/framing_tracks.json
(1 Hz face positions per seat, used by the renderer to frame each shot).
"""

import os, sys, json, math, subprocess
import numpy as np
from collections import defaultdict
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import geometry

FACE_LANDMARKER_MODEL = os.environ.get(
    "FACE_LANDMARKER_MODEL",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "face_landmarker.task"),
)

TRACK_FPS    = 12.5     # mouth motion is 2–7 Hz; 12.5 fps samples it cleanly
CROP_PX      = 256
CHUNK_SEC    = 30.0
N_WORKERS    = max(1, min(3, (os.cpu_count() or 4) // 3))
STEP         = 0.5      # resolution of speaker_probs.json
MAX_LAG      = 3        # frames (±240ms) for audio↔mouth sync search
SOFTMAX_T    = 3.0      # sharpness of p(seat | evidence)
OFFCAM_EVID  = 0.15     # label whose best on-camera evidence stays below → off-camera
CACHE        = "data/mouth_tracks.npz"

LIP_UP, LIP_LO, FOREHEAD, CHIN = 13, 14, 10, 152


def load_json(p):
    with open(p) as f:
        return json.load(f)


def probe_duration(path):
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                        "-of", "default=noprint_wrappers=1:nokey=1", path],
                       capture_output=True, text=True)
    return float(r.stdout.strip())


# ── 1. Mouth tracking ─────────────────────────────────────────────────────────

def chunk_views(seats, tracks, t0, t1):
    """Per seat (key, yaw, pitch, fov) for a chunk, or None if never seen nearby."""
    interval = float(tracks["interval"])
    out = []
    for k, s in seats.items():
        tr = tracks["seats"].get(k, [])
        lo = max(0, int((t0 - 10) / interval)); hi = min(len(tr), int((t1 + 10) / interval) + 1)
        samples = [x for x in tr[lo:hi] if x]
        if not samples and tr:
            out.append((k, None)); continue
        if samples:
            yaws = np.array([x[0] for x in samples]); ref = samples[len(samples) // 2][0]
            rel = np.array([geometry.norm_yaw(y - ref) for y in yaws])
            yaw = geometry.norm_yaw(ref + float(np.median(rel)))
            pitch = float(np.median([x[1] for x in samples]))
            size = float(np.median([x[2] for x in samples]))
            spread = float(rel.max() - rel.min())
        else:
            yaw, pitch = s["yaw_degrees"], s["pitch_degrees"]
            size, spread = s.get("face_deg", 10.0), 0.0
        fov = min(70.0, max(22.0, 2.6 * size, spread + 1.8 * size))
        out.append((k, (yaw, pitch, fov)))
    return out


def _landmarker():
    import mediapipe as mp
    opts = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=FACE_LANDMARKER_MODEL),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_faces=2, output_face_blendshapes=True,
        min_face_detection_confidence=0.3, min_face_presence_confidence=0.3,
        min_tracking_confidence=0.3)
    return mp.tasks.vision.FaceLandmarker.create_from_options(opts)


def track_chunk(args):
    video, t0, dur, views, lens_json_dir, seat_yaws = args
    import mediapipe as mp
    os.chdir(lens_json_dir)               # geometry reads data/lens.json from cwd
    live = [(k, v) for k, v in views if v]
    n_exp = int(round(dur * TRACK_FPS)) + 2
    res = {k: np.full((n_exp, 5), np.nan, dtype=np.float32) for k, _ in views}
    if not live:
        return t0, 0, res
    graph = geometry.views_graph([(y, p, f, f, CROP_PX, CROP_PX) for _, (y, p, f) in live],
                                 layout="hstack", fps=TRACK_FPS)
    cmd = ["ffmpeg", "-v", "error", "-hwaccel", "videotoolbox", "-ss", f"{t0:.3f}",
           "-i", video, "-t", f"{dur:.3f}", "-filter_complex", graph, "-map", "[out]",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    W = CROP_PX * len(live); fb = W * CROP_PX * 3
    lms = {k: _landmarker() for k, _ in live}
    n = 0
    try:
        while n < n_exp:
            buf = proc.stdout.read(fb)
            if len(buf) < fb:
                break
            frame = np.frombuffer(buf, np.uint8).reshape(CROP_PX, W, 3)
            ts = int(n * 1000 / TRACK_FPS)
            for j, (k, (yaw, pitch, fov)) in enumerate(live):
                crop = np.ascontiguousarray(frame[:, j * CROP_PX:(j + 1) * CROP_PX])
                r = lms[k].detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=crop), ts)
                if not r.face_landmarks:
                    continue
                # the seat's face is the one nearest the crop centre
                best, bd = None, 9.0
                for i, lm in enumerate(r.face_landmarks):
                    d = abs(lm[1].x - 0.5) + abs(lm[1].y - 0.5)
                    if d < bd:
                        best, bd = i, d
                if bd > 0.45:
                    continue
                lm = r.face_landmarks[best]
                xs = [p.x for p in lm]; ys = [p.y for p in lm]
                gy, gp = geometry.view_point_to_yaw_pitch((min(xs) + max(xs)) / 2,
                                                          (min(ys) + max(ys)) / 2, yaw, pitch, fov, fov)
                # identity gate: a neighbour leaning into this crop is NOT this seat
                own = abs(geometry.norm_yaw(gy - seat_yaws[k]))
                if own > 15.0 or any(abs(geometry.norm_yaw(gy - y)) < own
                                     for kk, y in seat_yaws.items() if kk != k):
                    continue
                fh = math.hypot(lm[FOREHEAD].x - lm[CHIN].x, lm[FOREHEAD].y - lm[CHIN].y)
                if fh < 1e-3:
                    continue
                mouth = math.hypot(lm[LIP_UP].x - lm[LIP_LO].x, lm[LIP_UP].y - lm[LIP_LO].y) / fh
                jaw = 0.0
                for c in (r.face_blendshapes[best] if r.face_blendshapes else []):
                    if c.category_name == "jawOpen":
                        jaw = c.score; break
                size = fh * fov  # forehead→chin in degrees (small-angle)
                res[k][n] = (mouth, jaw, gy, gp, size)
            n += 1
    finally:
        proc.stdout.close(); proc.wait()
        for lm in lms.values():
            lm.close()
    return t0, n, res


def track_mouths(video, seats, tracks, end_time=None):
    dur = probe_duration(video)
    if end_time:
        dur = min(dur, end_time)
    starts = list(np.arange(0.0, dur, CHUNK_SEC))
    jobs = []
    for t0 in starts:
        d = min(CHUNK_SEC, dur - t0)
        if d < 0.2:
            continue
        views = chunk_views(seats, tracks, t0, t0 + d)
        seat_yaws = {k: (v[0] if v else seats[k]["yaw_degrees"]) for k, v in views}
        jobs.append((video, float(t0), float(d), views, os.getcwd(), seat_yaws))
    total_frames = int(math.ceil(dur * TRACK_FPS))
    feats = {k: np.full((total_frames, 5), np.nan, dtype=np.float32) for k in seats}
    print(f"Mouth tracking: {len(seats)} seats × {dur:.0f}s at {TRACK_FPS} fps "
          f"({len(jobs)} chunks, {N_WORKERS} workers)…", flush=True)
    done = 0
    with Pool(N_WORKERS) as pool:
        for t0, n, res in pool.imap_unordered(track_chunk, jobs):
            base = int(round(t0 * TRACK_FPS))
            for k, arr in res.items():
                m = min(n, total_frames - base)
                if m > 0:
                    feats[k][base:base + m] = arr[:m]
            done += 1
            print(f"PROGRESS {done}/{len(jobs)}", flush=True)
    for k in seats:
        seen = np.isfinite(feats[k][:, 0]).mean()
        print(f"  {k}: face tracked in {100 * seen:.0f}% of frames", flush=True)
    return feats


# ── Signals ───────────────────────────────────────────────────────────────────

def audio_envelope(audio_path, n_frames):
    """Log-RMS loudness per video analysis frame."""
    import wave
    with wave.open(audio_path, "r") as wf:
        sr = wf.getframerate(); ch = wf.getnchannels()
        raw = wf.readframes(wf.getnframes())
    x = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
    if ch > 1:
        x = x.reshape(-1, ch).mean(1)
    hop = sr / TRACK_FPS
    env = np.zeros(n_frames, dtype=np.float32)
    for i in range(n_frames):
        a = int(i * hop); b = int((i + 1) * hop)
        seg = x[a:b]
        env[i] = math.log10(float(np.sqrt(np.mean(seg ** 2))) + 1.0) if len(seg) else 0.0
    return env


def fill_short_gaps(x, max_gap):
    """Linear-interpolate NaN runs up to max_gap samples; longer runs stay NaN."""
    x = x.copy(); n = len(x); i = 0
    while i < n:
        if np.isnan(x[i]):
            j = i
            while j < n and np.isnan(x[j]):
                j += 1
            if 0 < i and j < n and j - i <= max_gap:
                x[i:j] = np.linspace(x[i - 1], x[j], j - i + 2)[1:-1]
            i = j
        else:
            i += 1
    return x


def mouth_signal(feat):
    """Openness in ABSOLUTE, face-size-independent units (lip gap ÷ face height
    plus jawOpen). Deliberately NOT normalised per person: z-scoring a listener
    whose mouth barely moves blows their jitter up to look like talking."""
    m = fill_short_gaps(feat[:, 0], int(TRACK_FPS * 0.6))
    j = fill_short_gaps(feat[:, 1], int(TRACK_FPS * 0.6))
    return m + 0.5 * j


def motion_energy(sig, win):
    """Rolling std of the openness signal (talking = oscillating mouth)."""
    out = np.full(len(sig), np.nan, dtype=np.float32)
    h = win // 2
    for i in range(len(sig)):
        w = sig[max(0, i - h):i + h + 1]
        w = w[np.isfinite(w)]
        if len(w) >= max(3, win // 2):
            out[i] = float(np.std(w))
    return out


def best_lag_corr(m, a):
    """Max Pearson r between mouth and audio over ±MAX_LAG frames."""
    best = -1.0
    for lag in range(-MAX_LAG, MAX_LAG + 1):
        if lag >= 0:
            x, y = m[lag:], a[:len(a) - lag]
        else:
            x, y = m[:lag], a[-lag:]
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() < 8:
            continue
        x, y = x[ok], y[ok]
        if x.std() < 1e-6 or y.std() < 1e-6:
            continue
        best = max(best, float(np.corrcoef(x, y)[0, 1]))
    return best


class Evidence:
    """Per-seat visual speaking evidence for any time window."""

    def __init__(self, feats, env, presence):
        self.keys = list(feats)
        self.sig = {k: mouth_signal(feats[k]) for k in self.keys}
        self.env = env
        win = int(TRACK_FPS * 0.8)
        me = {k: motion_energy(self.sig[k], win) for k in self.keys}
        # one shared scale for everyone (what "talking" looks like at this table);
        # each person's own resting floor (beard / mic-occlusion jitter) removed
        pooled = np.concatenate([v[np.isfinite(v)] for v in me.values()] or [np.zeros(1)])
        scale = max(1e-3, float(np.percentile(pooled, 90)) if len(pooled) else 1.0)
        self.mot = {}
        for k in self.keys:
            ok = np.isfinite(me[k])
            floor = float(np.percentile(me[k][ok], 10)) if ok.sum() > 10 else 0.0
            self.mot[k] = (me[k] - floor) / scale          # ~0 resting, ~1 talking
        self.presence = presence

    def present(self, k, t):
        if not self.presence:
            return True
        p = self.presence["positions"].get(k)
        if p is None:
            return True
        iv = self.presence["interval"]
        return int(round(t / iv)) in self._pset(k)

    def _pset(self, k):
        if not hasattr(self, "_ps"):
            iv = self.presence["interval"]
            self._ps = {kk: {int(round(x / iv)) for x in v}
                        for kk, v in self.presence["positions"].items()}
        return self._ps.get(k, set())

    def scores(self, t0, t1):
        """{seat: (score, coverage)} for [t0, t1]."""
        a = int(t0 * TRACK_FPS); b = max(a + 4, int(t1 * TRACK_FPS))
        env = self.env[a:b]
        out = {}
        for k in self.keys:
            s = self.sig[k][a:b]
            cov = float(np.isfinite(s).mean()) if len(s) else 0.0
            if cov < 0.3:
                out[k] = (None, cov); continue
            r = best_lag_corr(s, env) if len(env) == len(s) else 0.0
            mot = np.nanmean(self.mot[k][a:b]) if np.isfinite(self.mot[k][a:b]).any() else 0.0
            out[k] = (1.5 * max(r, 0.0) + 1.0 * math.tanh(mot), cov)
        return out

    def probs(self, t0, t1):
        """Softmax p(seat speaking | video) over present seats; absent → 0."""
        sc = self.scores(t0, t1)
        tm = (t0 + t1) / 2
        keys = [k for k in self.keys if self.present(k, tm)] or list(self.keys)
        vals = np.array([sc[k][0] if sc[k][0] is not None else 0.0 for k in keys])
        e = np.exp(SOFTMAX_T * (vals - vals.max()))
        p = e / e.sum()
        out = {k: 0.0 for k in self.keys}
        out.update({k: float(v) for k, v in zip(keys, p)})
        strength = float(max((sc[k][0] or 0.0) for k in keys))
        return out, strength


# ── 3. Label → seat ───────────────────────────────────────────────────────────

def map_labels(diar, ev):
    per_label = defaultdict(lambda: defaultdict(float))
    strength = defaultdict(list)
    dur = defaultdict(float)
    for s in diar:
        d = s["end"] - s["start"]
        dur[s["speaker"]] += d
        if d < 0.6:
            continue
        p, st = ev.probs(s["start"], s["end"])
        w = min(d, 8.0)
        for k, v in p.items():
            per_label[s["speaker"]][k] += w * v
        strength[s["speaker"]].append((st, w))

    label_seat, offcam = {}, []
    for L in sorted(dur, key=lambda x: -dur[x]):
        if L not in per_label:
            offcam.append(L); continue
        tot = sum(per_label[L].values()) or 1.0
        dist = {k: v / tot for k, v in per_label[L].items()}
        sw = strength[L]
        mean_st = sum(s * w for s, w in sw) / (sum(w for _, w in sw) or 1)
        best = max(dist, key=dist.get)
        flag = "OFF-CAMERA" if mean_st < OFFCAM_EVID else ""
        print(f"  {L:12s} {dur[L]:7.1f}s  evidence {mean_st:+.2f}  → {best} "
              f"(p={dist[best]:.2f})  {flag}", flush=True)
        if flag:
            offcam.append(L)
        else:
            label_seat[L] = dist
    return label_seat, offcam


# ── 4. Speaker probabilities over time ────────────────────────────────────────

def speaker_steps(diar, overlaps, label_seat, ev, duration):
    n = int(math.ceil(duration / STEP))
    active = [[] for _ in range(n)]
    for s in diar:
        if s["speaker"] not in label_seat:
            continue
        for i in range(int(s["start"] / STEP), min(n, int(math.ceil(s["end"] / STEP)))):
            active[i].append(s["speaker"])
    osd = np.zeros(n, dtype=bool)
    for o in overlaps or []:
        osd[int(o["start"] / STEP):int(math.ceil(o["end"] / STEP))] = True

    steps = []
    for i in range(n):
        t = i * STEP
        labs = sorted(set(active[i]))
        if not labs:
            steps.append(None); continue
        loc, strength = ev.probs(max(0, t - 0.75), t + STEP + 0.75)
        comb = defaultdict(float)
        for L in labs:
            prior = label_seat[L]
            # visual evidence counts more when it is strong; a weak/blank video
            # window leaves the label's learned seat in charge
            beta = 0.4 + 0.8 * min(1.0, max(0.0, strength))
            lp = {k: math.log(prior.get(k, 0) + 1e-3) + beta * math.log(loc.get(k, 0) + 1e-3)
                  for k in ev.keys}
            if not any(loc.get(k, 0) > 0 for k in ev.keys):
                lp = {k: math.log(prior.get(k, 0) + 1e-3) for k in ev.keys}
            # absent seats can't be speaking (unless nobody is present)
            for k in ev.keys:
                if not ev.present(k, t) and any(ev.present(kk, t) for kk in ev.keys):
                    lp[k] -= 8.0
            mx = max(lp.values())
            e = {k: math.exp(v - mx) for k, v in lp.items()}
            z = sum(e.values())
            for k in e:
                comb[k] += e[k] / z / len(labs)
        steps.append({"p": {k: round(v, 3) for k, v in comb.items() if v >= 0.005},
                      "overlap": bool(len(labs) > 1 or osd[i])})
    return steps


def fine_presence(feats, tracks, duration, pad=3.0, step=STEP):
    """Seat present at t if its face was tracked within ±pad seconds (12.5fps
    mouth pass) or seen by the coarse frame scan within ±pad. Much tighter than
    the coarse ±8s smoothing: someone who walks off drops out within seconds."""
    n = int(math.ceil(duration / step))
    iv = float(tracks["interval"])
    out = {}
    for k, f in feats.items():
        seen = np.zeros(n, dtype=bool)
        hit_t = np.where(np.isfinite(f[:, 0]))[0] / TRACK_FPS
        hit_t = np.concatenate([hit_t, [i * iv for i, x in enumerate(tracks["seats"].get(k, [])) if x]])
        for t in hit_t:
            seen[max(0, int((t - pad) / step)):min(n, int((t + pad) / step) + 1)] = True
        out[k] = [round(i * step, 2) for i in np.where(seen)[0]]
    return {"interval": step, "num_frames": n, "positions": out}


def framing_tracks(feats, duration):
    """1 Hz medians of the precise face positions from the mouth pass."""
    out = {}
    per = int(TRACK_FPS)
    for k, f in feats.items():
        rows = []
        for s in range(int(duration)):
            blk = f[s * per:(s + 1) * per]
            ok = np.isfinite(blk[:, 2])
            if ok.sum() >= 3:
                b = blk[ok]
                ref = float(b[0, 2])
                y = geometry.norm_yaw(ref + float(np.median([geometry.norm_yaw(v - ref) for v in b[:, 2]])))
                rows.append([s + 0.5, round(y, 2), round(float(np.median(b[:, 3])), 2),
                             round(float(np.median(b[:, 4])), 2)])
        out[k] = rows
    return out


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("video_path")
    ap.add_argument("--end-time", type=float, default=None,
                    help="Only analyse up to this time (preview mode)")
    args = ap.parse_args()
    video = args.video_path

    audio_path = "data/audio.wav"
    if not os.path.exists(audio_path):
        print(f"{audio_path} not found — extracting camera audio from the video…", flush=True)
        r = subprocess.run(["ffmpeg", "-i", video, "-vn", "-acodec", "pcm_s16le", "-ar", "16000",
                            "-ac", "1", audio_path, "-y", "-loglevel", "error"],
                           capture_output=True, text=True)
        if r.returncode != 0:
            print("Error: could not obtain camera audio.\n" + r.stderr.strip(), flush=True)
            sys.exit(1)

    for need in ("data/diarization.json", "data/speaker_positions.json", "data/face_tracks.json"):
        if not os.path.exists(need):
            print(f"Error: {need} missing — run the earlier steps first.", flush=True)
            sys.exit(1)

    seats  = load_json("data/speaker_positions.json")
    tracks = load_json("data/face_tracks.json")
    diar   = load_json("data/diarization.json")
    overlaps = load_json("data/overlaps.json") if os.path.exists("data/overlaps.json") else []
    if args.end_time:
        diar = [s for s in diar if s["start"] < args.end_time]

    duration = probe_duration(video)
    if args.end_time:
        duration = min(duration, args.end_time)

    # mouth tracks are cached; reuse them if they cover the same seats and span
    feats = None
    if os.path.exists(CACHE):
        z = np.load(CACHE, allow_pickle=False)
        keys = [k for k in z.files if k != "_meta"]
        meta = z["_meta"] if "_meta" in z.files else None
        if sorted(keys) == sorted(seats) and meta is not None and meta[0] >= duration - 1:
            feats = {k: z[k] for k in keys}
            print(f"Reusing cached mouth tracks ({CACHE})", flush=True)
    if feats is None:
        feats = track_mouths(video, seats, tracks, end_time=args.end_time)
        np.savez_compressed(CACHE, _meta=np.array([duration]), **feats)

    n_frames = len(next(iter(feats.values())))
    env = audio_envelope(audio_path, n_frames)
    presence = fine_presence(feats, tracks, duration)
    with open("data/presence_fine.json", "w") as f:
        json.dump(presence, f)
    ev = Evidence(feats, env, presence)

    print("\nLabel → seat (pooled over every segment):", flush=True)
    label_seat, offcam = map_labels(diar, ev)
    manual = "data/speaker_map_manual.json"
    if os.path.exists(manual):
        # hand-made label → seat map wins; labels left out are treated as off-camera.
        # Per-moment mouth evidence still applies on top of it.
        fixed = load_json(manual)
        bad = [s for s in fixed.values() if s not in seats]
        if bad:
            print(f"Error: {manual} names unknown seats {bad}. Available: {sorted(seats)}", flush=True)
            sys.exit(1)
        label_seat = {L: {k: (0.9 if k == s else 0.1 / max(1, len(seats) - 1)) for k in seats}
                      for L, s in fixed.items()}
        offcam = sorted({x["speaker"] for x in diar} - set(fixed))
        print(f"Using manual map {manual}: {fixed}", flush=True)
    if not label_seat:
        print("No diarization label could be tied to an on-camera face — "
              "renderer will carousel on presence only.", flush=True)

    steps = speaker_steps(diar, overlaps, label_seat, ev, duration)
    speech = [s for s in steps if s]
    if speech:
        conf = np.array([max(s["p"].values()) for s in speech])
        print(f"\nSpeech steps: {len(speech)} × {STEP}s — confident (≥0.65): "
              f"{100 * (conf >= 0.65).mean():.0f}%, uncertain: {100 * (conf < 0.65).mean():.0f}%, "
              f"overlap: {100 * np.mean([s['overlap'] for s in speech]):.0f}%", flush=True)

    with open("data/speaker_probs.json", "w") as f:
        json.dump({"step": STEP, "seats": list(seats), "steps": steps,
                   "offcam_labels": offcam}, f)
    with open("data/framing_tracks.json", "w") as f:
        json.dump(framing_tracks(feats, duration), f)
    speaker_map = {L: max(d, key=d.get) for L, d in label_seat.items()}
    with open("data/speaker_map.json", "w") as f:
        json.dump(speaker_map, f, indent=2)
    print("\nSaved data/speaker_probs.json, data/speaker_map.json, data/framing_tracks.json", flush=True)
