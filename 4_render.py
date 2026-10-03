#!/usr/bin/env python3
"""
Step 4: Render the final 9:16 vertical video.

Shot types (all 1080×1920, stacked panes):
  single → one 1080×1920 pane
  split  → two 1080×960 panes   (speaker + listener, or two candidates)
  triple → speaker 1080×960 on top + two 540×960 boxes side by side below
A confident speaker's turn cycles: solo intro → split → triple → split → single …

Who is on screen comes from data/speaker_probs.json (step 3b): a probability
distribution over seats every 0.5s.
  • Turns are found by Viterbi smoothing with a switch penalty, so a 1-second
    "yeah" or a diarization blip doesn't yank the camera.
  • A CONFIDENT turn → brief solo intro on the speaker, then splits whose second
    pane cycles through the other present people (listener reactions).
  • An UNCERTAIN turn → the 2 or 3 likely speakers are shown together (double
    or triple shot), so whoever is actually talking is always on screen.
  • Overlapping speech → the overlapping people together.
  • No speech → carousel through whoever is physically present.
INVARIANT: the output always carousels among PRESENT people and never holds a
static single shot — even if mapping is wrong (see emit_gap_carousel).

Framing: every pane is aimed at the face's actual position during that shot
(data/framing_tracks.json, 1 Hz from the mouth-tracking pass; falls back to
data/face_tracks.json), zoomed by face size, with the face in the upper part
of the frame (geometry.frame_view). Panes are cut from the stitched 360° view,
so people sitting on the lens seam frame correctly.

Encoding: VideoToolbox HEVC; video-only segments concatenated, then ONE
continuous audio track laid over the whole video (no per-cut audio glitches);
cut points are frame-aligned so A/V never drifts.
"""

import json, math, os, sys, subprocess, tempfile, shutil
import numpy as np
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import geometry

N_RENDER_WORKERS = min(6, max(2, (os.cpu_count() or 4) - 2))

OUTPUT_WIDTH  = 1080
OUTPUT_HEIGHT = 1920
# pane sizes per shot: single = full frame; split = two stacked halves;
# triple = speaker on top half, two people side by side in the bottom half
PANES = {
    1: [(OUTPUT_WIDTH, OUTPUT_HEIGHT)],
    2: [(OUTPUT_WIDTH, OUTPUT_HEIGHT // 2)] * 2,
    3: [(OUTPUT_WIDTH, OUTPUT_HEIGHT // 2), (OUTPUT_WIDTH // 2, OUTPUT_HEIGHT // 2),
        (OUTPUT_WIDTH // 2, OUTPUT_HEIGHT // 2)],
}
LAYOUT = {1: "vstack", 2: "vstack", 3: "top_pair"}
# within a long confident turn the shot cycles through these (after the solo intro)
CYCLE = ["split", "triple", "split", "single"]

MIN_SEGMENT      = 0.8   # shortest shot (seconds)
SOLO_INTRO_DUR   = 2.0   # full-screen solo when the primary speaker changes
SOLO_REINTRO_GAP = 10.0  # seconds since last being primary before solo intro fires again
CAROUSEL_DUR     = 4.0   # how long each secondary slot shows before rotating
SWITCH_PENALTY   = 3.0   # Viterbi cost (nats) of changing speaker → min turn ≈ 1.5s
GAP_BRIDGE       = 1.5   # silences shorter than this stay inside the current turn
CONF_HI          = 0.60  # mean p(primary) at/above this → confident single-speaker turn
CAND_MIN_P       = 0.15  # a seat needs this mean p to be a candidate in an uncertain turn
SECONDARY_P      = 0.20  # a listener this likely to be talking takes the second pane
MIN_REGIME       = 3.0   # confident/uncertain stretches shorter than this are merged


def load_json(p):
    with open(p) as f:
        return json.load(f)


def _run_ffmpeg(cmd):
    """Run an ffmpeg command; on failure re-raise with stderr included."""
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode != 0:
        err = r.stderr.decode(errors='replace').strip()
        tail = '\n'.join(err.splitlines()[-12:])
        raise subprocess.CalledProcessError(r.returncode, cmd, output=r.stdout, stderr=tail.encode())


# ── Seats & framing ───────────────────────────────────────────────────────────

def load_positions(path="data/speaker_positions.json"):
    """Seats in the geometry-v2 global frame. Older job dirs stored back-lens yaw
    mirrored (180 − local); convert so they render at the right spot."""
    pos = load_json(path)
    scale = geometry.lens_fov() / 180.0 if geometry.video_mode() == "fisheye" else 1.0
    for k, p in pos.items():
        if p.get("geom") != 2:
            # old angles were measured assuming a 180° lens; rescale to the real FOV
            if p.get("lens") == "back":
                local = geometry.norm_yaw(180.0 - p["yaw_degrees"])
                p["yaw_degrees"] = geometry.norm_yaw(180.0 + local * scale)
            else:
                p["yaw_degrees"] = geometry.norm_yaw(geometry.norm_yaw(p["yaw_degrees"]) * scale)
            p["pitch_degrees"] = p.get("pitch_degrees", 0.0) * scale
            p["geom"] = 2
        p.setdefault("face_deg", geometry.DEFAULT_FACE_DEG)
    return pos


class Framer:
    """Where each seat's face is at any time → v360 view for a pane."""

    def __init__(self, positions):
        self.pos = positions
        # fine: 1 Hz from the 12.5fps mouth pass; coarse: every frame of step 2
        self.fine, self.coarse = defaultdict(list), defaultdict(list)
        if os.path.exists("data/framing_tracks.json"):
            for k, rows in load_json("data/framing_tracks.json").items():
                self.fine[k] = [tuple(r) for r in rows]
        if os.path.exists("data/face_tracks.json"):
            ft = load_json("data/face_tracks.json")
            iv = float(ft["interval"])
            for k, tr in ft["seats"].items():
                self.coarse[k] = [(i * iv, x[0], x[1], x[2]) for i, x in enumerate(tr) if x]

    @staticmethod
    def _window(rows, t0, t1, pad):
        return [r for r in rows if t0 - pad <= r[0] <= t1 + pad]

    def face_at(self, seat, t0, t1):
        """(yaw, pitch, size, yaw_spread) of the seat's face over [t0, t1]."""
        p = self.pos[seat]
        base = (p["yaw_degrees"], p["pitch_degrees"], p.get("face_deg", 10.0), 0.0)
        rows = []
        # nearest-in-time evidence first; only widen when there's nothing closer
        for src, pad in (("fine", 0.5), ("coarse", 1.5), ("fine", 3.0),
                         ("coarse", 4.0), ("fine", 8.0), ("coarse", 10.0)):
            rows = self._window((self.fine if src == "fine" else self.coarse).get(seat, []), t0, t1, pad)
            if rows:
                break
        if not rows:
            return base
        ref = base[0]
        rel = np.array([geometry.norm_yaw(r[1] - ref) for r in rows])
        med = float(np.median(rel))
        rel_in = rel[np.abs(rel - med) < 12]      # drop tracker glitches
        if not len(rel_in):
            rel_in = rel
        yaw = geometry.norm_yaw(ref + float(np.median(rel_in)))
        pitch = float(np.median([r[2] for r in rows]))
        size = float(np.median([r[3] for r in rows]))
        spread = float(rel_in.max() - rel_in.min()) if len(rel_in) > 1 else 0.0
        return yaw, pitch, size, spread

    def dedupe(self, seats, t0, t1):
        """Drop a pane whose face is where an earlier pane already points (a
        tracker that drifted onto a neighbour would show one person twice)."""
        kept, aims = [], []
        for s in seats:
            y, p, size, _ = self.face_at(s, t0, t1)
            if all(abs(geometry.norm_yaw(y - ay)) > max(6.0, 0.8 * size) or abs(p - ap) > size
                   for ay, ap in aims):
                kept.append(s); aims.append((y, p))
        return kept

    def views(self, seats, t0, t1):
        out = []
        for s, (w, h) in zip(seats, PANES[len(seats)]):
            yaw, pitch, size, spread = self.face_at(s, t0, t1)
            cy, cp, hf, vf = geometry.frame_view(yaw, pitch, size, w, h, spread)
            out.append((cy, cp, hf, vf, w, h))
        return out


def same_framing(a, b):
    """Is view b close enough to view a that re-aiming would only add a twitch?
    Dead-zone: the face may drift ~15% of the frame / the zoom ~15% before we
    re-frame."""
    ya, pa, ha, va = a[:4]
    yb, pb, hb, vb = b[:4]
    return (abs(geometry.norm_yaw(yb - ya)) <= 0.15 * ha
            and abs(pb - pa) <= 0.12 * va
            and abs(hb - ha) <= 0.15 * ha)


def encode_shot(video_path, views, start, end, out_path):
    """VIDEO ONLY. Audio is laid on as one continuous track after concat."""
    _run_ffmpeg([
        "ffmpeg", "-hwaccel", "videotoolbox", "-ss", f"{start:.6f}", "-i", video_path,
        "-t", f"{end - start:.6f}",
        "-filter_complex", geometry.views_graph(views, layout=LAYOUT[len(views)]),
        "-map", "[out]", "-an",
        "-c:v", "hevc_videotoolbox", "-b:v", "8M",
        out_path, "-y"
    ])


# ── Speaker probabilities ─────────────────────────────────────────────────────

def load_speaker_steps(positions, diarization, speaker_map, overlaps):
    """data/speaker_probs.json if present; otherwise derive the same structure
    from diarization + speaker_map (older job dirs): mapped seat gets p=1."""
    if os.path.exists("data/speaker_probs.json"):
        sp = load_json("data/speaker_probs.json")
        return float(sp["step"]), sp["steps"]
    step = 0.5
    end = max([s["end"] for s in diarization] or [0])
    n = int(math.ceil(end / step))
    active = [set() for _ in range(n)]
    for s in diarization:
        seat = speaker_map.get(s["speaker"])
        if seat in positions:
            for i in range(int(s["start"] / step), min(n, int(math.ceil(s["end"] / step)))):
                active[i].add(seat)
    osd = set()
    for o in overlaps or []:
        osd.update(range(int(o["start"] / step), int(math.ceil(o["end"] / step))))
    steps = []
    for i, a in enumerate(active):
        if not a:
            steps.append(None)
        else:
            steps.append({"p": {k: 1.0 / len(a) for k in a}, "overlap": len(a) > 1 or i in osd})
    return step, steps


def viterbi(emis, penalty):
    """emis: (T, S) log-likelihoods → best state path with a switch penalty."""
    T, S = emis.shape
    score = emis[0].copy()
    back = np.zeros((T, S), dtype=np.int32)
    for t in range(1, T):
        best_prev = int(np.argmax(score))
        stay = score
        switch = score[best_prev] - penalty
        back[t] = np.where(stay >= switch, np.arange(S), best_prev)
        score = np.maximum(stay, switch) + emis[t]
    path = np.zeros(T, dtype=np.int32)
    path[-1] = int(np.argmax(score))
    for t in range(T - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


# ── Timeline ──────────────────────────────────────────────────────────────────

def resolve_timeline(steps, step, seats, video_end, presence=None):
    """
    Returns [{type: single|split|triple, speakers: [...], start, end, gap_fill}]
    tiling [0, video_end] with no gaps.
    """
    seats = list(seats)
    S = len(seats)
    idx = {k: i for i, k in enumerate(seats)}

    # ── presence ──
    if presence:
        iv = float(presence.get("interval", 10) or 10)
        N = int(presence.get("num_frames", 0))
        pres = {k: {int(round(t / iv)) for t in v} for k, v in presence.get("positions", {}).items()}

        def is_present(k, t):
            if k not in pres:
                return True
            i = int(round(t / iv))
            if N:
                i = max(0, min(N - 1, i))
            return i in pres[k]
    else:
        def is_present(k, t):
            return True

    def present_at(t):
        return [k for k in seats if is_present(k, t)]

    def present_during(k, t0, t1):
        """Present at the start, middle and end of [t0, t1] — someone who walks
        off mid-shot must not leave an empty chair on screen."""
        return all(is_present(k, x) for x in (t0, (t0 + t1) / 2, max(t0, t1 - 0.25)))

    n = int(math.ceil(video_end / step))
    steps = (list(steps) + [None] * n)[:n]

    # ── speech blocks (short silences bridged) ──
    speech = [s is not None and bool(s.get("p")) for s in steps]
    blocks, i = [], 0
    bridge = int(round(GAP_BRIDGE / step))
    while i < n:
        if not speech[i]:
            i += 1; continue
        j = i
        while j < n:
            if speech[j]:
                j += 1; continue
            k = j
            while k < n and not speech[k]:
                k += 1
            if k < n and k - j <= bridge:
                j = k
            else:
                break
        blocks.append((i, j)); i = j

    min_regime = int(round(MIN_REGIME / step))

    def certainty_runs(a, b, primary):
        """Split one speaker's turn into confident / uncertain stretches (each at
        least MIN_REGIME long) so a sure stretch isn't diluted by an unsure one."""
        flags, last = [], True
        for x in range(a, b):
            st = steps[x]
            if st and st.get("p"):
                last = st["p"].get(primary, 0.0) >= CONF_HI and not st.get("overlap")
            flags.append(last)
        runs = []
        for i, f in enumerate(flags):
            if runs and runs[-1][2] == f:
                runs[-1][1] = a + i + 1
            else:
                runs.append([a + i, a + i + 1, f])
        merged = []
        for r in runs:                       # absorb short flickers into the previous run
            if merged and (r[1] - r[0] < min_regime or merged[-1][1] - merged[-1][0] < min_regime):
                prev_len, cur_len = merged[-1][1] - merged[-1][0], r[1] - r[0]
                merged[-1][1] = r[1]
                if cur_len > prev_len:       # the longer stretch decides the flag
                    merged[-1][2] = r[2]
            else:
                merged.append(list(r))
        return [tuple(r) for r in merged]

    # ── turns inside each block via Viterbi ──
    turns = []        # (a, b, primary, mean_p dict, overlap_frac, sure)
    for a, b in blocks:
        T = b - a
        emis = np.zeros((T, S))
        for t in range(T):
            st = steps[a + t]
            if st and st.get("p"):
                for k in seats:
                    emis[t, idx[k]] = math.log(st["p"].get(k, 0.0) + 1e-3)
            tm = (a + t) * step
            for k in seats:
                if not is_present(k, tm) and present_at(tm):
                    emis[t, idx[k]] -= 6.0
        path = viterbi(emis, SWITCH_PENALTY)
        t = 0
        while t < T:
            u = t
            while u < T and path[u] == path[t]:
                u += 1
            for ra, rb, sure in certainty_runs(a + t, a + u, seats[path[t]]):
                ps = [steps[x] for x in range(ra, rb) if steps[x] and steps[x].get("p")]
                mean = defaultdict(float)
                for st in ps:
                    for k, v in st["p"].items():
                        mean[k] += v / max(1, len(ps))
                ovl = (sum(1 for st in ps if st.get("overlap")) / len(ps)) if ps else 0.0
                turns.append((ra, rb, seats[path[t]], dict(mean), ovl, sure))
            t = u

    timeline = []
    sec_idx = defaultdict(int)
    last_primary_end = {}
    last_primary = [turns[0][2] if turns else (seats[0] if seats else None)]
    gap_ptr = [0]
    cycle_pos = {}    # primary → position in CYCLE, carried across that speaker's segments

    def add(kind, spk, t0, t1, gap=False):
        if t1 - t0 > 1e-3:
            timeline.append({"type": kind, "speakers": list(spk), "start": t0, "end": t1,
                             "gap_fill": gap})

    def kind_for(n):
        return {1: "single", 2: "split", 3: "triple"}[n]

    def slot_p(t0, t1):
        """Mean speaking probability per seat over [t0, t1]."""
        a, b = int(t0 / step), max(int(t0 / step) + 1, int(math.ceil(t1 / step)))
        m = defaultdict(float); c = 0
        for x in range(a, min(b, n)):
            st = steps[x]
            if st and st.get("p"):
                c += 1
                for k, v in st["p"].items():
                    m[k] += v
        return {k: v / c for k, v in m.items()} if c else {}

    def emit_confident(t0, t1, primary, is_new):
        t = t0
        others_any = [k for k in seats if k != primary]
        since = t0 - last_primary_end.get(primary, float("-inf"))
        if (is_new and others_any and since >= SOLO_REINTRO_GAP
                and t1 - t0 > SOLO_INTRO_DUR + MIN_SEGMENT):
            add("single", [primary], t, t + SOLO_INTRO_DUR); t += SOLO_INTRO_DUR
        step_i = 0 if is_new else cycle_pos.get(primary, 0)
        while t < t1 - 1e-6:
            want = CYCLE[step_i % len(CYCLE)]
            step_i += 1
            dur = SOLO_INTRO_DUR if want == "single" else CAROUSEL_DUR
            slot_end = min(t + dur, t1)
            if t1 - slot_end < MIN_SEGMENT:
                slot_end = t1
            mates = [k for k in others_any if present_during(k, t, slot_end)]
            sp = slot_p(t, slot_end)
            if want == "single" and any(sp.get(k, 0) >= SECONDARY_P for k in mates):
                want = "split"        # someone is chiming in — don't cut them out for a solo
            if want == "single" or not mates:
                add("single", [primary], t, slot_end)
                t = slot_end
                cycle_pos[primary] = step_i
                continue
            # anyone chiming in goes first; then rotate through the listeners
            talkers = sorted([k for k in mates if sp.get(k, 0) >= SECONDARY_P], key=lambda k: -sp[k])
            order = talkers + [mates[(sec_idx[primary] + i) % len(mates)] for i in range(len(mates))]
            picks = list(dict.fromkeys(order))
            sec_idx[primary] += 1
            if want == "triple" and len(picks) >= 2:
                add("triple", [primary] + picks[:2], t, slot_end)
            else:
                add("split", [primary, picks[0]], t, slot_end)
            t = slot_end
        cycle_pos[primary] = step_i

    def emit_uncertain(t0, t1, cands):
        """Show the likely speakers together; alternate double ↔ triple on long
        turns so the frame keeps moving."""
        t, flip = t0, 0
        while t < t1 - 1e-6:
            slot_end = min(t + CAROUSEL_DUR * 1.5, t1)
            if t1 - slot_end < MIN_SEGMENT:
                slot_end = t1
            here = [k for k in cands if present_during(k, t, slot_end)] or cands[:1]
            extra = [k for k in present_at(t) if k not in here and present_during(k, t, slot_end)]
            if len(here) >= 3:
                shot = here[:3] if flip % 2 == 0 else here[:2]
            elif len(here) == 2 and extra and flip % 2 == 1 and t1 - t0 > 2 * CAROUSEL_DUR:
                shot = here + [extra[sec_idx["_u"] % len(extra)]]; sec_idx["_u"] += 1
            else:
                shot = here[:2]
            if len(shot) == 1:
                mates = [k for k in extra]
                shot = shot + ([mates[sec_idx["_u"] % len(mates)]] if mates else [])
                sec_idx["_u"] += 1
            add(kind_for(len(shot)), shot, t, slot_end)
            t = slot_end; flip += 1

    def emit_gap_carousel(t0, t1):
        """No one is (mappably) talking. INVARIANT: never hold a static single —
        rotate through the PRESENT seats, alternating a brief single with a
        split, regardless of diarization/mapping quality."""
        t = t0
        while t < t1 - 1e-6:
            present = present_at(t) or [last_primary[0] if last_primary[0] in seats else seats[0]]
            primary = present[gap_ptr[0] % len(present)]
            solo_end = min(t + SOLO_INTRO_DUR, t1)
            if t1 - solo_end < MIN_SEGMENT:
                solo_end = t1
            add("single", [primary], t, solo_end, gap=True)
            t = solo_end
            if t >= t1 - 1e-6:
                break
            slot_end = min(t + CAROUSEL_DUR, t1)
            if t1 - slot_end < MIN_SEGMENT:
                slot_end = t1
            mates = [k for k in seats if k != primary and present_during(k, t, slot_end)]
            rot = [mates[(gap_ptr[0] + i) % len(mates)] for i in range(len(mates))] if mates else []
            if len(rot) >= 2 and gap_ptr[0] % 2 == 1:
                add("triple", [primary] + rot[:2], t, slot_end, gap=True)
            elif rot:
                add("split", [primary, rot[0]], t, slot_end, gap=True)
            else:
                add("single", [primary], t, slot_end, gap=True)
            t = slot_end
            gap_ptr[0] += 1
            last_primary[0] = primary

    # ── walk turns + gaps ──
    t = 0.0
    prev = None
    for a, b, primary, mean, ovl, sure in turns:
        t0, t1 = a * step, min(b * step, video_end)
        if t0 - t >= MIN_SEGMENT:
            emit_gap_carousel(t, t0)
        elif timeline and t0 > t:
            timeline[-1]["end"] = t0
        t0 = max(t0, t)
        if t1 - t0 < 1e-3:
            continue
        conf = mean.get(primary, 0.0)
        ranked = sorted((k for k in mean if mean[k] >= CAND_MIN_P), key=lambda k: -mean[k])
        if primary in ranked:
            ranked.remove(primary)
        cands = ([primary] + ranked)[:3]
        if ovl >= 0.5 and len(cands) < 2:
            others = sorted((k for k in mean if k != primary), key=lambda k: -mean[k])
            cands += others[:1]
        if (sure and conf >= CONF_HI * 0.9 and ovl < 0.5) or len(cands) < 2:
            emit_confident(t0, t1, primary, primary != prev)
            if prev and primary != prev:
                last_primary_end[prev] = t0
            prev = primary
            last_primary[0] = primary
        else:
            emit_uncertain(t0, t1, cands)
            prev = None
        t = t1
    if video_end - t >= MIN_SEGMENT:
        emit_gap_carousel(t, video_end)
    elif timeline:
        timeline[-1]["end"] = video_end

    # ── gap-free tiling of [0, video_end] ──
    timeline.sort(key=lambda s: s["start"])
    cleaned = []
    for shot in timeline:
        if cleaned and shot["start"] < cleaned[-1]["end"] - 1e-3:
            continue
        cleaned.append(shot)
    for i in range(len(cleaned) - 1):
        cleaned[i] = dict(cleaned[i], end=cleaned[i + 1]["start"])
    if cleaned:
        cleaned[0] = dict(cleaned[0], start=0.0)
        cleaned[-1] = dict(cleaned[-1], end=max(cleaned[-1]["end"], video_end))
    # merge sub-MIN_SEGMENT slivers into their neighbour
    merged = []
    for s in cleaned:
        if merged and s["end"] - s["start"] < MIN_SEGMENT:
            merged[-1] = dict(merged[-1], end=s["end"])
        else:
            merged.append(s)
    return [s for s in merged if s["end"] - s["start"] > 0.01]


# ── Render ────────────────────────────────────────────────────────────────────

def render(video_path, output_path="output/final.mp4", end_time=None, external_audio=None):
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    segments_dir = tempfile.mkdtemp(prefix="360autocam_segs_")
    try:
        _render_inner(video_path, output_path, end_time, external_audio, segments_dir)
    finally:
        shutil.rmtree(segments_dir, ignore_errors=True)


def _render_inner(video_path, output_path, end_time, external_audio, segments_dir):
    if external_audio is None and os.path.exists("data/external_audio_aligned.wav"):
        external_audio = "data/external_audio_aligned.wav"
        print(f"Using external audio: {external_audio}")

    positions   = load_positions()
    diarization = load_json("data/diarization.json") if os.path.exists("data/diarization.json") else []
    speaker_map = load_json("data/speaker_map.json") if os.path.exists("data/speaker_map.json") else {}
    overlaps    = load_json("data/overlaps.json") if os.path.exists("data/overlaps.json") else []
    for d, p in speaker_map.items():
        if p not in positions:
            print(f"Error: speaker_map.json maps '{d}' → '{p}', but '{p}' is not in "
                  f"speaker_positions.json. Available: {sorted(positions)}")
            sys.exit(1)

    # fine presence (0.5s, from the 12.5fps mouth-tracking pass) beats the coarse
    # frame-scan presence: it notices someone leaving within seconds
    presence = None
    for pp in ("data/presence_fine.json", "data/presence.json"):
        if os.path.exists(pp):
            presence = load_json(pp)
            print(f"Presence: {pp}")
            break
    if not presence:
        print("No data/presence.json — everyone treated as present.")

    step, steps = load_speaker_steps(positions, diarization, speaker_map, overlaps)

    dur_result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", video_path],
        capture_output=True, text=True)
    video_dur = float(dur_result.stdout.strip())
    fps_res = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=r_frame_rate", "-of", "default=noprint_wrappers=1:nokey=1",
         video_path], capture_output=True, text=True)
    try:
        num, den = fps_res.stdout.strip().split("/")
        fps = float(num) / float(den)
    except Exception:
        fps = 30.0
    print(f"Video: {video_dur:.1f}s at {fps:.3f} fps; lens FOV {geometry.lens_fov():.1f}°")

    video_end = min(video_dur, end_time) if end_time else video_dur
    timeline = resolve_timeline(steps, step, positions.keys(), video_end, presence)

    # ── Frame-align all boundaries (CRITICAL for A/V sync) ──
    # ffmpeg's -t rounds a segment UP to the frame covering its end. Summed over
    # hundreds of segments that drifts video against the single continuous audio
    # track. Snapping every boundary to the frame grid makes durations exact.
    def _snap(t):
        return round(t * fps) / fps
    snapped = []
    for s in timeline:
        b = _snap(s["end"])
        a = snapped[-1]["end"] if snapped else _snap(s["start"])
        if b - a >= 0.5 / fps:
            snapped.append(dict(s, start=a, end=b))
        elif snapped and b > snapped[-1]["end"]:
            snapped[-1] = dict(snapped[-1], end=b)
    timeline = snapped

    kinds = defaultdict(float)
    for s in timeline:
        kinds[s["type"]] += s["end"] - s["start"]
    total = sum(kinds.values()) or 1
    longest_single = max([s["end"] - s["start"] for s in timeline if s["type"] == "single"] or [0])
    print(f"Timeline: {len(timeline)} shots — " +
          ", ".join(f"{k} {100 * v / total:.0f}%" for k, v in sorted(kinds.items())) +
          f"; longest single {longest_single:.1f}s", flush=True)
    with open("data/timeline.json", "w") as f:
        json.dump(timeline, f, indent=1)

    framer = Framer(positions)
    print(f"Encoding {len(timeline)} segments with {N_RENDER_WORKERS} workers…", flush=True)

    for i, shot in enumerate(timeline):
        spk = framer.dedupe(shot["speakers"], shot["start"], shot["end"])
        if len(spk) != len(shot["speakers"]):
            timeline[i] = dict(shot, speakers=spk, type={1: "single", 2: "split", 3: "triple"}[len(spk)])

    # Back-to-back shots of the same people are ONE shot: cutting between them and
    # re-aiming each half separately makes the picture visibly twitch.
    joined = []
    for shot in timeline:
        if joined and joined[-1]["speakers"] == shot["speakers"]:
            joined[-1] = dict(joined[-1], end=shot["end"])
        else:
            joined.append(dict(shot))
    if len(joined) != len(timeline):
        print(f"  merged {len(timeline) - len(joined)} cuts between identical shots", flush=True)
    timeline = joined
    with open("data/timeline.json", "w") as f:
        json.dump(timeline, f, indent=1)

    # Aim every pane up front, sequentially, so a person who stays on screen
    # across a cut keeps their exact crop unless their face really moved.
    shot_views = []
    held = {}                                  # (seat, pane_h) → (view, end_time)
    for shot in timeline:
        views = framer.views(shot["speakers"], shot["start"], shot["end"])
        for j, seat in enumerate(shot["speakers"]):
            key = (seat, views[j][4], views[j][5])
            prev = held.get(key)
            if prev and abs(prev[1] - shot["start"]) < 1e-3 and same_framing(prev[0], views[j]):
                views[j] = prev[0]
            held[key] = (views[j], shot["end"])
        shot_views.append(views)

    def job(i, shot):
        out = os.path.join(segments_dir, f"seg_{i:05d}.mp4")
        encode_shot(video_path, shot_views[i], shot["start"], shot["end"], out)
        return i, out

    clips_map, done = {}, 0
    with ThreadPoolExecutor(max_workers=N_RENDER_WORKERS) as pool:
        futures = {pool.submit(job, i, shot): i for i, shot in enumerate(timeline)}
        for fut in as_completed(futures):
            try:
                i, out = fut.result()
            except subprocess.CalledProcessError as e:
                stderr_text = e.stderr.decode(errors='replace') if isinstance(e.stderr, bytes) else str(e.stderr or '')
                print(f"\nFFmpeg error (exit {e.returncode}):\n{stderr_text}", flush=True)
                raise
            clips_map[i] = out
            done += 1
            print(f"[{done}/{len(timeline)}]", flush=True)

    clips = [clips_map[i] for i in range(len(timeline))]
    print(f"\nConcatenating {len(clips)} clips...")
    concat_list = os.path.join(segments_dir, "concat.txt")
    with open(concat_list, "w") as f:
        for c in clips:
            f.write(f"file '{c}'\n")
    silent_video = os.path.join(segments_dir, "video_only.mp4")
    subprocess.run(["ffmpeg", "-v", "error", "-f", "concat", "-safe", "0", "-i", concat_list,
                    "-c", "copy", "-an", silent_video, "-y"], check=True)

    # ONE continuous audio track over the whole video (gapless). Output time 0 ==
    # camera time timeline[0].start; video is the master clock (apad + shortest).
    t0 = timeline[0]["start"] if timeline else 0.0
    audio_src = external_audio if external_audio else (
        "data/audio.wav" if os.path.exists("data/audio.wav") else video_path)
    print(f"Overlaying continuous audio from {audio_src} (offset {t0:.2f}s)…")
    subprocess.run([
        "ffmpeg", "-v", "error",
        "-i", silent_video,
        "-ss", f"{t0:.6f}", "-i", audio_src,
        "-map", "0:v:0", "-map", "1:a:0",
        "-af", "apad",
        "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
        "-shortest",
        output_path, "-y"
    ], check=True)
    print(f"Done! Output: {output_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("video_path")
    parser.add_argument("output_path", nargs="?", default="output/final.mp4")
    parser.add_argument("--end-time", type=float, default=None,
                        help="Only render up to this time (seconds)")
    parser.add_argument("--external-audio", default=None,
                        help="Aligned external mic WAV to use instead of camera audio "
                             "(auto-detected from data/external_audio_aligned.wav if present)")
    args = parser.parse_args()
    render(args.video_path, args.output_path,
           end_time=args.end_time, external_audio=args.external_audio)
