#!/usr/bin/env python3
"""
Step 4: Render the final 9:16 vertical video.

Single speaker  → full 1080x1920 crop
Two speakers    → split-screen: top 1080x960 + bottom 1080x960 (vstack)
Cuts            → hard cuts via concat demuxer (no re-encoding overhead)
Encoding        → VideoToolbox hardware H.264

Reads:
  data/speaker_positions.json  — per-speaker yaw/pitch/fov/stream
  data/diarization.json        — who speaks when (including overlaps)
  data/speaker_map.json        — maps diarization IDs to position keys (auto or manual)

Output: output/final.mp4
"""

import json, math, os, sys, subprocess, tempfile, shutil
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

V360_INPUT = "equirect" if os.environ.get("VIDEO_MODE") == "equirect" else "fisheye"

N_RENDER_WORKERS  = min(6, max(2, (os.cpu_count() or 4) - 2))  # scales with core count

OUTPUT_WIDTH  = 1080
OUTPUT_HEIGHT = 1920
HALF_HEIGHT   = OUTPUT_HEIGHT // 2   # 960px per half in split-screen
MIN_SEGMENT      = 0.8   # skip segments shorter than this (seconds)
SOLO_INTRO_DUR   = 2.0   # full-screen solo when the primary speaker changes
SOLO_REINTRO_GAP = 10.0  # seconds since last being primary before solo intro fires again
CAROUSEL_DUR     = 4.0   # how long each secondary slot shows before rotating
ACTIVITY_WINDOW  = 60    # seconds without real speech → exclude from carousel
MIN_ACTIVITY_DUR = 2.0   # minimum segment length to count as "real" speech (filters pyannote blips)

def load_json(p):
    with open(p) as f: return json.load(f)

def calc_v_fov(h_fov, height):
    return round(2 * math.degrees(math.atan(math.tan(math.radians(h_fov / 2)) * (height / OUTPUT_WIDTH))), 1)

def norm_yaw(y):
    """Normalize yaw to [-180, 180] as required by ffmpeg v360."""
    return ((y + 180) % 360) - 180

def filter_yaw(pos):
    """
    Convert stored global yaw to lens-local yaw for the v360 filter.
    Detection stores global yaw with a +180° offset for the back lens;
    v360 needs yaw relative to the lens center (offset removed).
    """
    offset = 180.0 if pos.get("lens") == "back" else 0.0
    return norm_yaw(pos["yaw_degrees"] - offset)

def _safe_hfov(pos):
    """
    Clamp h_fov so the far horizontal edge of the crop stays within the
    equidistant fisheye circle (usable up to ~90° from the optical axis).
    Speakers near the lens edge need a tighter FOV to avoid black borders.
    Constraint: abs(filter_yaw) + h_fov/2 <= 88° (2° safety margin).
    """
    local_yaw_abs = abs(filter_yaw(pos))
    h_fov_max = max(20.0, 2.0 * (88.0 - local_yaw_abs))
    return min(float(pos.get("h_fov", 75)), h_fov_max)

def _audio_inputs(video_path, start, external_audio):
    """Return (ffmpeg input args, audio map flag) for camera or external audio."""
    if external_audio:
        # second input is the pre-aligned external WAV; seek it to match the segment
        return (["-ss", str(start), "-i", video_path,
                 "-ss", str(start), "-i", external_audio],
                "-map", "1:a:0")
    return (["-ss", str(start), "-i", video_path],
            "-map", "0:a:0")

def _run_ffmpeg(cmd):
    """Run an ffmpeg command; on failure re-raise with stderr included."""
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode != 0:
        err = r.stderr.decode(errors='replace').strip()
        # surface the last few lines of ffmpeg output so the error is visible
        tail = '\n'.join(err.splitlines()[-12:])
        raise subprocess.CalledProcessError(r.returncode, cmd, output=r.stdout, stderr=tail.encode())

def extract_segment(video_path, pos, start, end, out_path, height=OUTPUT_HEIGHT,
                    external_audio=None):
    # VIDEO ONLY. Audio is laid on as one continuous track after concat (see
    # _render_inner) — per-segment audio re-encoding would add AAC priming/padding
    # at every one of the ~hundreds of joins and make the audio audibly jump.
    stream = pos.get("stream", 0)
    yaw    = filter_yaw(pos)
    pitch  = pos.get("pitch_degrees", 0)
    h_fov  = _safe_hfov(pos)
    v_fov  = calc_v_fov(h_fov, height)
    _run_ffmpeg([
        "ffmpeg", "-ss", str(start), "-i", video_path,
        "-t", str(end - start),
        "-map", f"0:v:{stream}", "-an",
        "-vf", f"v360={V360_INPUT}:flat:yaw={yaw}:pitch={pitch}:h_fov={h_fov}:v_fov={v_fov}:w={OUTPUT_WIDTH}:h={height}",
        "-c:v", "hevc_videotoolbox", "-b:v", "8M",
        out_path, "-y"
    ])

def extract_split_segment(video_path, pos_top, pos_bot, start, end, out_path,
                          external_audio=None):
    s_top = pos_top.get("stream", 0)
    s_bot = pos_bot.get("stream", 0)

    def vf(pos, label):
        y, p = filter_yaw(pos), pos.get("pitch_degrees", 0)
        hf   = _safe_hfov(pos)
        vf_  = calc_v_fov(hf, HALF_HEIGHT)
        return f"v360={V360_INPUT}:flat:yaw={y}:pitch={p}:h_fov={hf}:v_fov={vf_}:w={OUTPUT_WIDTH}:h={HALF_HEIGHT}[{label}]"

    if s_top == s_bot:
        # Same stream — must split first; referencing [0:v:N] twice in filter_complex fails in FFmpeg 6+
        fc = f"[0:v:{s_top}]split=2[_a][_b];[_a]{vf(pos_top,'t')};[_b]{vf(pos_bot,'b')};[t][b]vstack=inputs=2[out]"
    else:
        fc = f"[0:v:{s_top}]{vf(pos_top,'t')};[0:v:{s_bot}]{vf(pos_bot,'b')};[t][b]vstack=inputs=2[out]"

    # VIDEO ONLY — see extract_segment; audio is muxed once, after concat.
    _run_ffmpeg([
        "ffmpeg", "-ss", str(start), "-i", video_path,
        "-t", str(end - start),
        "-filter_complex", fc,
        "-map", "[out]", "-an",
        "-c:v", "hevc_videotoolbox", "-b:v", "8M",
        out_path, "-y"
    ])


def resolve_timeline(diarization, speaker_to_pos, video_duration=None, overlaps=None,
                     presence=None):
    """
    Split-screen first: the default shot is ALWAYS a split.

    For every moment in the video we pick a primary speaker (whoever diarization
    says is speaking; last known speaker during gaps).  The timeline is:

      • Speaker changes → SOLO_INTRO_DUR seconds of full-screen solo ("intro")
      • Everything else  → split(primary, secondary) where secondary rotates
                           through all other speakers every CAROUSEL_DUR seconds

    This means the viewer always sees the speaking person AND at least one
    listener / co-host, with the listener slot cycling continuously.
    """
    # Sort speakers most-active first so the carousel opens with the next busiest talker
    spk_speech = defaultdict(float)
    for seg in diarization:
        if seg["speaker"] in speaker_to_pos:
            spk_speech[seg["speaker"]] += seg["end"] - seg["start"]
    pos_keys = sorted(speaker_to_pos.keys(), key=lambda k: spk_speech[k], reverse=True)

    # Distinct CAMERA SEATS, one representative speaker-label each. Several
    # diarization labels can map to the same seat (pyannote over-segments one
    # person into many IDs); the carousel must treat them as a single face, so
    # we dedupe by position-object identity and keep the busiest label per seat.
    _seat_repr = {}
    for k in pos_keys:
        pid = id(speaker_to_pos[k])
        if pid not in _seat_repr or spk_speech[k] > spk_speech[_seat_repr[pid]]:
            _seat_repr[pid] = k
    distinct_positions = sorted(_seat_repr.values(), key=lambda k: spk_speech[k], reverse=True)

    # Build per-speaker sorted end-time list for fast recency lookups (binary search)
    spk_ends = defaultdict(list)
    for seg in diarization:
        if seg["speaker"] in speaker_to_pos:
            spk_ends[seg["speaker"]].append(seg["end"])
    for spk in spk_ends:
        spk_ends[spk].sort()

    def last_spoke_before(spk, t):
        """Latest segment-end time for spk strictly before t; -inf if never spoke."""
        ends = spk_ends.get(spk, [])
        lo, hi = 0, len(ends)
        while lo < hi:
            mid = (lo + hi) // 2
            if ends[mid] < t:
                lo = mid + 1
            else:
                hi = mid
        return ends[lo - 1] if lo > 0 else float('-inf')

    # Substantial-speech lookup: only count segments >= MIN_ACTIVITY_DUR (ignores blips)
    spk_real_ends = defaultdict(list)
    for _seg in diarization:
        if (_seg["speaker"] in speaker_to_pos
                and _seg["end"] - _seg["start"] >= MIN_ACTIVITY_DUR):
            spk_real_ends[_seg["speaker"]].append(_seg["end"])
    for _spk in spk_real_ends:
        spk_real_ends[_spk].sort()

    def last_real_spoke_before(spk, t):
        """Latest end of a real (>= MIN_ACTIVITY_DUR) segment before t; -inf if none."""
        ends = spk_real_ends.get(spk, [])
        lo, hi = 0, len(ends)
        while lo < hi:
            mid = (lo + hi) // 2
            if ends[mid] < t:
                lo = mid + 1
            else:
                hi = mid
        return ends[lo - 1] if lo > 0 else float('-inf')

    # ── Physical-presence lookup (from face detection over time) ──────────────
    # presence = {"interval": s, "num_frames": N, "by_speaker": {spk: set(frame_idx)}}
    # is_present(spk, t): is this speaker's face visible at time t (smoothed)?
    # When no presence data is available we treat everyone as always present so
    # the carousel falls back to pure speech-recency behaviour.
    if presence:
        _pres_int   = presence.get("interval", 10) or 10
        _pres_N     = presence.get("num_frames", 0)
        _pres_by_sp = presence.get("by_speaker", {})

        def is_present(spk, t):
            frames = _pres_by_sp.get(spk)
            if frames is None:
                return True            # unknown speaker → don't suppress
            idx = int(round(t / _pres_int))
            if _pres_N:
                idx = max(0, min(_pres_N - 1, idx))
            return idx in frames
    else:
        def is_present(spk, t):
            return True

    # Fall back to last real-speaker end time; don't let broadcast-audio artifacts extend the video
    real_ends = [s["end"] for s in diarization if s["speaker"] in speaker_to_pos and s["end"]]
    video_end = video_duration or (max(real_ends) if real_ends else 0.0)

    # Build a lookup: at any time t, which real speaker is "primary"?
    # Merge consecutive same-speaker real segments for efficiency.
    real_segs = sorted(
        [s for s in diarization if s["speaker"] in speaker_to_pos],
        key=lambda s: s["start"]
    )
    merged = []
    for seg in real_segs:
        if (merged and merged[-1]["speaker"] == seg["speaker"]
                and seg["start"] <= merged[-1]["end"] + 0.15):
            merged[-1] = dict(merged[-1], end=max(merged[-1]["end"], seg["end"]))
        else:
            merged.append(dict(seg))

    def primary_at(t):
        """Nearest real speaker at time t; None if no real segment covers t."""
        for seg in merged:
            if seg["start"] <= t < seg["end"]:
                return seg["speaker"]
        return None

    timeline          = []
    sec_idx           = defaultdict(int)   # per-primary: which secondary is next
    last_primary      = merged[0]["speaker"] if merged else pos_keys[0]
    last_primary_end  = {}                 # spk → timestamp when they last stopped being primary
    gap_ptr           = 0                  # rotates the gap-carousel through present seats

    def emit_window(t_start, t_end, primary, is_new_primary):
        """Fill [t_start, t_end] with intro solo + carousel splits."""
        nonlocal last_primary
        if t_end - t_start < MIN_SEGMENT:
            return
        primary_pos = speaker_to_pos[primary]
        # Candidate secondaries = OTHER distinct seats, busiest-talker first.
        others = sorted(
            [s for s in distinct_positions if speaker_to_pos[s] is not primary_pos],
            key=lambda s: last_real_spoke_before(s, t_start),
            reverse=True,
        )

        def eligible_secs(t):
            """Secondaries we may show in a split at time t.

            Primary gate is PHYSICAL PRESENCE: never show a split with someone
            whose chair is empty. When presence data is unavailable, fall back
            to speech-recency (within ACTIVITY_WINDOW) as a presence proxy.
            """
            if presence:
                return [s for s in others if is_present(s, t)]
            return [s for s in others
                    if last_real_spoke_before(s, t) >= t - ACTIVITY_WINDOW]

        t = t_start

        # Brief solo intro when primary speaker changes — only if:
        #  • window is long enough
        #  • this speaker hasn't been primary recently (avoids solo↔split chop on rapid exchanges)
        gap_since_primary = t_start - last_primary_end.get(primary, float('-inf'))
        if (is_new_primary and others
                and gap_since_primary >= SOLO_REINTRO_GAP
                and (t_end - t_start) > SOLO_INTRO_DUR + MIN_SEGMENT):
            solo_end = min(t + SOLO_INTRO_DUR, t_end)
            if solo_end - t >= MIN_SEGMENT:
                timeline.append({"type": "single", "speakers": [primary],
                                 "start": t, "end": solo_end, "gap_fill": False})
                t = solo_end

        # Carousel splits for everything else. NEVER emit a single shot longer
        # than one slot — if no secondary is present this slot, re-check next slot.
        while t < t_end - MIN_SEGMENT:
            # Only rotate among seats physically present right now.
            active = eligible_secs(t)
            slot_end = min(t + CAROUSEL_DUR, t_end)
            if active:
                # Normal rotation among present speakers
                sec = active[sec_idx[primary] % len(active)]
                sec_idx[primary] += 1
                if slot_end - t >= MIN_SEGMENT:
                    timeline.append({"type": "split", "speakers": [primary, sec],
                                     "start": t, "end": slot_end, "gap_fill": False})
            else:
                # No active secondary — go solo for this slot, re-check next slot
                if slot_end - t >= MIN_SEGMENT:
                    timeline.append({"type": "single", "speakers": [primary],
                                     "start": t, "end": slot_end, "gap_fill": False})
            t = slot_end

        # Emit any remaining tail the loop skipped to avoid gaps in audio
        if t < t_end:
            remaining = t_end - t
            if remaining >= MIN_SEGMENT:
                if others:
                    active = eligible_secs(t)
                    if active:
                        sec = active[sec_idx[primary] % len(active)]
                        timeline.append({"type": "split", "speakers": [primary, sec],
                                         "start": t, "end": t_end, "gap_fill": False})
                    else:
                        timeline.append({"type": "single", "speakers": [primary],
                                         "start": t, "end": t_end, "gap_fill": False})
                else:
                    timeline.append({"type": "single", "speakers": [primary],
                                     "start": t, "end": t_end, "gap_fill": False})
            elif timeline:
                # Stretch the last segment to absorb a near-zero tail
                timeline[-1] = dict(timeline[-1], end=t_end)

        if is_new_primary and last_primary != primary:
            last_primary_end[last_primary] = t_start
        last_primary = primary

    def emit_gap_carousel(t_start, t_end):
        """No mapped speaker is talking across [t_start, t_end] (silence, broadcast
        audio, or speech we failed to map). The INVARIANT: never hold a static
        single shot. Rotate the primary through the PRESENT seats and alternate a
        brief single with a split, so the output always carousels among whoever is
        physically in frame — regardless of diarization/mapping quality.

        Falls back gracefully: if presence data is missing, is_present() is always
        True, so this still rotates through every seat instead of freezing."""
        nonlocal gap_ptr, last_primary
        if t_end - t_start < MIN_SEGMENT:
            return
        t = t_start
        while t < t_end - MIN_SEGMENT:
            present = [p for p in distinct_positions if is_present(p, t)]
            if not present:
                present = [last_primary if last_primary in distinct_positions
                           else distinct_positions[0]]
            primary = present[gap_ptr % len(present)]
            mates   = [p for p in present if speaker_to_pos[p] is not speaker_to_pos[primary]]

            # Brief single of the current seat …
            solo_end = min(t + SOLO_INTRO_DUR, t_end)
            if solo_end - t >= MIN_SEGMENT:
                timeline.append({"type": "single", "speakers": [primary],
                                 "start": t, "end": solo_end, "gap_fill": True})
                t = solo_end
            if t >= t_end - MIN_SEGMENT:
                break
            # … then a split pairing it with another present seat.
            slot_end = min(t + CAROUSEL_DUR, t_end)
            if mates:
                sec = mates[gap_ptr % len(mates)]
                timeline.append({"type": "split", "speakers": [primary, sec],
                                 "start": t, "end": slot_end, "gap_fill": True})
            else:
                timeline.append({"type": "single", "speakers": [primary],
                                 "start": t, "end": slot_end, "gap_fill": True})
            t = slot_end
            gap_ptr  += 1
            last_primary = primary
        if t < t_end and timeline:
            timeline[-1] = dict(timeline[-1], end=t_end)

    # Walk through all merged real-speaker windows + gaps between them
    prev_primary = None
    t = 0.0

    for seg in merged:
        # Gap before this segment (broadcast audio / silence)
        if seg["start"] > t + MIN_SEGMENT:
            emit_gap_carousel(t, seg["start"])

        # Clip start to t — pyannote can produce overlapping segments from different
        # speakers; without this clip the timeline goes backwards and content repeats.
        actual_start = max(seg["start"], t)
        seg_dur = seg["end"] - actual_start

        # Skip blip segments from speakers who have been inactive for > ACTIVITY_WINDOW.
        # A short utterance from someone who left long ago is a pyannote misidentification;
        # promoting them to primary would show their empty chair at the top of the frame.
        is_blip_from_absent = (
            seg_dur < MIN_ACTIVITY_DUR
            and last_real_spoke_before(seg["speaker"], seg["start"]) < seg["start"] - ACTIVITY_WINDOW
        )
        if actual_start < seg["end"] - MIN_SEGMENT and not is_blip_from_absent:
            is_new = seg["speaker"] != prev_primary
            emit_window(actual_start, seg["end"], seg["speaker"], is_new)
            prev_primary = seg["speaker"]
        t = max(t, seg["end"])  # never let t go backwards

    # Tail gap after last real segment
    if t < video_end - MIN_SEGMENT:
        emit_gap_carousel(t, video_end)

    # ── Make the timeline GAP-FREE ────────────────────────────────────────────
    # emit_window/emit_gap_carousel skip sub-MIN_SEGMENT slivers and leave small
    # gaps at diarization-segment boundaries. Those spans are never rendered, so
    # the output ends up shorter than the video AND the (per-segment-seeked) audio
    # skips at every hole. Absorb each gap into the preceding shot so the timeline
    # tiles [0, video_end] continuously: end[i] := start[i+1], last end := video_end.
    timeline.sort(key=lambda s: s["start"])
    cleaned = []
    for shot in timeline:
        if cleaned and shot["start"] < cleaned[-1]["end"] - 0.001:
            continue  # fully overlapped by previous shot — drop
        cleaned.append(shot)
    for i in range(len(cleaned) - 1):
        cleaned[i] = dict(cleaned[i], end=cleaned[i + 1]["start"])
    if cleaned:
        cleaned[-1] = dict(cleaned[-1], end=max(cleaned[-1]["end"], video_end))
    cleaned = [s for s in cleaned if s["end"] - s["start"] > 0.01]

    return cleaned

def render(video_path, output_path="output/final.mp4", end_time=None, external_audio=None):
    os.makedirs("output", exist_ok=True)
    segments_dir = tempfile.mkdtemp(prefix="360autocam_segs_")
    try:
      _render_inner(video_path, output_path, end_time, external_audio, segments_dir)
    finally:
      shutil.rmtree(segments_dir, ignore_errors=True)


def _render_inner(video_path, output_path, end_time, external_audio, segments_dir):
    # Auto-detect aligned external audio if not explicitly provided
    if external_audio is None and os.path.exists("data/external_audio_aligned.wav"):
        external_audio = "data/external_audio_aligned.wav"
        print(f"Using external audio: {external_audio}")

    positions   = load_json("data/speaker_positions.json")
    diarization = load_json("data/diarization.json")

    map_path = "data/speaker_map.json"
    if os.path.exists(map_path):
        speaker_map = load_json(map_path)
    else:
        diar_spk = sorted(set(s["speaker"] for s in diarization))
        pos_spk  = sorted(positions.keys())
        if len(diar_spk) == len(pos_spk):
            speaker_map = dict(zip(diar_spk, pos_spk))
            print(f"Auto-mapped: {speaker_map}")
            print("If wrong, create data/speaker_map.json manually.")
        else:
            print(f"Diarization: {len(diar_spk)} speakers — {diar_spk}")
            print(f"Positions:   {len(pos_spk)} speakers — {pos_spk}")
            print("Create data/speaker_map.json to map them manually.")
            sys.exit(1)

    speaker_to_pos = {}
    for d, p in speaker_map.items():
        if p not in positions:
            print(f"Error: speaker_map.json maps '{d}' → '{p}', "
                  f"but '{p}' is not in speaker_positions.json.\n"
                  f"Available positions: {sorted(positions)}")
            sys.exit(1)
        speaker_to_pos[d] = positions[p]
    if not speaker_to_pos:
        print("Error: speaker_map.json contains no valid speaker-to-position mappings. "
              "Check data/speaker_map.json.")
        sys.exit(1)

    # Diagnostic: show how much each mapped speaker speaks
    from collections import Counter
    spk_dur = Counter()
    for seg in diarization:
        spk_dur[seg["speaker"]] += seg["end"] - seg["start"]
    total_dur = sum(spk_dur.values()) or 1
    print("Speaker speech time:")
    for spk, pos_key in speaker_map.items():
        dur = spk_dur.get(spk, 0)
        print(f"  {spk} → {pos_key}  {dur:.1f}s  ({100*dur/total_dur:.1f}% of speech)", flush=True)

    overlaps = []
    ovlp_path = "data/overlaps.json"
    if os.path.exists(ovlp_path):
        overlaps = load_json(ovlp_path)
        print(f"Loaded {len(overlaps)} OSD overlap intervals from {ovlp_path}")

    # Physical-presence-over-time (data/presence.json from step 2b), keyed by
    # speaker label so the carousel can gate splits on who is actually present.
    presence = None
    pres_path = "data/presence.json"
    if os.path.exists(pres_path):
        raw_pres = load_json(pres_path)
        interval = raw_pres.get("interval", 10) or 10
        by_speaker = {}
        for spk, pos_key in speaker_map.items():
            times = raw_pres.get("positions", {}).get(pos_key)
            if times is not None:
                by_speaker[spk] = set(int(round(t / interval)) for t in times)
        presence = {"interval": interval,
                    "num_frames": raw_pres.get("num_frames", 0),
                    "by_speaker": by_speaker}
        print(f"Loaded presence for {len(by_speaker)} speakers from {pres_path}")
    else:
        print("No data/presence.json — carousel will use speech-recency presence proxy.")

    # Get video duration for gap-fill coverage to end of recording
    dur_result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", video_path],
        capture_output=True, text=True)
    video_dur = float(dur_result.stdout.strip()) if dur_result.returncode == 0 else None

    # Video frame rate — needed to frame-align segment durations (see below).
    fps_res = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=r_frame_rate", "-of", "default=noprint_wrappers=1:nokey=1",
         video_path], capture_output=True, text=True)
    try:
        num, den = fps_res.stdout.strip().split("/")
        fps = float(num) / float(den)
    except Exception:
        fps = 30.0
    print(f"Video frame rate: {fps:.3f} fps")

    timeline = resolve_timeline(diarization, speaker_to_pos,
                                video_duration=video_dur, overlaps=overlaps,
                                presence=presence)
    if end_time is not None:
        clipped = []
        for s in timeline:
            if s["start"] >= end_time:
                continue
            if s["end"] > end_time:
                s = dict(s, end=end_time)
            clipped.append(s)
        timeline = clipped

    # ── Frame-align all boundaries (CRITICAL for A/V sync) ─────────────────────
    # ffmpeg's -t rounds a segment UP to the frame covering its end (~+0–½ frame,
    # always positive). Summed over hundreds of segments that bias drifts the
    # video against the single continuous audio track by several seconds. Snapping
    # every boundary to the frame grid makes each segment an exact whole number of
    # frames → -t is exact → durations sum exactly → audio stays locked. Shared
    # boundaries snap identically, so the timeline stays gap-free.
    def _snap(t):
        return round(t * fps) / fps
    snapped = []
    for s in timeline:
        b = _snap(s["end"])
        a = snapped[-1]["end"] if snapped else _snap(s["start"])  # force contiguity
        if b - a >= 0.5 / fps:                      # ≥ ~half a frame → keep
            snapped.append(dict(s, start=a, end=b))
        elif snapped and b > snapped[-1]["end"]:    # sub-frame → absorb into previous
            snapped[-1] = dict(snapped[-1], end=b)
    timeline = snapped

    real   = sum(1 for s in timeline if not s.get("gap_fill"))
    fills  = sum(1 for s in timeline if s.get("gap_fill"))
    splits = sum(1 for s in timeline if s["type"] == "split")
    print(f"Timeline: {len(timeline)} shots ({real} speech, {fills} gap-fill holds, {splits} split-screen)")

    # ── Encode segments (v360 per segment, N_RENDER_WORKERS parallel) ───────────
    print(f"Encoding {len(timeline)} segments with {N_RENDER_WORKERS} workers…", flush=True)

    def encode_shot(args):
        i, shot = args
        out = os.path.join(segments_dir, f"seg_{i:05d}.mp4")
        if shot["type"] == "single":
            extract_segment(video_path, speaker_to_pos[shot["speakers"][0]],
                            shot["start"], shot["end"], out,
                            external_audio=external_audio)
        else:
            extract_split_segment(video_path,
                                  speaker_to_pos[shot["speakers"][0]],
                                  speaker_to_pos[shot["speakers"][1]],
                                  shot["start"], shot["end"], out,
                                  external_audio=external_audio)
        return i, out

    clips_map = {}
    done = 0
    with ThreadPoolExecutor(max_workers=N_RENDER_WORKERS) as pool:
        futures = {pool.submit(encode_shot, (i, shot)): i for i, shot in enumerate(timeline)}
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

    # 1) Join the video-only segments into one silent, full-length video.
    silent_video = os.path.join(segments_dir, "video_only.mp4")
    subprocess.run([
        "ffmpeg", "-f", "concat", "-safe", "0",
        "-i", concat_list,
        "-c", "copy", "-an",
        silent_video, "-y"
    ], check=True)

    # 2) Lay ONE continuous audio track over it in a single encode → gapless.
    #    Output time 0 == camera time timeline[0].start, so the audio (which is on
    #    the camera/aligned timeline) is seeked to that same offset.
    t0 = timeline[0]["start"] if timeline else 0.0
    audio_src = external_audio if external_audio else (
        "data/audio.wav" if os.path.exists("data/audio.wav") else None)

    # The VIDEO is the master clock: -af apad pads the audio with silence if it
    # runs short, and -shortest then trims the result to the exact video length,
    # so the output is always video-length with A/V kept in sync.
    if audio_src and os.path.exists(audio_src):
        print(f"Overlaying continuous audio from {audio_src} (offset {t0:.2f}s)…")
        subprocess.run([
            "ffmpeg",
            "-i", silent_video,
            "-ss", f"{t0:.6f}", "-i", audio_src,
            "-map", "0:v:0", "-map", "1:a:0",
            "-af", "apad",
            "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
            "-shortest",
            output_path, "-y"
        ], check=True)
    else:
        # No audio source available — fall back to the camera's own audio track.
        print(f"No separate audio track; pulling camera audio from the video (offset {t0:.2f}s)…")
        subprocess.run([
            "ffmpeg",
            "-i", silent_video,
            "-ss", f"{t0:.6f}", "-i", video_path,
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
                        help="Only render segments starting before this time (seconds)")
    parser.add_argument("--external-audio", default=None,
                        help="Aligned external mic WAV to use instead of camera audio "
                             "(auto-detected from data/external_audio_aligned.wav if present)")
    args = parser.parse_args()
    render(args.video_path, args.output_path,
           end_time=args.end_time, external_audio=args.external_audio)
