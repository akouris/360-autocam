#!/usr/bin/env python3
"""
Step 3b: Auto-map diarization speaker IDs → camera positions.

Method: audio-visual lip-synchrony scoring (same principle as SyncNet).
  For each (speaker × camera-position) pair:
    1. Sample N segments where that speaker is talking
    2. Extract frames at that position via FFmpeg v360 filter
    3. Measure lip aperture per frame using MediaPipe Face Mesh
    4. Measure audio RMS per frame window from data/audio.wav
    5. Compute Pearson r between lip-aperture and audio-RMS signals

  High r  → lips at this position move in sync with this speaker's audio
           → this position is this speaker's face

  Uses scipy.optimize.linear_sum_assignment (Hungarian) for optimal 1-to-1
  assignment, then writes data/speaker_map.json.

  Speakers whose best-position r < EXCLUDE_THRESHOLD are flagged as broadcast
  audio and excluded from the map automatically.
"""

import os, sys, json, math, tempfile, shutil, subprocess, glob
import numpy as np

V360_INPUT = "equirect" if os.environ.get("VIDEO_MODE") == "equirect" else "fisheye"
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict

SAMPLE_SEGS       = 8      # segments to sample per speaker
MIN_SEG_DUR       = 2.0    # minimum usable segment (seconds)
MAX_SEG_DUR       = 4.0    # clip each segment to this length
FRAME_RATE        = 8      # fps for extracted frames
EXCLUDE_THRESHOLD = 0.08   # speakers with max r below this → broadcast audio
N_WORKERS         = min(4, (os.cpu_count() or 4) - 1)

# MediaPipe Face Mesh inner-lip landmarks
UPPER_IDX = 13   # upper inner lip midpoint
LOWER_IDX = 14   # lower inner lip midpoint


def load_json(p):
    with open(p) as f:
        return json.load(f)


def _norm_yaw(y):
    return ((y + 180) % 360) - 180

def _filter_yaw(pos):
    """Remove back-lens 180° offset and normalize to [-180, 180] for v360."""
    offset = 180.0 if pos.get("lens") == "back" else 0.0
    return _norm_yaw(pos["yaw_degrees"] - offset)

def extract_frames(video_path, pos, t_start, t_end, out_dir):
    """Extract 256×256 frames at a camera position into out_dir."""
    stream = pos.get("stream", 0)
    yaw    = _filter_yaw(pos)
    pitch  = pos.get("pitch_degrees", 0)
    h_fov  = pos.get("h_fov", 75)
    v_fov  = round(2 * math.degrees(math.atan(math.tan(math.radians(h_fov / 2)))), 1)

    subprocess.run([
        "ffmpeg", "-ss", str(t_start), "-i", video_path,
        "-t", str(t_end - t_start),
        "-map", f"0:v:{stream}",
        "-vf", (f"v360={V360_INPUT}:flat:yaw={yaw}:pitch={pitch}"
                f":h_fov={h_fov}:v_fov={v_fov}:w=256:h=256"
                f",fps={FRAME_RATE}"),
        "-q:v", "5",
        os.path.join(out_dir, "f%04d.jpg"), "-y"
    ], check=True, capture_output=True)
    return sorted(glob.glob(os.path.join(out_dir, "f*.jpg")))


FACE_LANDMARKER_MODEL = os.environ.get(
    "FACE_LANDMARKER_MODEL",
    os.path.join(os.path.dirname(__file__), "data", "face_landmarker.task"),
)


def lip_aperture_signal(frame_paths):
    """
    Per-frame lip aperture via MediaPipe FaceLandmarker (Tasks API, v0.10+).
    Returns np.array (length = n_frames), or None if unavailable.
    NaN frames (face not detected) are linearly interpolated.
    """
    try:
        import mediapipe as mp
        import cv2
    except ImportError:
        return None

    if not os.path.exists(FACE_LANDMARKER_MODEL):
        return None

    VisionRunningMode = mp.tasks.vision.RunningMode
    FaceLandmarker    = mp.tasks.vision.FaceLandmarker
    FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
    BaseOptions = mp.tasks.BaseOptions

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=FACE_LANDMARKER_MODEL),
        running_mode=VisionRunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.3,
        min_face_presence_confidence=0.3,
        min_tracking_confidence=0.3,
    )

    apertures = []
    with FaceLandmarker.create_from_options(options) as fm:
        for fp in frame_paths:
            img = cv2.imread(fp)
            if img is None:
                apertures.append(np.nan)
                continue
            rgb    = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = fm.detect(mp_img)
            if result.face_landmarks:
                lm    = result.face_landmarks[0]
                h     = img.shape[0]
                upper = lm[UPPER_IDX].y * h
                lower = lm[LOWER_IDX].y * h
                apertures.append(abs(lower - upper))
            else:
                apertures.append(np.nan)

    arr  = np.array(apertures, dtype=float)
    nans = np.isnan(arr)
    if nans.all():
        return np.zeros(len(arr))
    if nans.any():
        idx = np.arange(len(arr))
        arr = np.interp(idx, idx[~nans], arr[~nans])
    return arr


def audio_rms_signal(audio_path, t_start, t_end, n_frames):
    """RMS envelope aligned to n_frames equal-width windows."""
    import wave
    with wave.open(audio_path, "r") as wf:
        sr      = wf.getframerate()
        s_start = max(0, int(t_start * sr))
        n_samp  = int((t_end - t_start) * sr)
        wf.setpos(s_start)
        raw = wf.readframes(n_samp)
    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
    if len(samples) < n_frames:
        samples = np.pad(samples, (0, n_frames - len(samples)))
    step    = max(1, len(samples) // n_frames)
    windows = [samples[i * step:(i + 1) * step] for i in range(n_frames)]
    return np.array([np.sqrt(np.mean(w ** 2)) if len(w) else 0.0 for w in windows])


def score_pair(video_path, audio_path, pos, segments):
    """
    Average Pearson r (lip aperture vs audio RMS) across sampled segments.
    Returns (mean_r, n_valid).
    """
    from scipy.stats import pearsonr

    correlations = []
    tmp = tempfile.mkdtemp(prefix="automap_")
    try:
        for seg in segments:
            t0 = seg["start"]
            t1 = min(t0 + MAX_SEG_DUR, seg["end"])
            if t1 - t0 < MIN_SEG_DUR:
                continue
            frame_dir = tempfile.mkdtemp(dir=tmp)
            try:
                frames = extract_frames(video_path, pos, t0, t1, frame_dir)
            except subprocess.CalledProcessError:
                continue
            if len(frames) < 4:
                continue

            lip = lip_aperture_signal(frames)
            if lip is None or len(lip) < 4:
                continue
            audio = audio_rms_signal(audio_path, t0, t1, len(lip))

            if lip.std() < 1e-6 or audio.std() < 1e-6:
                continue

            r, _ = pearsonr(lip, audio)
            if not math.isnan(r):
                correlations.append(r)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    if correlations:
        return float(np.mean(correlations)), len(correlations)
    return 0.0, 0


def sample_segments(segs, n):
    """Pick n segments spread evenly across the video."""
    usable = [s for s in segs if s["end"] - s["start"] >= MIN_SEG_DUR]
    if not usable:
        return []
    if len(usable) <= n:
        return usable
    idxs = np.linspace(0, len(usable) - 1, n, dtype=int)
    return [usable[int(i)] for i in idxs]


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("video_path")
    ap.add_argument("--end-time", type=float, default=None,
                    help="Only use segments before this time (preview mode)")
    args = ap.parse_args()

    video_path = args.video_path
    audio_path = "data/audio.wav"

    # In preview mode use trimmed audio
    preview_audio = audio_path.replace(".wav", f"_preview{int(args.end_time)}s.wav") if args.end_time else None
    if preview_audio and os.path.exists(preview_audio):
        audio_path = preview_audio

    # Lip-sync needs the CAMERA audio (it correlates camera-video mouth motion
    # with audio energy). If it isn't present — e.g. the run used an external mic
    # for diarization and skipped the camera-audio step — extract it from the
    # video here instead of failing. Never hard-stop the pipeline on this.
    if not os.path.exists(audio_path) and not (preview_audio and audio_path == preview_audio):
        print(f"{audio_path} not found — extracting camera audio from the video…",
              flush=True)
        r = subprocess.run(
            ["ffmpeg", "-i", video_path, "-vn", "-acodec", "pcm_s16le",
             "-ar", "16000", "-ac", "1", "data/audio.wav", "-y", "-loglevel", "error"],
            capture_output=True, text=True)
        if r.returncode == 0 and os.path.exists("data/audio.wav"):
            audio_path = "data/audio.wav"
        else:
            print("Error: could not obtain camera audio for lip-sync.", flush=True)
            if r.stderr.strip():
                print(r.stderr.strip(), flush=True)
            sys.exit(1)

    diarization = load_json("data/diarization.json")
    if args.end_time:
        diarization = [s for s in diarization if s["start"] < args.end_time]
    positions   = load_json("data/speaker_positions.json")
    pos_keys    = list(positions.keys())
    if not pos_keys:
        print("Error: speaker_positions.json contains no positions. Re-run step 2 (detect speakers).")
        sys.exit(1)

    spk_dur  = defaultdict(float)
    spk_segs = defaultdict(list)
    for seg in diarization:
        spk_dur[seg["speaker"]]  += seg["end"] - seg["start"]
        spk_segs[seg["speaker"]].append(seg)

    all_speakers = sorted(spk_dur, key=lambda s: spk_dur[s], reverse=True)

    # Which diarization IDs are real on-camera people vs. broadcast/background audio
    # is decided by LIP-SYNC below (a real speaker's audio correlates with a face's
    # mouth movement; broadcast does not). We deliberately do NOT pre-exclude by
    # duration: when pyannote over-segments a few people into many labels, the real
    # hosts become high-duration outliers and a duration filter would wrongly drop
    # the very people we need to map.
    #
    # Score generously: the dominant talkers plus enough extra labels to catch
    # over-segmented fragments of the same people. Capped to bound lip-sync compute.
    n_score           = min(len(all_speakers), 3 * len(pos_keys) + 2)
    speakers_to_score = all_speakers[:n_score]

    print(f"Speakers: {[(s, int(spk_dur[s])) for s in speakers_to_score]}")
    print(f"Positions: {pos_keys}")
    print(f"Workers:   {N_WORKERS}\n")

    # Submit all (speaker, position) pairs in parallel
    tasks = {}
    with ThreadPoolExecutor(max_workers=N_WORKERS) as pool:
        for sp in speakers_to_score:
            segs = sample_segments(spk_segs[sp], SAMPLE_SEGS)
            if not segs:
                continue
            for pk in pos_keys:
                fut = pool.submit(score_pair, video_path, audio_path,
                                  positions[pk], segs)
                tasks[fut] = (sp, pk)

        score_map = {}
        total_pairs = len(tasks)
        done_pairs  = 0
        for fut in as_completed(tasks):
            sp, pk   = tasks[fut]
            r, n     = fut.result()
            score_map[(sp, pk)] = r
            done_pairs += 1
            print(f"PROGRESS {done_pairs}/{total_pairs}", flush=True)
            print(f"  {sp} × {pk}: r={r:+.3f}  (n={n})", flush=True)

    # Print score matrix
    print("\nScore matrix:")
    print("             " + "  ".join(f"{pk:12s}" for pk in pos_keys))
    for sp in speakers_to_score:
        row = "  ".join(f"{score_map.get((sp, pk), 0):+.3f}       " for pk in pos_keys)
        print(f"  {sp}  {row}")

    # Broadcast/background audio = a speaker whose best position still scores below
    # threshold (its audio doesn't move any on-camera mouth).
    def best_r(s):
        return max(score_map.get((s, pk), 0) for pk in pos_keys)

    real_speakers = [s for s in speakers_to_score if best_r(s) >= EXCLUDE_THRESHOLD]
    excluded      = [s for s in speakers_to_score if s not in real_speakers]
    if excluded:
        print(f"\nExcluded (max r < {EXCLUDE_THRESHOLD}, likely broadcast/background): {excluded}")

    if not real_speakers:
        print("\nNo speakers passed the threshold — all diarization IDs look like broadcast audio.")
        print("Lower EXCLUDE_THRESHOLD or check speaker_positions.json and diarization.json.")
        sys.exit(1)

    from scipy.optimize import linear_sum_assignment

    # Seed: assign the len(pos_keys) busiest real speakers to distinct positions
    # 1-to-1 (Hungarian). This pins each camera seat to its main occupant and
    # prevents two seats collapsing onto one person.
    seed     = sorted(real_speakers, key=lambda s: spk_dur[s], reverse=True)[:len(pos_keys)]
    mat      = np.array([[score_map.get((sp, pk), 0.0) for pk in pos_keys] for sp in seed])
    ri, ci   = linear_sum_assignment(-mat)
    speaker_map = {}
    print("\nAssignment (seed, 1-to-1):")
    for i, j in zip(ri, ci):
        speaker_map[seed[i]] = pos_keys[j]
        print(f"  {seed[i]} → {pos_keys[j]}  (r={mat[i, j]:+.3f})")

    # Attach remaining real speakers (over-segmented fragments of the same people,
    # cross-talk) to their best-correlating position — MANY labels may map to ONE
    # position. Without this, every fragment pyannote split off would be unmapped
    # and the renderer would hold a static single shot through all their speech.
    extras = [s for s in real_speakers if s not in speaker_map]
    if extras:
        print("Attach (over-segmented fragments → best position):")
        for sp in sorted(extras, key=lambda s: spk_dur[s], reverse=True):
            best_pk = max(pos_keys, key=lambda pk: score_map.get((sp, pk), 0))
            speaker_map[sp] = best_pk
            print(f"  {sp} → {best_pk}  (r={score_map.get((sp, best_pk), 0):+.3f}, {spk_dur[sp]:.0f}s)")

    out_path = "data/speaker_map.json"
    with open(out_path, "w") as f:
        json.dump(speaker_map, f, indent=2)
    print(f"\nSaved {out_path}")
    print("Review the assignment. If wrong, edit data/speaker_map.json manually.")
