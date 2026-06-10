#!/usr/bin/env python3
"""
Step 3: Run speaker diarization on the video's audio track.
Requires a Hugging Face token with access to pyannote/speaker-diarization-3.1

Set your token: export HF_TOKEN=hf_xxxx
Output: data/diarization.json
"""

import os
import sys
import io
import json
import subprocess
import warnings
warnings.filterwarnings('ignore')  # suppress torchaudio/speechbrain deprecation spam

# Force stderr to be write-through so pyannote's tqdm \r updates flush immediately
if hasattr(sys.stderr, 'buffer'):
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, write_through=True)

from concurrent.futures import ThreadPoolExecutor

# PyTorch 2.6+ changed torch.load default to weights_only=True, which breaks
# pyannote checkpoints that store custom classes. Patch the default back to False
# before any model loading happens. Safe because we only load from HuggingFace.
try:
    import torch
    _orig_load = torch.load
    def _load_compat(*a, **kw):
        kw['weights_only'] = False  # force — lightning_fabric passes weights_only=True explicitly
        return _orig_load(*a, **kw)
    torch.load = _load_compat
except Exception:
    pass

# Load .env if present
env_path = os.path.join(os.path.dirname(__file__), ".env")
if os.path.exists(env_path):
    with open(env_path) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _v = _line.split("=", 1)
                os.environ.setdefault(_k.strip(), _v.strip())

def extract_audio(video_path: str, audio_path: str = "data/audio.wav"):
    os.makedirs("data", exist_ok=True)
    print(f"Extracting audio from {video_path}...")
    subprocess.run([
        "ffmpeg", "-i", video_path,
        "-vn", "-acodec", "pcm_s16le",
        "-ar", "16000", "-ac", "1",
        audio_path, "-y"
    ], check=True)
    print(f"Audio saved to {audio_path}")
    return audio_path

class _ProgressHook:
    """Emits PROGRESS N/M lines to stdout for pyannote sub-step progress."""
    def __init__(self):
        self._last_pct = -1

    def __call__(self, step_name, step_artifact=None, file=None, total=None, completed=None, **_):
        if total and completed is not None and total > 0:
            pct = int(completed / total * 100)
            if pct != self._last_pct:
                print(f"PROGRESS {completed}/{total}", flush=True)
                self._last_pct = pct


def run_diarization(audio_path: str, hf_token: str) -> list:
    from pyannote.audio import Pipeline

    print("Loading diarization pipeline (first run downloads ~1GB of models)...")
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=hf_token
    )

    try:
        import torch
        if torch.backends.mps.is_available():
            pipeline.to(torch.device("mps"))
            print("Using MPS (Apple Silicon GPU) for diarization", flush=True)
    except Exception as _e:
        pass  # CPU fallback is silent; pyannote will warn if needed

    print("Running diarization...", flush=True)
    hook = _ProgressHook()

    # Speaker-count control. Explicit NUM_SPEAKERS wins. Otherwise, cap the count
    # using how many faces were detected on camera (data/speaker_positions.json):
    # without a cap, pyannote over-segments noisy / music-heavy studio audio into
    # dozens of phantom speakers (seen: 37 from an external mic feed). Real talkers
    # ≈ faces on camera, plus a little headroom for off-camera voices / broadcast.
    num_speakers = int(os.environ.get("NUM_SPEAKERS", 0)) or None
    kwargs = {}
    if num_speakers:
        kwargs["num_speakers"] = num_speakers
        print(f"  num_speakers fixed to {num_speakers} (NUM_SPEAKERS env)", flush=True)
    else:
        n_pos = 0
        try:
            if os.path.exists("data/speaker_positions.json"):
                with open("data/speaker_positions.json") as f:
                    n_pos = len(json.load(f))
        except Exception:
            n_pos = 0
        if n_pos:
            # Generous cap: bound pathological over-segmentation (seen: 37 phantom
            # speakers) WITHOUT forcing real distinct hosts to merge. A tight cap
            # (e.g. n_pos+2) made pyannote fuse two separate people into one giant
            # cluster on far-field audio. automap seeds the n_pos busiest speakers
            # and absorbs the rest, so extra clusters are harmless but merges aren't.
            cap = 3 * n_pos + 2
            kwargs["max_speakers"] = cap
            print(f"  capping at max_speakers={cap} "
                  f"(generous bound from {n_pos} detected face positions)", flush=True)
        else:
            print("  no position count available — letting pyannote auto-detect "
                  "(may over-segment)", flush=True)

    diarization = pipeline(audio_path, hook=hook, **kwargs)

    segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segments.append({
            "speaker": speaker,
            "start": round(turn.start, 3),
            "end": round(turn.end, 3)
        })

    return segments


def run_overlap_detection(audio_path: str, hf_token: str) -> list:
    from pyannote.audio.pipelines import OverlappedSpeechDetection

    print("Loading overlap detection pipeline...")
    osd = OverlappedSpeechDetection(segmentation="pyannote/segmentation-3.0",
                                    use_auth_token=hf_token)
    osd.instantiate({
        "min_duration_on":  0.3,
        "min_duration_off": 0.1,
    })

    print("Running overlap detection...")
    result = osd(audio_path)

    overlaps = []
    for turn, _ in result.itertracks():
        overlaps.append({
            "start": round(turn.start, 3),
            "end":   round(turn.end, 3),
        })

    return overlaps

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("video_path")
    parser.add_argument("--audio", default=None,
                        help="Use this WAV instead of extracting from video "
                             "(e.g. data/external_audio_aligned.wav)")
    parser.add_argument("--end-time", type=float, default=None,
                        help="Trim audio to this many seconds before diarizing (preview mode)")
    args = parser.parse_args()

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        print("Error: HF_TOKEN environment variable not set.")
        print("Get a token at https://huggingface.co/settings/tokens")
        print("Then accept model terms at https://huggingface.co/pyannote/speaker-diarization-3.1")
        sys.exit(1)

    video_path = args.video_path
    if args.audio:
        audio_path = args.audio
        print(f"Using provided audio: {audio_path}")
    elif os.path.exists("data/audio.wav"):
        audio_path = "data/audio.wav"
        print(f"Reusing camera audio extracted in step 0: {audio_path}")
    else:
        audio_path = extract_audio(video_path)

    # In preview mode, trim audio to speed up diarization
    if args.end_time:
        trimmed = audio_path.replace(".wav", f"_preview{int(args.end_time)}s.wav")
        print(f"Preview mode: trimming audio to {args.end_time}s…", flush=True)
        subprocess.run([
            "ffmpeg", "-i", audio_path,
            "-t", str(args.end_time), trimmed, "-y"
        ], check=True, capture_output=True)
        audio_path = trimmed

    # Run diarization and OSD in parallel — they're fully independent
    print("Launching diarization + overlap detection in parallel...")
    with ThreadPoolExecutor(max_workers=2) as pool:
        f_diar = pool.submit(run_diarization, audio_path, hf_token)
        f_osd  = pool.submit(run_overlap_detection, audio_path, hf_token)
        segments = f_diar.result()
        try:
            overlaps = f_osd.result()
        except Exception as e:
            print(f"Warning: overlap detection failed ({e}), continuing without overlaps", flush=True)
            overlaps = []

    out_path = "data/diarization.json"
    with open(out_path, "w") as f:
        json.dump(segments, f, indent=2)
    print(f"\nDiarization saved to {out_path} ({len(segments)} segments)")
    speakers = set(s["speaker"] for s in segments)
    print(f"Detected {len(speakers)} speakers: {', '.join(sorted(speakers))}")

    ovlp_path = "data/overlaps.json"
    with open(ovlp_path, "w") as f:
        json.dump(overlaps, f, indent=2)
    print(f"Overlap detection saved to {ovlp_path} ({len(overlaps)} intervals)")
