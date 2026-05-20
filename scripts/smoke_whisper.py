"""Smoke test: load one Whisper-HF model, transcribe one .wav, print the result.

Usage:
    python scripts/smoke_whisper.py /path/to/clip.wav
    python scripts/smoke_whisper.py /path/to/clip.wav --model openai/whisper-large-v3-turbo --device cpu

This is the "is the adapter shape right" check. If this works on a single
clip, the per-clip loop in Cell 4 of the notebook will work too.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make `src` importable when running this file directly.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.transcribers.whisper_hf import WhisperHFTranscriber  # noqa: E402
from src.utils import setup_logging  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio_path", help="Path to a .wav file to transcribe")
    parser.add_argument(
        "--model",
        default="openai/whisper-large-v3-turbo",
        help="HF model id (default: whisper-large-v3-turbo — small enough for a quick check)",
    )
    parser.add_argument("--device", default="cuda", help="cuda | cpu")
    parser.add_argument("--language", default="bn", help="ISO code passed to Whisper generate (bn = Bengali)")
    args = parser.parse_args()

    setup_logging(REPO_ROOT / "logs")

    audio = Path(args.audio_path)
    if not audio.exists():
        print(f"ERROR: audio file not found: {audio}", file=sys.stderr)
        sys.exit(1)

    transcriber = WhisperHFTranscriber(
        model_id=args.model,
        device=args.device,
        language=args.language,
    )
    text, latency = transcriber.transcribe(str(audio))

    print()
    print(f"Model    : {args.model}")
    print(f"Audio    : {audio}")
    print(f"Latency  : {latency:.2f} s")
    print(f"Transcript:\n  {text}")
    print()

    transcriber.cleanup()


if __name__ == "__main__":
    main()
