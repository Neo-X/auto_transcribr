#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.13"
# dependencies = [
#   "faster-whisper>=1.2.1",
# ]
# ///
"""
Batch transcriber: transcribes any WAV recordings in ~/recordings/ that
don't already have a matching .txt transcript.

Usage:
    uv run transcribe.py [recordings_dir]

If recordings_dir is omitted, defaults to ~/recordings/.
"""

import sys
from pathlib import Path

from faster_whisper import WhisperModel

WHISPER_MODEL = "medium"   # tiny / base / small / medium / large-v3 / turbo
WHISPER_COMPUTE = "int8"   # int8 (recommended) or float16


def cuda_available() -> bool:
    try:
        import ctranslate2
        return ctranslate2.get_cuda_device_count() > 0
    except Exception:
        return False


def load_model() -> WhisperModel:
    device = "cuda" if cuda_available() else "cpu"
    print(f"[transcriber] Loading Whisper model '{WHISPER_MODEL}' ({WHISPER_COMPUTE}) on {device}...")
    try:
        model = WhisperModel(WHISPER_MODEL, device=device, compute_type=WHISPER_COMPUTE)
    except Exception as e:
        if device == "cuda":
            print(f"[transcriber] CUDA load failed ({e}), falling back to CPU...")
            model = WhisperModel(WHISPER_MODEL, device="cpu", compute_type=WHISPER_COMPUTE)
        else:
            print(f"[transcriber] Failed to load model: {e}")
            sys.exit(1)
    print("[transcriber] Model loaded.")
    return model


def transcribe_file(model: WhisperModel, audio_path: Path) -> None:
    txt_path = audio_path.with_suffix(".txt")
    print(f"[transcriber] Transcribing {audio_path.name}...")
    try:
        segments, _ = model.transcribe(str(audio_path), beam_size=5)
        with txt_path.open("w") as f:
            for segment in segments:
                f.write(segment.text.strip() + "\n")
        print(f"[transcriber] Saved → {txt_path}")
    except Exception as e:
        print(f"[transcriber] Error transcribing {audio_path.name}: {e}")


def main():
    recordings_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.home() / "recordings"

    if not recordings_dir.exists():
        print(f"[transcriber] Directory not found: {recordings_dir}")
        sys.exit(1)

    pending = sorted(
        wav for wav in recordings_dir.glob("*.wav")
        if not wav.with_suffix(".txt").exists()
    )

    if not pending:
        print("[transcriber] No unprocessed recordings found.")
        return

    print(f"[transcriber] Found {len(pending)} unprocessed recording(s) in {recordings_dir}")
    model = load_model()
    for audio_path in pending:
        transcribe_file(model, audio_path)

    print("[transcriber] Done.")


if __name__ == "__main__":
    main()
