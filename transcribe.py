#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.13"
# dependencies = [
#   "whisperx>=3.3.4",
# ]
# ///
"""
Batch transcriber: transcribes any WAV recordings in ~/recordings/ that
don't already have a matching .txt transcript.

Usage:
    uv run transcribe.py [recordings_dir]

If recordings_dir is omitted, defaults to ~/recordings/.

Set HF_TOKEN env var to enable speaker diarization via pyannote.
"""

import os
import sys
from pathlib import Path

import whisperx

WHISPER_MODEL = "medium"   # tiny / base / small / medium / large-v3 / turbo
WHISPER_COMPUTE = "int8"   # int8 (recommended) or float16
BATCH_SIZE = 16


def get_device() -> str:
    try:
        import ctranslate2
        if ctranslate2.get_cuda_device_count() > 0:
            return "cuda"
    except Exception:
        pass
    return "cpu"


def load_model(device: str) -> object:
    print(f"[transcriber] Loading WhisperX model '{WHISPER_MODEL}' ({WHISPER_COMPUTE}) on {device}...")
    model = whisperx.load_model(WHISPER_MODEL, device, compute_type=WHISPER_COMPUTE)
    print("[transcriber] Model loaded.")
    return model


def transcribe_file(model, audio_path: Path, device: str, hf_token: str | None) -> None:
    txt_path = audio_path.with_suffix(".txt")
    print(f"[transcriber] Transcribing {audio_path.name}...")
    try:
        audio = whisperx.load_audio(str(audio_path))

        # Step 1: transcribe
        result = model.transcribe(audio, batch_size=BATCH_SIZE)
        language = result["language"]
        print(f"[transcriber] Detected language: {language}")

        # Step 2: align word-level timestamps
        align_model, metadata = whisperx.load_align_model(language_code=language, device=device)
        result = whisperx.align(
            result["segments"], align_model, metadata, audio, device, return_char_alignments=False
        )

        # Step 3: diarize (requires HF token)
        if hf_token:
            print("[transcriber] Running speaker diarization...")
            diarize_model = whisperx.DiarizationPipeline(use_auth_token=hf_token, device=device)
            diarize_segments = diarize_model(audio)
            result = whisperx.assign_word_speakers(diarize_segments, result)
        else:
            print("[transcriber] HF_TOKEN not set — skipping diarization (no speaker labels).")

        # Step 4: write transcript
        with txt_path.open("w") as f:
            current_speaker = None
            for segment in result["segments"]:
                speaker = segment.get("speaker", "UNKNOWN")
                text = segment["text"].strip()
                if speaker != current_speaker:
                    if current_speaker is not None:
                        f.write("\n")
                    f.write(f"[{speaker}]\n")
                    current_speaker = speaker
                f.write(text + "\n")

        print(f"[transcriber] Saved → {txt_path}")
    except Exception as e:
        print(f"[transcriber] Error transcribing {audio_path.name}: {e}")


def main():
    recordings_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.home() / "recordings"
    hf_token = os.environ.get("HF_TOKEN")

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
    device = get_device()
    model = load_model(device)
    for audio_path in pending:
        transcribe_file(model, audio_path, device, hf_token)

    print("[transcriber] Done.")


if __name__ == "__main__":
    main()
