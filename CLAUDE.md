# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the tools

All scripts use `uv run` as their shebang and can be run directly:

```bash
uv run monitor.py          # auto-record meetings (runs system tray daemon)
uv run dictate.py          # voice dictation via hotkey
uv run transcribe.py       # batch-transcribe existing WAV files in ~/recordings/
uv run analyze.py          # analyze transcripts with Claude API
```

With environment variables:

```bash
HF_TOKEN=... uv run monitor.py          # enables speaker diarization
ANTHROPIC_API_KEY=... uv run analyze.py
ROCR_VISIBLE_DEVICES=0 uv run dictate.py  # AMD GPU fix for Strix Halo APUs
```

## Architecture

Four independent scripts — no shared library, no imports between them:

| Script | Purpose |
|--------|---------|
| `monitor.py` | Polls every 3s for Zoom/Meet, drives `ffmpeg` to record, runs WhisperX transcription in a background thread. System tray via `pystray`. |
| `transcribe.py` | Batch mode: scans a directory for `.wav` files without a matching `.txt` and transcribes them with WhisperX. |
| `dictate.py` | Holds a keyboard hotkey (Ctrl+Shift+Space toggle), records mic via `sounddevice`, transcribes with Qwen3-ASR 1.7B by default (`--model whisper/granite/cohere` for the others), biased towards the names/jargon in `vocabulary.txt` (re-read on every recording; add terms with `python3 add_vocab.py <term>...`), and types output into the focused window. Each clip is staged in `/tmp/dictate-clips` so mis-transcriptions can be added to the ASR dataset in `~/playground/asr-whisper-finetuning` (`src/add_clip.py`). |
| `analyze.py` | Reads `.txt` transcripts, sends them to Claude (`claude-sonnet-4-6`) with a structured prompt, and writes `_analysis.md` files into the Obsidian vault at `~/playground/Obsidian/Transcripts/`. |

### Two different speech-to-text stacks

- `monitor.py` and `transcribe.py` use **WhisperX** (`whisperx`) — adds word-level alignment and speaker diarization via pyannote. Runs on NVIDIA CUDA via `ctranslate2`.
- `dictate.py` defaults to **Qwen3-ASR** (`Qwen/Qwen3-ASR-1.7B-hf` via transformers), with `vocabulary.txt` passed as context/hotwords; **openai-whisper** turbo is still available with `--model whisper`. Single-speaker, runs on AMD ROCm via PyTorch (`torch==2.11.0+rocm7.2`). Models were compared on the dictation failure set in `~/playground/asr-whisper-finetuning` (`src/compare_dictation.py`).

GPU detection differs accordingly: WhisperX checks `ctranslate2.get_cuda_device_count()`; dictate uses `torch.cuda.is_available()`.

### dictate.py internals

Recording, transcription, and text injection are split across threads:
1. `_record()` — runs in a daemon thread while `_recording` is True, appends chunks to `_audio_chunks`.
2. `_Stream` (qwen3 only; off with `--no-stream` / `DICTATE_STREAM=0`) — while the recording is still going, a background thread transcribes the audio so far every `STREAM_STEP_S` (20 s; shorter steps cost accuracy, see the comment in the code). Each pass continues from the text of the pass before it (passed to the model as the start of its answer) minus the last `STREAM_HOLD` (10) tokens, which the next pass writes afresh with more audio. This is what keeps the wait after release short on long dictations: writing text costs about 0.2 s per second of audio.
3. `_transcribe_and_type()` — runs in a daemon thread after recording stops; `_Stream.finish()` does one last pass over the whole recording (a plain one-go transcription if the recording was shorter than one step), then `type_text()`.
4. Hotkey detection runs either via `pynput` keyboard listener (X11) or a `select`-based `evdev` loop (Wayland). Both call `_toggle_recording()`.

The model is loaded by `_load_model()` from the `__main__` block, not at import, so `tests/` can import `dictate`. `_get_device()` sets `MIOPEN_FIND_MODE=FAST`: without it ROCm spends ~25 s searching for convolution algorithms the first time it sees each new audio length.

### Tests

`uv run --no-sync python tests/test_streaming.py --quick` (about 5 minutes on the laptop, a fixed sample of 28 labelled clips listed in `tests/quick_clips.json`; without `--quick` it runs every clip, 40+ minutes; loads the model on the GPU) checks the model's word error rate on the labelled clips in `~/playground/asr-whisper-finetuning/data/dictation` against `tests/baseline.json`, and that transcribing while recording gives the same quality as one-go transcription with a much shorter wait, by replaying saved clips against a clock. Run it after changing the model, the prompt, or the `_Stream` code; `--update-baseline` accepts a new accuracy level. Results go to `tests/results/`.

Text injection tries backends in order: `wtype` → `ydotool` → `xdotool`+clipboard → clipboard-only.

### monitor.py internals

`monitor_loop()` runs in a background thread; the `pystray` tray loop owns the main thread. When a meeting ends, transcription is launched in yet another daemon thread so the monitor loop can resume immediately.

## Dependencies and GPU notes

- ROCm 7.x is required for `dictate.py`; ROCm 6.x does not support Strix Halo (gfx1100).
- `whisperx` is **not** in `pyproject.toml` (it has complex dependencies); install it separately.
- The `[tool.uv.sources]` section points `torch`/`torchaudio` to the ROCm 7.2 PyTorch index.
