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
| `dictate.py` | Holds a keyboard hotkey (Ctrl+Shift+Space toggle), records mic via `sounddevice`, transcribes with `openai-whisper`, and types output into the focused window. |
| `analyze.py` | Reads `.txt` transcripts, sends them to Claude (`claude-sonnet-4-6`) with a structured prompt, and writes `_analysis.md` files into the Obsidian vault at `~/playground/Obsidian/Transcripts/`. |

### Two different Whisper stacks

- `monitor.py` and `transcribe.py` use **WhisperX** (`whisperx`) — adds word-level alignment and speaker diarization via pyannote. Runs on NVIDIA CUDA via `ctranslate2`.
- `dictate.py` uses **openai-whisper** (`whisper`) — simpler, lower latency, single-speaker. Runs on AMD ROCm via PyTorch (`torch==2.11.0+rocm7.2`).

GPU detection differs accordingly: WhisperX checks `ctranslate2.get_cuda_device_count()`; dictate uses `torch.cuda.is_available()`.

### dictate.py internals

Recording, transcription, and text injection are split across threads:
1. `_record()` — runs in a daemon thread while `_recording` is True, appends chunks to `_audio_chunks`.
2. `_transcribe_and_type()` — runs in a daemon thread after recording stops; calls `_model.transcribe()` then `type_text()`.
3. Hotkey detection runs either via `pynput` keyboard listener (X11) or a `select`-based `evdev` loop (Wayland). Both call `_toggle_recording()`.

Text injection tries backends in order: `wtype` → `ydotool` → `xdotool`+clipboard → clipboard-only.

### monitor.py internals

`monitor_loop()` runs in a background thread; the `pystray` tray loop owns the main thread. When a meeting ends, transcription is launched in yet another daemon thread so the monitor loop can resume immediately.

## Dependencies and GPU notes

- ROCm 7.x is required for `dictate.py`; ROCm 6.x does not support Strix Halo (gfx1100).
- `whisperx` is **not** in `pyproject.toml` (it has complex dependencies); install it separately.
- The `[tool.uv.sources]` section points `torch`/`torchaudio` to the ROCm 7.2 PyTorch index.
