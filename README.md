# Transcriber

Automatically records audio when a Zoom or Google Meet meeting is detected, then transcribes it with speaker diarization using a local WhisperX model. Saves timestamped WAV and TXT files to `~/recordings/`.

## How It Works

`monitor.py` runs a polling loop every 3 seconds:

1. **Zoom detection** — checks the running process list for any `zoom` process using `psutil`.
2. **Google Meet detection** — checks all open window titles for `meet.google.com` using `wmctrl` or `xdotool`. Falls back to scanning browser process command-line arguments (e.g. when launched as a PWA).
3. **Recording** — when a meeting is detected, `ffmpeg` starts capturing audio:
   - Microphone via the PulseAudio/PipeWire default input source.
   - System audio via the default sink's monitor source (what plays through your speakers), if available.
   - Both streams are mixed into a stereo WAV file at 16 kHz / 16-bit. Stereo is preserved to give the diarization model better signal for separating speakers.
4. **Stop** — when the meeting ends (process gone / window closed), `ffmpeg` is stopped and the file is finalized.
5. **Transcription** — WhisperX automatically transcribes the recording in the background and saves a `.txt` file alongside the WAV. If `HF_TOKEN` is set, speaker diarization runs and labels each speaker in the output.

Recordings are saved to `~/recordings/meeting_YYYY-MM-DD_HH-MM-SS.wav` with a matching `meeting_YYYY-MM-DD_HH-MM-SS.txt` transcript.

## Transcript Format

Without diarization (`HF_TOKEN` not set), the transcript is plain text:

```
Let's get started with the agenda.
Sure, I wanted to follow up on last week's items.
```

With diarization, speakers are labeled and grouped:

```
[SPEAKER_00]
Let's get started with the agenda.

[SPEAKER_01]
Sure, I wanted to follow up on last week's items.

[SPEAKER_00]
Right, so the first item was...
```

## WhisperX Model

Transcription uses [WhisperX](https://github.com/m-bain/whisperX), which extends Whisper with:
- **Word-level timestamp alignment** for precise speaker attribution
- **Speaker diarization** via [pyannote.audio](https://github.com/pyannote/pyannote-audio)

The default model is **`medium` with INT8 quantization**, which is the recommended setting for a 4 GB GPU (e.g. RTX 4060 laptop):

| Model | Parameters | File size | VRAM (fp16) | VRAM (int8) |
|-------|-----------|-----------|-------------|-------------|
| tiny | 39M | 75 MB | ~1 GB | ~0.5 GB |
| base | 74M | 145 MB | ~1 GB | ~0.5 GB |
| small | 244M | 465 MB | ~2 GB | ~1 GB |
| medium | 769M | 1.5 GB | ~5 GB | ~2.5 GB |
| large-v3 | 1.55B | 3 GB | ~10 GB | ~5 GB |
| turbo | 809M | 1.6 GB | ~6 GB | ~3 GB |

Models are downloaded automatically from Hugging Face on first use and cached in `~/.cache/huggingface/`.

To change the model, edit `WHISPER_MODEL` and `WHISPER_COMPUTE` at the top of `monitor.py`.

## Speaker Diarization Setup

Diarization requires a free [Hugging Face](https://huggingface.co) account and accepting the terms for two models:

1. Visit [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1) and accept the terms.
2. Visit [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0) and accept the terms.
3. Generate a token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).
4. Export it before running:

```bash
export HF_TOKEN=your_token_here
uv run monitor.py
```

If `HF_TOKEN` is not set, transcription still works but speaker labels are omitted.

## Requirements

- Python 3.13+
- [uv](https://docs.astral.sh/uv/)
- `ffmpeg`
- PulseAudio or PipeWire (with PulseAudio compatibility layer)
- `wmctrl` or `xdotool` — for Google Meet window detection on X11
- NVIDIA GPU or AMD GPU with ROCm 7.x (CPU fallback supported, but slow)
  - AMD note: requires PyTorch ROCm 7.2+ (`torch==2.11.0+rocm7.2`). ROCm 6.x does not support Strix Halo APUs (Radeon 8060S / gfx1100). Uses `openai-whisper` + PyTorch instead of `faster-whisper` + ctranslate2.

```bash
sudo apt install ffmpeg wmctrl libportaudio2 xdotool
```

For Wayland, install `wtype` instead of (or in addition to) `xdotool`:
```bash
sudo apt install wtype
```

### dictate.py additional requirements

`dictate.py` requires `libportaudio2` for microphone input and `xdotool` (X11) or `wtype` (Wayland) to type transcribed text into the focused window. Also install `xclip` for clipboard support:

```bash
sudo apt install xclip
```

For better Wayland support, also install:

```bash
sudo apt install wl-clipboard ydotool
```

For global hotkeys on Wayland, install and allow `evdev` access:

```bash
uv add evdev
sudo usermod -aG input $USER
# log out and back in after adding the group
```

**Usage notes:**

- Run `dictate.py` in a **dedicated terminal**. Do not try to dictate into the terminal running the script — the Python process will receive the paste instead of the shell, producing `^[[200~` garbage.
- The transcribed text is automatically copied to the clipboard. If auto-paste does not work, paste manually with `Ctrl+Shift+V` (terminal) or `Ctrl+V` (browser).
- The active window is captured at the moment you press the hotkey, so focus does not need to remain on the target during transcription.
- If you see a `GLIBCXX` version error when running in a conda environment (conda's `libstdc++` is older than what `libjack` requires), preload the system library:

```bash
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 python dictate.py
```

### Dictation typing backends

`dictate.py` now supports multiple output backends and tries them in order:

1. `wtype` (Wayland virtual keyboard protocol)
2. `ydotool` (uinput-based keyboard injection)
3. clipboard + `xdotool` paste
4. clipboard only (manual paste)

Override backend selection with:

```bash
DICTATE_TYPE_BACKEND=auto    uv run dictate.py
DICTATE_TYPE_BACKEND=wtype   uv run dictate.py
DICTATE_TYPE_BACKEND=ydotool uv run dictate.py
DICTATE_TYPE_BACKEND=xdotool uv run dictate.py
```

If your compositor does not support the virtual keyboard protocol, `wtype` will fail with an error like "Compositor does not support the virtual keyboard protocol". In that case, use `ydotool` or fall back to clipboard paste.

### Wayland recording controls

`dictate.py` uses **toggle recording**: press `Ctrl+Shift+Space` once to start, press again to stop and transcribe.

On Wayland, `dictate.py` defaults to `auto` mode:
- Try global hotkeys with `evdev` first (works across applications if input permissions are available)
- Fall back to terminal mode when unavailable

Force specific mode with `DICTATE_WAYLAND_CONTROL`:

```bash
DICTATE_WAYLAND_CONTROL=auto    uv run dictate.py
DICTATE_WAYLAND_CONTROL=evdev   uv run dictate.py
DICTATE_WAYLAND_CONTROL=pynput  uv run dictate.py
DICTATE_WAYLAND_CONTROL=terminal uv run dictate.py
```

Mode notes:
- `evdev`: best option for global hotkeys on Wayland; requires access to `/dev/input/event*`
- `pynput`: can work on X11; often blocked on Wayland compositors
- `terminal`: Enter toggles recording in the terminal window

If you added yourself to the `input` group recently, start a new login session (or run `newgrp input` in the current shell) before starting `dictate.py`.

For auto-typing with `ydotool`, `/dev/uinput` must also be writable by your user (or by a group your user is in). If `dictate.py` logs `/dev/uinput not writable`, typing injection is blocked by system permissions; dictation will still copy text to clipboard for manual paste.

## Installation & Usage

**Run directly (no install needed):**
```bash
export HF_TOKEN=your_token_here
uv run monitor.py
```

**Or make it executable and run as a script:**
```bash
chmod +x monitor.py
./monitor.py
```

**Batch-transcribe existing recordings:**
```bash
export HF_TOKEN=your_token_here
uv run transcribe.py [recordings_dir]
```

If `recordings_dir` is omitted, defaults to `~/recordings/`. Only WAV files without a matching `.txt` are processed.

## Notes

- **Wayland**: `wmctrl` requires X11. On a pure Wayland session without XWayland, Google Meet detection falls back to checking browser process arguments (works if Meet is launched as a PWA, less reliable otherwise). Install `wmctrl` and ensure XWayland is active for best results.
- **System audio**: If no PulseAudio monitor source is found (e.g. on a headless machine), only the microphone is recorded.
- **Zoom**: Recording starts when the Zoom application launches, not when a call begins.
- **First run**: The Whisper model (~1.5 GB for medium) and pyannote diarization models are downloaded on first use and cached in `~/.cache/huggingface/`.
- **CPU fallback**: If no CUDA GPU is available, transcription falls back to CPU automatically. Expect ~10–20x slower than GPU.

## Stopping

Press `Ctrl-C` to stop the monitor. Any active recording will be finalized and queued for transcription before exit.
