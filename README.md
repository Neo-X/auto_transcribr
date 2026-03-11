# Transcriber

Automatically records audio when a Zoom or Google Meet meeting is detected, then transcribes it using a local Whisper model. Saves timestamped WAV and TXT files to `~/recordings/`.

## How It Works

`monitor.py` runs a polling loop every 3 seconds:

1. **Zoom detection** — checks the running process list for any `zoom` process using `psutil`.
2. **Google Meet detection** — checks all open window titles for `meet.google.com` using `wmctrl` or `xdotool`. Falls back to scanning browser process command-line arguments (e.g. when launched as a PWA).
3. **Recording** — when a meeting is detected, `ffmpeg` starts capturing audio:
   - Microphone via the PulseAudio/PipeWire default input source.
   - System audio via the default sink's monitor source (what plays through your speakers), if available.
   - Both streams are mixed into a single mono WAV file at 16 kHz / 16-bit (Whisper's native format).
4. **Stop** — when the meeting ends (process gone / window closed), `ffmpeg` is stopped and the file is finalized.
5. **Transcription** — `faster-whisper` automatically transcribes the recording in the background and saves a `.txt` file alongside the WAV.

Recordings are saved to `~/recordings/meeting_YYYY-MM-DD_HH-MM-SS.wav` with a matching `meeting_YYYY-MM-DD_HH-MM-SS.txt` transcript.

## Whisper Model

Transcription uses [faster-whisper](https://github.com/SYSTRAN/faster-whisper), a CTranslate2-based reimplementation of OpenAI Whisper that runs at ~4x the speed with half the VRAM via INT8 quantization.

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

## Requirements

- Python 3.13+
- [uv](https://docs.astral.sh/uv/)
- `ffmpeg`
- PulseAudio or PipeWire (with PulseAudio compatibility layer)
- `wmctrl` or `xdotool` — for Google Meet window detection on X11
- NVIDIA GPU with 4 GB+ VRAM recommended (CPU fallback supported, but slow)

```bash
sudo apt install ffmpeg wmctrl
```

## Installation & Usage

**Run directly (no install needed):**
```bash
uv run monitor.py
```

**Or make it executable and run as a script:**
```bash
chmod +x monitor.py
./monitor.py
```

**Or install as a system command:**
```bash
uv tool install .
transcriber
```

## Notes

- **Wayland**: `wmctrl` requires X11. On a pure Wayland session without XWayland, Google Meet detection falls back to checking browser process arguments (works if Meet is launched as a PWA, less reliable otherwise). Install `wmctrl` and ensure XWayland is active for best results.
- **System audio**: If no PulseAudio monitor source is found (e.g. on a headless machine), only the microphone is recorded.
- **Zoom**: Recording starts when the Zoom application launches, not when a call begins.
- **First run**: The Whisper model (~1.5 GB for medium) is downloaded on first transcription. Subsequent runs use the local cache.
- **CPU fallback**: If no CUDA GPU is available, transcription falls back to CPU automatically. Expect ~10–20x slower than GPU.

## Stopping

Press `Ctrl-C` to stop the monitor. Any active recording will be finalized and queued for transcription before exit.
