#!/usr/bin/env -S uv run
"""
dictate.py — voice-to-text via keyboard hotkey

Hold Ctrl+Shift+Space to record; release to transcribe and type the result
into the currently focused window.

Requirements (beyond base transcriber deps):
  - pynput       (uv add pynput)
  - sounddevice  (uv add sounddevice)
  - xdotool (X11) or wtype (Wayland) for typing output
  - libportaudio2 (sudo apt install libportaudio2)

If sounddevice fails to load with a GLIBCXX version error (conda environment
shadowing the system libstdc++), run with:
  LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 python dictate.py
"""

import os
import time
import threading
import sys

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel
from pynput import keyboard
import subprocess

SAMPLE_RATE = 16000
WHISPER_MODEL = "small"  # small balances speed and accuracy; use medium for more accuracy
WHISPER_COMPUTE = "int8"

_CTRL_KEYS = {keyboard.Key.ctrl, keyboard.Key.ctrl_l, keyboard.Key.ctrl_r}
_SHIFT_KEYS = {keyboard.Key.shift, keyboard.Key.shift_l, keyboard.Key.shift_r}


# ---------------------------------------------------------------------------
# Text output
# ---------------------------------------------------------------------------

_TERMINAL_CLASSES = {"gnome-terminal", "konsole", "xterm", "urxvt", "alacritty",
                     "kitty", "tilix", "terminator", "st", "rxvt", "xfce4-terminal"}
_TYPE_BACKEND = os.environ.get("DICTATE_TYPE_BACKEND", "auto").strip().lower()
_WAYLAND_CONTROL_MODE = os.environ.get("DICTATE_WAYLAND_CONTROL", "terminal").strip().lower()


def _get_active_window() -> tuple[str | None, bool]:
    """Returns (window_id, is_terminal)."""
    try:
        win_id = subprocess.run(
            ["xdotool", "getactivewindow"], capture_output=True, text=True
        ).stdout.strip()
        if not win_id:
            return None, False
        cls = subprocess.run(
            ["xprop", "-id", win_id, "WM_CLASS"],
            capture_output=True, text=True,
        ).stdout.strip().lower()
        is_term = any(t in cls for t in _TERMINAL_CLASSES)
        return win_id, is_term
    except FileNotFoundError:
        return None, False


def _set_clipboard(text: str) -> bool:
    # Prefer wl-copy on Wayland, then fall back to X11 clipboard tools.
    try:
        subprocess.run(["wl-copy"], input=text.encode(), check=True)
        return True
    except FileNotFoundError:
        pass

    try:
        subprocess.run(["xclip", "-selection", "clipboard"],
                       input=text.encode(), check=True)
        return True
    except FileNotFoundError:
        pass
    try:
        subprocess.run(["xsel", "--clipboard", "--input"],
                       input=text.encode(), check=True)
        return True
    except FileNotFoundError:
        pass
    return False


def _try_wtype(text: str) -> bool:
    try:
        subprocess.run(["wtype", text], check=True)
        return True
    except FileNotFoundError:
        return False
    except subprocess.CalledProcessError as e:
        print(f"[dictate] wtype failed ({e.returncode}); trying next backend.", flush=True)
        return False


def _try_ydotool(text: str) -> bool:
    try:
        # Requires ydotoold and uinput permissions.
        subprocess.run(["ydotool", "type", "--key-delay", "4", text], check=True)
        return True
    except FileNotFoundError:
        return False
    except subprocess.CalledProcessError as e:
        print(f"[dictate] ydotool failed ({e.returncode}); trying next backend.", flush=True)
        return False


def _try_xdotool_paste(window_id: str | None, is_terminal: bool) -> bool:
    paste_key = "ctrl+shift+v" if is_terminal else "ctrl+v"
    time.sleep(0.2)  # ensure hotkey modifiers are fully released before sending paste
    try:
        cmd = ["xdotool", "key", "--clearmodifiers"]
        if window_id:
            cmd += ["--window", window_id]
        cmd.append(paste_key)
        subprocess.run(cmd, check=True)
        return True
    except FileNotFoundError:
        return False
    except subprocess.CalledProcessError as e:
        print(f"[dictate] xdotool paste failed ({e.returncode}); text remains in clipboard.", flush=True)
        return False


def type_text(text: str, window_id: str | None = None, is_terminal: bool = False):
    session = os.environ.get("XDG_SESSION_TYPE", "").lower()
    backend = _TYPE_BACKEND

    if backend in {"auto", "wtype"} and "wayland" in session and _try_wtype(text):
        return
    if backend in {"auto", "ydotool"} and _try_ydotool(text):
        return

    # Clipboard is always the safest non-privileged fallback.
    if _set_clipboard(text):
        if backend in {"auto", "xdotool"} and _try_xdotool_paste(window_id, is_terminal):
            return
        print("[dictate] Text copied to clipboard. Paste with Ctrl+V (or Ctrl+Shift+V in terminal).", flush=True)
        return

    print(f"[dictate] No usable typing backend found (backend={backend}) — transcription: {text}")


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def _get_device() -> str:
    try:
        import ctranslate2
        if ctranslate2.get_cuda_device_count() > 0:
            return "cuda"
    except Exception:
        pass
    return "cpu"


print(f"[dictate] Loading Whisper {WHISPER_MODEL} model...", flush=True)
_device = _get_device()
_model = WhisperModel(WHISPER_MODEL, device=_device, compute_type=WHISPER_COMPUTE)
print(f"[dictate] Ready on {_device}. Hold Ctrl+Shift+Space to dictate.", flush=True)


# ---------------------------------------------------------------------------
# Recording state
# ---------------------------------------------------------------------------

_recording = False
_audio_chunks: list[np.ndarray] = []
_current_keys: set = set()
_rec_thread: threading.Thread | None = None
_lock = threading.Lock()
_target_window: str | None = None
_target_is_terminal: bool = False
_record_error: str | None = None


def _first_input_device() -> int | None:
    """Returns a usable input device index, or None to use sounddevice default."""
    try:
        default_in = sd.default.device[0]
        if isinstance(default_in, int) and default_in >= 0:
            return default_in
    except Exception:
        pass

    try:
        for idx, dev in enumerate(sd.query_devices()):
            if int(dev.get("max_input_channels", 0)) > 0:
                return idx
    except Exception:
        pass
    return None


def _input_devices_debug() -> str:
    try:
        parts = []
        for idx, dev in enumerate(sd.query_devices()):
            channels = int(dev.get("max_input_channels", 0))
            if channels > 0:
                parts.append(f"{idx}:{dev.get('name', 'unknown')} ({channels}ch)")
        return ", ".join(parts) if parts else "none"
    except Exception:
        return "unavailable"


def _record():
    global _record_error
    chunks = []
    _record_error = None
    device = _first_input_device()
    stream_kwargs = {"samplerate": SAMPLE_RATE, "channels": 1, "dtype": "float32"}
    if device is not None:
        stream_kwargs["device"] = device

    try:
        with sd.InputStream(**stream_kwargs) as stream:
            while _recording:
                data, _ = stream.read(1024)
                chunks.append(data.copy())
    except Exception as e:
        _record_error = str(e)
        print(f"\n[dictate] Audio capture failed: {_record_error}", flush=True)
        print(f"[dictate] Input devices seen: {_input_devices_debug()}", flush=True)

    with _lock:
        _audio_chunks.clear()
        _audio_chunks.extend(chunks)


def _transcribe_and_type():
    with _lock:
        chunks = list(_audio_chunks)
    if _record_error:
        print(f"[dictate] Recording failed: {_record_error}", flush=True)
        print("[dictate] Check microphone permissions and default input device.", flush=True)
        return
    if not chunks:
        print("[dictate] No audio captured.", flush=True)
        return
    audio = np.concatenate(chunks).flatten()
    print("\r[dictate] Transcribing...              ", end="", flush=True)
    segments, _ = _model.transcribe(audio, vad_filter=True)
    text = " ".join(s.text.strip() for s in segments).strip()
    if text:
        print(f"\r[dictate] → {text}                     ", flush=True)
        try:
            type_text(text, _target_window, _target_is_terminal)
        except Exception as e:
            print(f"[dictate] Failed to deliver text to target window: {e}", flush=True)
            print(f"[dictate] Transcription: {text}", flush=True)
    else:
        print("\r[dictate] No speech detected.          ", flush=True)
    print("[dictate] Hold Ctrl+Shift+Space to dictate.", flush=True)


def _hotkey_active() -> bool:
    return (
        any(k in _current_keys for k in _CTRL_KEYS)
        and any(k in _current_keys for k in _SHIFT_KEYS)
        and keyboard.Key.space in _current_keys
    )


def on_press(key):
    global _recording, _rec_thread, _target_window, _target_is_terminal
    if key == keyboard.Key.esc:
        print("\n[dictate] Exiting.", flush=True)
        return False  # stops the listener
    _current_keys.add(key)
    if _hotkey_active() and not _recording:
        _target_window, _target_is_terminal = _get_active_window()
        _recording = True
        _rec_thread = threading.Thread(target=_record, daemon=True)
        _rec_thread.start()
        print("\r[dictate] Recording...                 ", end="", flush=True)


def on_release(key):
    global _recording
    _current_keys.discard(key)
    if _recording and not _hotkey_active():
        _recording = False
        if _rec_thread:
            _rec_thread.join(timeout=3)
        threading.Thread(target=_transcribe_and_type, daemon=True).start()


def _toggle_recording_terminal() -> None:
    """Wayland-safe fallback: toggle recording from the script terminal."""
    global _recording, _rec_thread, _target_window, _target_is_terminal
    if not _recording:
        _target_window, _target_is_terminal = _get_active_window()
        _recording = True
        _rec_thread = threading.Thread(target=_record, daemon=True)
        _rec_thread.start()
        print("\r[dictate] Recording...                 ", end="", flush=True)
        return

    _recording = False
    if _rec_thread:
        _rec_thread.join(timeout=3)
    threading.Thread(target=_transcribe_and_type, daemon=True).start()


def _run_terminal_fallback() -> None:
    print("[dictate] Wayland fallback mode: press Enter to start/stop recording.", flush=True)
    print("[dictate] Press q + Enter to quit.", flush=True)
    while True:
        try:
            line = sys.stdin.readline()
        except KeyboardInterrupt:
            print("\n[dictate] Exiting.", flush=True)
            break
        if line == "":
            # EOF on stdin (e.g., terminal closed)
            break

        command = line.strip().lower()
        if command in {"q", "quit", "exit"}:
            print("[dictate] Exiting.", flush=True)
            break

        _toggle_recording_terminal()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

_BANNER = """
┌─────────────────────────────────────────┐
│              dictate.py                 │
├─────────────────────────────────────────┤
│  Hold  Ctrl+Shift+Space →  record voice  │
│  Release               →  transcribe    │
│  Wayland fallback: Enter toggles rec    │
│  Esc                   →  quit          │
│  Ctrl+C                →  quit          │
└─────────────────────────────────────────┘
"""

def main():
    print(_BANNER, flush=True)
    session = os.environ.get("XDG_SESSION_TYPE", "").lower()
    if "wayland" in session and _WAYLAND_CONTROL_MODE == "terminal":
        print("[dictate] Running on Wayland: global hotkeys may be unavailable.", flush=True)
        print("[dictate] Using terminal control mode. Set DICTATE_WAYLAND_CONTROL=hotkey to try Ctrl+Shift+Space.", flush=True)
        _run_terminal_fallback()
        return
    if "wayland" in session and _WAYLAND_CONTROL_MODE == "hotkey":
        print("[dictate] Running on Wayland hotkey mode. Ctrl+Shift+Space may still be blocked by your compositor.", flush=True)

    # pynput allows returning False from callbacks to stop the listener.
    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:  # type: ignore[arg-type]
        listener.join()


if __name__ == "__main__":
    main()
