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

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel
from pynput import keyboard
import subprocess

SAMPLE_RATE = 16000
WHISPER_MODEL = "small"  # small balances speed and accuracy; use medium for more accuracy
WHISPER_COMPUTE = "int8"

HOTKEY = frozenset([keyboard.Key.ctrl, keyboard.Key.shift, keyboard.Key.space])


# ---------------------------------------------------------------------------
# Text output
# ---------------------------------------------------------------------------

_TERMINAL_CLASSES = {"gnome-terminal", "konsole", "xterm", "urxvt", "alacritty",
                     "kitty", "tilix", "terminator", "st", "rxvt", "xfce4-terminal"}


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


def type_text(text: str, window_id: str | None = None, is_terminal: bool = False):
    session = os.environ.get("XDG_SESSION_TYPE", "").lower()
    if "wayland" in session:
        try:
            subprocess.run(["wtype", text], check=True)
            return
        except FileNotFoundError:
            pass

    # Clipboard-paste approach: reliable for both terminals and browsers
    if _set_clipboard(text):
        paste_key = "ctrl+shift+v" if is_terminal else "ctrl+v"
        time.sleep(0.2)  # ensure hotkey modifiers are fully released before sending paste
        try:
            cmd = ["xdotool", "key", "--clearmodifiers"]
            if window_id:
                cmd += ["--window", window_id]
            cmd.append(paste_key)
            subprocess.run(cmd, check=True)
            return
        except FileNotFoundError:
            pass

    print(f"[dictate] No xdotool/xclip found — transcription: {text}")


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


def _record():
    chunks = []
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32") as stream:
        while _recording:
            data, _ = stream.read(1024)
            chunks.append(data.copy())
    with _lock:
        _audio_chunks.clear()
        _audio_chunks.extend(chunks)


def _transcribe_and_type():
    with _lock:
        chunks = list(_audio_chunks)
    if not chunks:
        return
    audio = np.concatenate(chunks).flatten()
    print("\r[dictate] Transcribing...              ", end="", flush=True)
    segments, _ = _model.transcribe(audio, vad_filter=True)
    text = " ".join(s.text.strip() for s in segments).strip()
    if text:
        print(f"\r[dictate] → {text}                     ", flush=True)
        type_text(text, _target_window, _target_is_terminal)
    else:
        print("\r[dictate] No speech detected.          ", flush=True)
    print("[dictate] Hold Ctrl+Shift+Space to dictate.", flush=True)


def _hotkey_active() -> bool:
    return HOTKEY.issubset(_current_keys)


def on_press(key):
    global _recording, _rec_thread, _target_window
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


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

_BANNER = """
┌─────────────────────────────────────────┐
│              dictate.py                 │
├─────────────────────────────────────────┤
│  Hold  Ctrl+Shift+Space →  record voice  │
│  Release               →  transcribe    │
│  Esc                   →  quit          │
│  Ctrl+C                →  quit          │
└─────────────────────────────────────────┘
"""

def main():
    print(_BANNER, flush=True)
    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        listener.join()


if __name__ == "__main__":
    main()
