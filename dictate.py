#!/usr/bin/env -S uv run
"""
dictate.py — voice-to-text via keyboard hotkey

Hold Ctrl+Shift+Space to record; release to transcribe and type the result
into the currently focused window.

Requirements (beyond base transcriber deps):
  - pynput       (uv add pynput)
  - sounddevice  (uv add sounddevice)
  - xdotool (X11) or ydotool/wtype (Wayland) for typing output
  - libportaudio2 (sudo apt install libportaudio2)

Wayland setup:
  - sudo apt install ydotool wl-clipboard
  - sudo usermod -aG input $USER  (then log out/in)
  - Optional: systemctl --user enable --now ydotool

If sounddevice fails to load with a GLIBCXX version error (conda environment
shadowing the system libstdc++), run with:
  LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 python dictate.py
"""

import os
import time
import threading
import sys
import select
import termios
import tty

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel
from pynput import keyboard
import subprocess

SAMPLE_RATE = 16000
WHISPER_MODEL = "small"  # small balances speed and accuracy; use medium for more accuracy
WHISPER_COMPUTE = "int8"

HOTKEY = frozenset([keyboard.Key.ctrl, keyboard.Key.shift, keyboard.Key.space])
WAYLAND_CONTROL = os.environ.get("DICTATE_WAYLAND_CONTROL", "auto").lower()


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
    session = os.environ.get("XDG_SESSION_TYPE", "").lower()
    if "wayland" in session:
        try:
            subprocess.run(["wl-copy"], input=text.encode(), check=True)
            return True
        except (FileNotFoundError, subprocess.SubprocessError):
            pass
    try:
        subprocess.run(["xclip", "-selection", "clipboard"],
                       input=text.encode(), check=True)
        return True
    except (FileNotFoundError, subprocess.SubprocessError):
        pass
    try:
        subprocess.run(["xsel", "--clipboard", "--input"],
                       input=text.encode(), check=True)
        return True
    except (FileNotFoundError, subprocess.SubprocessError):
        pass
    return False


def _can_use_ydotool() -> bool:
    return os.path.exists("/dev/uinput") and os.access("/dev/uinput", os.W_OK)


def type_text(text: str, window_id: str | None = None, is_terminal: bool = False):
    session = os.environ.get("XDG_SESSION_TYPE", "").lower()
    copied = _set_clipboard(text)
    if "wayland" in session:
        # ydotool requires writable /dev/uinput.
        if _can_use_ydotool():
            try:
                subprocess.run(["ydotool", "type", "--", text], check=True)
                return
            except (FileNotFoundError, subprocess.SubprocessError, OSError) as exc:
                print(f"[dictate] ydotool type failed, falling back: {exc}", flush=True)
                pass
        else:
            print("[dictate] /dev/uinput not writable; skipping ydotool injection.", flush=True)

        # wtype as fallback (works on wlroots-based compositors)
        try:
            subprocess.run(["wtype", text], check=True)
            return
        except (FileNotFoundError, subprocess.SubprocessError, OSError) as exc:
            print(f"[dictate] wtype failed, falling back: {exc}", flush=True)
            pass

        # Clipboard-paste fallback for Wayland
        if copied and _can_use_ydotool():
            paste_key = "ctrl+shift+v" if is_terminal else "ctrl+v"
            time.sleep(0.2)
            try:
                subprocess.run(["ydotool", "key", paste_key], check=True)
                return
            except (FileNotFoundError, subprocess.SubprocessError, OSError) as exc:
                print(f"[dictate] ydotool key failed, text left in clipboard: {exc}", flush=True)
                pass

    # X11/XWayland: clipboard-paste approach.
    if copied:
        paste_key = "ctrl+shift+v" if is_terminal else "ctrl+v"
        time.sleep(0.2)  # ensure hotkey modifiers are fully released before sending paste
        try:
            cmd = ["xdotool", "key", "--clearmodifiers"]
            if window_id:
                cmd += ["--window", window_id]
            cmd.append(paste_key)
            subprocess.run(cmd, check=True)
            return
        except (FileNotFoundError, subprocess.SubprocessError, OSError) as exc:
            print(f"[dictate] xdotool paste failed, text left in clipboard: {exc}", flush=True)
            pass

    if copied:
        key_hint = "Ctrl+Shift+V" if is_terminal else "Ctrl+V"
        print(f"[dictate] Text copied to clipboard. Paste manually with {key_hint}.", flush=True)
    else:
        print(f"[dictate] No suitable typing backend and clipboard unavailable — transcription: {text}", flush=True)


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
_combo_latched = False


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
        try:
            type_text(text, _target_window, _target_is_terminal)
        except Exception as exc:
            print(f"[dictate] Typing backend error: {exc}", flush=True)
            if _set_clipboard(text):
                print("[dictate] Transcription copied to clipboard.", flush=True)
    else:
        print("\r[dictate] No speech detected.          ", flush=True)
    print("[dictate] Press Ctrl+Shift+Space to toggle dictation.", flush=True)


def _start_recording():
    global _recording, _rec_thread, _target_window, _target_is_terminal
    if _recording:
        return
    _target_window, _target_is_terminal = _get_active_window()
    _recording = True
    _rec_thread = threading.Thread(target=_record, daemon=True)
    _rec_thread.start()
    print("\r[dictate] Recording...                 ", end="", flush=True)


def _stop_recording(async_transcribe: bool = True):
    global _recording
    if not _recording:
        return
    _recording = False
    if _rec_thread:
        _rec_thread.join(timeout=3)
    if async_transcribe:
        threading.Thread(target=_transcribe_and_type, daemon=True).start()
    else:
        _transcribe_and_type()


def _toggle_recording(async_transcribe: bool = True):
    if _recording:
        _stop_recording(async_transcribe=async_transcribe)
    else:
        _start_recording()


def _hotkey_active() -> bool:
    return HOTKEY.issubset(_current_keys)


def _normalize_key(key):
    # Treat left/right modifier keys as the generic modifier so hotkey checks work.
    if key in (keyboard.Key.ctrl_l, keyboard.Key.ctrl_r):
        return keyboard.Key.ctrl
    if key in (keyboard.Key.shift_l, keyboard.Key.shift_r):
        return keyboard.Key.shift
    return key


def on_press(key):
    global _combo_latched
    if key == keyboard.Key.esc:
        print("\n[dictate] Exiting.", flush=True)
        return False  # stops the listener
    _current_keys.add(_normalize_key(key))
    if _hotkey_active() and not _combo_latched:
        _combo_latched = True
        _toggle_recording(async_transcribe=True)


def on_release(key):
    global _combo_latched
    _current_keys.discard(_normalize_key(key))
    if not _hotkey_active():
        _combo_latched = False


def _run_wayland_evdev_hotkey_listener():
    try:
        from evdev import InputDevice, ecodes, list_devices
    except Exception as exc:
        print(f"[dictate] evdev unavailable ({exc}).", flush=True)
        return False

    hotkey_tokens = {"ctrl", "shift", "space"}
    key_map = {
        ecodes.KEY_LEFTCTRL: "ctrl",
        ecodes.KEY_RIGHTCTRL: "ctrl",
        ecodes.KEY_LEFTSHIFT: "shift",
        ecodes.KEY_RIGHTSHIFT: "shift",
        ecodes.KEY_SPACE: "space",
        ecodes.KEY_ESC: "esc",
    }

    devices = []
    permission_denied = 0
    open_errors = 0
    for path in list_devices():
        try:
            dev = InputDevice(path)
            caps = dev.capabilities().get(ecodes.EV_KEY, [])
            key_codes = set()
            for item in caps:
                if isinstance(item, tuple):
                    key_codes.add(item[0])
                else:
                    key_codes.add(item)
            if any(code in key_codes for code in key_map):
                devices.append(dev)
        except PermissionError:
            permission_denied += 1
            continue
        except Exception:
            open_errors += 1
            continue

    if not devices:
        if permission_denied > 0:
            print("[dictate] evdev permission denied for /dev/input/event*.", flush=True)
            print("[dictate] Add user to input group and re-login: sudo usermod -aG input $USER", flush=True)
            print("[dictate] Temporary workaround for this shell: newgrp input", flush=True)
        elif open_errors > 0:
            print("[dictate] evdev could not open input devices.", flush=True)
        else:
            print("[dictate] No readable input devices found for evdev.", flush=True)
        return False

    print("[dictate] Wayland global hotkey mode (evdev): Ctrl+Shift+Space toggles, Esc quits.", flush=True)
    pressed: set[str] = set()
    combo_latched = False

    try:
        while True:
            ready, _, _ = select.select(devices, [], [], 0.2)
            if not ready:
                continue

            for dev in ready:
                for event in dev.read():
                    if event.type != ecodes.EV_KEY:
                        continue

                    token = key_map.get(event.code)
                    if not token:
                        continue

                    if token == "esc" and event.value == 1:
                        print("\n[dictate] Exiting.", flush=True)
                        if _recording:
                            _stop_recording(async_transcribe=False)
                        return True

                    if event.value in (1, 2):
                        pressed.add(token)
                    elif event.value == 0:
                        pressed.discard(token)

                    combo_active = hotkey_tokens.issubset(pressed)
                    if combo_active and not combo_latched:
                        combo_latched = True
                        _toggle_recording(async_transcribe=True)
                    elif not combo_active:
                        combo_latched = False
    except PermissionError:
        print("[dictate] No permission to read /dev/input/event*. Add user to input group and re-login.", flush=True)
        return False
    except Exception as exc:
        print(f"[dictate] evdev listener failed: {exc}", flush=True)
        return False
    finally:
        for dev in devices:
            try:
                dev.close()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

_BANNER = """
┌─────────────────────────────────────────┐
│              dictate.py                 │
├─────────────────────────────────────────┤
│  Press Ctrl+Shift+Space → toggle record │
│  Press Esc              → quit          │
│  Ctrl+C                →  quit          │
└─────────────────────────────────────────┘
"""

def main():
    print(_BANNER, flush=True)
    session = os.environ.get("XDG_SESSION_TYPE", "").lower()
    wayland = "wayland" in session

    if wayland and WAYLAND_CONTROL in ("auto", "evdev"):
        if _run_wayland_evdev_hotkey_listener():
            return
        if WAYLAND_CONTROL == "evdev":
            print("[dictate] Exiting because DICTATE_WAYLAND_CONTROL=evdev and evdev listener was unavailable.", flush=True)
            return
        print("[dictate] Falling back to terminal mode because Wayland global hotkey was unavailable.", flush=True)
        WAYLAND_FALLBACK = "terminal"
    else:
        WAYLAND_FALLBACK = WAYLAND_CONTROL

    use_terminal_control = wayland and WAYLAND_FALLBACK == "terminal"
    if use_terminal_control:
        print("[dictate] Wayland terminal mode: Enter toggles record, Esc/Q quits.", flush=True)
        if not sys.stdin.isatty():
            print("[dictate] stdin is not a TTY; set DICTATE_WAYLAND_CONTROL=pynput to try pynput.", flush=True)
            return

        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)
            while True:
                ready, _, _ = select.select([sys.stdin], [], [], 0.1)
                if not ready:
                    continue
                ch = sys.stdin.read(1)
                if ch in ("\x1b", "q", "Q"):
                    print("\n[dictate] Exiting.", flush=True)
                    if _recording:
                        _stop_recording(async_transcribe=False)
                    break
                if ch in ("\r", "\n"):
                    if _recording:
                        _stop_recording(async_transcribe=True)
                    else:
                        _start_recording()
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return

    if wayland and WAYLAND_FALLBACK == "pynput":
        print("[dictate] Wayland pynput mode: Ctrl+Shift+Space toggles (may not work on many compositors).", flush=True)
    else:
        print("[dictate] Hotkey mode: Ctrl+Shift+Space toggles dictation (Esc quits).", flush=True)
    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:  # type: ignore[arg-type]
        listener.join()


if __name__ == "__main__":
    main()
