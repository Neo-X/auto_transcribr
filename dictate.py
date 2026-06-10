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

AMD GPU (ROCm) notes:
  GPU acceleration uses openai-whisper + PyTorch ROCm 7.2 (torch==2.11.0+rocm7.2).
  Requires ROCm 7.x; ROCm 6.x does NOT support Strix Halo (Radeon 8060S / gfx1100 APU).
  If the script falls back to CPU unexpectedly, set ROCR_VISIBLE_DEVICES=0 to force
  device enumeration, e.g.:
    ROCR_VISIBLE_DEVICES=0 uv run dictate.py
"""

import os
import time
import threading
import sys
import select
import signal
import faulthandler
import termios
import traceback
import tty
import atexit

import numpy as np
import sounddevice as sd
import whisper
from pynput import keyboard
import subprocess

_LOG_PATH = os.path.expanduser("~/.cache/dictate.log")
_log_file = open(_LOG_PATH, "a", buffering=1)


def _log(msg: str):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    print(line, file=_log_file, flush=True)


# Dump segfault tracebacks to the log file.
faulthandler.enable(file=_log_file, all_threads=True)


def _signal_handler(signum, frame):
    _log(f"Received signal {signum} ({signal.Signals(signum).name}); exiting.")
    traceback.print_stack(frame, file=_log_file)
    _log_file.flush()
    sys.exit(0)


for _sig in (signal.SIGTERM, signal.SIGHUP):
    signal.signal(_sig, _signal_handler)
# SIGINT is left as default (raises KeyboardInterrupt) so genuine Ctrl+C still works.

atexit.register(lambda: _log("Process exiting via atexit."))


SAMPLE_RATE = 16000
WHISPER_MODEL = "turbo"  # large-v3-turbo: near large-v3 accuracy at ~8x speed; "medium"/"small" are lighter fallbacks

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
    paste_key = "ctrl+shift+v" if is_terminal else "ctrl+v"

    if "wayland" in session:
        # wtype injects text directly into the focused window in one shot (no per-char delay).
        try:
            subprocess.run(["wtype", "--", text], check=True)
            return
        except (FileNotFoundError, subprocess.SubprocessError, OSError) as exc:
            print(f"[dictate] wtype failed, falling back: {exc}", flush=True)

        # ydotool type with zero key-delay — fast and reliable if /dev/uinput is accessible.
        if _can_use_ydotool():
            try:
                subprocess.run(["ydotool", "type", "--key-delay", "0", "--", text], check=True)
                return
            except (FileNotFoundError, subprocess.SubprocessError, OSError) as exc:
                print(f"[dictate] ydotool type failed, falling back: {exc}", flush=True)

        # Clipboard-paste via ydotool key as last resort.
        if copied and _can_use_ydotool():
            time.sleep(0.2)
            try:
                subprocess.run(["ydotool", "key", paste_key], check=True)
                return
            except (FileNotFoundError, subprocess.SubprocessError, OSError) as exc:
                print(f"[dictate] ydotool key failed: {exc}", flush=True)
    else:
        # X11/XWayland: clipboard-paste targets the specific window — instant and reliable.
        if copied:
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

    if copied:
        key_hint = "Ctrl+Shift+V" if is_terminal else "Ctrl+V"
        print(f"[dictate] Text copied to clipboard. Paste manually with {key_hint}.", flush=True)
    else:
        print(f"[dictate] No suitable typing backend and clipboard unavailable — transcription: {text}", flush=True)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def _get_device() -> str:
    # On AMD APUs (e.g. Strix Halo / Radeon 8060S), ROCm requires ROCR_VISIBLE_DEVICES=0
    # to enumerate devices correctly; set it here if not already provided.
    if "ROCR_VISIBLE_DEVICES" not in os.environ:
        os.environ["ROCR_VISIBLE_DEVICES"] = "0"
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"  # works for both NVIDIA CUDA and AMD ROCm via HIP
    except Exception:
        pass
    return "cpu"


_log(f"[dictate] Loading Whisper {WHISPER_MODEL} model...")
_device = _get_device()
_model = whisper.load_model(WHISPER_MODEL, device=_device)
_log(f"[dictate] Ready on {_device}. Hold Ctrl+Shift+Space to dictate.")


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
    try:
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32") as stream:
            while _recording:
                data, _ = stream.read(1024)
                chunks.append(data.copy())
    except Exception:
        _log("[dictate] ERROR in recording thread:")
        traceback.print_exc(file=_log_file)
        traceback.print_exc()
    with _lock:
        _audio_chunks.clear()
        _audio_chunks.extend(chunks)


def _transcribe_and_type():
    try:
        with _lock:
            chunks = list(_audio_chunks)
        if not chunks:
            return
        audio = np.concatenate(chunks).flatten()
        print("\r[dictate] Transcribing...              ", end="", flush=True)
        result = _model.transcribe(audio, fp16=(_device == "cuda"))
        text = result["text"].strip()
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
    except Exception:
        print("\n[dictate] ERROR in transcription thread:", flush=True)
        traceback.print_exc()


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
    try:
        _current_keys.add(_normalize_key(key))
        if _hotkey_active() and not _combo_latched:
            _combo_latched = True
            _log("[dictate] Hotkey triggered → toggling recording.")
            _toggle_recording(async_transcribe=True)
    except Exception:
        _log("[dictate] ERROR in on_press callback:")
        traceback.print_exc(file=_log_file)
        traceback.print_exc()


def on_release(key):
    global _combo_latched
    try:
        _current_keys.discard(_normalize_key(key))
        if not _hotkey_active():
            _combo_latched = False
    except Exception:
        _log("[dictate] ERROR in on_release callback:")
        traceback.print_exc(file=_log_file)
        traceback.print_exc()


def _on_listener_error(exc):
    _log("[dictate] ERROR in keyboard listener thread:")
    traceback.print_exception(type(exc), exc, exc.__traceback__, file=_log_file)
    traceback.print_exception(type(exc), exc, exc.__traceback__)


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
            print("[dictate] evdev: permission denied. Run: sudo usermod -aG input $USER  (or: newgrp input)", flush=True)
        else:
            print("[dictate] evdev: no readable input devices found.", flush=True)
        return False

    print("[dictate] Wayland global hotkey mode (evdev): Ctrl+Shift+Space toggles, Ctrl+C quits.", flush=True)
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

                    if event.value in (1, 2):
                        pressed.add(token)
                    elif event.value == 0:
                        pressed.discard(token)

                    combo_active = hotkey_tokens.issubset(pressed)
                    if combo_active and not combo_latched:
                        combo_latched = True
                        _log("[dictate] Hotkey triggered via evdev → toggling recording.")
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
│  Ctrl+Shift+Space → toggle record       │
│  Ctrl+C           → quit               │
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
    try:
        with keyboard.Listener(  # type: ignore[arg-type]
            on_press=on_press,
            on_release=on_release,
            on_error=_on_listener_error,
        ) as listener:
            listener.join()
        _log("[dictate] Keyboard listener stopped (join returned).")
    except Exception:
        _log("[dictate] ERROR in keyboard listener:")
        traceback.print_exc(file=_log_file)
        traceback.print_exc()


if __name__ == "__main__":
    _log(f"[dictate] Started. PID={os.getpid()}")
    try:
        main()
        _log("[dictate] main() returned normally.")
    except KeyboardInterrupt:
        import traceback as _tb
        _log("[dictate] KeyboardInterrupt (SIGINT) received. Stack at interrupt:")
        _tb.print_stack(file=_log_file)
        _log_file.flush()
        print("\n[dictate] Interrupted (Ctrl+C).", flush=True)
    except Exception:
        _log("[dictate] FATAL ERROR:")
        traceback.print_exc(file=_log_file)
        traceback.print_exc()
        sys.exit(1)
