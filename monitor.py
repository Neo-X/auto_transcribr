#!/usr/bin/env -S uv run
"""
Meeting recorder: auto-records audio when Zoom or Google Meet is detected.
Saves recordings to ~/recordings/ as timestamped WAV files.

Requirements:
  - ffmpeg
  - PulseAudio or PipeWire (with PulseAudio compatibility)
  - wmctrl or xdotool  (for Google Meet detection on X11)
  - psutil, pystray, pillow  (uv add psutil pystray pillow)
"""

import re
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import psutil
import pystray
from faster_whisper import WhisperModel
from PIL import Image, ImageDraw

RECORDINGS_DIR = Path.home() / "recordings"
POLL_INTERVAL = 3  # seconds between checks

WHISPER_MODEL   = "medium"   # tiny / base / small / medium / large-v3 / turbo
WHISPER_COMPUTE = "int8"     # int8 (recommended) or float16


# ---------------------------------------------------------------------------
# Tray icon
# ---------------------------------------------------------------------------

def _make_icon(color: str) -> Image.Image:
    """Draw a solid circle of the given color on a transparent background."""
    size = 64
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    margin = 4
    draw.ellipse([margin, margin, size - margin, size - margin], fill=color)
    return img


ICON_IDLE        = _make_icon("#888888")   # grey   — monitoring, not recording
ICON_REC         = _make_icon("#e03030")   # red    — actively recording
ICON_TRANSCRIBING = _make_icon("#e08800")  # orange — transcribing


class TrayIcon:
    def __init__(self, on_quit):
        self._icon = pystray.Icon(
            "transcriber",
            icon=ICON_IDLE,
            title="Transcriber - idle",
            menu=pystray.Menu(
                pystray.MenuItem("Open recordings folder", self._open_recordings),
                pystray.MenuItem("Quit", lambda: on_quit()),
            ),
        )

    def set_recording(self, recording: bool):
        if recording:
            self._icon.icon = ICON_REC
            self._icon.title = "Transcriber - recording"
        else:
            self._icon.icon = ICON_IDLE
            self._icon.title = "Transcriber - idle"

    def set_transcribing(self):
        self._icon.icon = ICON_TRANSCRIBING
        self._icon.title = "Transcriber - transcribing"

    def _open_recordings(self):
        RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
        subprocess.Popen(["xdg-open", str(RECORDINGS_DIR)])

    def run(self):
        """Block and run the tray event loop (call from main thread)."""
        self._icon.run()

    def stop(self):
        self._icon.stop()


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------

def find_monitor_source() -> str | None:
    """Return the name of the default sink's monitor source, or None."""
    try:
        out = subprocess.check_output(
            ["pactl", "list", "sources", "short"], text=True, timeout=5
        )
        for line in out.splitlines():
            if ".monitor" in line:
                return line.split()[1]
    except Exception:
        pass
    return None


def build_ffmpeg_cmd(output_path: Path) -> list[str]:
    """
    Build an ffmpeg command that captures:
      - default microphone (pulse default source)
      - system audio monitor (if available)
    and mixes them into a single mono WAV file.
    """
    monitor = find_monitor_source()

    # Whisper expects 16 kHz, 16-bit, mono PCM
    whisper_args = ["-ar", "16000", "-ac", "1", "-sample_fmt", "s16", "-vn"]

    if monitor:
        cmd = [
            "ffmpeg", "-y",
            "-f", "pulse", "-i", "default",          # microphone
            "-f", "pulse", "-i", monitor,             # system audio
            "-filter_complex", "amix=inputs=2:duration=longest",
            *whisper_args,
            str(output_path),
        ]
    else:
        print("[recorder] Warning: no monitor source found — recording mic only.")
        cmd = [
            "ffmpeg", "-y",
            "-f", "pulse", "-i", "default",
            *whisper_args,
            str(output_path),
        ]

    return cmd


# ---------------------------------------------------------------------------
# Meeting detection
# ---------------------------------------------------------------------------

def is_zoom_running() -> bool:
    # Zoom keeps its process alive after a meeting ends, so we must check window
    # titles to determine whether an active meeting is in progress.
    titles = get_window_titles()
    return "zoom meeting" in titles


def get_window_titles() -> str:
    # wmctrl (X11)
    try:
        out = subprocess.check_output(
            ["wmctrl", "-l"], text=True, timeout=3, stderr=subprocess.DEVNULL
        )
        return out.lower()
    except FileNotFoundError:
        pass
    except subprocess.TimeoutExpired:
        return ""

    # xdotool (X11 / XWayland)
    try:
        ids = subprocess.check_output(
            ["xdotool", "search", "--name", ""],
            text=True, timeout=3, stderr=subprocess.DEVNULL
        ).split()
        titles = []
        for wid in ids[:50]:
            try:
                t = subprocess.check_output(
                    ["xdotool", "getwindowname", wid],
                    text=True, timeout=1, stderr=subprocess.DEVNULL
                ).strip()
                titles.append(t)
            except Exception:
                pass
        return "\n".join(titles).lower()
    except FileNotFoundError:
        pass
    except subprocess.TimeoutExpired:
        pass

    return ""


_warned_no_window_tool = False


def is_google_meet_running() -> bool:
    global _warned_no_window_tool

    titles = get_window_titles()
    if titles:
        return "meet.google.com" in titles or "google meet" in titles

    for proc in psutil.process_iter(["cmdline"]):
        try:
            cmdline = " ".join(proc.info["cmdline"] or [])
            if re.search(r"meet\.google\.com", cmdline, re.IGNORECASE):
                return True
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

    if not _warned_no_window_tool:
        print(
            "[monitor] Warning: neither wmctrl nor xdotool found.\n"
            "          Google Meet detection may be unreliable.\n"
            "          Install one of them:  sudo apt install wmctrl"
        )
        _warned_no_window_tool = True

    return False


def is_meeting_active() -> bool:
    return is_zoom_running() or is_google_meet_running()


# ---------------------------------------------------------------------------
# Recorder
# ---------------------------------------------------------------------------

class MeetingRecorder:
    def __init__(self):
        self._proc: subprocess.Popen | None = None
        self.output_file: Path | None = None

    @property
    def is_recording(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def start(self):
        RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.output_file = RECORDINGS_DIR / f"meeting_{ts}.wav"
        self._proc = subprocess.Popen(
            build_ffmpeg_cmd(self.output_file),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        print(f"[recorder] Started  → {self.output_file}")

    def stop(self):
        if self._proc and self._proc.poll() is None:
            self._proc.send_signal(signal.SIGINT)
            try:
                self._proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        print(f"[recorder] Stopped  → {self.output_file}")
        self._proc = None


# ---------------------------------------------------------------------------
# Transcription
# ---------------------------------------------------------------------------

_whisper_model: WhisperModel | None = None
_whisper_lock = threading.Lock()


def get_whisper_model() -> WhisperModel:
    global _whisper_model
    with _whisper_lock:
        if _whisper_model is None:
            print(f"[transcriber] Loading Whisper model '{WHISPER_MODEL}' ({WHISPER_COMPUTE})...")
            device = "cuda" if _cuda_available() else "cpu"
            _whisper_model = WhisperModel(WHISPER_MODEL, device=device, compute_type=WHISPER_COMPUTE)
            print(f"[transcriber] Model loaded on {device}.")
    return _whisper_model


def _cuda_available() -> bool:
    try:
        import ctranslate2
        return ctranslate2.get_cuda_device_count() > 0
    except Exception:
        return False


def transcribe(audio_path: Path, tray: TrayIcon):
    """Transcribe audio_path and save a .txt file alongside it."""
    txt_path = audio_path.with_suffix(".txt")
    print(f"[transcriber] Transcribing {audio_path.name}...")
    tray.set_transcribing()

    try:
        model = get_whisper_model()
        segments, info = model.transcribe(str(audio_path), beam_size=5)
        with txt_path.open("w") as f:
            for segment in segments:
                f.write(segment.text.strip() + "\n")
        print(f"[transcriber] Transcript saved → {txt_path}")
    except Exception as e:
        print(f"[transcriber] Error: {e}")
    finally:
        tray.set_recording(False)


# ---------------------------------------------------------------------------
# Monitor loop (runs in background thread)
# ---------------------------------------------------------------------------

def monitor_loop(recorder: MeetingRecorder, tray: TrayIcon, stop_event: threading.Event):
    print("[monitor] Watching for Zoom and Google Meet...")
    print(f"[monitor] Recordings will be saved to: {RECORDINGS_DIR}")

    while not stop_event.is_set():
        meeting_on = is_meeting_active()

        if meeting_on and not recorder.is_recording:
            print("[monitor] Meeting detected!")
            recorder.start()
            tray.set_recording(True)
        elif not meeting_on and recorder.is_recording:
            print("[monitor] Meeting ended.")
            recorder.stop()
            audio_file = recorder.output_file
            threading.Thread(
                target=transcribe, args=(audio_file, tray), daemon=True
            ).start()

        stop_event.wait(POLL_INTERVAL)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    recorder = MeetingRecorder()
    stop_event = threading.Event()

    def quit_app():
        print("\n[monitor] Shutting down...")
        stop_event.set()
        if recorder.is_recording:
            recorder.stop()
        tray.stop()

    tray = TrayIcon(on_quit=quit_app)

    def handle_signal(sig, frame):
        quit_app()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    # Run monitor loop in background thread; tray runs on main thread
    thread = threading.Thread(target=monitor_loop, args=(recorder, tray, stop_event), daemon=True)
    thread.start()

    tray.run()  # blocks until tray.stop() is called


if __name__ == "__main__":
    main()
