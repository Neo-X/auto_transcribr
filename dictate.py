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
import json
import time
import threading
import wave
import sys
import select
import signal
import faulthandler
import termios
import traceback
import tty
import atexit

import argparse

import numpy as np
import sounddevice as sd
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


def _install_handlers():
    for _sig in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(_sig, _signal_handler)
    # SIGINT is left as default (raises KeyboardInterrupt) so genuine Ctrl+C still works.
    atexit.register(lambda: _log("Process exiting via atexit."))


SAMPLE_RATE = 16000
WHISPER_MODEL = "turbo"  # large-v3-turbo: near large-v3 accuracy at ~8x speed; "medium"/"small" are lighter fallbacks

# Selectable speech-to-text models (--model / DICTATE_MODEL). All are open weights
# and run locally; the HF ones are downloaded on first use.
#   qwen3    Qwen3-ASR 1.7B — the default; biased towards vocabulary.txt (names, jargon).
#            Best at the names/jargon in the dictation failure set (see
#            asr-whisper-finetuning/data/dictation-results/comparison_summary.csv)
#   whisper  openai-whisper large-v3-turbo (809M) — the original default
#   granite  IBM Granite Speech 4.1 2B — supports keyword biasing (--keywords)
#   cohere   Cohere Transcribe 03-2026 (2B) — gated on Hugging Face: accept the
#            terms on its model page and run `hf auth login` once before using it
MODELS = {
    "qwen3": "Qwen/Qwen3-ASR-1.7B-hf",
    "whisper": f"openai-whisper/{WHISPER_MODEL}",
    "granite": "ibm-granite/granite-speech-4.1-2b",
    "cohere": "CohereLabs/cohere-transcribe-03-2026",
}

_parser = argparse.ArgumentParser(description="Hold-to-dictate voice typing")
_parser.add_argument("--model", choices=sorted(MODELS), default=os.environ.get("DICTATE_MODEL", "qwen3"),
                     help="Speech-to-text model (default: $DICTATE_MODEL or qwen3)")
_parser.add_argument("--keywords", default=os.environ.get("DICTATE_KEYWORDS", ""),
                     help="qwen3/granite: comma-separated words to bias towards, e.g. 'Asana, RLC, LoRA' "
                          "(default: the contents of vocabulary.txt next to this script)")
_parser.add_argument("--no-stream", action="store_true", default=os.environ.get("DICTATE_STREAM", "1") == "0",
                     help="qwen3: transcribe only after the recording stops, instead of also while you speak "
                          "(also: DICTATE_STREAM=0)")
_args, _ = _parser.parse_known_args()
MODEL = _args.model
STREAM = not _args.no_stream
VOCAB_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vocabulary.txt")


def _load_keywords() -> str:
    """--keywords / $DICTATE_KEYWORDS if given, else vocabulary.txt (re-read on every call,
    so terms added while dictate.py is running apply to the next recording)."""
    if _args.keywords.strip():
        return _args.keywords.strip()
    try:
        with open(VOCAB_FILE) as f:
            return " ".join(f.read().split())
    except OSError:
        return ""


KEYWORDS = _load_keywords()

# Whisper defaults to American spelling. Seeding the decoder with an initial_prompt
# written in British English biases it to continue in the same style/spelling
# (there's no dedicated "dialect" flag — this prompt-conditioning trick is the
# standard workaround).
WHISPER_INITIAL_PROMPT = (
    "Transcribed in British/Canadian English: colour, favourite, organise, realise, "
    "centre, behaviour, analyse, travelling, licence."
)

HOTKEY = frozenset([keyboard.Key.ctrl, keyboard.Key.shift, keyboard.Key.space])

# Every dictation is saved (WAV + a line in metadata.jsonl) so clips can later be
# reviewed and the interesting ones — especially failures — promoted into a
# validation/training set. The directory is in /tmp on purpose: it is a staging
# area that gets wiped on reboot, not the permanent dataset. Layout follows the
# Hugging Face "audiofolder" format: load_dataset("audiofolder", data_dir=...).
# Set DICTATE_SAVE_AUDIO=0 to disable.
SAVE_AUDIO = os.environ.get("DICTATE_SAVE_AUDIO", "1") != "0"
SAVE_DIR = os.path.expanduser(os.environ.get("DICTATE_SAVE_DIR", "/tmp/dictate-clips"))
WAYLAND_CONTROL = os.environ.get("DICTATE_WAYLAND_CONTROL", "auto").lower()


# ---------------------------------------------------------------------------
# Text output
# ---------------------------------------------------------------------------

_TERMINAL_CLASSES = {"gnome-terminal", "konsole", "xterm", "urxvt", "alacritty",
                     "kitty", "tilix", "terminator", "st", "rxvt", "xfce4-terminal",
                     "foot", "wezterm", "ghostty", "terminology", "deepin-terminal",
                     "lxterminal", "mate-terminal", "terminal", "guake", "tilda"}


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

        # Clipboard-paste via ydotool key: one synthetic paste event for the whole
        # text, so it can't drop characters the way per-key injection can. Preferred
        # over `ydotool type` below, which sends one synthetic keystroke per character
        # and has been observed to silently drop characters (partial paste) when kernel/
        # uinput event timing changes — e.g. after a kernel update — outpaces what the
        # focused app's input handling can keep up with.
        if copied and _can_use_ydotool():
            time.sleep(0.2)
            try:
                subprocess.run(["ydotool", "key", paste_key], check=True)
                return
            except (FileNotFoundError, subprocess.SubprocessError, OSError) as exc:
                print(f"[dictate] ydotool key failed, falling back: {exc}", flush=True)

        # ydotool type as last resort (per-character synthetic keystrokes — see note above).
        if _can_use_ydotool():
            try:
                subprocess.run(["ydotool", "type", "--key-delay", "8", "--", text], check=True)
                return
            except (FileNotFoundError, subprocess.SubprocessError, OSError) as exc:
                print(f"[dictate] ydotool type failed: {exc}", flush=True)
    else:
        # X11/XWayland: clipboard-paste into the currently focused window.
        #
        # Deliberately NOT using `xdotool key --window <id>`: that sends a synthetic
        # XSendEvent targeted at a specific window, which GTK/GNOME apps (gnome-terminal,
        # xfce4-terminal, nautilus, etc.) silently ignore for security reasons — the paste
        # would appear to succeed but nothing would happen. Plain `xdotool key` uses XTEST
        # (a fake hardware event delivered to whatever currently has focus), which every
        # app accepts. This assumes focus hasn't moved away from the target window since
        # recording started, which holds for the normal hold-hotkey-and-speak workflow.
        if window_id:
            try:
                subprocess.run(["xdotool", "windowactivate", "--sync", window_id],
                               capture_output=True, check=False)
            except (FileNotFoundError, subprocess.SubprocessError, OSError):
                pass
        if copied:
            time.sleep(0.2)  # ensure hotkey modifiers are fully released before sending paste
            try:
                subprocess.run(["xdotool", "key", "--clearmodifiers", paste_key], check=True)
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
    # MIOpen (ROCm) otherwise runs a ~25 s convolution-algorithm search the first time it
    # sees each new audio length, which makes long (rarely seen) clip lengths very slow.
    os.environ.setdefault("MIOPEN_FIND_MODE", "FAST")
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"  # works for both NVIDIA CUDA and AMD ROCm via HIP
    except Exception:
        pass
    return "cpu"


def _load_whisper(device: str):
    import whisper

    model = whisper.load_model(WHISPER_MODEL, device=device)

    def transcribe(audio: np.ndarray) -> str:
        result = model.transcribe(audio, fp16=(device == "cuda"), initial_prompt=WHISPER_INITIAL_PROMPT)
        return result["text"].strip()

    return transcribe


def _load_granite(device: str):
    import torch
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

    processor = AutoProcessor.from_pretrained(MODELS["granite"])
    tokenizer = processor.tokenizer
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        MODELS["granite"], device_map=device, dtype=torch.bfloat16
    ).eval()

    def transcribe(audio: np.ndarray) -> str:
        instruction = "<|audio|>transcribe the speech with proper punctuation and capitalization."
        keywords = _load_keywords()
        if keywords:
            instruction += f" Keywords: {keywords}"
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": instruction}], tokenize=False, add_generation_prompt=True
        )
        wav = torch.from_numpy(audio).float().unsqueeze(0)
        inputs = processor(prompt, wav, device=device, return_tensors="pt").to(device)
        # Leave room for fast speech: ~4 tokens per second of audio, at least 200.
        max_new_tokens = max(200, int(4 * len(audio) / SAMPLE_RATE))
        with torch.no_grad():
            ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, num_beams=1)
        new_tokens = ids[:, inputs["input_ids"].shape[-1]:]
        return tokenizer.batch_decode(new_tokens, add_special_tokens=False, skip_special_tokens=True)[0].strip()

    return transcribe


def _load_cohere(device: str):
    import torch
    from transformers import AutoProcessor, CohereAsrForConditionalGeneration

    processor = AutoProcessor.from_pretrained(MODELS["cohere"])
    model = CohereAsrForConditionalGeneration.from_pretrained(MODELS["cohere"], device_map=device).eval()

    def transcribe(audio: np.ndarray) -> str:
        # The processor splits long audio into chunks; audio_chunk_index stitches them back.
        inputs = processor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt", language="en")
        audio_chunk_index = inputs.get("audio_chunk_index")
        inputs.to(model.device, dtype=model.dtype)
        max_new_tokens = max(256, int(4 * len(audio) / SAMPLE_RATE))
        with torch.no_grad():
            ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
        text = processor.decode(ids, skip_special_tokens=True, audio_chunk_index=audio_chunk_index, language="en")
        if isinstance(text, list):
            text = text[0]
        return text.strip()

    return transcribe


def _load_qwen3(device: str):
    import torch
    from transformers import AutoProcessor, Qwen3ASRForConditionalGeneration

    processor = AutoProcessor.from_pretrained(MODELS["qwen3"])
    model = Qwen3ASRForConditionalGeneration.from_pretrained(
        MODELS["qwen3"], device_map=device, dtype=torch.bfloat16
    ).eval()

    eos = model.generation_config.eos_token_id
    eos = set(eos if isinstance(eos, (list, tuple)) else [eos])

    def transcribe(audio: np.ndarray, prefix: list[int] | None = None, return_ids: bool = False):
        """Transcribe `audio`. `prefix` is the start of the answer as token ids (from an earlier
        call on the first part of the same audio): the model continues from it instead of
        writing it again. With return_ids, returns (text, answer token ids without the end token)."""
        # vocabulary.txt goes in the system prompt as context/hotwords; forcing English skips language ID.
        inputs = processor.apply_transcription_request(audio, language="English", prompt=_load_keywords() or None)
        inputs = inputs.to(model.device, dtype=model.dtype)
        n_prompt = inputs["input_ids"].shape[-1]
        if prefix:
            extra = torch.tensor([prefix], device=model.device, dtype=inputs["input_ids"].dtype)
            inputs["input_ids"] = torch.cat([inputs["input_ids"], extra], dim=-1)
            inputs["attention_mask"] = torch.cat([inputs["attention_mask"], torch.ones_like(extra)], dim=-1)
        max_new_tokens = max(256, int(5 * len(audio) / SAMPLE_RATE))
        with torch.no_grad():
            ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, num_beams=1)
        answer = ids[0, n_prompt:].tolist()
        while answer and answer[-1] in eos:
            answer.pop()
        text = processor.decode(answer, return_format="transcription_only").strip()
        return (text, answer) if return_ids else text

    transcribe.supports_prefix = True
    return transcribe


_LOADERS = {"qwen3": _load_qwen3, "whisper": _load_whisper, "granite": _load_granite, "cohere": _load_cohere}

_transcribe = None  # set by _load_model()


def _load_model():
    """Load the speech model. Called once at start-up (not at import, so tests can import this file)."""
    global _transcribe
    _log(f"[dictate] Loading {MODEL} ({MODELS[MODEL]})" + (f", {len(KEYWORDS.split(','))} keywords" if MODEL in ("qwen3", "granite") and KEYWORDS else "") + "...")
    device = _get_device()
    try:
        _transcribe = _LOADERS[MODEL](device)
    except OSError as exc:
        if "gated" in str(exc).lower():
            _log(f"[dictate] {MODELS[MODEL]} is gated: accept its terms at https://huggingface.co/{MODELS[MODEL]} "
                 "and run `hf auth login`, then restart.")
        raise
    _log(f"[dictate] Ready on {device}. Hold Ctrl+Shift+Space to dictate.")
    return _transcribe


# ---------------------------------------------------------------------------
# Recording state
# ---------------------------------------------------------------------------

_recording = False
_audio_chunks: list[np.ndarray] = []
_audio_stream = None  # the finished recording's _Stream (what was transcribed while it was recording)
_current_keys: set = set()
_rec_thread: threading.Thread | None = None
_lock = threading.Lock()
_target_window: str | None = None
_target_is_terminal: bool = False
_combo_latched = False


# Transcribing while speaking (qwen3 only). Writing the text costs about 0.2 s per second of
# audio, so a long dictation transcribed only after the key is released means a long wait
# (20 s for 100 s of speech). Instead, while the recording is still going, the audio so far is
# transcribed every STREAM_STEP_S seconds, each pass continuing from the text of the pass
# before it rather than writing it again. The last STREAM_HOLD tokens of a pass are not kept:
# they were written without knowing what comes next, so the next pass writes them afresh.
# On release, one last pass over the whole recording only has to write the final stretch.
# Recordings shorter than STREAM_STEP_S are transcribed in one go, as before.
# The step is a trade: text settled early is written with less of the recording to go on and
# is slightly less accurate (numbers spelled out, an odd changed word). On tests/test_streaming.py
# --quick, a 10 s step raised the word error rate from 5.9% to 7.5%; 20 s stayed within 0.5 points.
# Under 1 point is the accepted price for the shorter wait.
STREAM_STEP_S = float(os.environ.get("DICTATE_STREAM_STEP", "20"))
STREAM_HOLD = int(os.environ.get("DICTATE_STREAM_HOLD", "10"))


class _Stream:
    """One recording's transcription while it is still being recorded."""

    def __init__(self, transcribe):
        self._transcribe = transcribe
        self.kept: list[int] = []   # tokens of the answer settled so far
        self.covered = 0            # samples of audio seen by the last pass
        self.passes = 0
        self.failed = False
        self._done = threading.Event()
        self._thread: threading.Thread | None = None

    def step(self, audio: np.ndarray):
        """One pass over the recording so far; keeps all of its text but the last STREAM_HOLD tokens."""
        _, ids = self._transcribe(audio, prefix=self.kept, return_ids=True)
        self.kept = ids[: max(len(self.kept), len(ids) - STREAM_HOLD)]
        self.covered = len(audio)
        self.passes += 1

    def finish(self, audio: np.ndarray) -> str:
        """The text of the whole recording, continuing from what the passes settled."""
        if self._thread is not None:
            self._done.set()
            self._thread.join()
        if self.failed:
            self.kept = []
        text, _ = self._transcribe(audio, prefix=self.kept, return_ids=True)
        return text

    def follow(self, chunks: list[np.ndarray]):
        """Start a background thread that runs a pass whenever `chunks` (the list the recording
        loop is appending to) has grown by STREAM_STEP_S since the last pass."""
        def run():
            while not self._done.wait(0.1):
                n = len(chunks)
                if n * 1024 - self.covered < STREAM_STEP_S * SAMPLE_RATE:
                    continue
                try:
                    self.step(np.concatenate(chunks[:n]).flatten())
                except Exception:
                    self.failed = True
                    _log("[dictate] ERROR transcribing while recording (the clip will be transcribed in one go):")
                    traceback.print_exc(file=_log_file)
                    return
        self._thread = threading.Thread(target=run, daemon=True)
        self._thread.start()


def _record():
    global _audio_stream
    chunks = []
    stream_state = None
    if STREAM and getattr(_transcribe, "supports_prefix", False):
        stream_state = _Stream(_transcribe)
        stream_state.follow(chunks)
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
        _audio_stream = stream_state


def _save_clip(audio: np.ndarray) -> str | None:
    """Write the clip as 16-bit mono WAV under SAVE_DIR/clips; returns its clip id."""
    if not SAVE_AUDIO:
        return None
    try:
        clips_dir = os.path.join(SAVE_DIR, "clips")
        os.makedirs(clips_dir, exist_ok=True)
        clip_id = time.strftime("%Y%m%d_%H%M%S") + f"_{int(time.time() * 1000) % 1000:03d}"
        pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
        with wave.open(os.path.join(clips_dir, f"{clip_id}.wav"), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(pcm.tobytes())
        return clip_id
    except Exception:
        _log("[dictate] ERROR saving audio clip:")
        traceback.print_exc(file=_log_file)
        return None


def _save_metadata(clip_id: str, audio: np.ndarray, hypothesis: str):
    """Append the clip's record to SAVE_DIR/metadata.jsonl.

    `transcription` is the reference text: empty until the clip is reviewed and
    corrected. `hypothesis` is what the model produced.
    """
    record = {
        "file_name": f"clips/{clip_id}.wav",
        "id": clip_id,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "duration_s": round(len(audio) / SAMPLE_RATE, 2),
        "model": WHISPER_MODEL if MODEL == "whisper" else MODELS[MODEL],
        "initial_prompt": WHISPER_INITIAL_PROMPT if MODEL == "whisper" else "",
        "keywords": _load_keywords() if MODEL in ("qwen3", "granite") else "",
        "hypothesis": hypothesis,
        "transcription": "",
    }
    try:
        with open(os.path.join(SAVE_DIR, "metadata.jsonl"), "a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        _log("[dictate] ERROR writing clip metadata:")
        traceback.print_exc(file=_log_file)


def _transcribe_and_type():
    try:
        with _lock:
            chunks = list(_audio_chunks)
            stream_state = _audio_stream
        if not chunks:
            return
        audio = np.concatenate(chunks).flatten()
        # Save before transcribing so the audio survives a transcription crash.
        clip_id = _save_clip(audio)
        print("\r[dictate] Transcribing...              ", end="", flush=True)
        if stream_state is not None:
            text = stream_state.finish(audio)
            if stream_state.passes:
                _log(f"[dictate] {stream_state.covered / SAMPLE_RATE:.0f} s of {len(audio) / SAMPLE_RATE:.0f} s "
                     f"transcribed while recording ({stream_state.passes} passes)")
        else:
            text = _transcribe(audio)
        if clip_id:
            _save_metadata(clip_id, audio, text)
            _log(f"[dictate] clip {clip_id} → {text!r}")
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

    # Keyboards are tracked by device path. The set is rescanned periodically so
    # keyboards plugged in (or wireless keyboards reconnecting after sleep, which
    # come back as a new /dev/input/event* node) after startup are picked up, and
    # ones that disappear are dropped instead of crashing the listener.
    RESCAN_INTERVAL = 300.0  # seconds
    devices: dict[str, "InputDevice"] = {}
    # Per-device pressed keys, so a keyboard unplugged mid-press can't leave a
    # modifier stuck down.
    pressed: dict[str, set[str]] = {}
    ignored_paths: set[str] = set()

    def _scan() -> int:
        permission_denied = 0
        for path in list_devices():
            if path in devices or path in ignored_paths:
                continue
            try:
                dev = InputDevice(path)
            except PermissionError:
                permission_denied += 1
                continue
            except Exception:
                continue
            try:
                caps = dev.capabilities().get(ecodes.EV_KEY, [])
                key_codes = {item[0] if isinstance(item, tuple) else item for item in caps}
            except Exception:
                key_codes = set()
            if ecodes.KEY_SPACE in key_codes:
                devices[path] = dev
                pressed[path] = set()
                _log(f"[dictate] evdev: listening on {dev.name} ({path})")
            else:
                ignored_paths.add(path)
                dev.close()
        return permission_denied

    def _drop(path: str):
        dev = devices.pop(path, None)
        pressed.pop(path, None)
        if dev is not None:
            _log(f"[dictate] evdev: device removed: {dev.name} ({path})")
            try:
                dev.close()
            except Exception:
                pass

    permission_denied = _scan()
    if not devices:
        if permission_denied > 0:
            print("[dictate] evdev: permission denied. Run: sudo usermod -aG input $USER  (or: newgrp input)", flush=True)
        else:
            print("[dictate] evdev: no readable input devices found.", flush=True)
        return False

    print("[dictate] Wayland global hotkey mode (evdev): Ctrl+Shift+Space toggles, Ctrl+C quits.", flush=True)
    combo_latched = False
    last_scan = time.monotonic()

    try:
        while True:
            now = time.monotonic()
            if now - last_scan >= RESCAN_INTERVAL:
                last_scan = now
                # Forget ignored paths that no longer exist so a reused node number
                # gets re-checked, then pick up any new keyboards.
                existing = set(list_devices())
                ignored_paths.intersection_update(existing)
                for path in [p for p in devices if p not in existing]:
                    _drop(path)
                _scan()

            if not devices:
                time.sleep(0.2)
                continue

            try:
                ready, _, _ = select.select(list(devices.values()), [], [], 0.2)
            except (OSError, ValueError):
                # A device fd went bad between rescans; force a rescan.
                last_scan = 0.0
                continue

            for dev in ready:
                path = dev.path
                try:
                    events = list(dev.read())
                except OSError:
                    _drop(path)
                    continue

                for event in events:
                    if event.type != ecodes.EV_KEY:
                        continue

                    token = key_map.get(event.code)
                    if not token:
                        continue

                    if event.value in (1, 2):
                        pressed[path].add(token)
                    elif event.value == 0:
                        pressed[path].discard(token)

                    combo_active = hotkey_tokens.issubset(pressed[path])
                    if combo_active and not combo_latched:
                        combo_latched = True
                        _log(f"[dictate] Hotkey triggered via evdev ({dev.name}) → toggling recording.")
                        _toggle_recording(async_transcribe=True)
                    elif not any(hotkey_tokens.issubset(keys) for keys in pressed.values()):
                        combo_latched = False
    except PermissionError:
        print("[dictate] No permission to read /dev/input/event*. Add user to input group and re-login.", flush=True)
        return False
    except Exception as exc:
        print(f"[dictate] evdev listener failed: {exc}", flush=True)
        return False
    finally:
        for dev in devices.values():
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
    _install_handlers()
    _load_model()
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
