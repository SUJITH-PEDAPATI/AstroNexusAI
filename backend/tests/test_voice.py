"""
AstroNexus AI — Backend Voice Test  (diagnostic edition)
=========================================================
Verifies ONLY:
    Microphone → Recording → WAV → WhisperService → Transcript

Nothing else runs. No LLM, RAG, TTS, Neo4j, LangGraph.

Usage (from project root):
    python backend/tests/test_voice.py              # press Enter to stop
    python backend/tests/test_voice.py --duration 5 # fixed 5-second clip

Dependencies (install once):
    pip install sounddevice soundfile openai-whisper
    # Windows/Linux also needs ffmpeg on PATH for Whisper
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import time
import threading
import traceback
from pathlib import Path

# ── Args ──────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description="AstroNexus Voice Test")
parser.add_argument("--duration",    type=float, default=None)
parser.add_argument("--sample-rate", type=int,   default=16_000)
parser.add_argument("--channels",    type=int,   default=1)
args = parser.parse_args()

SAMPLE_RATE = args.sample_rate
CHANNELS    = args.channels
DURATION    = args.duration

# ── Helpers ───────────────────────────────────────────────────────────────────

def sep(c="=", n=44): print(c * n)
def line(c="-", n=44): print(c * n)

def section(title: str):
    print()
    line()
    print(f"  {title}")
    line()

def ok(msg):  print(f"  ✓  {msg}")
def err(msg): print(f"  ✗  {msg}")
def info(msg):print(f"  ·  {msg}")

# ── Banner ────────────────────────────────────────────────────────────────────

print()
sep()
print("  AstroNexus AI — Voice Diagnostic Test")
sep()

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Environment
# ══════════════════════════════════════════════════════════════════════════════

section("1 · Python Environment")

info(f"Executable : {sys.executable}")
info(f"Version    : {sys.version.split()[0]}")
info(f"Platform   : {sys.platform}")

# Virtual environment detection
venv = os.environ.get("VIRTUAL_ENV") or os.environ.get("CONDA_PREFIX")
if venv:
    ok(f"Virtual env: {venv}")
else:
    err("No virtual environment detected — wrong Python might be active")
    info("Activate your venv:  .venv\\Scripts\\activate  (Windows)")
    info("                  or source .venv/bin/activate  (Linux/Mac)")

# sys.path (for import diagnosis)
section("2 · sys.path (import search order)")
for i, p in enumerate(sys.path):
    print(f"  [{i}] {p}")

# Project root calculation
_here    = Path(__file__).resolve()
_project = _here.parents[2]  # backend/tests/test_voice.py → 3 up → project root
info(f"This file  : {_here}")
info(f"Project root (calculated): {_project}")

if str(_project) not in sys.path:
    sys.path.insert(0, str(_project))
    ok(f"Added to sys.path: {_project}")
else:
    ok(f"Project root already in sys.path")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — Dependency checks
# ══════════════════════════════════════════════════════════════════════════════

section("3 · Dependency Check")

# numpy
try:
    import numpy as np
    ok(f"numpy {np.__version__}")
except ImportError:
    err("numpy missing — run: pip install numpy")
    sys.exit(1)

# sounddevice
try:
    import sounddevice as sd
    ok(f"sounddevice {sd.__version__}")
except ImportError:
    err("sounddevice missing — run: pip install sounddevice")
    sys.exit(1)

# soundfile
try:
    import soundfile as sf
    ok(f"soundfile {sf.__version__}")
except ImportError:
    err("soundfile missing — run: pip install soundfile")
    sys.exit(1)

# openai-whisper
try:
    import whisper as _w
    ok(f"openai-whisper  (whisper module found at: {_w.__file__})")
except ImportError:
    err("openai-whisper missing — run: pip install openai-whisper")
    info("Install into THIS interpreter:")
    info(f"  {sys.executable} -m pip install openai-whisper")
    sys.exit(1)

# ffmpeg (Whisper requires it to decode audio)
ffmpeg_ok = shutil.which("ffmpeg") is not None
if ffmpeg_ok:
    try:
        v = subprocess.check_output(
            ["ffmpeg", "-version"], stderr=subprocess.STDOUT, text=True
        ).splitlines()[0]
        ok(f"ffmpeg: {v}")
    except Exception:
        ok("ffmpeg found on PATH")
else:
    err("ffmpeg NOT found on PATH")
    info("Whisper requires ffmpeg to decode audio files.")
    info("Windows : https://ffmpeg.org/download.html  (add to PATH)")
    info("Linux   : sudo apt install ffmpeg")
    info("Mac     : brew install ffmpeg")
    info("Continuing — transcription will likely fail without ffmpeg.")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — WhisperService import (full traceback always shown)
# ══════════════════════════════════════════════════════════════════════════════

section("4 · WhisperService Import")

# Show exactly where Python will look for the module
voice_candidate = _project / "backend" / "voice" / "whisper_service.py"
info(f"Looking for: {voice_candidate}")
info(f"Exists     : {voice_candidate.exists()}")

if not voice_candidate.exists():
    # Search the whole project tree as a fallback hint
    found = list(_project.rglob("whisper_service.py"))
    if found:
        err("whisper_service.py is in a DIFFERENT location than expected:")
        for f in found:
            info(f"  Found at: {f}")
        info("The import path 'backend.voice.whisper_service' expects the file at:")
        info(f"  {voice_candidate}")
        info("Either move the file there or adjust the import in this script.")
    else:
        err("whisper_service.py not found anywhere under the project root.")
    sys.exit(1)

# Check __init__.py chain
for pkg_dir in [
    _project / "backend",
    _project / "backend" / "voice",
]:
    init = pkg_dir / "__init__.py"
    if init.exists():
        ok(f"__init__.py present: {init}")
    else:
        err(f"__init__.py MISSING: {init}")
        info(f"Create it:  echo. > {init}  (Windows)  or  touch {init}")

# Attempt import — always print full traceback on failure
WhisperService = None
WHISPER_MODEL  = "base"

try:
    from backend.voice.whisper_service import WhisperService
    WHISPER_MODEL = getattr(WhisperService, '_model_name', None) or 'base'
    ok(f"WhisperService imported successfully")
    ok(f"Model configured: {WHISPER_MODEL}")
except Exception:
    err("Import failed — full traceback:")
    print()
    traceback.print_exc()
    print()
    err("Diagnosis: see traceback above for the real cause.")
    info("Common causes:")
    info("  1. Wrong Python interpreter / venv (not the project's venv)")
    info("  2. openai-whisper not installed in the active interpreter")
    info("  3. Circular import in backend/__init__.py")
    info("  4. Missing __init__.py in backend/voice/")
    info("  5. backend/ has an __init__.py that imports something broken")
    print()
    info("Quick test — run this to isolate:")
    info(f"  {sys.executable} -c \"from backend.voice.whisper_service import WhisperService\"")
    sys.exit(1)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — Microphone detection
# ══════════════════════════════════════════════════════════════════════════════

section("5 · Microphone Detection")

try:
    all_devices = sd.query_devices()
    default_in  = sd.query_devices(kind="input")
    ok(f"Default input : {default_in['name']}")
    info(f"Max channels  : {default_in['max_input_channels']}")
    info(f"Default SR    : {int(default_in['default_samplerate'])} Hz")
    info(f"Using SR      : {SAMPLE_RATE} Hz")
except Exception:
    err("Failed to query audio devices:")
    traceback.print_exc()
    sys.exit(1)

# List all input devices so user can identify the correct one
print()
info("All available input devices:")
try:
    for i, d in enumerate(all_devices):
        if d["max_input_channels"] > 0:
            marker = " ← default" if d["name"] == default_in["name"] else ""
            print(f"    [{i}] {d['name']}{marker}")
except Exception:
    pass

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — Recording
# ══════════════════════════════════════════════════════════════════════════════

section("6 · Recording")

if DURATION:
    print(f"  Recording for {DURATION:.0f} seconds...")
else:
    print("  Recording...  press Enter to stop")
print("  Speak now.")
print()

frames: list[np.ndarray] = []
recording       = True
callback_errors: list[str] = []

def _callback(indata, frame_count, time_info, status):
    if status:
        callback_errors.append(str(status))
    if recording:
        frames.append(indata.copy())

def _record_thread():
    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype="float32",
            callback=_callback,
            blocksize=1024,
        ):
            while recording:
                time.sleep(0.05)
    except Exception:
        print()
        err("InputStream error:")
        traceback.print_exc()

t = threading.Thread(target=_record_thread, daemon=True)
t.start()
t0 = time.perf_counter()

if DURATION:
    time.sleep(DURATION)
else:
    try:
        input()
    except (EOFError, KeyboardInterrupt):
        pass

recording = False
t.join(timeout=2)
elapsed_rec = time.perf_counter() - t0

if callback_errors:
    err(f"Audio callback reported {len(callback_errors)} status warnings:")
    for e in set(callback_errors):
        info(f"  {e}")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — Audio diagnostics (always printed, even if silent)
# ══════════════════════════════════════════════════════════════════════════════

section("7 · Audio Diagnostics")

if not frames:
    err("No frames captured at all. Recording thread may have crashed.")
    sys.exit(1)

audio = np.concatenate(frames, axis=0)
duration_sec  = len(audio) / SAMPLE_RATE
size_kb       = audio.nbytes / 1024

# Flatten to mono for energy analysis
mono = audio[:, 0] if audio.ndim == 2 else audio
rms            = float(np.sqrt(np.mean(mono ** 2)))
max_amplitude  = float(np.max(np.abs(mono)))
is_silent      = max_amplitude < 1e-4

info(f"Frames captured  : {len(frames)}")
info(f"Duration         : {duration_sec:.3f} s")
info(f"Sample rate      : {SAMPLE_RATE} Hz")
info(f"Channels         : {CHANNELS}")
info(f"Samples          : {len(audio):,}")
info(f"Audio size       : {size_kb:.1f} KB")
info(f"Max amplitude    : {max_amplitude:.6f}  (0 = silent, 1 = full scale)")
info(f"RMS energy       : {rms:.6f}")
info(f"Silent?          : {'YES — no speech detected' if is_silent else 'NO — audio contains signal'}")

if is_silent:
    print()
    err("Recording is SILENT (max amplitude < 0.0001)")
    info("Possible causes:")
    info("  1. Microphone is muted in OS sound settings")
    info("  2. Wrong input device selected (see device list above)")
    info("  3. Microphone permissions denied at OS level")
    info("     Windows: Settings → Privacy → Microphone → allow for Python")
    info("  4. sounddevice not reading from the physical microphone")
    info("     Try specifying the device index explicitly:")
    info(f"     Add  device=N  to sd.InputStream()  (check device list above)")
    sys.exit(1)

if duration_sec < 0.5:
    err(f"Recording too short ({duration_sec:.2f}s). Minimum ~0.5s needed.")
    sys.exit(1)

ok(f"Audio contains real signal — ready to transcribe")

# ── Save WAV ──────────────────────────────────────────────────────────────────

# Save to project output dir (keep a copy for manual inspection)
out_dir = _project / "output" / "voice_results"
out_dir.mkdir(parents=True, exist_ok=True)
wav_path = out_dir / "astronexus_voice_test.wav"

try:
    sf.write(str(wav_path), audio, SAMPLE_RATE, subtype="PCM_16")
    wav_size_kb = wav_path.stat().st_size / 1024
    ok(f"WAV saved : {wav_path}")
    info(f"File size : {wav_size_kb:.1f} KB")
except Exception:
    err("Failed to save WAV:")
    traceback.print_exc()
    sys.exit(1)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — Whisper transcription
# ══════════════════════════════════════════════════════════════════════════════

section("8 · Whisper Transcription")

ok(f"Model : {WHISPER_MODEL}")
print("  Transcribing...  (this takes a few seconds on first run)")
t1 = time.perf_counter()

try:
    result = WhisperService().transcribe(wav_path)
except Exception:
    err("Transcription failed — full traceback:")
    traceback.print_exc()
    wav_path.unlink(missing_ok=True)
    sys.exit(1)

process_time = time.perf_counter() - t1

# Keep the WAV so user can play it back and verify the recording
info(f"WAV kept at: {wav_path}  (play it to verify recording)")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 9 — Results
# ══════════════════════════════════════════════════════════════════════════════

section("9 · Results")

transcript = result.get("text", "").strip()
language   = result.get("language", "unknown")
device     = result.get("device",   "cpu")
model_used = result.get("model",    WHISPER_MODEL)

lang_label = {
    "en": "English", "hi": "Hindi", "fr": "French", "de": "German",
    "es": "Spanish", "zh": "Chinese", "ja": "Japanese", "ar": "Arabic",
}.get(language, language.title() if language else "Unknown")

info(f"Detected language : {lang_label} ({language})")
info(f"Processing time   : {process_time:.2f} sec")
info(f"Whisper model     : {model_used}  [{device.upper()}]")
info(f"Transcript length : {len(transcript)} chars")

print()
line()
print("  Transcript:")
print()
if transcript:
    for ln in textwrap.wrap(transcript, width=68):
        print(f"    {ln}")
else:
    print("    (empty)")
    info("Whisper returned nothing. Try WHISPER_MODEL = 'small' for better accuracy.")
print()
line()

# ── Final status ──────────────────────────────────────────────────────────────

print()
if transcript:
    sep()
    ok(" Voice Test Successful")
    sep()
else:
    sep()
    err(" Transcription empty")
    info("Switch to a more accurate model: set WHISPER_MODEL='small'")
    info(f"in: {voice_candidate}")
    sep()

print()