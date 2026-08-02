"""
AstroNexus AI — Whisper Speech-to-Text Service

Transcribes audio files to text using OpenAI Whisper.
Supports WAV (no ffmpeg required), MP3, M4A, OGG, FLAC, WEBM (need ffmpeg).

Model: base (good speed/accuracy balance for RTX 4060)
       small — better accuracy, slightly slower
       medium — best quality, more VRAM

Install: pip install openai-whisper soundfile numpy

GPU: automatic if CUDA available
CPU: fallback, expect ~2-4x realtime

──────────────────────────────────────────────────────────────────────────────
WinError 2 — root cause and fix
──────────────────────────────────────────────────────────────────────────────
openai-whisper decodes audio via ffmpeg subprocess:
    subprocess.run(["ffmpeg", "-i", path, ...])

On Windows, if ffmpeg.exe is NOT on PATH, Python raises:
    FileNotFoundError: [WinError 2] The system cannot find the file specified

The "file" Windows cannot find is ffmpeg.exe, NOT the audio file.

This service fixes the problem two ways:

1. For WAV files (the common case in this pipeline): loads audio with
   soundfile + numpy directly — NO ffmpeg subprocess at all.

2. For other formats: checks ffmpeg availability FIRST and raises a clear
   error if missing, rather than letting the subprocess crash cryptically.

To install ffmpeg (only needed for non-WAV formats):
    Windows: https://www.gyan.dev/ffmpeg/builds/  → add bin/ folder to PATH
    Linux:   sudo apt install ffmpeg
    Mac:     brew install ffmpeg
──────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
import traceback
import time
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

WHISPER_MODEL  = "base"
SAMPLE_RATE    = 16_000   # Whisper always works at 16 kHz

# WAV is loaded without ffmpeg. All other formats need ffmpeg.
WAV_ONLY_EXTS  = {".wav"}
FFMPEG_EXTS    = {".mp3", ".m4a", ".ogg", ".flac", ".webm", ".mp4", ".mkv"}
SUPPORTED_EXTS = WAV_ONLY_EXTS | FFMPEG_EXTS

_model:  Optional[object] = None
_device: Optional[str]    = None


# ── FFmpeg probe ──────────────────────────────────────────────────────────────

def _ffmpeg_path() -> str | None:
    """Return the absolute path of ffmpeg if available on PATH, else None."""
    local_ffmpeg = Path(__file__).resolve().parents[2] / "bin" / "ffmpeg" / "bin" / "ffmpeg.exe"
    if local_ffmpeg.exists():
        return str(local_ffmpeg)
    return shutil.which("ffmpeg") or shutil.which("ffmpeg.exe")


def _require_ffmpeg(suffix: str) -> None:
    """
    Raise a clear RuntimeError if ffmpeg is needed but missing.
    Called before attempting to decode non-WAV formats.
    """
    path = _ffmpeg_path()
    if path:
        logger.info(f"[Whisper] ffmpeg found: {path}")
        return
    msg = (
        f"[WinError 2 FIX] Cannot transcribe {suffix!r} files: ffmpeg is not installed "
        f"or not on PATH.\n"
        f"The file you want to transcribe exists, but Whisper cannot decode it "
        f"because it relies on ffmpeg as a subprocess.\n\n"
        f"To fix:\n"
        f"  Windows → download from https://www.gyan.dev/ffmpeg/builds/\n"
        f"            extract the zip, add the bin\\ folder to your PATH\n"
        f"            restart your terminal and run: ffmpeg -version\n"
        f"  Linux   → sudo apt install ffmpeg\n"
        f"  Mac     → brew install ffmpeg\n\n"
        f"Alternatively, record audio as WAV — WAV files are loaded without "
        f"ffmpeg and will work immediately."
    )
    logger.error(msg)
    raise RuntimeError(msg)


# ── WAV loader (no ffmpeg) ────────────────────────────────────────────────────

def _load_wav_direct(path: Path) -> np.ndarray:
    """
    Load a WAV file into a float32 numpy array at 16 kHz WITHOUT ffmpeg.

    Uses soundfile (libsndfile) to read the file, then resamples if needed.
    This completely bypasses the ffmpeg subprocess that causes WinError 2.

    Args:
        path: Path to a WAV file that exists on disk

    Returns:
        float32 numpy array at 16 kHz, values in [-1, 1]

    Raises:
        FileNotFoundError if path does not exist
        RuntimeError if soundfile cannot read the file
    """
    # Verify existence here so the error message is clear
    if not path.exists():
        raise FileNotFoundError(
            f"Audio file not found: {path.resolve()}\n"
            f"  cwd={os.getcwd()}\n"
            f"  path.parent exists: {path.parent.exists()}"
        )

    file_bytes = path.stat().st_size
    logger.info(
        f"[Whisper/WAV] Loading: {path.resolve()}  "
        f"size={file_bytes} bytes  exists=True"
    )

    if file_bytes < 44:
        raise RuntimeError(
            f"WAV file is too small ({file_bytes} bytes). "
            "The recording may be empty or corrupt."
        )

    try:
        import soundfile as sf
    except ImportError as e:
        raise ImportError(
            "soundfile is required for WAV loading without ffmpeg. "
            "Run: pip install soundfile"
        ) from e

    try:
        audio, file_sr = sf.read(str(path), dtype="float32", always_2d=False)
    except Exception as e:
        raise RuntimeError(
            f"soundfile could not read {path.name}: {e}\n"
            f"The file may be corrupt or in an unsupported sub-format."
        ) from e

    logger.info(
        f"[Whisper/WAV] Read: shape={audio.shape} "
        f"sr={file_sr} Hz  duration={len(audio)/file_sr:.2f}s"
    )

    # Convert stereo → mono
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
        logger.info("[Whisper/WAV] Converted stereo → mono")

    # Resample to 16 kHz if needed
    if file_sr != SAMPLE_RATE:
        logger.info(f"[Whisper/WAV] Resampling {file_sr} → {SAMPLE_RATE} Hz")
        try:
            # scipy gives better quality than linear interpolation
            from scipy.signal import resample_poly
            from math import gcd
            g   = gcd(SAMPLE_RATE, file_sr)
            up  = SAMPLE_RATE // g
            down = file_sr    // g
            audio = resample_poly(audio, up, down).astype(np.float32)
        except ImportError:
            # Fallback: numpy linear interpolation
            ratio  = SAMPLE_RATE / file_sr
            n_out  = int(len(audio) * ratio)
            audio  = np.interp(
                np.linspace(0, len(audio) - 1, n_out),
                np.arange(len(audio)),
                audio,
            ).astype(np.float32)
        logger.info(f"[Whisper/WAV] Resampled → {len(audio)} samples")

    rms = float(np.sqrt(np.mean(audio ** 2)))
    logger.info(
        f"[Whisper/WAV] Ready: samples={len(audio)} "
        f"rms={rms:.4f} max={float(np.abs(audio).max()):.4f}"
    )

    if rms < 1e-5:
        logger.warning(
            "[Whisper/WAV] Audio RMS is near zero — recording may be silent. "
            "Check that the microphone was not muted."
        )

    return audio


# ── WhisperService ────────────────────────────────────────────────────────────

class WhisperService:
    """
    Singleton Whisper transcription service.

    For WAV input: loads audio via soundfile (no ffmpeg required).
    For other formats: checks ffmpeg first, then lets whisper decode.

    Usage:
        svc    = WhisperService()
        result = svc.transcribe("recording.wav")
        print(result["text"])
    """

    _instance: Optional["WhisperService"] = None

    def __new__(cls) -> "WhisperService":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._loaded = False
        return cls._instance

    # ── Model loading ──────────────────────────────────────────────────────────

    def _load(self) -> None:
        """Lazy-load Whisper model onto GPU or CPU."""
        if self._loaded:
            return

        global _model, _device

        logger.info("[Whisper] Loading model — pre-flight checks")
        logger.info(f"  Python cwd: {os.getcwd()}")
        logger.info(f"  WHISPER_MODEL: {WHISPER_MODEL}")
        logger.info(f"  ffmpeg on PATH: {_ffmpeg_path() or 'NOT FOUND'}")

        try:
            import whisper
        except ImportError as e:
            raise ImportError(
                "openai-whisper is not installed. "
                "Run: pip install openai-whisper"
            ) from e

        import torch
        _device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info(f"[Whisper] Loading '{WHISPER_MODEL}' on {_device.upper()}...")
        t0     = time.perf_counter()
        _model = whisper.load_model(WHISPER_MODEL, device=_device)
        logger.info(
            f"[Whisper] Model ready in {(time.perf_counter()-t0)*1000:.0f}ms "
            f"device={_device}"
        )
        self._loaded = True

    # ── Public API ─────────────────────────────────────────────────────────────

    def transcribe(self, audio_path: str | Path) -> dict:
        """
        Transcribe an audio file to text.

        WAV files are decoded with soundfile (no ffmpeg needed).
        Other formats use whisper's internal ffmpeg decoder.

        Args:
            audio_path: Absolute or relative path to the audio file.

        Returns:
            {
                "text":     str,
                "language": str,
                "duration": float,
                "segments": list,
                "model":    str,
                "device":   str,
            }

        Raises:
            FileNotFoundError  — audio file does not exist
            ValueError         — unsupported file format
            RuntimeError       — ffmpeg missing for non-WAV, or Whisper error
        """
        self._load()

        path = Path(audio_path).resolve()  # always absolute on Windows

        # ── Pre-flight logging ──────────────────────────────────────────────────
        logger.info("=" * 60)
        logger.info("[Whisper] Transcription request")
        logger.info(f"  input path (raw):     {audio_path!r}")
        logger.info(f"  resolved (absolute):  {path}")
        logger.info(f"  file exists:          {path.exists()}")
        logger.info(f"  file size:            {path.stat().st_size if path.exists() else 'N/A'} bytes")
        logger.info(f"  parent dir exists:    {path.parent.exists()}")
        logger.info(f"  cwd:                  {os.getcwd()}")
        logger.info(f"  ffmpeg:               {_ffmpeg_path() or 'NOT FOUND'}")
        logger.info("=" * 60)

        # ── Existence guard — must come before any processing ──────────────────
        if not path.exists():
            raise FileNotFoundError(
                f"Audio file not found: {path}\n"
                f"  Raw input was: {audio_path!r}\n"
                f"  CWD: {os.getcwd()}\n"
                f"  Parent exists: {path.parent.exists()}\n"
                f"  This is NOT a WinError 2 / ffmpeg issue. "
                f"The WAV file itself was not saved or was deleted before transcription."
            )

        # ── Format check ───────────────────────────────────────────────────────
        suffix = path.suffix.lower()
        if suffix not in SUPPORTED_EXTS:
            raise ValueError(
                f"Unsupported audio format: {suffix!r}. "
                f"Supported: {sorted(SUPPORTED_EXTS)}"
            )

        # ── WAV path: bypass ffmpeg entirely ──────────────────────────────────
        if suffix == ".wav":
            logger.info("[Whisper] WAV detected → using soundfile (no ffmpeg)")
            try:
                audio_np = _load_wav_direct(path)
            except Exception:
                traceback.print_exc()
                raise

            logger.info(f"[Whisper] Calling _model.transcribe() on numpy array")
            t0 = time.perf_counter()
            try:
                result = _model.transcribe(
                    audio_np,           # pass numpy array, not file path
                    fp16=    (_device == "cuda"),
                    verbose= False,
                )
            except Exception as e:
                traceback.print_exc()
                raise RuntimeError(
                    f"Whisper model.transcribe() failed on numpy array: {e}\n"
                    f"This is NOT a file path or ffmpeg issue."
                ) from e

        # ── Non-WAV path: check ffmpeg then let whisper decode ─────────────────
        else:
            logger.info(
                f"[Whisper] {suffix!r} format → requires ffmpeg for decoding"
            )
            _require_ffmpeg(suffix)   # raises RuntimeError with fix instructions

            logger.info(
                f"[Whisper] Calling _model.transcribe() with file path: {path}"
            )
            t0 = time.perf_counter()
            try:
                result = _model.transcribe(
                    str(path),
                    fp16=    (_device == "cuda"),
                    verbose= False,
                )
            except FileNotFoundError as e:
                traceback.print_exc()
                raise RuntimeError(
                    f"[WinError 2] Whisper triggered a FileNotFoundError even though "
                    f"ffmpeg was detected at {_ffmpeg_path()!r}.\n"
                    f"Original error: {e}\n"
                    f"Check that ffmpeg works: ffmpeg -version"
                ) from e
            except Exception as e:
                traceback.print_exc()
                raise

        elapsed = time.perf_counter() - t0

        segs     = result.get("segments", [])
        duration = float(segs[-1].get("end", 0.0)) if segs else 0.0
        text     = result.get("text", "").strip()
        language = result.get("language", "unknown")

        logger.info(
            f"[Whisper] Done in {elapsed:.2f}s — "
            f"lang={language}  chars={len(text)}  duration={duration:.1f}s"
        )
        logger.info(f"[Whisper] Transcript: {text[:100]!r}")

        return {
            "text":     text,
            "language": language,
            "duration": round(duration, 2),
            "segments": [
                {
                    "start": round(s.get("start", 0), 2),
                    "end":   round(s.get("end",   0), 2),
                    "text":  s.get("text", "").strip(),
                }
                for s in segs
            ],
            "model":  WHISPER_MODEL,
            "device": _device,
        }

    def is_loaded(self) -> bool:
        return self._loaded

    def get_device(self) -> str | None:
        return _device


# ── Module-level convenience function ─────────────────────────────────────────

def transcribe(audio_path: str | Path) -> dict:
    """Convenience wrapper — same as WhisperService().transcribe(audio_path)."""
    return WhisperService().transcribe(audio_path)