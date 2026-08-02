"""
AstroNexus AI — Whisper Speech-to-Text Service

Transcribes audio files to text using OpenAI Whisper.
Supports MP3, WAV, M4A, OGG, FLAC, WEBM.

Model: base (good speed/accuracy balance for RTX 4060)
       small — better accuracy, slightly slower
       medium — best quality, more VRAM

Install: pip install openai-whisper

GPU: automatic if CUDA available
CPU: fallback, expect ~2-4x realtime
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

import os
import shutil

logger = logging.getLogger(__name__)

WHISPER_MODEL  = "base"            # change to "small" for better accuracy
SUPPORTED_EXTS = {
    ".mp3", ".wav", ".m4a", ".ogg",
    ".flac", ".webm", ".mp4", ".mkv",
}

_model  = None
_device = None


# ── FFmpeg auto-discovery ─────────────────────────────────────────────────────

def _find_ffmpeg() -> str | None:
    """
    Search for ffmpeg.exe in multiple locations.
    Returns the full path if found, None otherwise.
    """
    # 1. Local project bin/ folder (manually placed)
    local = Path(__file__).resolve().parents[2] / "bin" / "ffmpeg" / "bin" / "ffmpeg.exe"
    if local.exists():
        return str(local)

    # 2. Standard system PATH
    found = shutil.which("ffmpeg") or shutil.which("ffmpeg.exe")
    if found:
        return found

    # 3. WinGet install location (winget install Gyan.FFmpeg)
    winget_base = Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft" / "WinGet" / "Packages"
    if winget_base.exists():
        for pkg_dir in winget_base.glob("Gyan.FFmpeg*"):
            for candidate in sorted(pkg_dir.rglob("ffmpeg.exe")):
                if candidate.exists():
                    return str(candidate)

    return None


def _ensure_ffmpeg_on_path() -> str | None:
    """
    Find ffmpeg and inject its bin directory into os.environ PATH so that
    Whisper's internal subprocess call can locate it automatically.
    Returns the ffmpeg path if found, None otherwise.
    """
    ffmpeg = _find_ffmpeg()
    if ffmpeg:
        bin_dir = str(Path(ffmpeg).parent)
        current_path = os.environ.get("PATH", "")
        if bin_dir not in current_path:
            os.environ["PATH"] = bin_dir + os.pathsep + current_path
            logger.info(f"[Whisper] Injected ffmpeg into PATH: {bin_dir}")
        else:
            logger.info(f"[Whisper] ffmpeg already on PATH: {ffmpeg}")
    else:
        logger.warning(
            "[Whisper] ffmpeg not found. Non-WAV transcription may fail.\n"
            "Install: winget install Gyan.FFmpeg  (then restart the server)"
        )
    return ffmpeg


# Inject ffmpeg into PATH at import time so all subsequent calls work
_ffmpeg_exe = _ensure_ffmpeg_on_path()




class WhisperService:
    """
    Singleton-backed Whisper transcription service.

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

        try:
            import whisper
        except ImportError as e:
            raise ImportError(
                "Install Whisper: pip install openai-whisper"
            ) from e

        import torch
        _device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info(
            f"[Whisper] Loading model '{WHISPER_MODEL}' "
            f"on {_device.upper()}..."
        )
        t0     = time.perf_counter()
        _model = whisper.load_model(WHISPER_MODEL, device=_device)
        logger.info(
            f"[Whisper] Model ready in "
            f"{(time.perf_counter()-t0)*1000:.0f}ms  "
            f"device={_device}"
        )
        self._loaded = True

    # ── Public API ─────────────────────────────────────────────────────────────

    def transcribe(self, audio_path: str | Path) -> dict:
        """
        Transcribe an audio file to text.

        Args:
            audio_path: Path to audio file (WAV, MP3, M4A, OGG, FLAC...)

        Returns:
            {
                "text":     str,    # transcribed text
                "language": str,    # detected language code e.g. "en"
                "duration": float,  # audio duration in seconds
                "segments": list,   # word-level segments (optional detail)
                "model":    str,    # model name used
                "device":   str,    # "cuda" or "cpu"
            }

        Raises:
            FileNotFoundError if audio_path does not exist
            ValueError if file format is unsupported
            RuntimeError if Whisper fails
        """
        self._load()

        path = Path(audio_path)

        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {path}")

        if path.suffix.lower() not in SUPPORTED_EXTS:
            raise ValueError(
                f"Unsupported format: {path.suffix}. "
                f"Supported: {', '.join(sorted(SUPPORTED_EXTS))}"
            )

        logger.info(f"[Whisper] Transcribing: {path.name}")
        t0 = time.perf_counter()

        result = _model.transcribe(
            str(path),
            fp16=   (_device == "cuda"),  # fp16 only on GPU
            verbose=False,
        )

        elapsed = time.perf_counter() - t0

        # Extract duration from segments if available
        duration = 0.0
        segs     = result.get("segments", [])
        if segs:
            duration = float(segs[-1].get("end", 0.0))

        text     = result.get("text", "").strip()
        language = result.get("language", "unknown")

        logger.info(
            f"[Whisper] Done in {elapsed:.2f}s — "
            f"lang={language}  chars={len(text)}  duration={duration:.1f}s"
        )

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
    """
    Transcribe audio — convenience wrapper around WhisperService.

    Args:
        audio_path: Path to audio file

    Returns:
        {"text": str, "language": str, "duration": float, ...}
    """
    return WhisperService().transcribe(audio_path)