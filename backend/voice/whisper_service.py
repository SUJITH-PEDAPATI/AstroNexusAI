"""
AstroNexus AI — Whisper Speech-to-Text Service

Converts audio files (WAV, MP3, M4A, FLAC, OGG) to text using OpenAI Whisper.

Primary:  openai-whisper — local PyTorch-based model (tiny / base / small / medium / large)
Fallback: speech_recognition (Google Speech Recognition API / Offline Sphinx)

Install:
    pip install openai-whisper torch soundfile
"""
from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Optional, Any

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "base"

_whisper_model  = None
_whisper_device = None


def _load_audio_to_numpy(audio_path: Path) -> np.ndarray:
    """
    Load an audio file into a 16kHz float32 mono NumPy array.
    Bypasses ffmpeg dependency required by whisper.load_audio().
    """
    # 1. Try soundfile (handles WAV, FLAC, OGG, etc.)
    try:
        import soundfile as sf
        data, samplerate = sf.read(str(audio_path), dtype="float32")
        if data.ndim > 1:
            data = data.mean(axis=1)  # convert stereo → mono

        if samplerate != 16000:
            try:
                import scipy.signal
                num_samples = int(len(data) * 16000 / samplerate)
                data = scipy.signal.resample(data, num_samples).astype(np.float32)
            except ImportError:
                # Basic linear interpolation fallback if scipy isn't installed
                old_indices = np.arange(len(data))
                new_indices = np.linspace(0, len(data) - 1, int(len(data) * 16000 / samplerate))
                data = np.interp(new_indices, old_indices, data).astype(np.float32)
        return data
    except (ImportError, Exception) as e:
        logger.debug(f"[Whisper] soundfile load failed: {e}")

    # 2. Try standard library wave module for WAV files
    try:
        import wave
        with wave.open(str(audio_path), "rb") as wf:
            n_channels = wf.getnchannels()
            sampwidth  = wf.getsampwidth()
            framerate  = wf.getframerate()
            n_frames   = wf.getnframes()
            raw_bytes  = wf.readframes(n_frames)

        if sampwidth == 2:
            data = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        elif sampwidth == 4:
            data = np.frombuffer(raw_bytes, dtype=np.int32).astype(np.float32) / 2147483648.0
        elif sampwidth == 1:
            data = (np.frombuffer(raw_bytes, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
        else:
            raise ValueError(f"Unsupported sample width: {sampwidth}")

        if n_channels > 1:
            data = data.reshape(-1, n_channels).mean(axis=1)

        if framerate != 16000:
            old_indices = np.arange(len(data))
            new_indices = np.linspace(0, len(data) - 1, int(len(data) * 16000 / framerate))
            data = np.interp(new_indices, old_indices, data).astype(np.float32)

        return data
    except Exception as e:
        logger.debug(f"[Whisper] wave module load failed: {e}")

    # 3. Fallback to whisper.load_audio (requires ffmpeg)
    import whisper
    return whisper.load_audio(str(audio_path))


class WhisperService:
    """
    Singleton-backed Whisper speech recognition service.

    Usage:
        svc    = WhisperService()
        result = svc.transcribe("path/to/audio.wav")
        print(result["text"])
    """

    _instance: Optional["WhisperService"] = None

    def __new__(cls) -> "WhisperService":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._loaded = False
            cls._instance._backend = None
            cls._instance._model_name = DEFAULT_MODEL
        return cls._instance

    # ── Model loading ──────────────────────────────────────────────────────────

    def _load_whisper(self) -> bool:
        """Try to load OpenAI Whisper. Returns True on success."""
        global _whisper_model, _whisper_device

        try:
            import whisper
            import torch

            _whisper_device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(
                f"[Whisper] Loading model '{self._model_name}' on {_whisper_device.upper()}..."
            )
            t0 = time.perf_counter()

            _whisper_model = whisper.load_model(self._model_name, device=_whisper_device)

            logger.info(
                f"[Whisper] Model '{self._model_name}' ready in "
                f"{(time.perf_counter()-t0)*1000:.0f}ms"
            )
            self._backend = "whisper"
            return True

        except ImportError:
            logger.warning(
                "[Whisper] openai-whisper package not installed. "
                "Install: pip install openai-whisper"
            )
            return False
        except Exception as e:
            logger.warning(f"[Whisper] Load failed: {e}")
            return False

    def _load_speech_recognition(self) -> bool:
        """Try to load SpeechRecognition fallback. Returns True on success."""
        try:
            import speech_recognition as sr

            self._recognizer = sr.Recognizer()
            self._backend = "speech_recognition"
            logger.info("[Whisper] SpeechRecognition fallback ready")
            return True
        except ImportError:
            logger.warning(
                "[Whisper] SpeechRecognition not installed. "
                "Install: pip install SpeechRecognition"
            )
            return False
        except Exception as e:
            logger.warning(f"[Whisper] SpeechRecognition load failed: {e}")
            return False

    def _load(self) -> None:
        if self._loaded:
            return
        if not self._load_whisper():
            if not self._load_speech_recognition():
                raise RuntimeError(
                    "No STT backend available.\n"
                    "Install at least one:\n"
                    "  pip install openai-whisper   # OpenAI Whisper (recommended)\n"
                    "  pip install SpeechRecognition # fallback"
                )
        self._loaded = True

    # ── Public API ─────────────────────────────────────────────────────────────

    def transcribe(
        self,
        audio_path: str | Path,
        language:   Optional[str] = None,
    ) -> dict:
        """
        Transcribe an audio file to text.

        Args:
            audio_path: Path to audio file (WAV, MP3, M4A, FLAC, OGG, etc.)
            language:   Optional 2-letter language code (e.g. 'en', 'es'). Autodetected if None.

        Returns:
            {
                "text":     str,   Full transcribed text
                "language": str,   Detected or specified language
                "duration": float, Audio duration in seconds (approx)
                "model":    str,   Model name used
                "device":   str,   Device used ('cpu' or 'cuda')
                "segments": list,  Detailed segment timestamps (Whisper backend only)
            }
        """
        path = Path(audio_path)
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {path}")

        self._load()

        logger.info(f"[Whisper] Transcribing '{path.name}' via {self._backend}...")
        t0 = time.perf_counter()

        if self._backend == "whisper":
            options: dict[str, Any] = {}
            if language:
                options["language"] = language

            audio_input = _load_audio_to_numpy(path)
            result = _whisper_model.transcribe(audio_input, **options)
            elapsed = time.perf_counter() - t0

            text     = result.get("text", "").strip()
            detected_lang = result.get("language", language or "en")
            segments = result.get("segments", [])

            # Duration estimate from last segment end time or file
            duration = round(segments[-1]["end"], 2) if segments else 0.0

            logger.info(
                f"[Whisper] Done in {elapsed:.2f}s  "
                f"lang={detected_lang}  "
                f"text='{text[:60]}...'"
            )

            return {
                "text":     text,
                "language": detected_lang,
                "duration": duration,
                "model":    self._model_name,
                "device":   _whisper_device or "cpu",
                "segments": segments,
            }

        elif self._backend == "speech_recognition":
            import speech_recognition as sr

            with sr.AudioFile(str(path)) as source:
                audio_data = self._recognizer.record(source)

            try:
                text = self._recognizer.recognize_google(audio_data, language=language or "en-US")
            except sr.UnknownValueError:
                text = ""
            except sr.RequestError as e:
                raise RuntimeError(f"Google Speech Recognition service error: {e}") from e

            elapsed = time.perf_counter() - t0
            logger.info(f"[Whisper] SpeechRecognition done in {elapsed:.2f}s")

            return {
                "text":     text,
                "language": language or "en",
                "duration": 0.0,
                "model":    "google-speech-recognition",
                "device":   "cloud",
                "segments": [],
            }

        raise RuntimeError("No active STT backend loaded")

    def get_backend(self) -> str | None:
        return self._backend


# ── Module-level convenience function ─────────────────────────────────────────

def transcribe(
    audio_path: str | Path,
    language:   Optional[str] = None,
) -> dict:
    """
    Transcribe audio file — convenience wrapper.

    Args:
        audio_path: Path to audio file
        language:   Optional language code

    Returns:
        Dict with text, language, duration, model details
    """
    return WhisperService().transcribe(audio_path, language=language)
