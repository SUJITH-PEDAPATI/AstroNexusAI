"""
AstroNexus AI — Text-to-Speech Service

Converts text to speech using Coqui TTS (XTTS-v2).
Saves output as WAV to output/tts_output.wav.

Primary:  XTTS-v2 (Coqui) — highest quality, multi-lingual
Fallback: pyttsx3          — offline, no quality but always works

Install:
    pip install TTS          # Coqui XTTS-v2
    pip install pyttsx3      # fallback

XTTS-v2 requires a reference speaker WAV (3-10 seconds of speech).
A default reference is generated on first run if none is provided.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

DEFAULT_OUT_PATH = Path("output/tts_output.wav")
DEFAULT_REF_WAV  = Path("output/tts_reference.wav")
LANGUAGE         = "en"

_tts_model  = None
_tts_device = None


class TTSService:
    """
    Singleton-backed text-to-speech service.

    Usage:
        svc      = TTSService()
        out_path = svc.speak("The satellite image shows urban development.")
        print(f"Audio saved to {out_path}")
    """

    _instance: Optional["TTSService"] = None

    def __new__(cls) -> "TTSService":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._loaded = False
            cls._instance._backend = None
        return cls._instance

    # ── Model loading ──────────────────────────────────────────────────────────

    def _load_xtts(self) -> bool:
        """Try to load Coqui XTTS-v2. Returns True on success."""
        global _tts_model, _tts_device

        try:
            from TTS.api import TTS
            import torch

            _tts_device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(
                f"[TTS] Loading XTTS-v2 on {_tts_device.upper()}..."
            )
            t0 = time.perf_counter()

            _tts_model = TTS(
                model_name="tts_models/multilingual/multi-dataset/xtts_v2",
                progress_bar=False,
            ).to(_tts_device)

            logger.info(
                f"[TTS] XTTS-v2 ready in "
                f"{(time.perf_counter()-t0)*1000:.0f}ms"
            )
            self._backend = "xtts"
            return True

        except ImportError:
            logger.warning(
                "[TTS] Coqui TTS not installed. "
                "Install: pip install TTS"
            )
            return False
        except Exception as e:
            logger.warning(f"[TTS] XTTS-v2 load failed: {e}")
            return False

    def _load_pyttsx3(self) -> bool:
        """Try to load pyttsx3 fallback. Returns True on success."""
        try:
            import pyttsx3
            engine = pyttsx3.init()
            engine.setProperty("rate", 175)
            self._pyttsx3_engine = engine
            self._backend = "pyttsx3"
            logger.info("[TTS] pyttsx3 fallback ready")
            return True
        except ImportError:
            logger.warning(
                "[TTS] pyttsx3 not installed. "
                "Install: pip install pyttsx3"
            )
            return False
        except Exception as e:
            logger.warning(f"[TTS] pyttsx3 load failed: {e}")
            return False

    def _load(self) -> None:
        if self._loaded:
            return
        if not self._load_xtts():
            if not self._load_pyttsx3():
                raise RuntimeError(
                    "No TTS backend available.\n"
                    "Install at least one:\n"
                    "  pip install TTS        # Coqui XTTS-v2 (recommended)\n"
                    "  pip install pyttsx3    # fallback"
                )
        self._loaded = True

    # ── Reference WAV for XTTS ─────────────────────────────────────────────────

    def _ensure_reference_wav(self, ref_path: Path) -> Path:
        """
        XTTS-v2 requires a short reference WAV for voice cloning.
        If none is provided, generate a silent placeholder using scipy.
        In production: replace with a real 5-10 second speech WAV.
        """
        if ref_path.exists():
            return ref_path

        ref_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(
            f"[TTS] No reference WAV found at {ref_path}. "
            f"Generating silent placeholder — replace with real speech WAV "
            f"for better voice quality."
        )

        try:
            import numpy as np
            import scipy.io.wavfile as wavfile

            # 5 seconds of silence at 22050 Hz
            sample_rate = 22050
            samples     = np.zeros(5 * sample_rate, dtype=np.int16)
            wavfile.write(str(ref_path), sample_rate, samples)
            logger.info(f"[TTS] Placeholder reference WAV → {ref_path}")
        except ImportError:
            # scipy not available — try with wave stdlib
            import wave, struct
            sample_rate = 22050
            n_samples   = 5 * sample_rate
            with wave.open(str(ref_path), "w") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sample_rate)
                wf.writeframes(b"\x00\x00" * n_samples)

        return ref_path

    # ── Public API ─────────────────────────────────────────────────────────────

    def speak(
        self,
        text:     str,
        out_path: str | Path = DEFAULT_OUT_PATH,
        ref_wav:  str | Path = DEFAULT_REF_WAV,
    ) -> Path:
        """
        Convert text to speech and save as WAV.

        Args:
            text:     Text to speak (max ~200 words for XTTS)
            out_path: Output WAV path
            ref_wav:  Speaker reference WAV (XTTS only, 3-10 seconds)

        Returns:
            Path to output WAV file
        """
        self._load()

        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # Truncate very long text to avoid XTTS memory issues
        max_chars = 800
        if len(text) > max_chars:
            text = text[:max_chars].rsplit(" ", 1)[0] + "..."
            logger.warning(f"[TTS] Text truncated to {max_chars} chars")

        logger.info(
            f"[TTS] Synthesising {len(text)} chars "
            f"via {self._backend}..."
        )
        t0 = time.perf_counter()

        if self._backend == "xtts":
            ref_wav = self._ensure_reference_wav(Path(ref_wav))
            _tts_model.tts_to_file(
                text=            text,
                speaker_wav=     str(ref_wav),
                language=        LANGUAGE,
                file_path=       str(out_path),
            )

        elif self._backend == "pyttsx3":
            # pyttsx3 can save to file via runAndWait + save_to_file
            self._pyttsx3_engine.save_to_file(text, str(out_path))
            self._pyttsx3_engine.runAndWait()

        elapsed = time.perf_counter() - t0
        logger.info(
            f"[TTS] Done in {elapsed:.2f}s  "
            f"backend={self._backend}  "
            f"output={out_path}"
        )

        return out_path

    def get_backend(self) -> str | None:
        return self._backend


# ── Module-level convenience function ─────────────────────────────────────────

def speak(
    text:     str,
    out_path: str | Path = DEFAULT_OUT_PATH,
    ref_wav:  str | Path = DEFAULT_REF_WAV,
) -> Path:
    """
    Convert text to speech — convenience wrapper.

    Args:
        text:     Text to synthesise
        out_path: Output WAV path
        ref_wav:  Speaker reference WAV (for XTTS voice cloning)

    Returns:
        Path to output WAV file
    """
    return TTSService().speak(text, out_path=out_path, ref_wav=ref_wav)