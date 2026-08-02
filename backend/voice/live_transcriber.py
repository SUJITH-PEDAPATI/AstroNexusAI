"""
AstroNexus AI — Live Chunked Transcriber
==========================================
Implements chunked audio recording with rolling Whisper transcription,
producing a live-updating partial transcript as the user speaks.

Architecture:
    sounddevice.InputStream (callback)
          ↓
    audio chunks (configurable window, e.g. 2s)
          ↓
    WhisperService.transcribe() on each chunk
          ↓
    DeduplicatingMerger (removes repeated words at chunk boundaries)
          ↓
    on_partial(text)  ← called after every chunk

    Final call to transcribe(full_recording) produces the clean transcript.

Design notes:
    • openai-whisper does NOT support streaming audio natively.
      We implement chunk-based transcription: record N seconds, run Whisper,
      emit the new words, then continue recording the next chunk.
    • The DeduplicatingMerger handles the boundary overlap problem where
      Whisper repeats words across chunk boundaries.
    • This architecture is future-ready: swap transcribe_chunk() with a
      faster-whisper / WhisperX streaming call when available.
    • Thread-safe: recording runs in a daemon thread; callbacks are delivered
      from the main thread after each chunk completes.

Dependencies:
    pip install sounddevice soundfile openai-whisper numpy
"""
from __future__ import annotations

import io
import logging
import os
import tempfile
import threading
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────

SAMPLE_RATE        = int(os.environ.get("VOICE_SAMPLE_RATE",    "16000"))
CHANNELS           = int(os.environ.get("VOICE_CHANNELS",       "1"))
CHUNK_SECONDS      = float(os.environ.get("VOICE_CHUNK_SECS",   "2.0"))   # how often to transcribe
SILENCE_THRESHOLD  = float(os.environ.get("VOICE_SILENCE_THR",  "0.005")) # RMS below = silence
SILENCE_GAP_SECS   = float(os.environ.get("VOICE_SILENCE_GAP",  "2.0"))   # seconds of silence → auto-stop
MAX_DURATION_SECS  = float(os.environ.get("VOICE_MAX_DURATION", "60.0"))  # safety limit

# ── Deduplicating word merger ─────────────────────────────────────────────────

class _Merger:
    """
    Merges chunk transcripts while suppressing repeated words at boundaries.

    Example:
        chunk 1 → "The James Webb"
        chunk 2 → "Webb Space Telescope"   ← "Webb" is repeated
        merged  → "The James Webb Space Telescope"

    Algorithm: compare the tail of the accumulated text with the head of the
    new chunk and remove the longest matching overlap.
    """

    def __init__(self) -> None:
        self._words: list[str] = []

    @property
    def text(self) -> str:
        return " ".join(self._words)

    def add(self, chunk_text: str) -> str:
        """
        Merge a new chunk into the accumulated transcript.

        Returns:
            Only the NEW words added (useful for streaming display).
        """
        new_words = chunk_text.strip().split()
        if not new_words:
            return ""

        if not self._words:
            self._words.extend(new_words)
            return chunk_text.strip()

        # Find the longest suffix of self._words that is a prefix of new_words
        overlap = 0
        tail    = self._words[-min(8, len(self._words)):]  # look back up to 8 words
        for n in range(min(len(tail), len(new_words)), 0, -1):
            if tail[-n:] == new_words[:n]:
                overlap = n
                break

        added = new_words[overlap:]
        self._words.extend(added)
        logger.debug(f"[Merger] overlap={overlap} added={added}")
        return " ".join(added)

    def reset(self) -> None:
        self._words.clear()


# ── Live transcriber ──────────────────────────────────────────────────────────

class LiveTranscriber:
    """
    Records audio from the default microphone in chunks.
    After each chunk, runs Whisper and emits the new words via on_partial().
    After stop(), runs Whisper on the full recording for a clean final transcript.

    Usage (blocking — press-Enter mode):
        def show(text):
            print("\\r" + text, end="", flush=True)

        lt = LiveTranscriber(on_partial=show)
        lt.start()
        input()           # user presses Enter when done
        result = lt.stop()
        print("\\nFinal:", result.final_transcript)

    Usage (fixed duration):
        lt = LiveTranscriber(on_partial=show, duration=5.0)
        result = lt.start_and_wait()
        print("Final:", result.final_transcript)
    """

    def __init__(
        self,
        on_partial:       Callable[[str], None] | None = None,
        duration:         float | None = None,
        chunk_seconds:    float = CHUNK_SECONDS,
        sample_rate:      int   = SAMPLE_RATE,
        channels:         int   = CHANNELS,
        auto_stop_silence: float = SILENCE_GAP_SECS,
    ) -> None:
        self._on_partial       = on_partial or (lambda t: None)
        self._duration         = duration
        self._chunk_secs       = chunk_seconds
        self._sr               = sample_rate
        self._ch               = channels
        self._auto_stop        = auto_stop_silence

        # Recording state
        self._frames:   list[np.ndarray] = []
        self._chunk_buf: list[np.ndarray] = []
        self._merger     = _Merger()
        self._running    = False
        self._lock       = threading.Lock()

        # Silence tracking
        self._last_sound_time: float = 0.0

        # Result
        self._result: Optional[TranscriptionResult] = None

    # ── Public API ────────────────────────────────────────────────────────────

    def start(self) -> None:
        """
        Begin recording and live transcription in a background thread.
        Call stop() to finish, or wait for auto-stop on silence/duration.
        """
        self._running        = True
        self._frames         = []
        self._chunk_buf      = []
        self._last_sound_time = time.perf_counter()
        self._merger.reset()

        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        logger.info("[LiveTranscriber] Recording started")

    def stop(self) -> "TranscriptionResult":
        """
        Stop recording, transcribe the full audio, return the result.
        """
        self._running = False
        if hasattr(self, "_thread"):
            self._thread.join(timeout=5)
        result = self._build_result()
        logger.info(f"[LiveTranscriber] Stopped — {len(self._frames)} frames")
        return result

    def start_and_wait(self) -> "TranscriptionResult":
        """
        Convenience: start, block until done (duration or silence), return result.
        """
        self.start()
        if self._duration:
            time.sleep(self._duration)
        else:
            # wait until auto-stopped or KeyboardInterrupt
            try:
                while self._running:
                    time.sleep(0.1)
            except KeyboardInterrupt:
                pass
        return self.stop()

    # ── Recording thread ──────────────────────────────────────────────────────

    def _run(self) -> None:
        try:
            import sounddevice as sd
        except ImportError:
            logger.error("[LiveTranscriber] sounddevice not installed")
            return

        chunk_samples = int(self._sr * self._chunk_secs)
        t_start       = time.perf_counter()

        def _callback(indata: np.ndarray, frames: int, time_info, status):
            if status:
                logger.debug(f"[LiveTranscriber] stream status: {status}")
            data = indata.copy()
            with self._lock:
                self._frames.append(data)
                self._chunk_buf.append(data)

            # Update silence tracker
            rms = float(np.sqrt(np.mean(data ** 2)))
            if rms > SILENCE_THRESHOLD:
                self._last_sound_time = time.perf_counter()

        try:
            with sd.InputStream(
                samplerate=self._sr,
                channels=  self._ch,
                dtype=     "float32",
                callback=  _callback,
                blocksize= 1024,
            ):
                while self._running:
                    time.sleep(0.05)

                    elapsed = time.perf_counter() - t_start
                    if elapsed > MAX_DURATION_SECS:
                        logger.info("[LiveTranscriber] Max duration reached")
                        self._running = False
                        break

                    # Auto-stop on silence
                    silence_dur = time.perf_counter() - self._last_sound_time
                    if silence_dur > self._auto_stop and elapsed > self._chunk_secs:
                        logger.info(f"[LiveTranscriber] Silence {silence_dur:.1f}s → auto-stop")
                        self._running = False
                        break

                    # Transcribe chunk when enough audio is buffered
                    with self._lock:
                        buf_samples = sum(len(f) for f in self._chunk_buf)

                    if buf_samples >= chunk_samples:
                        self._transcribe_chunk()

        except Exception:
            logger.exception("[LiveTranscriber] Stream error")

    def _transcribe_chunk(self) -> None:
        """Transcribe the current chunk buffer and emit partial text."""
        with self._lock:
            chunk = self._chunk_buf[:]
            self._chunk_buf.clear()

        if not chunk:
            return

        audio = np.concatenate(chunk, axis=0)
        rms   = float(np.sqrt(np.mean(audio ** 2)))
        if rms < SILENCE_THRESHOLD:
            logger.debug("[LiveTranscriber] Chunk is silent — skipping")
            return

        try:
            chunk_text = _whisper_numpy(audio, self._sr)
            if chunk_text.strip():
                new_words = self._merger.add(chunk_text)
                if new_words:
                    logger.debug(f"[LiveTranscriber] +words: {new_words!r}")
                    self._on_partial(self._merger.text)
        except Exception:
            logger.warning("[LiveTranscriber] Chunk transcription failed", exc_info=True)

    # ── Final result ──────────────────────────────────────────────────────────

    def _build_result(self) -> "TranscriptionResult":
        """Transcribe the full recording for a clean final transcript."""
        with self._lock:
            all_frames = self._frames[:]

        if not all_frames:
            return TranscriptionResult(
                final_transcript="",
                partial_transcript=self._merger.text,
                duration_sec=0.0,
                language="unknown",
            )

        full_audio = np.concatenate(all_frames, axis=0)
        duration   = len(full_audio) / self._sr

        # Full-audio transcription is cleaner than stitched partials
        try:
            full_text, language = _whisper_numpy_full(full_audio, self._sr)
        except Exception:
            logger.warning("[LiveTranscriber] Full transcription failed — using partial")
            full_text = self._merger.text
            language  = "unknown"

        return TranscriptionResult(
            final_transcript=  full_text.strip(),
            partial_transcript= self._merger.text,
            duration_sec=       round(duration, 2),
            language=           language,
        )


# ── Result dataclass ──────────────────────────────────────────────────────────

class TranscriptionResult:
    """
    Holds the output of a live transcription session.
    future-ready: add word_timestamps, confidence, segments as needed.
    """
    __slots__ = ("final_transcript", "partial_transcript", "duration_sec", "language")

    def __init__(
        self,
        final_transcript:   str,
        partial_transcript: str,
        duration_sec:       float,
        language:           str,
    ) -> None:
        self.final_transcript   = final_transcript
        self.partial_transcript = partial_transcript
        self.duration_sec       = duration_sec
        self.language           = language

    def __repr__(self) -> str:
        return (
            f"TranscriptionResult("
            f"duration={self.duration_sec:.1f}s "
            f"lang={self.language!r} "
            f"text={self.final_transcript[:40]!r})"
        )


# ── Whisper helpers (operate on numpy arrays — no temp file needed) ───────────

def _whisper_numpy(audio: np.ndarray, sr: int) -> str:
    """
    Transcribe a numpy float32 array without saving to disk.
    Whisper's load_audio() accepts a filepath; we write a minimal WAV
    to a BytesIO buffer and pass it as a named temp file.
    """
    import whisper
    from backend.voice.whisper_service import WhisperService

    svc = WhisperService()
    svc._load()  # ensures _model is loaded

    # Whisper expects a 1-D float32 array at its internal sample rate (16 kHz)
    mono = audio[:, 0] if audio.ndim == 2 else audio
    if sr != 16000:
        # Simple linear resampling (sounddevice already provides 16 kHz)
        import numpy as np
        ratio = 16000 / sr
        n_out = int(len(mono) * ratio)
        mono  = np.interp(
            np.linspace(0, len(mono) - 1, n_out),
            np.arange(len(mono)),
            mono,
        ).astype(np.float32)

    # Access the underlying whisper model directly (no file I/O)
    from backend.voice.whisper_service import _model
    result = _model.transcribe(mono, fp16=False, verbose=False)
    return result.get("text", "").strip()


def _whisper_numpy_full(audio: np.ndarray, sr: int) -> tuple[str, str]:
    """Like _whisper_numpy but also returns detected language."""
    import soundfile as sf
    from backend.voice.whisper_service import WhisperService

    svc = WhisperService()
    svc._load()

    mono = audio[:, 0] if audio.ndim == 2 else audio

    # Write temp WAV so WhisperService.transcribe() path validation passes
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        sf.write(str(tmp_path), mono, sr, subtype="PCM_16")
        result = svc.transcribe(tmp_path)
        return result.get("text", "").strip(), result.get("language", "unknown")
    finally:
        tmp_path.unlink(missing_ok=True)