"""
AstroNexus AI — Live Voice Pipeline
=====================================
Orchestrates the complete live voice workflow:

    🎤 Microphone
          ↓
    LiveTranscriber  (chunked Whisper, partial callbacks)
          ↓
    KeywordExtractor  (pattern → spaCy → LLM fallback)
          ↓
    VoiceGraphStore   (MERGE entities into Neo4j)
          ↓
    orchestrator.run()  (existing RAG + KG + multi-agent)
          ↓
    VoicePipelineResult (printed to terminal)

This module is the single entry point for all voice features.
It does NOT modify WhisperService, orchestrator, retriever,
neo4j_client, or any other existing module.

Future hooks (no refactoring needed):
    • Voice Activity Detection:  inject a VAD gate before LiveTranscriber.start()
    • Wake word:                 call start() only after wake word detected
    • Streaming Whisper:         swap _whisper_numpy() in live_transcriber.py
    • Streaming LLM:             add on_token callback to PipelineConfig
    • Streaming TTS:             await TTS as LLM tokens arrive

Usage:
    # Terminal test — blocking
    python -m backend.voice.live_voice_pipeline

    # Programmatic
    from backend.voice.live_voice_pipeline import run_voice_pipeline, PipelineConfig
    result = run_voice_pipeline(PipelineConfig(duration=5.0))
    print(result.final_answer)
"""
from __future__ import annotations

import logging
import sys
import textwrap
import time
import uuid
from dataclasses import dataclass, field
from typing import Callable, Optional

logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────

@dataclass
class PipelineConfig:
    """
    All tuneable knobs in one place.
    Override any field for different contexts (CLI, API, tests).
    """
    # Recording
    duration:          Optional[float] = None    # None = press-Enter mode
    sample_rate:       int             = 16_000
    channels:          int             = 1
    chunk_seconds:     float           = 2.0     # live transcription cadence
    auto_stop_silence: float           = 2.5     # seconds of silence → stop

    # Keyword extraction
    use_llm_fallback:  bool            = True

    # RAG
    run_rag:           bool            = True    # set False to skip orchestrator

    # Callbacks (all optional)
    on_partial:        Optional[Callable[[str], None]] = None   # live text update
    on_stage:          Optional[Callable[[str], None]] = None   # pipeline stage log

    # Output
    paper_id:          Optional[str]   = None   # pre-loaded paper for RAG context
    session_id:        Optional[str]   = field(default_factory=lambda: str(uuid.uuid4())[:8])


# ── Result ────────────────────────────────────────────────────────────────────

@dataclass
class VoicePipelineResult:
    """Complete output of one voice interaction."""
    session_id:       str
    transcript:       str
    language:         str
    duration_sec:     float
    keywords:         list[str]        = field(default_factory=list)
    entities:         list[str]        = field(default_factory=list)
    scientific:       list[str]        = field(default_factory=list)
    graph_status:     dict             = field(default_factory=dict)
    rag_context:      str              = ""
    final_answer:     str              = ""
    query_type:       str              = ""
    error:            Optional[str]    = None
    timing:           dict             = field(default_factory=dict)


# ── Pipeline ──────────────────────────────────────────────────────────────────

def run_voice_pipeline(cfg: PipelineConfig = PipelineConfig()) -> VoicePipelineResult:
    """
    Run the complete voice pipeline end-to-end.

    Stages:
        1. Record audio with live partial transcription
        2. Final clean transcription (full audio)
        3. Keyword / entity extraction
        4. Neo4j knowledge graph update
        5. Existing orchestrator (RAG + multi-agent + LLM)

    Args:
        cfg: PipelineConfig with all settings

    Returns:
        VoicePipelineResult with all outputs
    """
    session_id = cfg.session_id or str(uuid.uuid4())[:8]
    timings: dict[str, float] = {}

    def _stage(name: str) -> None:
        if cfg.on_stage:
            cfg.on_stage(name)
        logger.info(f"[VoicePipeline] ── {name}")

    # ══════════════════════════════════════════════════════════════════════════
    # Stage 1 — Live recording + partial transcription
    # ══════════════════════════════════════════════════════════════════════════

    _stage("Recording")
    t0 = time.perf_counter()

    try:
        from backend.voice.live_transcriber import LiveTranscriber, TranscriptionResult

        partial_buf: list[str] = []

        def _on_partial(text: str) -> None:
            partial_buf.append(text)
            if cfg.on_partial:
                cfg.on_partial(text)
            else:
                # Default: overwrite current line
                print(f"\r  {text}", end="", flush=True)

        lt = LiveTranscriber(
            on_partial=        _on_partial,
            duration=          cfg.duration,
            chunk_seconds=     cfg.chunk_seconds,
            sample_rate=       cfg.sample_rate,
            channels=          cfg.channels,
            auto_stop_silence= cfg.auto_stop_silence,
        )

        lt.start()

        if cfg.duration:
            time.sleep(cfg.duration + 0.5)  # wait for duration + margin
        else:
            try:
                input()    # blocks until Enter key
            except (EOFError, KeyboardInterrupt):
                pass

        result: TranscriptionResult = lt.stop()

    except ImportError as e:
        logger.error(f"[VoicePipeline] LiveTranscriber unavailable: {e}")
        logger.info("[VoicePipeline] Falling back to non-live transcription")
        result = _fallback_record(cfg)

    timings["recording"] = time.perf_counter() - t0

    transcript   = result.final_transcript
    language     = result.language
    duration_sec = result.duration_sec

    if not transcript:
        return VoicePipelineResult(
            session_id=   session_id,
            transcript=   "",
            language=     language,
            duration_sec= duration_sec,
            error=        "Empty transcript — no speech detected",
            timing=       timings,
        )

    logger.info(f"[VoicePipeline] Transcript ({len(transcript)} chars): {transcript[:80]!r}")

    # ══════════════════════════════════════════════════════════════════════════
    # Stage 2 — Keyword & entity extraction
    # ══════════════════════════════════════════════════════════════════════════

    _stage("Extracting keywords and entities")
    t1 = time.perf_counter()

    try:
        from backend.voice.keyword_extractor import extract, KeywordResult
        kw_result: KeywordResult = extract(transcript, use_llm_fallback=cfg.use_llm_fallback)
        logger.info(f"[VoicePipeline] Extraction: {kw_result.summary()}")
    except Exception as e:
        logger.warning(f"[VoicePipeline] Extraction failed (non-fatal): {e}")
        kw_result = _empty_keywords()

    timings["extraction"] = time.perf_counter() - t1

    # ══════════════════════════════════════════════════════════════════════════
    # Stage 3 — Neo4j knowledge graph update
    # ══════════════════════════════════════════════════════════════════════════

    _stage("Updating Knowledge Graph")
    t2 = time.perf_counter()
    graph_status: dict = {}

    try:
        from backend.voice.voice_graph_store import store_voice_session
        graph_status = store_voice_session(
            session_id=   session_id,
            transcript=   transcript,
            keywords=     kw_result,
            duration_sec= duration_sec,
            language=     language,
        )
    except Exception as e:
        logger.warning(f"[VoicePipeline] Graph store failed (non-fatal): {e}")
        graph_status = {"status": "skipped", "error": str(e)}

    timings["graph"] = time.perf_counter() - t2

    # ══════════════════════════════════════════════════════════════════════════
    # Stage 4 — Existing RAG + multi-agent pipeline
    # ══════════════════════════════════════════════════════════════════════════

    final_answer = ""
    rag_context  = ""
    query_type   = ""

    if cfg.run_rag and transcript:
        _stage("Running RAG pipeline")
        t3 = time.perf_counter()

        try:
            from backend.agents.orchestrator import run
            orch_result = run(
                query=        transcript,
                paper_loaded= bool(cfg.paper_id),
                paper_id=     cfg.paper_id,
            )
            final_answer = orch_result.get("final_answer", "")
            rag_context  = orch_result.get("rag_context",  "") or ""
            query_type   = orch_result.get("query_type",   "")
            meta         = orch_result.get("metadata") or {}
            logger.info(f"[VoicePipeline] Orchestrator done: query_type={query_type}")
        except Exception as e:
            logger.error(f"[VoicePipeline] Orchestrator failed: {e}")
            final_answer = f"Pipeline error: {e}"

        timings["rag"] = time.perf_counter() - t3

    timings["total"] = sum(timings.values())

    return VoicePipelineResult(
        session_id=   session_id,
        transcript=   transcript,
        language=     language,
        duration_sec= duration_sec,
        keywords=     kw_result.keywords,
        entities=     [e.name for e in kw_result.entities],
        scientific=   kw_result.scientific,
        graph_status= graph_status,
        rag_context=  rag_context[:500] if rag_context else "",
        final_answer= final_answer,
        query_type=   query_type,
        timing=       timings,
    )


# ── Fallback: non-live recording when sounddevice is absent ──────────────────

def _fallback_record(cfg: PipelineConfig):
    """
    Non-live fallback: record to WAV using sounddevice directly, then transcribe.
    Used when live_transcriber's chunked mode is unavailable.
    """
    from backend.voice.live_transcriber import TranscriptionResult

    try:
        import numpy as np
        import sounddevice as sd
        import soundfile as sf
        import tempfile
        from pathlib import Path
        from backend.voice.whisper_service import WhisperService

        frames: list[np.ndarray] = []
        recording = True

        def _cb(indata, *_):
            if recording:
                frames.append(indata.copy())

        dur = cfg.duration or 10.0
        with sd.InputStream(samplerate=cfg.sample_rate, channels=cfg.channels,
                             dtype="float32", callback=_cb, blocksize=1024):
            logger.info(f"[VoicePipeline/fallback] Recording {dur}s")
            time.sleep(dur)

        recording = False
        if not frames:
            return TranscriptionResult("", "", 0.0, "unknown")

        audio = np.concatenate(frames, axis=0)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            sf.write(tmp.name, audio, cfg.sample_rate, subtype="PCM_16")
            stt = WhisperService().transcribe(tmp.name)
            Path(tmp.name).unlink(missing_ok=True)

        return TranscriptionResult(
            final_transcript=   stt.get("text", "").strip(),
            partial_transcript= "",
            duration_sec=       len(audio) / cfg.sample_rate,
            language=           stt.get("language", "unknown"),
        )
    except Exception as e:
        logger.error(f"[VoicePipeline/fallback] Failed: {e}")
        return TranscriptionResult("", "", 0.0, "unknown")


def _empty_keywords():
    from backend.voice.keyword_extractor import KeywordResult
    return KeywordResult()


# ── Terminal runner ───────────────────────────────────────────────────────────

def _print_result(r: VoicePipelineResult) -> None:
    """Pretty-print the pipeline result to the terminal."""
    sep  = "=" * 50
    line = "-" * 50

    print(f"\n{sep}")
    print("  AstroNexus AI — Voice Pipeline Result")
    print(sep)

    print(f"\n  Session ID   : {r.session_id}")
    print(f"  Language     : {r.language}")
    print(f"  Duration     : {r.duration_sec:.2f}s")
    print(f"  Query Type   : {r.query_type or 'n/a'}")

    print(f"\n{line}")
    print("  Transcript:")
    print()
    for ln in textwrap.wrap(r.transcript or "(empty)", width=68):
        print(f"    {ln}")

    if r.scientific:
        print(f"\n{line}")
        print("  Scientific Terms:")
        for t in r.scientific:
            print(f"    • {t}")

    if r.entities:
        print(f"\n{line}")
        print("  Named Entities:")
        for e in r.entities:
            print(f"    • {e}")

    if r.keywords:
        print(f"\n{line}")
        print(f"  Keywords ({len(r.keywords)}):")
        print(f"    {', '.join(r.keywords[:12])}")

    print(f"\n{line}")
    print("  Knowledge Graph:")
    gs = r.graph_status
    if gs.get("status") == "ok":
        print(f"    ✓ entities_written={gs.get('entities_written', 0)}")
        print(f"    ✓ keywords_written={gs.get('keywords_written', 0)}")
    else:
        print(f"    ⚠ {gs.get('error', 'skipped')}")

    if r.rag_context:
        print(f"\n{line}")
        print("  RAG Context (preview):")
        for ln in textwrap.wrap(r.rag_context[:300], width=68):
            print(f"    {ln}")

    if r.final_answer:
        print(f"\n{line}")
        print("  Final Response:")
        print()
        for ln in textwrap.wrap(r.final_answer, width=68):
            print(f"    {ln}")

    print(f"\n{line}")
    print("  Timing:")
    for stage, secs in r.timing.items():
        print(f"    {stage:<15} {secs:.2f}s")

    print()
    if r.error:
        print(f"  ⚠  Error: {r.error}")
        print(sep)
    else:
        print(sep)
        print("  ✓  Voice Pipeline Complete")
        print(sep)
    print()


if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=   logging.INFO,
        format=  "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    p = argparse.ArgumentParser(description="AstroNexus AI Live Voice Pipeline")
    p.add_argument("--duration",    type=float, default=None,
                   help="Fixed recording duration in seconds (default: press Enter)")
    p.add_argument("--no-rag",      action="store_true",
                   help="Skip RAG pipeline (transcription + KG only)")
    p.add_argument("--chunk",       type=float, default=2.0,
                   help="Live transcription chunk size in seconds")
    p.add_argument("--paper-id",    type=str,   default=None)
    a = p.parse_args()

    sep = "=" * 50
    print(f"\n{sep}")
    print("  AstroNexus AI — Live Voice Pipeline")
    print(sep)

    if a.duration:
        print(f"\n  Recording for {a.duration:.0f} seconds...")
    else:
        print("\n  Recording...  press Enter to stop")
    print("  Speak now.\n")

    result = run_voice_pipeline(PipelineConfig(
        duration=      a.duration,
        run_rag=       not a.no_rag,
        chunk_seconds= a.chunk,
        paper_id=      a.paper_id,
    ))

    _print_result(result)