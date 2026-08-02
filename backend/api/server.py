"""
AstroNexus AI — FastAPI HTTP Server
====================================
A minimal HTTP wrapper around the existing agent pipeline.
Zero model logic lives here — everything is delegated to
backend.agents.orchestrator.run(), which is exactly what the
terminal CLI (python -m backend.tests.chat) calls.

Start the server from your project root:

    uvicorn backend.api.server:app --reload --port 8000

Then set in astronexus-app/.env.local:

    NEXT_PUBLIC_API_URL=http://localhost:8000
"""
from __future__ import annotations

import logging
import os
import time

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="AstroNexus AI",
    description="Space Research Intelligence Platform",
    version="1.0.0",
)

# CORS — allow the Next.js dev server and any production origin you add
_origins = os.environ.get(
    "CORS_ORIGINS",
    "http://localhost:3000,http://localhost:3001",
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Warm up the pipeline once on startup so the first chat request is fast ────

@app.on_event("startup")
async def _warmup():
    try:
        from backend.agents.orchestrator import get_graph
        get_graph()
        logger.info("[server] Pipeline ready.")
    except Exception as exc:
        # Non-fatal: the first request will warm it up instead
        logger.warning(f"[server] Warmup skipped: {exc}")


# ── Request / Response models ─────────────────────────────────────────────────

class ChatRequest(BaseModel):
    """
    Matches exactly what the frontend sends:
        { "query": "...", "paper_id": "..." }
    paper_id is optional — omit it for general questions.
    conversation_history lets the frontend pass previous turns for
    multi-turn context (the orchestrator already supports this).
    """
    query:                str
    paper_id:             str | None  = None
    conversation_history: list | None = None   # [{turn, query, answer}, ...]


class CitationOut(BaseModel):
    section: str
    page:    int
    score:   float


class ChatResponse(BaseModel):
    """
    Matches what chat.service.ts (frontend) reads:
        { answer, grade, citations }
    Extra fields are included so nothing is thrown away if the
    frontend is later extended to display them.
    """
    answer:      str
    grade:       str          # "A" | "B" | "C"
    citations:   list[CitationOut]
    # ── extras (displayed when frontend is ready) ──
    confidence:      str   = ""
    grounding_score: float = 0.0
    query_type:      str   = ""
    is_reliable:     bool  = False


# ── /chat endpoint ────────────────────────────────────────────────────────────

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """
    Primary chat endpoint consumed by the Next.js frontend.

    Delegates to the same orchestrator.run() the CLI uses, so every
    agent (research, graph, satellite, general) is available.
    """
    logger.info(f"[/chat] query='{req.query[:60]}' paper_id={req.paper_id}")
    t0 = time.perf_counter()

    try:
        from backend.agents.orchestrator import run

        result = run(
            query=                req.query,
            paper_loaded=         bool(req.paper_id),
            paper_id=             req.paper_id,
            conversation_history= req.conversation_history or [],
        )
    except Exception as exc:
        logger.error(f"[/chat] pipeline error: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))

    elapsed = time.perf_counter() - t0
    logger.info(f"[/chat] done in {elapsed:.1f}s")

    answer     = result.get("final_answer") or ""
    meta       = result.get("metadata") or {}
    evaluation = meta.get("evaluation") or {}

    # ── Map confidence → grade (same scale the frontend shows) ───────────────
    confidence = evaluation.get("confidence", "")
    grade_map  = {"HIGH": "A", "MEDIUM": "B", "LOW": "C"}
    grade      = grade_map.get(confidence, "B")

    # ── Normalise citations from the evaluation dict ──────────────────────────
    raw_citations = evaluation.get("citations", [])
    citations: list[CitationOut] = []
    for c in raw_citations:
        if isinstance(c, dict):
            citations.append(CitationOut(
                section=str(c.get("section") or ""),
                page=   int(c.get("page")    or 0),
                score=  float(c.get("score") or 0.0),
            ))

    return ChatResponse(
        answer=          answer,
        grade=           grade,
        citations=       citations,
        confidence=      confidence,
        grounding_score= float(evaluation.get("grounding_score", 0.0)),
        query_type=      str(result.get("query_type") or ""),
        is_reliable=     bool(evaluation.get("is_reliable", False)),
    )



# ── Imports for file handling (voice endpoints need these) ────────────────────

import shutil
import tempfile
import time as _time
import uuid as _uuid
from pathlib import Path

from fastapi import File, Form, UploadFile

_AUDIO_DIR = Path(tempfile.gettempdir()) / "astronexus_audio"
_AUDIO_DIR.mkdir(exist_ok=True)


# ── Voice response models ─────────────────────────────────────────────────────

class VoiceResponse(BaseModel):
    transcript:  str
    answer:      str
    audio_url:   str
    confidence:  str
    is_reliable: bool

class TranscribeResponse(BaseModel):
    success:         bool
    transcript:      str
    language:        str
    duration_sec:    float
    processing_time: float
    model:           str
    device:          str
    audio_format:    str
    audio_size_kb:   float
    segments:        list = []
    error:           str  = ""


# ── POST /voice/chat ──────────────────────────────────────────────────────────

@app.post("/voice/chat", response_model=VoiceResponse)
async def voice_chat(
    audio:    UploadFile    = File(...),
    paper_id: str | None    = Form(None),
):
    """
    Voice note → Whisper STT → orchestrator → edge TTS → MP3.
    Identical pipeline to main.py /voice/chat.
    """
    audio_path = _AUDIO_DIR / f"{_uuid.uuid4()}_{audio.filename or 'recording.webm'}"

    # Save upload
    with open(audio_path, "wb") as f:
        shutil.copyfileobj(audio.file, f)

    # Step 1: Whisper STT
    try:
        from backend.voice.whisper_service import WhisperService
        stt    = WhisperService().transcribe(audio_path)
        transcript = stt["text"]
        logger.info(f"[/voice/chat] transcript: {transcript[:80]}")
    except Exception as e:
        audio_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"STT failed: {e}")

    # Step 2: Keyword extraction → Neo4j (non-fatal)
    try:
        from backend.agents.tier_classifier     import extract_keywords
        from backend.agents.query_keyword_store import store_query_keywords
        kw = extract_keywords(transcript)
        if kw:
            store_query_keywords(query=transcript, keywords=kw, query_type="voice")
    except Exception as e:
        logger.warning(f"[/voice/chat] keyword storage skipped: {e}")

    # Step 3: Research pipeline
    try:
        from backend.agents.orchestrator import run
        result = run(
            query=        transcript,
            paper_loaded= bool(paper_id),
            paper_id=     paper_id,
        )
        answer = result.get("final_answer") or "No answer generated."
        meta   = result.get("metadata") or {}
        evl    = meta.get("evaluation") or {}
    except Exception as e:
        audio_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Research failed: {e}")

    # Step 4: TTS → MP3 (non-fatal)
    audio_url = ""
    try:
        import edge_tts
        out_mp3 = _AUDIO_DIR / f"response_{_uuid.uuid4()}.mp3"
        voice   = os.environ.get("TTS_VOICE", "en-US-AriaNeural")
        await edge_tts.Communicate(answer[:800], voice).save(str(out_mp3))
        audio_url = f"/audio/{out_mp3.name}"
    except Exception as e:
        logger.warning(f"[/voice/chat] TTS skipped: {e}")

    audio_path.unlink(missing_ok=True)

    return VoiceResponse(
        transcript=  transcript,
        answer=      answer,
        audio_url=   audio_url,
        confidence=  evl.get("confidence",  "LOW"),
        is_reliable= evl.get("is_reliable", False),
    )


# ── GET /audio/{filename} ─────────────────────────────────────────────────────

@app.get("/audio/{filename}")
async def serve_audio(filename: str):
    from fastapi.responses import FileResponse
    p = _AUDIO_DIR / filename
    if not p.exists():
        raise HTTPException(status_code=404, detail="Audio file not found")
    return FileResponse(str(p), media_type="audio/mpeg")


# ── POST /voice/transcribe — isolated STT test ───────────────────────────────

@app.post("/voice/transcribe", response_model=TranscribeResponse)
async def voice_transcribe(audio: UploadFile = File(...)):
    """Isolated Whisper STT test — no other services called."""
    audio_path = _AUDIO_DIR / f"test_{_uuid.uuid4()}_{audio.filename or 'rec.webm'}"
    audio_fmt  = Path(audio.filename or "rec.webm").suffix.lstrip(".").upper() or "WEBM"

    with open(audio_path, "wb") as f:
        shutil.copyfileobj(audio.file, f)
    size_kb = audio_path.stat().st_size / 1024

    t0 = _time.time()
    try:
        from backend.voice.whisper_service import WhisperService, WHISPER_MODEL
        result   = WhisperService().transcribe(audio_path)
        elapsed  = _time.time() - t0
        audio_path.unlink(missing_ok=True)
        return TranscribeResponse(
            success=True,
            transcript=      result["text"],
            language=        result.get("language", ""),
            duration_sec=    round(result.get("duration", 0.0), 2),
            processing_time= round(elapsed, 2),
            model=           result.get("model", WHISPER_MODEL),
            device=          result.get("device", "cpu"),
            audio_format=    audio_fmt,
            audio_size_kb=   round(size_kb, 1),
            segments=        result.get("segments", []),
        )
    except Exception as e:
        elapsed = _time.time() - t0
        audio_path.unlink(missing_ok=True)
        return TranscribeResponse(
            success=False, transcript="", language="", duration_sec=0,
            processing_time=round(elapsed, 2), model="", device="",
            audio_format=audio_fmt, audio_size_kb=round(size_kb, 1),
            error=str(e),
        )


# ── /health ───────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    """Quick liveness check — useful for verifying the server is running."""
    return {"status": "ok", "service": "AstroNexus AI"}


# ── Run directly (alternative to uvicorn CLI) ─────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.api.server:app",
        host=   os.environ.get("API_HOST",   "0.0.0.0"),
        port=   int(os.environ.get("API_PORT", "8000")),
        reload= os.environ.get("API_RELOAD", "true").lower() == "true",
    )