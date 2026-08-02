"""
AstroNexus AI — FastAPI Backend

Endpoints:
    POST /upload          — ingest a PDF paper
    POST /caption         — caption a satellite image
    POST /segment         — run SAM2 segmentation
    POST /research        — ask a question about uploaded papers
    POST /graph           — query the knowledge graph
    POST /voice/chat      — voice note → answer → MP3
    GET  /health          — system health check

Run:
    uvicorn backend.api.main:app --reload --port 8000
"""
from __future__ import annotations

import logging
import os
import shutil
import tempfile
import time
import uuid
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# ── Output dirs ───────────────────────────────────────────────────────────────
UPLOAD_DIR = Path("data/uploads")
OUTPUT_DIR = Path("output")
AUDIO_DIR  = Path("output/audio")

for d in [UPLOAD_DIR, OUTPUT_DIR, AUDIO_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
# APP
# ══════════════════════════════════════════════════════════════════════════════

app = FastAPI(
    title=       "AstroNexus AI",
    description= "Hybrid Research Intelligence Platform",
    version=     "1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=     os.environ.get("CORS_ORIGINS", "http://localhost:3000").split(","),
    allow_credentials= True,
    allow_methods=     ["*"],
    allow_headers=     ["*"],
)


# ══════════════════════════════════════════════════════════════════════════════
# RESPONSE MODELS
# ══════════════════════════════════════════════════════════════════════════════

class HealthResponse(BaseModel):
    status:   str
    services: dict
    version:  str = "1.0.0"


class UploadResponse(BaseModel):
    paper_id:        str
    title:           str
    chunks:          int
    authors:         list[str]
    keywords:        list[str]
    domain:          str
    pipeline_stages: list[str]
    message:         str


class ResearchResponse(BaseModel):
    answer:          str
    confidence:      str
    grounding_score: float
    relevance_score: float
    citations:       list[dict]
    warnings:        list[str]
    is_reliable:     bool
    sources:         str
    ollama_used:     bool
    gemini_used:     bool


class GraphResponse(BaseModel):
    answer:  str
    results: list[dict]
    query:   str


class CaptionResponse(BaseModel):
    caption:          str
    detailed_caption: str
    bounding_boxes:   list[dict]
    keywords:         list[str]
    model:            str


class SegmentResponse(BaseModel):
    mask_count: int
    segments:   list[dict]
    viz_path:   str


class VoiceResponse(BaseModel):
    transcript:  str
    answer:      str
    audio_url:   str
    confidence:  str
    is_reliable: bool


# ══════════════════════════════════════════════════════════════════════════════
# HEALTH
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/health", response_model=HealthResponse)
async def health():
    """Check status of all services."""
    services = {}

    # Qdrant
    try:
        from backend.rag.vector_store import get_collection_info
        info = get_collection_info()
        services["qdrant"] = {
            "status": "online",
            "points": info.get("total_points", 0),
        }
    except Exception as e:
        services["qdrant"] = {"status": "offline", "error": str(e)}

    # Neo4j
    try:
        from backend.graph.neo4j_client import get_graph_stats
        stats = get_graph_stats()
        services["neo4j"] = {
            "status": "online",
            "nodes":  stats.get("total_nodes", 0),
        }
    except Exception as e:
        services["neo4j"] = {"status": "offline", "error": str(e)}

    # Ollama
    try:
        import urllib.request
        urllib.request.urlopen(
            f"{os.environ.get('OLLAMA_BASE_URL','http://localhost:11434')}/api/tags",
            timeout=3,
        )
        services["ollama"] = {"status": "online"}
    except Exception:
        services["ollama"] = {"status": "offline"}

    # Gemini
    services["gemini"] = {
        "status": "configured" if os.environ.get("GEMINI_API_KEY") else "not configured"
    }

    overall = (
        "healthy"
        if services.get("qdrant", {}).get("status") == "online"
        and services.get("neo4j", {}).get("status") == "online"
        else "degraded"
    )

    return HealthResponse(status=overall, services=services)


# ══════════════════════════════════════════════════════════════════════════════
# UPLOAD — PDF ingestion
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/upload", response_model=UploadResponse)
async def upload_paper(file: UploadFile = File(...)):
    """
    Ingest a PDF research paper.

    Pipeline:
        PDF → chunks → embeddings → Qdrant
            → entity extraction → Neo4j
            → domain classification → keywords → Neo4j
    """
    if not file.filename.lower().endswith((".pdf", ".docx", ".md", ".txt")):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format: {file.filename}. Supported: PDF, DOCX, MD, TXT",
        )

    # Save upload
    save_path = UPLOAD_DIR / f"{uuid.uuid4()}_{file.filename}"
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    logger.info(f"[API/upload] Saved: {save_path}")

    try:
        # Ingest
        from backend.ingestion.paper_ingestion import ingest_paper
        from backend.ingestion.chunking        import chunk_document
        from backend.embeddings                import embed_chunks
        from backend.rag                       import upsert_chunks
        from backend.graph.graph_builder       import build_graph_from_document

        t0  = time.perf_counter()
        doc = ingest_paper(save_path)

        chunks   = chunk_document(doc)
        embedded = embed_chunks(chunks)
        upsert_chunks(embedded)

        # Graph — entities + keywords
        extraction = build_graph_from_document(doc)

        elapsed = time.perf_counter() - t0
        logger.info(f"[API/upload] Done in {elapsed:.1f}s")

        # Collect keyword names from Neo4j
        keywords = []
        try:
            from backend.graph.neo4j_client import _get_driver
            driver = _get_driver()
            with driver.session() as s:
                rows = s.run(
                    "MATCH (p:Paper {paper_id:$pid})-[:TAGGED]->(k:Keyword) "
                    "RETURN k.name AS kw",
                    pid=doc.paper_id,
                ).data()
            keywords = [r["kw"] for r in rows]
        except Exception:
            pass

        # Domain
        domain = ""
        try:
            from backend.graph.neo4j_client import _get_driver
            driver = _get_driver()
            with driver.session() as s:
                row = s.run(
                    "MATCH (p:Paper {paper_id:$pid})-[:BELONGS_TO]->(d:Domain) "
                    "RETURN d.name AS name LIMIT 1",
                    pid=doc.paper_id,
                ).single()
            domain = row["name"] if row else ""
        except Exception:
            pass

        return UploadResponse(
            paper_id=        doc.paper_id,
            title=           doc.metadata.title or file.filename,
            chunks=          len(chunks),
            authors=         [a.name for a in extraction.authors],
            keywords=        keywords,
            domain=          domain,
            pipeline_stages= ["ingestion","chunking","embedding",
                              "qdrant","entity_extraction","neo4j","keywords"],
            message=         f"Ingested in {elapsed:.1f}s",
        )

    except Exception as e:
        save_path.unlink(missing_ok=True)
        logger.error(f"[API/upload] Failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ══════════════════════════════════════════════════════════════════════════════
# CAPTION — satellite image
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/caption", response_model=CaptionResponse)
async def caption_image(file: UploadFile = File(...)):
    """
    Caption a satellite image via Gemini Vision.
    Extracts keywords and writes them to Neo4j.
    """
    suffix    = Path(file.filename).suffix.lower()
    save_path = UPLOAD_DIR / f"{uuid.uuid4()}{suffix}"

    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        from backend.vision.florence2_captioner import caption_image as cap

        result = cap(save_path, write_to_graph=True)

        keywords = getattr(result, "image_keywords", [])

        return CaptionResponse(
            caption=          result.caption,
            detailed_caption= result.detailed_caption or "",
            bounding_boxes=   result.bounding_boxes,
            keywords=         keywords,
            model=            result.model_name,
        )
    except Exception as e:
        logger.error(f"[API/caption] Failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ══════════════════════════════════════════════════════════════════════════════
# SEGMENT — SAM2
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/segment", response_model=SegmentResponse)
async def segment_image(file: UploadFile = File(...)):
    """Run SAM2 automatic segmentation on a satellite image."""
    suffix    = Path(file.filename).suffix.lower()
    save_path = UPLOAD_DIR / f"{uuid.uuid4()}{suffix}"

    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        from backend.vision.sam2_segmentor import segment_image as seg

        viz_path = OUTPUT_DIR / f"segmented_{save_path.stem}.png"
        result   = seg(save_path, save_viz=True, viz_path=viz_path)

        return SegmentResponse(
            mask_count= result["mask_count"],
            segments=   result["segments"],
            viz_path=   str(viz_path),
        )
    except Exception as e:
        logger.error(f"[API/segment] Failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ══════════════════════════════════════════════════════════════════════════════
# RESEARCH — Q&A over uploaded papers
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/research", response_model=ResearchResponse)
async def research(
    query:      str           = Form(...),
    paper_id:   Optional[str] = Form(None),
    image_path: Optional[str] = Form(None),
):
    """
    Answer a question using the full pipeline:
    Qdrant + Neo4j + Knowledge Fusion + Ollama + Gemini + Evaluator
    """
    try:
        from backend.agents.orchestrator import run

        result = run(
            query=        query,
            paper_loaded= bool(paper_id),
            paper_id=     paper_id,
            image_path=   image_path,
        )

        meta  = result.get("metadata") or {}
        evl   = meta.get("evaluation") or {}

        return ResearchResponse(
            answer=          result.get("final_answer", ""),
            confidence=      evl.get("confidence",      "LOW"),
            grounding_score= evl.get("grounding_score", 0.0),
            relevance_score= evl.get("relevance_score", 0.0),
            citations=       evl.get("citations",       []),
            warnings=        evl.get("warnings",        []),
            is_reliable=     evl.get("is_reliable",     False),
            sources=         result.get("rag_context",  "")[:500],
            ollama_used=     meta.get("ollama_used",    False),
            gemini_used=     meta.get("gemini_used",    False),
        )
    except Exception as e:
        logger.error(f"[API/research] Failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ══════════════════════════════════════════════════════════════════════════════
# GRAPH — knowledge graph queries
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/graph", response_model=GraphResponse)
async def graph_query(query: str = Form(...)):
    """Query the Neo4j knowledge graph."""
    try:
        from backend.agents.orchestrator import run

        result = run(query=query, paper_loaded=False)

        return GraphResponse(
            answer=  result.get("final_answer", ""),
            results= [],
            query=   query,
        )
    except Exception as e:
        logger.error(f"[API/graph] Failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ══════════════════════════════════════════════════════════════════════════════
# VOICE CHAT
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/voice/chat", response_model=VoiceResponse)
async def voice_chat(
    audio:    UploadFile      = File(...),
    paper_id: Optional[str]   = Form(None),
):
    """
    Voice note → STT → Research pipeline → TTS → MP3

    Flow:
        1. Save audio
        2. Whisper STT → transcript
        3. Research agent (Qdrant + Neo4j + Ollama + Gemini + Evaluator)
        4. Edge TTS → MP3
        5. Return transcript + answer + audio URL
    """
    audio_path = AUDIO_DIR / f"{uuid.uuid4()}_{audio.filename}"
    with open(audio_path, "wb") as f:
        shutil.copyfileobj(audio.file, f)

    # Step 1: Transcribe
    try:
        from backend.voice.whisper_service import WhisperService
        stt_result = WhisperService().transcribe(audio_path)
        transcript = stt_result["text"]
        logger.info(f"[API/voice] Transcript: {transcript}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"STT failed: {e}")

    # Step 2: Extract keywords → store in Neo4j Knowledge Graph
    try:
        from backend.agents.tier_classifier     import extract_keywords
        from backend.agents.query_keyword_store import store_query_keywords
        keywords = extract_keywords(transcript)
        if keywords:
            store_query_keywords(
                query=      transcript,
                keywords=   keywords,
                query_type= "voice",
            )
            logger.info(f"[API/voice] Stored {len(keywords)} keywords: {keywords[:5]}")
    except Exception as e:
        logger.warning(f"[API/voice] Keyword storage skipped: {e}")

    # Step 3: Research pipeline
    try:
        from backend.agents.orchestrator import run
        result = run(
            query=        transcript,
            audio_path=   str(audio_path),
            paper_loaded= bool(paper_id),
            paper_id=     paper_id,
        )
        answer = result.get("final_answer", "No answer generated.")
        meta   = result.get("metadata") or {}
        evl    = meta.get("evaluation") or {}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Research failed: {e}")

    # Step 3: TTS → MP3
    audio_out_url = ""
    try:
        import edge_tts, asyncio

        out_mp3 = AUDIO_DIR / f"response_{uuid.uuid4()}.mp3"
        voice   = os.environ.get("TTS_VOICE", "en-US-AriaNeural")

        communicate = edge_tts.Communicate(answer[:800], voice)
        await communicate.save(str(out_mp3))

        audio_out_url = f"/audio/{out_mp3.name}"
        logger.info(f"[API/voice] TTS saved: {out_mp3}")
    except Exception as e:
        logger.warning(f"[API/voice] TTS failed (non-fatal): {e}")

    return VoiceResponse(
        transcript=  transcript,
        answer=      answer,
        audio_url=   audio_out_url,
        confidence=  evl.get("confidence",  "LOW"),
        is_reliable= evl.get("is_reliable", False),
    )


# ══════════════════════════════════════════════════════════════════════════════
# AUDIO FILE SERVE
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/audio/{filename}")
async def serve_audio(filename: str):
    """Serve generated TTS audio files."""
    path = AUDIO_DIR / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="Audio file not found")
    return FileResponse(path, media_type="audio/mpeg")


# ══════════════════════════════════════════════════════════════════════════════
# CHAT — JSON endpoint consumed by the Next.js frontend
# ══════════════════════════════════════════════════════════════════════════════

class ChatRequest(BaseModel):
    """
    Frontend sends:  POST /chat
                     Content-Type: application/json
                     { "query": "...", "paper_id": null, "conversation_history": null }
    """
    query:                str
    paper_id:             Optional[str] = None
    conversation_history: Optional[list] = None


class ChatResponse(BaseModel):
    answer:      str
    grade:       str         # "A" | "B" | "C"
    citations:   list[dict]
    confidence:  str   = ""
    is_reliable: bool  = False
    search_type: str   = "local_rag"   # local_rag | web_search | hybrid
    search_label: str  = "📄 Research Papers"
    web_sources: list[dict] = []       # [{title, url, source, snippet}]


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """
    Primary chat endpoint for the web frontend.
    Accepts JSON (not Form), delegates to the same orchestrator pipeline
    as /research — no model logic is duplicated.
    """
    try:
        from backend.agents.orchestrator import run

        result = run(
            query=                req.query,
            paper_loaded=         bool(req.paper_id),
            paper_id=             req.paper_id,
            conversation_history= req.conversation_history or [],
        )

        meta  = result.get("metadata") or {}
        evl   = meta.get("evaluation") or {}
        conf  = evl.get("confidence", "LOW")

        grade_map = {"HIGH": "A", "MEDIUM": "B", "LOW": "C"}

        search_type  = meta.get("search_type",  "local_rag")
        search_label = meta.get("search_label", "📄 Research Papers")
        web_sources  = meta.get("web_sources",  [])

        return ChatResponse(
            answer=       result.get("final_answer", ""),
            grade=        grade_map.get(conf, "B"),
            citations=    evl.get("citations", []),
            confidence=   conf,
            is_reliable=  evl.get("is_reliable", False),
            search_type=  search_type,
            search_label= search_label,
            web_sources=  web_sources,
        )
    except Exception as e:
        logger.error(f"[API/chat] Failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ══════════════════════════════════════════════════════════════════════════════
# WEB SEARCH — direct test endpoint
# ══════════════════════════════════════════════════════════════════════════════

class WebSearchRequest(BaseModel):
    query:    str
    n:        int = 5

class WebSearchResponse(BaseModel):
    results:  list[dict]
    provider: str
    count:    int

@app.post("/web-search", response_model=WebSearchResponse)
async def web_search_direct(req: WebSearchRequest):
    """
    Direct web search endpoint — bypasses the agent pipeline.
    Useful for testing the search integration independently.
    """
    try:
        from backend.services.web_search import web_search_service
        results = web_search_service.search(req.query, n=req.n)
        return WebSearchResponse(
            results=[
                {
                    "title":   r.title,
                    "url":     r.url,
                    "snippet": r.snippet,
                    "source":  r.source,
                    "score":   r.score,
                    "date":    r.date,
                }
                for r in results
            ],
            provider= web_search_service.provider,
            count=    len(results),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



# ══════════════════════════════════════════════════════════════════════════════
# VOICE TRANSCRIBE — isolated STT test endpoint
# Only WhisperService is called. No RAG, Neo4j, LangGraph, LLM, or TTS.
# ══════════════════════════════════════════════════════════════════════════════

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
    segments:        list[dict] = []
    error:           str        = ""


@app.post("/voice/transcribe", response_model=TranscribeResponse)
async def voice_transcribe(audio: UploadFile = File(...)):
    """
    Isolated STT test endpoint.

    Accepts any audio file, passes it directly to WhisperService,
    and returns ONLY the transcript. No other service is called.

    Useful for verifying:  🎤 → MediaRecorder → Backend → Whisper → Transcript
    """
    size_bytes = 0
    audio_path = AUDIO_DIR / f"test_{uuid.uuid4()}_{audio.filename or 'recording.webm'}"

    print("\n" + "="*44)
    print(" Voice Test Started")
    print("="*44)

    # ── Save upload ────────────────────────────────────────────────────────────
    try:
        with open(audio_path, "wb") as f:
            shutil.copyfileobj(audio.file, f)
        size_bytes = audio_path.stat().st_size
        audio_fmt  = audio_path.suffix.lstrip(".").upper() or "WEBM"

        print(f" Audio Received")
        print(f"   Filename : {audio_path.name}")
        print(f"   Format   : {audio_fmt}")
        print(f"   Size     : {size_bytes/1024:.1f} KB")
    except Exception as e:
        logger.error(f"[/voice/transcribe] Save failed: {e}")
        return TranscribeResponse(
            success=False, transcript="", language="", duration_sec=0,
            processing_time=0, model="", device="", audio_format="",
            audio_size_kb=0, error=f"Could not save audio: {e}",
        )

    # ── Whisper transcription (ONLY service called) ────────────────────────────
    t0 = time.time()
    try:
        print("\n Loading Whisper...")
        from backend.voice.whisper_service import WhisperService, WHISPER_MODEL
        svc    = WhisperService()
        result = svc.transcribe(audio_path)

        elapsed    = time.time() - t0
        transcript = result["text"]
        language   = result.get("language", "unknown")
        duration   = result.get("duration",  0.0)
        segments   = result.get("segments",  [])
        model      = result.get("model",     WHISPER_MODEL)
        device     = result.get("device",    "cpu")

        print("\n Transcribing complete")
        print(f"   Detected language : {language}")
        print(f"   Audio duration    : {duration:.2f}s")
        print(f"   Processing time   : {elapsed:.2f}s")
        print(f"   Whisper model     : {model}  [{device.upper()}]")
        print(f"\n Transcript:")
        print(f"   {transcript}")
        print("\n" + "="*44)
        print(" Voice Test Complete")
        print("="*44 + "\n")

        return TranscribeResponse(
            success=         True,
            transcript=      transcript,
            language=        language,
            duration_sec=    round(duration,   2),
            processing_time= round(elapsed,    2),
            model=           model,
            device=          device,
            audio_format=    audio_fmt,
            audio_size_kb=   round(size_bytes / 1024, 1),
            segments=        segments,
        )

    except Exception as e:
        elapsed = time.time() - t0
        logger.error(f"[/voice/transcribe] Whisper failed: {e}")
        print(f"\n ERROR: {e}")
        print("="*44 + "\n")
        return TranscribeResponse(
            success=False, transcript="", language="", duration_sec=0,
            processing_time=round(elapsed, 2), model="", device="",
            audio_format=audio_fmt, audio_size_kb=round(size_bytes/1024, 1),
            error=str(e),
        )
    finally:
        # Always clean up the temp file after transcription
        try:
            audio_path.unlink(missing_ok=True)
        except Exception:
            pass

# ══════════════════════════════════════════════════════════════════════════════
# RUN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.api.main:app",
        host=    os.environ.get("API_HOST", "0.0.0.0"),
        port=    int(os.environ.get("API_PORT", "8000")),
        reload=  os.environ.get("API_RELOAD", "false").lower() == "true",
    )