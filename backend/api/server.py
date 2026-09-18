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

# CORS — open to all origins.
#
# Authentication has been removed from AstroNexusAI, so there are no
# session cookies or credentials that need strict origin protection.
# Using allow_origins=["*"] + allow_credentials=False is the correct
# configuration: it allows any frontend origin (localhost:3000, any port,
# any hostname) without the strict origin-matching requirement that caused
# the OPTIONS 400 errors when allow_credentials=True was set.
#
# Previous bug: allow_credentials=True forces the browser to send a CORS
# preflight for every custom header (e.g. Content-Type: application/json).
# FastAPI's CORSMiddleware rejects the preflight with 400 if the Origin
# doesn't exactly match allow_origins — even with correct origins listed,
# subtle mismatches (port variations, IP vs hostname) caused failures.

logger.info("[CORS] Open to all origins (credentials=False)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Timing"],
)

# ── Warm up the pipeline once on startup so the first chat request is fast ────

@app.on_event("startup")
async def _warmup():
    """
    Pre-load the LangGraph pipeline and embedding model at startup.
    This moves the ~60s Ollama model-load cost to startup time instead of
    making the first user request wait for it.
    """
    import asyncio

    # 1. Compile the LangGraph (fast — just Python graph construction)
    try:
        from backend.agents.orchestrator import get_graph
        get_graph()
        logger.info("[server] LangGraph pipeline ready.")
    except Exception as exc:
        logger.warning(f"[server] LangGraph warmup skipped: {exc}")

    # 2. Warm up the embedding model in a background thread so startup
    #    doesn't block (Ollama model load can take 30-90s on first call).
    #    The model is a singleton; once loaded it stays in memory.
    async def _warm_embedder():
        try:
            import asyncio
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, _load_embedder)
        except Exception as exc:
            logger.warning(f"[server] Embedder warmup skipped: {exc}")

    asyncio.create_task(_warm_embedder())


def _load_embedder():
    """Load the embedding model singleton (runs in a thread pool)."""
    try:
        from backend.embeddings.embedder import embed_query
        embed_query("warmup")   # triggers _get_model() singleton init
        logger.info("[server] Embedding model warmed up.")
    except Exception as exc:
        logger.warning(f"[server] Embedder warmup failed: {exc}")


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
        { answer, grade, citations, search_type, search_label, web_sources }
    """
    answer:       str
    grade:        str                  # "A" | "B" | "C"
    citations:    list[CitationOut]
    search_type:  str             = "local_rag"
    search_label: str             = "📄 Research Papers"
    web_sources:  list[dict]      = []
    # ── extras ──
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

    # ── Write topic subgraph to Neo4j (non-fatal) ────────────────────────────
    # Every text query creates/updates a topic subgraph.
    # Same topic is reused when follow-up queries share top keywords.
    try:
        try:
            from backend.agents.topic_graph_store import store_query_topic
        except ImportError:
            from topic_graph_store import store_query_topic
        import uuid as _uuid_mod

        # Extract paper_ids from Qdrant result if available in the response
        _paper_ids = [req.paper_id] if req.paper_id else []
        _rag_ctx = result.get("rag_context", "") or ""
        # Also try to extract paper_ids from the retrieved context metadata
        if hasattr(result, "get"):
            _metadata = result.get("metadata") or {}
            _retrieved_pids = _metadata.get("paper_ids", [])
            if _retrieved_pids:
                _paper_ids = list(set(_paper_ids + _retrieved_pids))

        _topic_result = store_query_topic(
            query=      req.query,
            session_id= str(_uuid_mod.uuid4())[:12],
            paper_ids=  _paper_ids,
        )
        logger.info(
            f"[/chat] topic_id={_topic_result.topic_id} "
            f"label='{_topic_result.topic_label}' "
            f"new={_topic_result.is_new} "
            f"kws={len(_topic_result.keywords)} "
            f"entities={len(_topic_result.entities)}"
        )
    except Exception as _qe:
        logger.debug(f"[/chat] Topic graph write skipped: {_qe}")
    logger.info(f"[/chat] done in {elapsed:.1f}s")

    answer     = result.get("final_answer") or ""
    meta       = result.get("metadata") or {}
    evaluation = meta.get("evaluation") or {}

    # ── Component timing from pipeline result (for X-Timing header) ──────────
    _timings  = result.get("timings") or {}
    _timing_parts = []
    for _k in ["router", "retrieval", "neo4j", "agent", "llm", "evaluation"]:
        _v = _timings.get(_k)
        if _v is not None:
            _timing_parts.append(f"{_k}={round(float(_v)*1000,1)}")
    _timing_parts.append(f"total={round(elapsed*1000,1)}")
    _x_timing = "; ".join(_timing_parts)

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

    # Determine search type from the pipeline result
    _qtype = str(result.get("query_type") or "research")
    if _qtype in ("web_search", "web"):
        _search_type  = "web_search"
        _search_label = "🌐 Web Search"
    elif _qtype == "hybrid":
        _search_type  = "hybrid"
        _search_label = "📄 + 🌐 Hybrid"
    else:
        _search_type  = "local_rag"
        _search_label = "📄 Research Papers"

    from fastapi.responses import JSONResponse
    _resp_data = ChatResponse(
        answer=          answer,
        grade=           grade,
        citations=       citations,
        search_type=     _search_type,
        search_label=    _search_label,
        web_sources=     meta.get("web_sources", []),
        confidence=      confidence,
        grounding_score= float(evaluation.get("grounding_score", 0.0)),
        query_type=      _qtype,
        is_reliable=     bool(evaluation.get("is_reliable", False)),
    )
    return JSONResponse(
        content=_resp_data.model_dump(),
        headers={"X-Timing": _x_timing} if _x_timing else {},
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




# ══════════════════════════════════════════════════════════════════════════════
# INGEST — PDF ingestion endpoint
# Alias: frontend calls POST /ingest; this runs the full ingestion pipeline
# from main.py (parse → chunk → embed → Qdrant → Neo4j entity/keyword graph).
# ══════════════════════════════════════════════════════════════════════════════

class IngestResponse(BaseModel):
    paper_id: str
    title:    str
    chunks:   int
    entities: int      = 0
    keywords: list     = []
    domain:   str      = ""
    message:  str      = "Ingested successfully"


@app.post("/ingest", response_model=IngestResponse)
async def ingest_paper_endpoint(file: UploadFile = File(...)):
    """
    Ingest a research paper (PDF / DOCX / MD / TXT).

    Full pipeline:
        File → parse → chunk (1024 tok, 128 overlap)
             → BGE-M3 embeddings → Qdrant upsert
             → entity extraction → Neo4j Knowledge Graph
             → domain classification + keyword tagging → Neo4j

    Called by the frontend Research page (POST /ingest via httpUpload).
    Reuses the same pipeline as main.py POST /upload — zero code duplication.
    """
    import shutil as _sh
    import time as _t2

    # ── Validate format ───────────────────────────────────────────────────────
    fname = file.filename or ""
    if not fname.lower().endswith((".pdf", ".docx", ".md", ".txt")):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format: {fname}. Accepted: PDF, DOCX, MD, TXT",
        )

    # ── Save to temp dir ──────────────────────────────────────────────────────
    _AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    save_path = _AUDIO_DIR / f"{_uuid.uuid4()}_{fname}"
    with open(save_path, "wb") as fh:
        _sh.copyfileobj(file.file, fh)

    logger.info(
        f"[/ingest] saved {fname} → {save_path} "
        f"({save_path.stat().st_size/1024:.1f} KB)"
    )

    t0 = _time.perf_counter()

    try:
        # ── Ingestion pipeline (same as main.py /upload) ──────────────────────
        from backend.ingestion.paper_ingestion import ingest_paper
        from backend.ingestion.chunking        import chunk_document
        from backend.embeddings                import embed_chunks
        from backend.rag                       import upsert_chunks
        from backend.graph.graph_builder       import build_graph_from_document

        doc      = ingest_paper(save_path)
        chunks   = chunk_document(doc)
        embedded = embed_chunks(chunks)
        upsert_chunks(embedded)
        extraction = build_graph_from_document(doc)

        elapsed = _time.perf_counter() - t0
        logger.info(
            f"[/ingest] done in {elapsed:.1f}s  "
            f"paper_id={doc.paper_id}  chunks={len(chunks)}"
        )

        # ── Collect keyword and domain info from Neo4j for the response ───────
        keywords: list = []
        domain:   str  = ""
        entities: int  = 0
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

            with driver.session() as s:
                row = s.run(
                    "MATCH (p:Paper {paper_id:$pid})-[:BELONGS_TO]->(d:Domain) "
                    "RETURN d.name AS name LIMIT 1",
                    pid=doc.paper_id,
                ).single()
            domain = row["name"] if row else ""

            with driver.session() as s:
                row2 = s.run(
                    "MATCH (p:Paper {paper_id:$pid})-[:HAS_ENTITY]->(e) "
                    "RETURN count(e) AS n",
                    pid=doc.paper_id,
                ).single()
            entities = int(row2["n"]) if row2 else 0
        except Exception as eg:
            logger.debug(f"[/ingest] Neo4j meta query skipped: {eg}")

        save_path.unlink(missing_ok=True)

        return IngestResponse(
            paper_id= doc.paper_id,
            title=    getattr(doc.metadata, "title", None) or fname,
            chunks=   len(chunks),
            entities= entities,
            keywords= keywords[:20],
            domain=   domain,
            message=  f"Ingested in {elapsed:.1f}s",
        )

    except Exception as e:
        save_path.unlink(missing_ok=True)
        logger.error(f"[/ingest] Pipeline failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))




# ══════════════════════════════════════════════════════════════════════════════
# VISION — unified satellite image endpoint (detect / segment / explain)
# Dispatches to the existing vision modules. Does NOT alter image logic.
# Frontend calls POST /vision with FormData {file, mode} or JSON {image_url, mode}.
# ══════════════════════════════════════════════════════════════════════════════

from fastapi import Request

class VisionResponse(BaseModel):
    result: str
    mode:   str


@app.post("/vision", response_model=VisionResponse)
async def vision_analyze(request: Request):
    """
    Unified vision endpoint used by the Vision AI page.

    Accepts either:
      - multipart/form-data:  file=<image>, mode=detect|segment|explain
      - application/json:     { "image_url": "...", "mode": "..." }

    Dispatches to the existing vision modules unchanged:
      explain / detect -> florence2_captioner.caption_image
      segment          -> sam2_segmentor.segment_image
    """
    import shutil as _sh

    content_type = request.headers.get("content-type", "")
    mode         = "explain"
    save_path    = None

    # ── Resolve the image + mode from either payload form ─────────────────────
    try:
        if "multipart/form-data" in content_type:
            form   = await request.form()
            mode   = (form.get("mode") or "explain").lower()
            upload = form.get("file")
            if upload is None:
                raise HTTPException(status_code=400, detail="No 'file' provided")
            suffix    = Path(getattr(upload, "filename", "img.png")).suffix.lower() or ".png"
            save_path = _AUDIO_DIR / f"vision_{_uuid.uuid4()}{suffix}"
            _AUDIO_DIR.mkdir(parents=True, exist_ok=True)
            with open(save_path, "wb") as f:
                _sh.copyfileobj(upload.file, f)
        else:
            body      = await request.json()
            mode      = (body.get("mode") or "explain").lower()
            image_url = body.get("image_url", "")
            # Sample image path served by the frontend (public/). Resolve locally
            # if it exists; otherwise report a clear error rather than 404.
            candidate = Path(image_url.lstrip("/"))
            if candidate.exists():
                save_path = candidate
            else:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "Sample image not accessible from the backend. "
                        "Upload an image file to run analysis."
                    ),
                )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid request: {e}")

    logger.info(f"[/vision] mode={mode} image={save_path}")

    # ── Dispatch to the EXISTING vision modules (logic unchanged) ─────────────
    try:
        if mode == "segment":
            from backend.vision.sam2_segmentor import segment_image as seg
            out = seg(save_path, save_viz=False)
            n   = out.get("mask_count", 0)
            segs = out.get("segments", [])
            parts = [f"Segmented {n} regions."]
            for s in segs[:6]:
                label = s.get("label") or s.get("class") or "region"
                area  = s.get("area_pct") or s.get("area") or ""
                parts.append(f"{label}{f' ({area})' if area else ''}")
            result_text = " ".join(parts)

        else:
            # detect and explain both use the captioner's description
            from backend.vision.florence2_captioner import caption_image as cap
            res = cap(save_path, write_to_graph=False)
            if mode == "detect":
                boxes = getattr(res, "bounding_boxes", []) or []
                kws   = getattr(res, "image_keywords", []) or []
                result_text = (
                    f"Detected {len(boxes)} objects. "
                    + (res.detailed_caption or res.caption)
                    + (f" Keywords: {', '.join(kws[:8])}." if kws else "")
                )
            else:  # explain
                result_text = res.detailed_caption or res.caption

        return VisionResponse(result=result_text, mode=mode)

    except Exception as e:
        logger.error(f"[/vision] {mode} failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ══════════════════════════════════════════════════════════════════════════════
# GET /papers — list all ingested papers
# Frontend Papers page calls this via React Query.
# ══════════════════════════════════════════════════════════════════════════════

class PaperItem(BaseModel):
    id:         str
    title:      str
    authors:    list[str]  = []
    chunks:     int        = 0
    status:     str        = "ready"
    uploadedAt: int        = 0      # Unix ms timestamp
    keywords:   list[str]  = []


@app.get("/papers", response_model=list[PaperItem])
async def list_papers():
    """
    Return all ingested Paper nodes from the Neo4j knowledge graph.

    Each Paper node contains:
        paper_id, title, year, created_at
        -[:AUTHORED_BY]-> Author  (name)
        -[:TAGGED]->      Keyword (name)

    Chunk count is stored on the Paper node as a property set during ingestion.
    Status is always 'ready' for nodes that reached Neo4j.

    Frontend expects:
        { id, title, authors, chunks, status, uploadedAt, keywords }
    """
    import traceback as _tb
    logger.info("=" * 44)
    logger.info("[GET /papers] Request received")

    try:
        from backend.graph.neo4j_client import _get_driver
        driver = _get_driver()
    except Exception as e:
        logger.error(f"[GET /papers] Neo4j connection failed: {e}")
        _tb.print_exc()
        raise HTTPException(
            status_code=503,
            detail=f"Neo4j unavailable: {e}",
        )

    try:
        with driver.session() as session:
            # Fetch all Paper nodes with authors and keywords in one query
            rows = session.run(
                """
                MATCH (p:Paper)
                OPTIONAL MATCH (p)-[:AUTHORED_BY]->(a:Author)
                OPTIONAL MATCH (p)-[:TAGGED]->(k:Keyword)
                WITH p,
                     collect(DISTINCT a.name) AS authors,
                     collect(DISTINCT k.name) AS keywords
                RETURN
                    coalesce(p.paper_id, p.id, id(p)) AS paper_id,
                    coalesce(p.title, 'Untitled')      AS title,
                    coalesce(p.chunks, 0)               AS chunks,
                    coalesce(p.created_at, 0)           AS created_at,
                    authors,
                    keywords
                ORDER BY p.created_at DESC
                """
            ).data()

        logger.info(f"[GET /papers] Fetched {len(rows)} papers from Neo4j")

        papers: list[PaperItem] = []
        for row in rows:
            papers.append(PaperItem(
                id=         str(row["paper_id"]),
                title=      str(row["title"]),
                authors=    [a for a in row["authors"]    if a],
                keywords=   [k for k in row["keywords"]   if k][:15],
                chunks=     int(row["chunks"] or 0),
                status=     "ready",
                uploadedAt= int(row["created_at"] or 0),
            ))

        logger.info(
            f"[GET /papers] Returning {len(papers)} papers  "
            f"titles={[p.title[:30] for p in papers[:3]]}"
        )
        logger.info("=" * 44)
        return papers

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[GET /papers] Query failed: {e}")
        _tb.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch papers: {e}",
        )


# ══════════════════════════════════════════════════════════════════════════════
# VOICE WEBSOCKET — live chunked transcription + full pipeline
# ══════════════════════════════════════════════════════════════════════════════
#
# Protocol (client → server):
#   Binary frames:  raw audio chunks (PCM float32 LE, 16 kHz, mono)
#   JSON text:      { "action": "stop", "paper_id": "..." }
#
# Protocol (server → client, all JSON text frames):
#   { "type": "partial",    "text": "The James Webb..." }
#   { "type": "transcript", "text": "The James Webb Space Telescope..." }
#   { "type": "stage",      "stage": "Extracting keywords" }
#   { "type": "keywords",   "keywords": ["JWST", "exoplanets"] }
#   { "type": "graph",      "entities": 3, "keywords": 5 }
#   { "type": "answer",     "text": "...", "grade": "A", "citations": [...] }
#   { "type": "error",      "message": "..." }
#   { "type": "done" }
#
# The WebSocket approach lets the frontend show each pipeline stage as it
# completes rather than waiting for the full response.  Existing REST
# endpoints are completely untouched.

import asyncio
import json
import struct
from fastapi import WebSocket, WebSocketDisconnect
from fastapi.websockets import WebSocketState

# In-memory audio accumulator keyed by WebSocket id
_ws_audio: dict[int, list[bytes]] = {}


def _emit(ws_id: int, msg: dict) -> None:
    """Stash an outbound message (caller sends asynchronously)."""
    pass   # sending is done inline via await ws.send_text()


async def _send(ws: WebSocket, msg: dict) -> None:
    """Send one JSON frame to the client; ignore if connection already closed."""
    try:
        if ws.client_state == WebSocketState.CONNECTED:
            await ws.send_text(json.dumps(msg))
    except Exception:
        pass


@app.websocket("/voice/stream")
async def voice_stream(ws: WebSocket):
    """
    WebSocket endpoint for live voice transcription.

    Audio arrives as binary frames (float32 LE, 16 kHz, mono).
    When the client sends {"action":"stop"}, the server:
        1. Runs Whisper on the accumulated audio
        2. Extracts keywords
        3. Updates Neo4j knowledge graph
        4. Runs the existing orchestrator (RAG + multi-agent)
        5. Streams back stage-by-stage status messages
    """
    await ws.accept()
    ws_id   = id(ws)
    audio_chunks: list[bytes] = []
    paper_id: str | None      = None

    logger.info(f"[WS /voice/stream] client connected  id={ws_id}")

    try:
        while True:
            # Each receive could be binary (audio) or text (control)
            msg = await ws.receive()

            # ── Binary audio chunk ─────────────────────────────────────────
            if "bytes" in msg and msg["bytes"]:
                chunk = msg["bytes"]
                audio_chunks.append(chunk)

                # Run partial Whisper every ~2s of audio (16kHz float32 = 64 KB)
                total_bytes = sum(len(c) for c in audio_chunks)
                CHUNK_BYTES = 16_000 * 4 * 2   # 2 seconds of float32 16kHz mono

                if total_bytes % CHUNK_BYTES < len(chunk):
                    # Non-blocking partial transcription
                    asyncio.create_task(
                        _partial_transcribe(ws, audio_chunks[:])
                    )

            # ── Text control message ───────────────────────────────────────
            elif "text" in msg and msg["text"]:
                try:
                    ctrl = json.loads(msg["text"])
                except json.JSONDecodeError:
                    continue

                if ctrl.get("action") == "stop":
                    paper_id = ctrl.get("paper_id")
                    break   # fall through to full processing

                elif ctrl.get("action") == "ping":
                    await _send(ws, {"type": "pong"})

    except WebSocketDisconnect:
        logger.info(f"[WS /voice/stream] client disconnected mid-recording  id={ws_id}")
        return
    except Exception as e:
        logger.error(f"[WS /voice/stream] receive loop error: {e}")
        await _send(ws, {"type": "error", "message": str(e)})
        return

    # ── Full pipeline after stop ───────────────────────────────────────────────
    if not audio_chunks:
        await _send(ws, {"type": "error", "message": "No audio received"})
        await ws.close()
        return

    await _run_voice_pipeline(ws, audio_chunks, paper_id)

    try:
        await ws.close()
    except Exception:
        pass
    logger.info(f"[WS /voice/stream] session complete  id={ws_id}")


async def _partial_transcribe(ws: WebSocket, chunks: list[bytes]) -> None:
    """
    Transcribe audio accumulated so far and emit a partial transcript.
    Runs in a background task — non-blocking.
    """
    try:
        import numpy as np
        import soundfile as sf

        # Decode float32 LE bytes → numpy
        raw   = b"".join(chunks)
        count = len(raw) // 4
        if count < 1600:   # < 0.1s at 16 kHz → skip
            return

        audio = np.frombuffer(raw, dtype="<f4")  # little-endian float32
        if audio.max() == 0:
            return

        # Write temp WAV
        tmp = Path(tempfile.gettempdir()) / f"partial_{id(ws)}.wav"
        sf.write(str(tmp), audio, 16_000, subtype="PCM_16")

        from backend.voice.whisper_service import WhisperService
        result = WhisperService().transcribe(tmp)
        tmp.unlink(missing_ok=True)

        text = result.get("text", "").strip()
        if text:
            await _send(ws, {"type": "partial", "text": text})
            logger.debug(f"[WS partial] {text[:60]}")

    except Exception as e:
        logger.debug(f"[WS partial] transcription skipped: {e}")


async def _run_voice_pipeline(
    ws:           WebSocket,
    audio_chunks: list[bytes],
    paper_id:     str | None,
) -> None:
    """
    Full pipeline after user stops speaking.
    Emits stage-by-stage WebSocket messages so the frontend
    can show each step as it completes.
    """
    import numpy as np

    t_total = _time.perf_counter()

    # ── 1. Final transcription ────────────────────────────────────────────────
    await _send(ws, {"type": "stage", "stage": "Transcribing audio"})

    # ── Pre-flight: verify ffmpeg is reachable (WinError 2 root cause) ────────
    import shutil as _shutil
    import traceback as _tb

    _ffmpeg = _shutil.which("ffmpeg") or _shutil.which("ffmpeg.exe")
    if _ffmpeg:
        logger.info(f"[WS pipeline] ffmpeg found: {_ffmpeg}")
    else:
        _msg = (
            "ffmpeg not found on PATH. "
            "Whisper requires ffmpeg to decode audio. "
            "Install from https://ffmpeg.org/download.html and add to PATH, "
            "then restart the server."
        )
        logger.error(f"[WS pipeline] {_msg}")
        await _send(ws, {"type": "error", "message": _msg})
        return

    tmp_wav = None
    try:
        import soundfile as sf

        # ── Build and verify the temp directory ───────────────────────────────
        _AUDIO_DIR.mkdir(parents=True, exist_ok=True)
        logger.info(f"[WS pipeline] audio dir: {_AUDIO_DIR}  exists={_AUDIO_DIR.exists()}")

        # ── Decode raw PCM bytes → numpy → WAV ────────────────────────────────
        raw   = b"".join(audio_chunks)
        count = len(raw) // 4
        logger.info(
            f"[WS pipeline] audio_chunks={len(audio_chunks)} "
            f"raw_bytes={len(raw)} samples={count}"
        )

        if count < 1600:
            raise ValueError(
                f"Audio too short ({count} samples = {count/16000:.2f}s). "
                "Speak for at least 0.5 seconds."
            )

        audio = np.frombuffer(raw, dtype="<f4")
        logger.info(
            f"[WS pipeline] audio shape={audio.shape} "
            f"max={float(audio.max()):.4f} rms={float((audio**2).mean()**0.5):.4f}"
        )

        # ── Write WAV to disk ─────────────────────────────────────────────────
        tmp_wav = _AUDIO_DIR / f"ws_{_uuid.uuid4()}.wav"
        sf.write(str(tmp_wav), audio, 16_000, subtype="PCM_16")

        # ── Verify the file actually exists and has content ───────────────────
        if not tmp_wav.exists():
            raise FileNotFoundError(
                f"WAV write appeared to succeed but file is missing: {tmp_wav}"
            )
        wav_bytes = tmp_wav.stat().st_size
        logger.info(
            f"[WS pipeline] WAV written: {tmp_wav}  "
            f"size={wav_bytes} bytes  exists=True"
        )
        if wav_bytes < 44:
            raise ValueError(
                f"WAV file is too small ({wav_bytes} bytes) — write may have failed."
            )

        # ── Whisper transcription (file must still exist at this point) ───────
        logger.info(f"[WS pipeline] Starting Whisper on: {tmp_wav}")
        from backend.voice.whisper_service import WhisperService
        t0  = _time.perf_counter()
        stt = WhisperService().transcribe(tmp_wav)
        stt_time = _time.perf_counter() - t0

        # ── Only delete AFTER transcription fully returns ─────────────────────
        tmp_wav.unlink(missing_ok=True)
        logger.info(f"[WS pipeline] WAV cleaned up")

        transcript = stt.get("text", "").strip()
        language   = stt.get("language", "en")

        logger.info(
            f"[WS pipeline] transcript in {stt_time:.1f}s  "
            f"lang={language}  chars={len(transcript)}: {transcript[:80]}"
        )

    except Exception as e:
        # Always print the full traceback so the exact failing line is visible
        _tb.print_exc()
        logger.error(
            f"[WS pipeline] STT failed: {type(e).__name__}: {e}\n"
            f"  tmp_wav={tmp_wav}  "
            f"  exists={tmp_wav.exists() if tmp_wav else 'N/A'}  "
            f"  ffmpeg={_ffmpeg}"
        )
        # Clean up if the file is still there
        if tmp_wav and tmp_wav.exists():
            try:
                tmp_wav.unlink()
            except Exception:
                pass
        await _send(ws, {
            "type":    "error",
            "message": f"Transcription failed ({type(e).__name__}): {e}",
        })
        return

    if not transcript:
        await _send(ws, {"type": "error", "message": "No speech detected"})
        return

    await _send(ws, {"type": "transcript", "text": transcript, "language": language})

    # ── 2. Keyword extraction ─────────────────────────────────────────────────
    await _send(ws, {"type": "stage", "stage": "Extracting keywords"})

    keywords: list[str] = []
    entities: list[str] = []
    scientific: list[str] = []

    try:
        # Try the richer extractor first
        from backend.voice.keyword_extractor import extract as kw_extract
        kw_result = kw_extract(transcript)
        keywords   = kw_result.keywords
        entities   = [e.name for e in kw_result.entities]
        scientific = kw_result.scientific
        all_terms  = kw_result.all_terms
    except ImportError:
        # Fall back to the existing tier_classifier (always present)
        try:
            from backend.agents.tier_classifier import extract_keywords
            all_terms = keywords = extract_keywords(transcript)
        except Exception as e2:
            logger.warning(f"[WS pipeline] keyword extraction failed: {e2}")
            all_terms = []

    logger.info(
        f"[WS pipeline] keywords={len(keywords)} entities={len(entities)} "
        f"scientific={len(scientific)}"
    )
    await _send(ws, {
        "type":       "keywords",
        "keywords":   keywords[:12],
        "entities":   entities[:8],
        "scientific": scientific[:8],
    })

    # ── 3. Neo4j knowledge graph update ──────────────────────────────────────
    await _send(ws, {"type": "stage", "stage": "Updating Knowledge Graph"})
    graph_status: dict = {}

    try:
        from backend.voice.voice_graph_store import store_voice_session
        from backend.voice.keyword_extractor import KeywordResult, Entity
        # Build a KeywordResult even if we fell back to tier_classifier
        try:
            kr = kw_result
        except NameError:
            from backend.voice.keyword_extractor import KeywordResult
            kr = KeywordResult(keywords=keywords, all_terms=keywords)

        graph_status = store_voice_session(
            session_id=   str(_uuid.uuid4())[:8],
            transcript=   transcript,
            keywords=     kr,
            language=     language,
        )
        logger.info(f"[WS pipeline] graph: {graph_status}")
    except Exception as e:
        logger.warning(f"[WS pipeline] graph store skipped: {e}")
        graph_status = {"status": "skipped", "error": str(e)}

    # Also persist via existing keyword store (belt + braces)
    try:
        from backend.agents.query_keyword_store import store_query_keywords
        store_query_keywords(
            query=      transcript,
            keywords=   keywords or entities,
            query_type= "voice_ws",
        )
    except Exception:
        pass

    await _send(ws, {
        "type":    "graph",
        "entities": graph_status.get("entities_written", 0),
        "keywords": graph_status.get("keywords_written", 0),
        "status":   graph_status.get("status", "ok"),
    })

    # ── 4. RAG + multi-agent pipeline ─────────────────────────────────────────
    await _send(ws, {"type": "stage", "stage": "Searching Research Papers"})

    try:
        from backend.agents.orchestrator import run

        await _send(ws, {"type": "stage", "stage": "Running Multi-Agent reasoning"})

        t1     = _time.perf_counter()
        result = run(
            query=        transcript,
            paper_loaded= bool(paper_id),
            paper_id=     paper_id,
        )
        rag_time = _time.perf_counter() - t1

        answer     = result.get("final_answer") or "No answer generated."
        meta       = result.get("metadata") or {}
        evl        = meta.get("evaluation") or {}
        confidence = evl.get("confidence", "LOW")
        grade_map  = {"HIGH": "A", "MEDIUM": "B", "LOW": "C"}

        raw_cites = evl.get("citations", [])
        citations = [
            {"section": c.get("section",""), "page": c.get("page",0), "score": c.get("score",0.0)}
            for c in raw_cites if isinstance(c, dict)
        ]

        search_type  = meta.get("search_type",  "local_rag")
        search_label = meta.get("search_label", "📄 Research Papers")
        web_sources  = meta.get("web_sources",  [])

        logger.info(
            f"[WS pipeline] answer in {rag_time:.1f}s  "
            f"type={result.get('query_type')}  conf={confidence}"
        )

        await _send(ws, {
            "type":         "answer",
            "text":         answer,
            "grade":        grade_map.get(confidence, "B"),
            "citations":    citations,
            "search_type":  search_type,
            "search_label": search_label,
            "web_sources":  web_sources,
            "query_type":   result.get("query_type", ""),
        })

    except Exception as e:
        logger.error(f"[WS pipeline] orchestrator failed: {e}")
        await _send(ws, {"type": "error", "message": f"Pipeline failed: {e}"})
        return

    total_time = _time.perf_counter() - t_total
    logger.info(f"[WS pipeline] total time: {total_time:.1f}s")

    await _send(ws, {"type": "done", "elapsed_sec": round(total_time, 2)})



# ══════════════════════════════════════════════════════════════════════════════
# DELETE /papers — clear ALL ingested papers
# Removes: Qdrant "papers" collection (recreated fresh) + all Neo4j Paper nodes
# PDFs are deleted at ingest-time so there is nothing to clean on disk.
# ══════════════════════════════════════════════════════════════════════════════

class ClearPapersResponse(BaseModel):
    success:        bool
    message:        str
    deleted_papers: int = 0
    deleted_chunks: int = 0


@app.delete("/papers", response_model=ClearPapersResponse)
async def clear_all_papers():
    """
    Permanently remove all ingested papers from the knowledge base.

    Cleans up:
      1. Qdrant "papers" collection — all embedded chunks deleted
         (collection is dropped and recreated so a new ingest works immediately)
      2. Neo4j — all Paper nodes and their relationships DETACH DELETE'd
         (Author / Keyword / Domain / Entity nodes that have no other Paper
          relationship are also removed to avoid orphans)

    Safe to call when there are zero papers (idempotent).
    Does NOT touch the satellite_images Qdrant collection or any unrelated data.
    """
    import traceback as _tb

    logger.info("[DELETE /papers] Clear request received")

    deleted_papers = 0
    deleted_chunks = 0
    warnings:  list[str] = []

    # ── Step 1: Qdrant ────────────────────────────────────────────────────────
    try:
        from backend.rag.vector_store import (
            COLLECTION_NAME as PAPERS_COLLECTION,
            delete_collection,
            ensure_collection,
            _get_client as _qdrant_client,
        )

        client = _qdrant_client()
        existing = [c.name for c in client.get_collections().collections]

        if PAPERS_COLLECTION in existing:
            # Count points before deletion so we can report them
            try:
                info = client.get_collection(PAPERS_COLLECTION)
                deleted_chunks = info.points_count or 0
            except Exception:
                deleted_chunks = -1  # unknown but proceeding

            delete_collection()          # drops the collection
            ensure_collection()          # recreates it fresh
            logger.info(
                f"[DELETE /papers] Qdrant '{PAPERS_COLLECTION}' cleared "
                f"({deleted_chunks} chunks removed)"
            )
        else:
            ensure_collection()          # make sure it exists for next ingest
            logger.info(f"[DELETE /papers] Qdrant '{PAPERS_COLLECTION}' was already empty")

    except Exception as e:
        _tb.print_exc()
        logger.error(f"[DELETE /papers] Qdrant cleanup failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Qdrant cleanup failed: {e}",
        )

    # ── Step 2: Neo4j ─────────────────────────────────────────────────────────
    try:
        from backend.graph.neo4j_client import _get_driver

        driver = _get_driver()
        with driver.session() as s:
            # Count papers first
            row = s.run("MATCH (p:Paper) RETURN count(p) AS n").single()
            deleted_papers = int(row["n"]) if row else 0

            if deleted_papers > 0:
                # Delete all Paper nodes and their direct relationships.
                # Also remove now-orphaned Author/Keyword/Domain/Entity nodes
                # that have no remaining connection to any other Paper.
                s.run("MATCH (p:Paper) DETACH DELETE p")
                logger.info(
                    f"[DELETE /papers] Neo4j: {deleted_papers} Paper nodes deleted"
                )

                # Remove orphan nodes (Author, Keyword, Domain, Entity, etc.)
                # that are no longer connected to anything.
                orphan_q = """
                    MATCH (n)
                    WHERE NOT (n:Paper)
                      AND NOT ()-[]->(n)
                      AND NOT (n)-[]->()
                    DELETE n
                """
                result = s.run(orphan_q)
                summary = result.consume()
                orphans = summary.counters.nodes_deleted
                if orphans:
                    logger.info(
                        f"[DELETE /papers] Neo4j: {orphans} orphan nodes removed"
                    )
            else:
                logger.info("[DELETE /papers] Neo4j: no Paper nodes found — nothing to delete")

    except Exception as e:
        _tb.print_exc()
        logger.error(f"[DELETE /papers] Neo4j cleanup failed: {e}")
        warnings.append(f"Neo4j cleanup failed: {e}")
        # Do NOT raise — Qdrant already cleaned; partial success is better
        # than a 500 that leaves Qdrant and Neo4j inconsistent.

    # ── Step 3: Build response ────────────────────────────────────────────────
    if warnings:
        return ClearPapersResponse(
            success=        False,
            message=        f"Partial cleanup. Qdrant cleared. Warnings: {'; '.join(warnings)}",
            deleted_papers= deleted_papers,
            deleted_chunks= deleted_chunks,
        )

    message = (
        f"All ingested papers have been cleared."
        if deleted_papers > 0 or deleted_chunks > 0
        else "Knowledge base was already empty."
    )
    logger.info(f"[DELETE /papers] Done — {deleted_papers} papers, {deleted_chunks} chunks removed")

    return ClearPapersResponse(
        success=        True,
        message=        message,
        deleted_papers= deleted_papers,
        deleted_chunks= deleted_chunks,
    )


# ══════════════════════════════════════════════════════════════════════════════
# GET /dashboard/stats — live statistics for the dashboard page
# ══════════════════════════════════════════════════════════════════════════════

class DashboardStats(BaseModel):
    papers:          int    = 0
    chunks:          int    = 0
    conversations:   int    = 0
    queries:         int    = 0
    avg_reliability: float  = 0.0
    usage_trend:     list   = []
    query_bars:      list   = []
    recent_activity: list   = []
    bookmarks:       list   = []


@app.get("/dashboard/stats", response_model=DashboardStats)
async def dashboard_stats():
    """
    Aggregate statistics shown on the dashboard.
    Paper/chunk counts come from Qdrant + Neo4j (real numbers).
    Other fields return safe defaults for now.
    """
    papers = 0
    chunks = 0

    # Paper count from Neo4j
    try:
        from backend.graph.neo4j_client import _get_driver
        driver = _get_driver()
        with driver.session() as s:
            row = s.run("MATCH (p:Paper) RETURN count(p) AS n").single()
            papers = int(row["n"]) if row else 0
    except Exception as e:
        logger.debug(f"[/dashboard/stats] Neo4j unavailable: {e}")

    # Chunk count from Qdrant
    try:
        from backend.rag.vector_store import _get_client, COLLECTION_NAME
        client = _get_client()
        existing = [c.name for c in client.get_collections().collections]
        if COLLECTION_NAME in existing:
            info = client.get_collection(COLLECTION_NAME)
            chunks = info.points_count or 0
    except Exception as e:
        logger.debug(f"[/dashboard/stats] Qdrant unavailable: {e}")

    return DashboardStats(
        papers=        papers,
        chunks=        chunks,
        conversations= 0,
        queries=       0,
        avg_reliability= 0.0,
        usage_trend=   [],
        query_bars=    [],
        recent_activity= [],
        bookmarks=     [],
    )



# ══════════════════════════════════════════════════════════════════════════════
# GET /graph — live knowledge graph data for visualization
# Returns nodes and links from Neo4j filtered by paper_id or topic_id
# ══════════════════════════════════════════════════════════════════════════════

class GraphNode(BaseModel):
    id:    str
    label: str
    type:  str
    val:   float = 1.0

class GraphLink(BaseModel):
    source: str
    target: str
    label:  str

class GraphResponse(BaseModel):
    nodes: list[GraphNode]
    links: list[GraphLink]
    meta:  dict = {}


@app.get("/graph", response_model=GraphResponse)
async def get_graph_data(
    paper_id: str | None = None,
    topic_id: str | None = None,
    limit:    int        = 120,
):
    """
    Return live knowledge graph data from Neo4j.

    Query params:
      paper_id  – filter to one ingested paper  (Paper subgraph)
      topic_id  – filter to one query topic     (Topic subgraph)
      (none)    – return full overview graph
    """
    try:
        from backend.graph.neo4j_client import _get_driver
        driver = _get_driver()

        nodes: dict[str, GraphNode] = {}
        links: list[GraphLink]      = []

        TYPE_VAL = {
            "Paper": 3.0, "Topic": 2.5, "Author": 1.8,
            "Domain": 2.0, "Keyword": 1.2, "Entity": 1.4,
            "Model": 1.6, "Dataset": 1.6, "Task": 1.2,
            "VoiceInput": 1.3, "Query": 1.2,
        }

        with driver.session() as s:

            if paper_id:
                # Paper-isolated subgraph
                rows = s.run(
                    """
                    MATCH (p:Paper {paper_id: $pid})-[r]->(n)
                    RETURN p.paper_id AS src_id,
                           coalesce(p.title, p.paper_id) AS src_label,
                           'Paper' AS src_type,
                           coalesce(n.name, n.title, elementId(n)) AS tgt_label,
                           labels(n)[0] AS tgt_type,
                           type(r) AS rel
                    LIMIT $lim
                    """,
                    pid=paper_id, lim=limit,
                ).data()

                for row in rows:
                    sid = f"paper_{row['src_id']}"
                    tid = f"{row['tgt_type']}_{row['tgt_label']}"
                    if sid not in nodes:
                        nodes[sid] = GraphNode(id=sid, label=row["src_label"][:40],
                                               type="Paper", val=3.0)
                    if tid not in nodes:
                        nodes[tid] = GraphNode(id=tid, label=row["tgt_label"][:30],
                                               type=row["tgt_type"],
                                               val=TYPE_VAL.get(row["tgt_type"], 1.2))
                    links.append(GraphLink(source=sid, target=tid, label=row["rel"]))

            elif topic_id:
                # Topic-isolated subgraph
                rows = s.run(
                    """
                    MATCH (t:Topic {topic_id: $tid})-[r]->(n)
                    RETURN t.topic_id AS src_id,
                           coalesce(t.label, t.topic_id) AS src_label,
                           'Topic' AS src_type,
                           coalesce(n.name, n.title, elementId(n)) AS tgt_label,
                           labels(n)[0] AS tgt_type,
                           type(r) AS rel
                    LIMIT $lim
                    """,
                    tid=topic_id, lim=limit,
                ).data()

                for row in rows:
                    sid = f"topic_{row['src_id']}"
                    tid = f"{row['tgt_type']}_{row['tgt_label']}"
                    if sid not in nodes:
                        nodes[sid] = GraphNode(id=sid, label=row["src_label"][:40],
                                               type="Topic", val=2.5)
                    if tid not in nodes:
                        nodes[tid] = GraphNode(id=tid, label=row["tgt_label"][:30],
                                               type=row["tgt_type"],
                                               val=TYPE_VAL.get(row["tgt_type"], 1.2))
                    links.append(GraphLink(source=sid, target=tid, label=row["rel"]))

            else:
                # Overview: Papers + Topics + their direct children
                rows = s.run(
                    """
                    MATCH (root)-[r]->(child)
                    WHERE root:Paper OR root:Topic
                    RETURN coalesce(root.paper_id, root.topic_id) AS src_id,
                           coalesce(root.title, root.label, elementId(root)) AS src_label,
                           labels(root)[0] AS src_type,
                           coalesce(child.name, child.title, child.label,
                                    elementId(child)) AS tgt_label,
                           labels(child)[0] AS tgt_type,
                           type(r) AS rel
                    LIMIT $lim
                    """,
                    lim=limit,
                ).data()

                for row in rows:
                    sid = f"{row['src_type']}_{row['src_id']}"
                    tid = f"{row['tgt_type']}_{row['tgt_label']}"
                    if sid not in nodes:
                        nodes[sid] = GraphNode(id=sid, label=row["src_label"][:40],
                                               type=row["src_type"],
                                               val=TYPE_VAL.get(row["src_type"], 2.0))
                    if tid not in nodes:
                        nodes[tid] = GraphNode(id=tid, label=row["tgt_label"][:30],
                                               type=row["tgt_type"],
                                               val=TYPE_VAL.get(row["tgt_type"], 1.2))
                    links.append(GraphLink(source=sid, target=tid, label=row["rel"]))

        meta = {
            "node_count": len(nodes),
            "link_count":  len(links),
            "filter":      "paper" if paper_id else ("topic" if topic_id else "overview"),
        }
        logger.info(f"[/graph] {meta}")
        return GraphResponse(nodes=list(nodes.values()), links=links, meta=meta)

    except Exception as e:
        logger.error(f"[/graph] {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/graph/topics")
async def list_graph_topics():
    """Return all topics for the topic filter dropdown."""
    try:
        try:
            from backend.agents.topic_graph_store import list_topics
        except ImportError:
            from topic_graph_store import list_topics
        return list_topics(limit=50)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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