"""
AstroNexus AI — FastAPI HTTP Server
====================================
A minimal HTTP wrapper around the existing agent pipeline.
Zero model logic lives here — everything is delegated to
backend.agents.orchestrator.run(), which is exactly what the
terminal CLI (python -m backend.tests.chat) calls.

Start the server from your project root:

    uvicorn backend.api.main:app --reload --port 8000

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


# ── /dashboard/stats ────────────────────────────────────────────────────────────

@app.get("/dashboard/stats")
async def dashboard_stats():
    """Mock stats endpoint for the frontend dashboard."""
    return {
        "conversations": 42,
        "papers": 12,
        "queries": 156,
        "avg_reliability": 0.92,
        "usage_trend": [5, 10, 15, 12, 22, 18, 30, 25, 40, 35, 50, 42],
        "query_bars": [12, 15, 8, 22, 30, 25, 44],
        "recent_activity": [
            {"action": "Queried", "target": "Origins Space Telescope", "time": "2m ago"},
            {"action": "Ingested", "target": "Exoplanet Atmospheres.pdf", "time": "1h ago"},
            {"action": "Queried", "target": "SAR Flood Detection", "time": "3h ago"}
        ],
        "bookmarks": ["James Webb Space Telescope", "Mars Rover Analysis"]
    }

# ── /health ───────────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    """Root endpoint for the API."""
    return {"message": "AstroNexus AI API is running. Visit /docs for documentation."}

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