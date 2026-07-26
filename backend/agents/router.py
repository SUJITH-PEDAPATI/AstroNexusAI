"""
AstroNexus AI — Query Router (v2.0)

Architecture: 3-layer routing

    Layer 1 — Resource Guard   (zero LLM cost, eliminates obvious misroutes)
    Layer 2 — Context Default  (fixes general vs research distinction)  
    Layer 3 — LLM Classifier   (Ollama, only for ambiguous multi-signal queries)

Key fixes over v1:
    - "satellite" words no longer route to Satellite Agent without an image
    - No paper loaded → default is General, not Research
    - Conversation state used as prior for follow-up queries
    - LLM classifier resolves multi-signal ambiguity instead of silent priority
"""
from __future__ import annotations

import json
import logging
import re
import urllib.request
import urllib.error
import os
from typing import Optional

from backend.agents.state import AgentState

logger = logging.getLogger(__name__)

OLLAMA_BASE  = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL",    "qwen3:4b")

VALID_ROUTES = {"research", "graph", "satellite", "general", "voice"}

# ══════════════════════════════════════════════════════════════════════════════
# SIGNAL PATTERNS
# ══════════════════════════════════════════════════════════════════════════════

# Vision INTENT — user wants to look at / analyse an image
# Only relevant when an image is actually present
_VISION_INTENT_RE = re.compile(
    r"\b(analyse|analyze|identify|detect|segment|classify|describe|"
    r"show|visualise|visualize|caption|what.{0,10}image|"
    r"land.?cover|flood.?extent|visible.{0,20}in|"
    r"estimate.{0,20}percent|environmental.{0,20}hazard)\b",
    re.IGNORECASE,
)

# Satellite TOPIC — mentions satellite concepts but NOT necessarily needing image
_SATELLITE_TOPIC_RE = re.compile(
    r"\b(satellite|sentinel|landsat|modis|sar|ndvi|spectral|"
    r"remote.?sensing|aerial|geospatial|lidar|sar|earth.?observation|"
    r"multispectral|hyperspectral|geotiff|raster)\b",
    re.IGNORECASE,
)

# Graph signal — entity / relationship lookups
_GRAPH_RE = re.compile(
    r"\b(who.{0,10}(authored|wrote|published)|"
    r"which.{0,30}(paper|satellite|sensor|algorithm|mission|institution|model)s?|"
    r"what.{0,30}(paper|satellite|sensor|mission|institution|dataset|algorithm).{0,30}"
    r"(use|has|belong|related|part.?of|carry|operate|in|solve|about)|"
    r"list.{0,20}(paper|author|satellite|sensor|algorithm|metric|keyword|institution)|"
    r"citation|cite|knowledge.?graph|entity|relationship|neo4j|graph|"
    r"authored.?by|belong.?to|tagged|related.?to)\b",
    re.IGNORECASE,
)

# Research signal — paper content questions
_RESEARCH_RE = re.compile(
    r"\b(main.?contribution|proposed|methodology|ablation|"
    r"hyperparameter|equation|baseline|table|figure|section|"
    r"appendix|conclusion|abstract)\b",
    re.IGNORECASE,
)

# Voice signal
_VOICE_RE = re.compile(
    r"\b(speak|voice|aloud|read.?out|tts|text.?to.?speech|"
    r"say.?it|listen|audio.?response)\b",
    re.IGNORECASE,
)

# General world knowledge — NOT scientific paper content
_GENERAL_RE = re.compile(
    r"\b(president|prime.?minister|governor|politician|"
    r"weather|temperature|forecast|rain|"
    r"recipe|cook|food|restaurant|"
    r"sport|football|cricket|baseball|nba|nfl|"
    r"movie|film|actor|actress|music|song|artist|"
    r"stock|share.?price|crypto|bitcoin|"
    r"history|geography|capital.?of|population.?of|"
    r"joke|funny|tell.?me.?a|poem|story|"
    r"translate|meaning.?of.{0,15}in|"
    r"shopping|buy|price.?of|"
    r"news|latest|current.?event)\b",
    re.IGNORECASE,
)

# Short follow-up phrases that should inherit previous route
_FOLLOWUP_RE = re.compile(
    r"^(explain\s+(that|it|more|further)|"
    r"what\s+do\s+you\s+mean|"
    r"can\s+you\s+elaborate|"
    r"tell\s+me\s+more|"
    r"give\s+me\s+more\s+detail|"
    r"in\s+simple\s+(terms|words)|"
    r"summarise\s+(that|it)|"
    r"summarize\s+(that|it)|"
    r"repeat\s+(that|the\s+last)|"
    r"what\s+about\s+(the\s+)?limitations|"
    r"who\s+proposed\s+(it|this)|"
    r"can\s+you\s+compare)[\s?]*$",
    re.IGNORECASE,
)


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 1 — RESOURCE GUARD
# ══════════════════════════════════════════════════════════════════════════════

def _resource_guard(query: str, state: AgentState) -> Optional[str]:
    """
    Route purely based on available resources.

    Returns a forced route string if context makes intent unambiguous.
    Returns None to pass through to the LLM classifier.

    Rules:
        - Voice: audio_path present and query is empty → voice
        - Satellite: ONLY route here when image_path is set AND
                     query has vision-intent words
        - Satellite topic words WITHOUT an image → pass through (not satellite)
        - General world-knowledge words → general (no paper needed)
        - Short follow-up → inherit previous route
    """
    metadata   = state.get("metadata") or {}
    image_path = metadata.get("image_path")
    audio_path = state.get("audio_path")

    # Voice: audio present, no text yet
    if audio_path and not query.strip():
        logger.info("[Router] Resource guard → voice (audio present)")
        return "voice"

    # Explicit voice intent in text
    if _VOICE_RE.search(query):
        logger.info("[Router] Resource guard → voice (voice keyword)")
        return "voice"

    # Satellite: MUST have image AND vision-intent words
    if image_path:
        if _VISION_INTENT_RE.search(query):
            logger.info("[Router] Resource guard → satellite (image + vision intent)")
            return "satellite"
        # Image present but question is not about the image content
        # Fall through — might be research or graph

    # Satellite topic words WITHOUT image → not a satellite agent task
    # Do NOT route to satellite; let graph/research handle it
    if _SATELLITE_TOPIC_RE.search(query) and not image_path:
        logger.info("[Router] Resource guard: satellite topic but no image → not satellite")
        # Return None — let classifier decide between graph/research/general

    # Short follow-up: inherit previous route
    last_route = state.get("query_type")
    if last_route and last_route in VALID_ROUTES:
        if _FOLLOWUP_RE.match(query.strip()) or len(query.split()) <= 5:
            logger.info(f"[Router] Resource guard → {last_route} (follow-up inherited)")
            return last_route

    # Strong general signal with no domain signal → general
    has_domain = (
        _GRAPH_RE.search(query) or
        _RESEARCH_RE.search(query) or
        _SATELLITE_TOPIC_RE.search(query)
    )
    if _GENERAL_RE.search(query) and not has_domain:
        logger.info("[Router] Resource guard → general (world knowledge, no domain)")
        return "general"

    return None  # pass through to classifier


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 2 — CONTEXT-AWARE DEFAULT
# ══════════════════════════════════════════════════════════════════════════════

def _context_default(state: AgentState) -> str:
    """
    Choose default route based on available context.

    v1 always defaulted to "research" — this caused 0/15 general knowledge.
    v2 defaults to "general" when no paper is loaded, "research" when it is.
    """
    metadata = state.get("metadata") or {}

    has_paper = bool(
        metadata.get("paper_id") or
        metadata.get("image_path") is None and state.get("rag_context")
    )

    # Check if any paper has been ingested (Qdrant has data)
    # Light check — don't call Qdrant on every query, use session state
    session_paper_loaded = metadata.get("paper_loaded", False)

    if has_paper or session_paper_loaded:
        logger.info("[Router] Context default → research (paper in session)")
        return "research"

    logger.info("[Router] Context default → general (no paper loaded)")
    return "general"


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 3 — LLM CLASSIFIER (ambiguous multi-signal queries only)
# ══════════════════════════════════════════════════════════════════════════════

_CLASSIFY_PROMPT = """\
You are a query router for AstroNexus, a scientific research platform.

Route the query to exactly one agent:
  research   — questions about content of an uploaded scientific paper
               (methodology, results, datasets, equations, limitations)
  graph      — entity/relationship lookups (who authored X, which satellites
               use Y sensor, what papers are tagged flood detection)
  satellite  — satellite image analysis (ONLY if image is provided)
  general    — general world knowledge not requiring any uploaded paper
  voice      — user explicitly wants a spoken audio response

Context:
  Paper uploaded : {has_paper}
  Image provided : {has_image}
  Last route     : {last_route}

Query: {query}

Reply with ONE word only: research, graph, satellite, general, or voice"""


def _llm_classify(
    query:      str,
    has_paper:  bool,
    has_image:  bool,
    last_route: str,
) -> str:
    """
    Use Ollama to classify ambiguous queries.
    Only called when multiple regex signals fire simultaneously.
    Falls back to context_default if Ollama is unavailable.
    """
    prompt = _CLASSIFY_PROMPT.format(
        query=      query,
        has_paper=  "yes" if has_paper else "no",
        has_image=  "yes" if has_image else "no",
        last_route= last_route or "none",
    )

    payload = json.dumps({
        "model":   OLLAMA_MODEL,
        "prompt":  prompt,
        "stream":  False,
        "options": {"temperature": 0.0, "num_predict": 5},
    }).encode("utf-8")

    try:
        req = urllib.request.Request(
            f"{OLLAMA_BASE}/api/generate",
            data=    payload,
            headers= {"Content-Type": "application/json"},
            method=  "POST",
        )
        with urllib.request.urlopen(req, timeout=8) as resp:
            data   = json.loads(resp.read())
            label  = data.get("response", "").strip().lower().split()[0]
            if label in VALID_ROUTES:
                logger.info(f"[Router] LLM classifier → {label}")
                return label
    except Exception as e:
        logger.warning(f"[Router] LLM classifier unavailable: {e}")

    return ""   # signal failure — caller uses context_default


# ══════════════════════════════════════════════════════════════════════════════
# SIGNAL COUNT — decides whether LLM classifier is needed
# ══════════════════════════════════════════════════════════════════════════════

def _count_signals(query: str) -> dict[str, bool]:
    return {
        "graph":    bool(_GRAPH_RE.search(query)),
        "research": bool(_RESEARCH_RE.search(query)),
        "general":  bool(_GENERAL_RE.search(query)),
    }


def _is_ambiguous(signals: dict[str, bool]) -> bool:
    """True when multiple domain signals fire — LLM classifier needed."""
    return sum(signals.values()) >= 2


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ══════════════════════════════════════════════════════════════════════════════

def classify_query(query: str, state: Optional[AgentState] = None) -> str:
    """
    Classify a user query into one of:
        research | graph | satellite | general | voice

    Three-layer pipeline:
        1. Resource guard  — eliminates obvious misroutes using context
        2. Signal counting — if unambiguous, route directly
        3. LLM classifier  — only for multi-signal ambiguous queries
        4. Context default — fallback when everything else is uncertain

    Args:
        query: User query string
        state: LangGraph AgentState (provides session context)

    Returns:
        One of: "research" | "graph" | "satellite" | "general" | "voice"
    """
    state = state or {}

    # ── Layer 1: Resource guard ────────────────────────────────────────────────
    forced = _resource_guard(query, state)
    if forced:
        return forced

    # ── Signal counting ────────────────────────────────────────────────────────
    signals  = _count_signals(query)
    metadata = state.get("metadata") or {}
    has_paper = bool(
        metadata.get("paper_id") or
        metadata.get("paper_loaded") or
        state.get("rag_context")
    )
    has_image  = bool(metadata.get("image_path"))
    last_route = state.get("query_type", "")

    # Unambiguous single-signal cases — no LLM needed
    if signals["graph"] and not signals["research"] and not signals["general"]:
        logger.info("[Router] Single signal → graph")
        return "graph"

    if signals["research"] and not signals["graph"] and not signals["general"]:
        logger.info("[Router] Single signal → research")
        return "research"

    # ── Layer 3: LLM classifier for ambiguous queries ──────────────────────────
    if _is_ambiguous(signals):
        result = _llm_classify(query, has_paper, has_image, last_route)
        if result:
            return result

    # ── Layer 2: Context-aware default ────────────────────────────────────────
    return _context_default(state)


def router_node(state: AgentState) -> AgentState:
    """
    LangGraph node: classify query and set state["query_type"].
    Handles audio transcription if audio_path is provided.
    """
    query      = state.get("query", "")
    audio_path = state.get("audio_path")

    # Transcribe audio if provided
    if audio_path and not query.strip():
        try:
            from backend.voice.whisper_service import WhisperService
            result = WhisperService().transcribe(audio_path)
            query  = result["text"]
            logger.info(f"[Router] Transcribed: {query}")
        except Exception as e:
            logger.error(f"[Router] Transcription failed: {e}")
            return {**state, "error": f"Transcription failed: {e}"}

    query_type = classify_query(query, state)
    logger.info(f"[Router] '{query[:60]}' → {query_type}")

    return {**state, "query": query, "query_type": query_type}


def route_decision(state: AgentState) -> str:
    """LangGraph conditional edge — returns next node name."""
    return state.get("query_type", "general")