"""
AstroNexus AI — Router v3.4

Fix: paper_loaded preserved correctly through all metadata copies.
Priority: resource_guard → graph → tier
"""
from __future__ import annotations

import logging
import os
import re
from typing import Optional

from backend.agents.state import AgentState

logger = logging.getLogger(__name__)
VALID_ROUTES = {"research","graph","satellite","general","voice"}

_GRAPH_RE = re.compile(
    r'\b(who.{0,10}(authored|wrote|published|created)|'
    r'which.{0,20}(paper|satellite|sensor|algorithm|mission|institution)|'
    r'what.{0,20}(sensor|mission|satellite|institution).{0,20}(has|carry|use|belong)|'
    r'list.{0,20}(paper|author|satellite|algorithm|keyword)|'
    r'authored.?by|belong.?to|tagged|related.?to|'
    r'knowledge.?graph|entity|relationship|'
    r'part.?of.?mission|operated.?by|has.?sensor)\b',
    re.IGNORECASE,
)
_VOICE_RE    = re.compile(r'\b(speak|voice|aloud|read.?out|tts|say.?it)\b', re.IGNORECASE)
_VISION_RE   = re.compile(r'\b(analyse|analyze|segment|caption|what.{0,10}image|land.?cover)\b', re.IGNORECASE)
_FOLLOWUP_RE = re.compile(
    r'^(explain\s+(that|it|more)|what\s+do\s+you\s+mean|'
    r'tell\s+me\s+more|in\s+simple\s+(terms|words)|'
    r'summari[sz]e\s+(that|it))[\s?]*$', re.IGNORECASE
)


def _safe_metadata(state: AgentState) -> dict:
    """Always return a fresh copy — never mutate shared state."""
    return dict(state.get("metadata") or {})


def _get_paper_loaded(state: AgentState) -> bool:
    """Read paper_loaded from both state levels."""
    meta = _safe_metadata(state)
    return meta.get("paper_loaded", False) or state.get("paper_loaded", False)


def _resource_guard(query: str, state: AgentState) -> Optional[str]:
    meta       = _safe_metadata(state)
    image_path = meta.get("image_path")
    audio_path = state.get("audio_path")

    if audio_path and not query.strip(): return "voice"
    if _VOICE_RE.search(query):          return "voice"
    if image_path and _VISION_RE.search(query): return "satellite"

    last = state.get("query_type")
    if last and last in VALID_ROUTES:
        if _FOLLOWUP_RE.match(query.strip()) or len(query.split()) <= 4:
            return last

    return None


def _tier_route(query: str, state: AgentState) -> str:
    from backend.agents.tier_classifier import classify_tier
    paper_loaded = _get_paper_loaded(state)
    paper_id     = _safe_metadata(state).get("paper_id")

    tier_info = classify_tier(
        query=        query,
        paper_loaded= paper_loaded,
        paper_id=     paper_id,
    )

    # Write tier_info while preserving paper_loaded
    meta = _safe_metadata(state)
    meta["tier_info"]    = tier_info
    meta["paper_loaded"] = paper_loaded   # ← never lose this
    state["metadata"]    = meta

    logger.info(
        f"[Router] tier={tier_info['tier']} "
        f"paper_loaded={paper_loaded} "
        f"is_science={tier_info['is_science']}"
    )
    return "research" if tier_info["tier"] == 1 else "general"


def _store_keywords(query: str, state: AgentState, route: str) -> list[str]:
    try:
        from backend.agents.tier_classifier     import extract_keywords
        from backend.agents.query_keyword_store import store_query_keywords
        keywords = extract_keywords(query)
        if keywords:
            store_query_keywords(
                query=      query,
                keywords=   keywords,
                session_id= _safe_metadata(state).get("session_id"),
                query_type= route,
            )
        return keywords
    except Exception as e:
        logger.debug(f"[Router] Keyword store skipped: {e}")
        return []


def classify_query(query: str, state: Optional[AgentState] = None) -> str:
    state = state or {}
    forced = _resource_guard(query, state)
    if forced: return forced
    if _GRAPH_RE.search(query):
        logger.info("[Router] Graph signal detected")
        return "graph"
    return _tier_route(query, state)


def router_node(state: AgentState) -> AgentState:
    query      = state.get("query", "")
    audio_path = state.get("audio_path")

    if audio_path and not query.strip():
        try:
            from backend.voice.whisper_service import WhisperService
            query = WhisperService().transcribe(audio_path)["text"]
        except Exception as e:
            return {**state, "error": f"Transcription failed: {e}"}

    query_type = classify_query(query, state)
    keywords   = _store_keywords(query, state, query_type)

    meta = _safe_metadata(state)
    meta["query_keywords"] = keywords
    meta["paper_loaded"]   = _get_paper_loaded(state)

    logger.info(f"[Router] '{query[:60]}' → {query_type} paper_loaded={meta['paper_loaded']}")

    return {
        **state,
        "query":      query,
        "query_type": query_type,
        "metadata":   meta,
    }


def route_decision(state: AgentState) -> str:
    return state.get("query_type", "general")