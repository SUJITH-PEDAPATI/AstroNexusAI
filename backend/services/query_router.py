"""
AstroNexus AI — Hybrid Query Router
======================================
Classifies every incoming query into one of four routes BEFORE
the LangGraph orchestrator runs. This sits at the very top of
the pipeline, is stateless, and never calls any external service.

Routes:
    local_rag    — answer from uploaded papers + knowledge graph
    web_search   — requires live/current web data
    hybrid       — needs both local corpus AND web information
    general_llm  — pure LLM reasoning, no retrieval needed

Design principle:
    - Fast: regex + keyword matching, no LLM call
    - Conservative: biases toward local_rag when uncertain
    - Transparent: explains its decision in metadata
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# ── Signals that require live web data ───────────────────────────────────────

_WEB_TEMPORAL = re.compile(
    r"\b(today|yesterday|this week|this month|this year|right now|currently|"
    r"latest|recent|breaking|just happened|last (?:week|month|year|night|hour)|"
    r"as of|breaking news|what happened|news about|recent (?:launch|discovery|"
    r"mission|update|announcement|report|study|findings))\b",
    re.IGNORECASE,
)

_WEB_LIVE = re.compile(
    r"\b(launch(?:ed)?|landing|rover|mission status|spacewalk|eclipse|"
    r"price|stock|weather|forecast|election|president|prime minister|"
    r"CEO|founded|released|announced|discovered|won|lost|"
    r"current[ly]?|live|real.?time|up.?to.?date|2024|2025|2026)\b",
    re.IGNORECASE,
)

# Signals that the user is asking about their uploaded papers
_LOCAL_SIGNALS = re.compile(
    r"\b(this paper|my paper|uploaded|the document|figure \d|table \d|"
    r"section \d|the authors?|according to|cited in|methodology|"
    r"our (?:paper|model|approach|method|results?|findings?)|"
    r"summarize|summarise|abstract of|conclusion of|what does the paper)\b",
    re.IGNORECASE,
)

# Signals that require both sources
_HYBRID_SIGNALS = re.compile(
    r"\b(compare (?:with|to) (?:latest|recent|current|new)|"
    r"how does .{0,40} compare to (?:latest|current|new)|"
    r"what does the (?:paper|literature) say .{0,30} (?:versus|vs|compared to)|"
    r"latest (?:research|findings|discoveries) (?:on|about|related to)|"
    r"my paper .{0,50} (?:and|with) (?:latest|recent|current))\b",
    re.IGNORECASE,
)

# General knowledge that doesn't need retrieval
_GENERAL_LLM = re.compile(
    r"\b(what is|explain|define|how does|describe|tell me about|"
    r"history of|basics of|introduction to|overview of)\b",
    re.IGNORECASE,
)


@dataclass
class RouteDecision:
    route:      str        # local_rag | web_search | hybrid | general_llm
    confidence: float      # 0–1 confidence in the routing decision
    reason:     str        # human-readable explanation
    signals:    list[str]  # matched signal strings


def classify(query: str, has_paper: bool = False) -> RouteDecision:
    """
    Classify a user query into a search strategy.

    Args:
        query:     The user's question
        has_paper: Whether the user has an uploaded paper in the session

    Returns:
        RouteDecision with route, confidence, reason, and matched signals
    """
    q = query.strip()
    signals: list[str] = []
    scores: dict[str, float] = {
        "local_rag":   0.0,
        "web_search":  0.0,
        "hybrid":      0.0,
        "general_llm": 0.0,
    }

    # ── Hybrid check (must be first — most specific pattern) ──────────────────
    if m := _HYBRID_SIGNALS.search(q):
        signals.append(f"hybrid_signal: '{m.group()}'")
        scores["hybrid"] += 0.85
        if has_paper:
            scores["hybrid"] += 0.1   # stronger signal if paper exists

    # ── Local paper signals ───────────────────────────────────────────────────
    local_matches = _LOCAL_SIGNALS.findall(q)
    if local_matches:
        signals.extend([f"local: '{m}'" for m in local_matches[:3]])
        scores["local_rag"] += 0.3 * min(len(local_matches), 3)
    if has_paper:
        scores["local_rag"] += 0.2   # baseline boost when paper is loaded

    # ── Web search signals ────────────────────────────────────────────────────
    if m := _WEB_TEMPORAL.search(q):
        signals.append(f"temporal: '{m.group()}'")
        scores["web_search"] += 0.75
    if m := _WEB_LIVE.search(q):
        signals.append(f"live_data: '{m.group()}'")
        scores["web_search"] += 0.45

    # ── General LLM ──────────────────────────────────────────────────────────
    if m := _GENERAL_LLM.search(q):
        signals.append(f"general: '{m.group()}'")
        scores["general_llm"] += 0.35

    # ── Default: if paper is loaded, lean local; else lean general ────────────
    if all(v == 0 for v in scores.values()):
        if has_paper:
            scores["local_rag"]   = 0.55
            scores["general_llm"] = 0.35
            signals.append("default: paper_loaded → local_rag")
        else:
            scores["general_llm"] = 0.55
            scores["local_rag"]   = 0.20
            signals.append("default: no_paper → general_llm")

    # ── Hybrid promotion: if both local AND web signals are strong ────────────
    if scores["local_rag"] >= 0.3 and scores["web_search"] >= 0.4:
        scores["hybrid"] = max(scores["hybrid"], (scores["local_rag"] + scores["web_search"]) * 0.6)
        signals.append("promoted_to_hybrid")

    # ── Pick winner ───────────────────────────────────────────────────────────
    route      = max(scores, key=lambda k: scores[k])
    confidence = scores[route]

    reasons = {
        "local_rag":   "Query refers to uploaded documents or domain knowledge",
        "web_search":  "Query requires live/current information from the web",
        "hybrid":      "Query needs both uploaded papers and live web sources",
        "general_llm": "Query can be answered from LLM knowledge alone",
    }

    logger.info(
        f"[QueryRouter] '{q[:55]}' → {route} "
        f"(conf={confidence:.2f}, has_paper={has_paper})"
    )

    return RouteDecision(
        route=      route,
        confidence= round(confidence, 3),
        reason=     reasons[route],
        signals=    signals,
    )