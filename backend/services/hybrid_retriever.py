"""
AstroNexus AI — Hybrid Retriever
==================================
Merges local RAG context with web search results into a single
ranked context string for the LLM. Handles attribution clearly
so the model can cite "📄 Paper" vs "🌐 Web" sources correctly.

This module is called by the WebSearchAgent and does NOT modify
any existing RAG or retrieval logic.

Priority ranking:
    1. Uploaded document chunks (highest trust, always first)
    2. Neo4j knowledge graph context
    3. Trusted web sources (nasa.gov, arxiv.org, esa.int > others)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

# Minimum RAG score to consider a chunk "confident"
RAG_CONFIDENCE_THRESHOLD = 0.40
# If top RAG score is below this, trigger web search fallback
RAG_FALLBACK_THRESHOLD   = 0.35


@dataclass
class HybridContext:
    """Merged context ready for the LLM prompt."""
    text:            str              # full merged context string
    search_type:     str              # local_rag | web_search | hybrid
    has_rag:         bool
    has_web:         bool
    rag_top_score:   float
    web_sources:     list[dict]       # [{title, url, source, snippet}]
    rag_chunks_used: int


def merge(
    query:       str,
    rag_context: Optional[str]  = None,
    rag_chunks:  Optional[list] = None,   # list[RetrievedChunk]
    web_results: Optional[list] = None,   # list[WebResult]
    graph_context: Optional[str] = None,
) -> HybridContext:
    """
    Merge local and web contexts with clear source attribution.

    Args:
        query:         Original user query
        rag_context:   Pre-formatted RAG context string (from retriever)
        rag_chunks:    Raw RetrievedChunk objects (for score inspection)
        web_results:   WebResult objects from web_search_service
        graph_context: Neo4j context string

    Returns:
        HybridContext with merged text and metadata.
    """
    sections: list[str] = []
    web_sources: list[dict] = []
    has_rag = False
    has_web = False
    rag_top_score = 0.0
    rag_chunks_used = 0

    # ── 1. Local RAG (highest priority) ──────────────────────────────────────
    if rag_context and rag_context.strip():
        has_rag = True
        rag_chunks_used = rag_context.count("[") or 1

        # Extract top score from chunks if available
        if rag_chunks:
            scores = [getattr(c, "score", 0) for c in rag_chunks]
            rag_top_score = max(scores) if scores else 0.0

        sections.append(
            "=== 📄 RESEARCH PAPER CONTEXT ===\n"
            "(Information retrieved from your uploaded documents — "
            "cite using [Section, p.N] format)\n\n"
            + rag_context
        )

    # ── 2. Knowledge graph (second priority) ─────────────────────────────────
    if graph_context and graph_context.strip():
        sections.append(
            "\n=== 🕸 KNOWLEDGE GRAPH CONTEXT ===\n"
            "(Entity relationships from Neo4j graph)\n\n"
            + graph_context
        )

    # ── 3. Web search results (third priority) ────────────────────────────────
    if web_results:
        has_web = True
        web_lines = [
            "\n=== 🌐 WEB SEARCH RESULTS ===\n"
            "(Live information from trusted web sources — "
            "cite using [🌐 source_name] format)"
        ]
        for i, r in enumerate(web_results, 1):
            web_lines.append(
                f"\n[🌐 {i}] {r.title}\n"
                f"Source: {r.source}  URL: {r.url}\n"
                f"{r.snippet}\n"
                + (f"Date: {r.date}\n" if r.date else "")
            )
            web_sources.append({
                "title":   r.title,
                "url":     r.url,
                "source":  r.source,
                "snippet": r.snippet[:200],
            })
        sections.append("\n".join(web_lines))

    # ── Determine search type ─────────────────────────────────────────────────
    if has_rag and has_web:
        search_type = "hybrid"
    elif has_web:
        search_type = "web_search"
    elif has_rag:
        search_type = "local_rag"
    else:
        search_type = "general_llm"

    merged_text = "\n\n".join(sections) if sections else ""

    if not merged_text:
        logger.warning(f"[HybridRetriever] No context available for: {query[:50]}")

    logger.info(
        f"[HybridRetriever] search_type={search_type} "
        f"rag_chunks={rag_chunks_used} web_results={len(web_results or [])} "
        f"rag_top_score={rag_top_score:.3f}"
    )

    return HybridContext(
        text=            merged_text,
        search_type=     search_type,
        has_rag=         has_rag,
        has_web=         has_web,
        rag_top_score=   rag_top_score,
        web_sources=     web_sources,
        rag_chunks_used= rag_chunks_used,
    )


def should_fallback_to_web(rag_chunks: list) -> bool:
    """
    Return True if RAG confidence is too low and web search should augment.

    Args:
        rag_chunks: List of RetrievedChunk objects

    Returns:
        True if fallback to web is warranted.
    """
    if not rag_chunks:
        return True
    top_score = max((getattr(c, "score", 0) for c in rag_chunks), default=0)
    return top_score < RAG_FALLBACK_THRESHOLD