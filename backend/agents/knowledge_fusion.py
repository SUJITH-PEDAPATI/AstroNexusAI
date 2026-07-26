"""
AstroNexus AI — Knowledge Fusion Layer

Merges Qdrant semantic chunks + Neo4j graph context + paper metadata
into a single structured context before sending to the LLM.

Called by Research Agent on every query.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class FusedContext:
    """Single structured context object passed to LLM."""
    qdrant_chunks:    list[dict] = field(default_factory=list)
    graph_entities:   dict       = field(default_factory=dict)
    paper_metadata:   dict       = field(default_factory=dict)
    image_keywords:   list[str]  = field(default_factory=list)
    prompt_text:      str        = ""
    chunk_dicts:      list[dict] = field(default_factory=list)  # for evaluator


def _get_graph_context(query: str, paper_node_id: str | None) -> dict:
    """
    Pull structured facts from Neo4j relevant to this query and paper.
    Returns dict of entity lists.
    """
    context = {
        "authors":      [],
        "models":       [],
        "datasets":     [],
        "tasks":        [],
        "keywords":     [],
        "satellites":   [],
        "domain":       "",
        "related_facts":[],
    }

    try:
        from backend.graph.neo4j_client import _get_driver

        driver   = _get_driver()
        keywords = [
            w.strip("?.,!").lower()
            for w in query.split()
            if len(w.strip("?.,!")) > 3
        ][:6]

        with driver.session() as s:

            # If we know the paper, get its full graph context
            if paper_node_id:
                try:
                    from backend.graph.neo4j_client import get_paper_graph_context
                    ctx = get_paper_graph_context(paper_node_id)
                    if ctx:
                        context.update({
                            "authors":    ctx.get("authors",    []),
                            "models":     ctx.get("models",     []) + ctx.get("algorithms", []),
                            "datasets":   ctx.get("datasets",   []),
                            "tasks":      ctx.get("tasks",      []),
                            "keywords":   ctx.get("keywords",   []),
                            "satellites": ctx.get("satellites", []),
                            "domain":     ctx.get("domain",     ""),
                        })
                except Exception:
                    pass

            # Also search graph for query keywords
            for kw in keywords:
                rows = s.run(
                    """
                    MATCH (n)
                    WHERE (n:Keyword OR n:Algorithm OR n:Dataset OR n:Task)
                      AND toLower(n.name) CONTAINS toLower($kw)
                    OPTIONAL MATCH (p:Paper)-[:TAGGED|USES|SOLVES]->(n)
                    RETURN n.name AS entity, labels(n)[0] AS type,
                           collect(DISTINCT p.title)[..3] AS papers
                    LIMIT 5
                    """,
                    kw=kw,
                ).data()

                for row in rows:
                    fact = (
                        f"{row['type']}: {row['entity']}"
                        + (f" (used in: {', '.join(row['papers'])})"
                           if row.get("papers") else "")
                    )
                    if fact not in context["related_facts"]:
                        context["related_facts"].append(fact)

    except Exception as e:
        logger.warning(f"[KnowledgeFusion] Neo4j context failed (non-fatal): {e}")

    return context


def _build_prompt(
    query:        str,
    chunks:       list[dict],
    graph_ctx:    dict,
    image_kws:    list[str],
) -> str:
    """
    Assemble the final structured prompt from all sources.
    """
    parts = []

    # ── Qdrant chunks ──────────────────────────────────────────────────────────
    if chunks:
        parts.append("=== DOCUMENT EXCERPTS ===")
        for i, c in enumerate(chunks[:5], 1):
            payload = c.get("payload", c)
            section = payload.get("section", "Unknown")
            page    = payload.get("page_num", "?")
            score   = round(c.get("score", 0.0), 3)
            text    = payload.get("text", "")[:600]
            parts.append(
                f"[{i}] Section: {section} | Page: {page} | Score: {score}\n{text}"
            )

    # ── Graph context ──────────────────────────────────────────────────────────
    graph_lines = []

    if graph_ctx.get("domain"):
        graph_lines.append(f"Domain: {graph_ctx['domain']}")
    if graph_ctx.get("authors"):
        graph_lines.append(f"Authors: {', '.join(graph_ctx['authors'][:5])}")
    if graph_ctx.get("models"):
        graph_lines.append(f"Models/Algorithms: {', '.join(graph_ctx['models'][:5])}")
    if graph_ctx.get("datasets"):
        graph_lines.append(f"Datasets: {', '.join(graph_ctx['datasets'][:5])}")
    if graph_ctx.get("tasks"):
        graph_lines.append(f"Tasks: {', '.join(graph_ctx['tasks'][:5])}")
    if graph_ctx.get("keywords"):
        graph_lines.append(f"Keywords: {', '.join(graph_ctx['keywords'][:8])}")
    if graph_ctx.get("satellites"):
        graph_lines.append(f"Satellites: {', '.join(graph_ctx['satellites'][:5])}")
    if graph_ctx.get("related_facts"):
        graph_lines.append("Related graph facts:")
        for fact in graph_ctx["related_facts"][:5]:
            graph_lines.append(f"  • {fact}")

    if graph_lines:
        parts.append("=== KNOWLEDGE GRAPH CONTEXT ===")
        parts.extend(graph_lines)

    # ── Image keywords ─────────────────────────────────────────────────────────
    if image_kws:
        parts.append("=== IMAGE ANALYSIS KEYWORDS ===")
        parts.append(", ".join(image_kws))

    # ── Question ───────────────────────────────────────────────────────────────
    parts.append(f"=== QUESTION ===\n{query}")

    return "\n\n".join(parts)


def fuse(
    query:          str,
    qdrant_chunks:  list[dict],
    paper_node_id:  str | None = None,
    image_keywords: list[str]  = None,
) -> FusedContext:
    """
    Merge all knowledge sources into one structured context.

    Args:
        query:          User question
        qdrant_chunks:  Retrieved chunks from Qdrant (list of dicts with score)
        paper_node_id:  Neo4j node_id of the current paper (optional)
        image_keywords: Keywords extracted from satellite image caption (optional)

    Returns:
        FusedContext with prompt_text ready for LLM
    """
    image_keywords = image_keywords or []

    logger.info(
        f"[KnowledgeFusion] Fusing — "
        f"chunks={len(qdrant_chunks)} "
        f"paper_id={paper_node_id is not None} "
        f"image_kws={len(image_keywords)}"
    )

    # Pull graph context
    graph_ctx = _get_graph_context(query, paper_node_id)

    logger.info(
        f"[KnowledgeFusion] Graph — "
        f"authors={len(graph_ctx['authors'])} "
        f"models={len(graph_ctx['models'])} "
        f"keywords={len(graph_ctx['keywords'])} "
        f"facts={len(graph_ctx['related_facts'])}"
    )

    # Build unified prompt
    prompt = _build_prompt(query, qdrant_chunks, graph_ctx, image_keywords)

    # Chunk dicts for evaluator
    chunk_dicts = []
    for c in qdrant_chunks:
        payload = c.get("payload", c)
        chunk_dicts.append({
            "score":   c.get("score", 0.0),
            "text":    payload.get("text", ""),
            "section": payload.get("section", ""),
            "page_num":payload.get("page_num", "?"),
            "title":   payload.get("title", ""),
        })

    return FusedContext(
        qdrant_chunks=  qdrant_chunks,
        graph_entities= graph_ctx,
        image_keywords= image_keywords,
        prompt_text=    prompt,
        chunk_dicts=    chunk_dicts,
    )
