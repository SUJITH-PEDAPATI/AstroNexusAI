"""
AstroNexus AI — Knowledge Fusion v4.0

When paper is uploaded, EVERY answer pulls from three sources:
    1. Qdrant chunks      — semantic retrieval
    2. Neo4j graph        — structured entities (authors, models, keywords, domain)
    3. Conversation history — last 3 turns for pronoun resolution

Graph lookup tries paper_id, node_id, id in sequence.
Logs exactly what was found so empty graph is visible in logs.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

MAX_HISTORY_TURNS = 3
MAX_HISTORY_CHARS = 800


@dataclass
class FusedContext:
    qdrant_chunks:        list[dict] = field(default_factory=list)
    graph_entities:       dict       = field(default_factory=dict)
    image_keywords:       list[str]  = field(default_factory=list)
    conversation_history: list[dict] = field(default_factory=list)
    prompt_text:          str        = ""
    chunk_dicts:          list[dict] = field(default_factory=list)


# ══════════════════════════════════════════════════════════════════════════════
# GRAPH CONTEXT
# ══════════════════════════════════════════════════════════════════════════════

def _get_graph_context(query: str, paper_id: str | None) -> dict:
    """
    Pull all entities linked to the paper from Neo4j.
    Tries paper_id → node_id → id until one matches.
    Also runs keyword-based search for query terms.
    """
    empty = {
        "authors": [], "models": [], "datasets": [], "tasks": [],
        "keywords": [], "satellites": [], "domain": "", "venue": "",
        "related_facts": [],
    }

    if not paper_id:
        return empty

    try:
        from backend.graph.neo4j_client import _get_driver
        driver = _get_driver()

        # ── Step 1: Find paper internal ID ────────────────────────────────────
        internal_id = None
        with driver.session() as s:
            for prop in ["paper_id", "node_id", "id"]:
                row = s.run(
                    f"MATCH (p:Paper {{{prop}: $pid}}) "
                    "RETURN id(p) AS iid, p.title AS title LIMIT 1",
                    pid=paper_id,
                ).single()
                if row:
                    internal_id = row["iid"]
                    logger.info(
                        f"[Fusion] Paper found via {prop}: '{(row['title'] or '')[:50]}'"
                    )
                    break

        if internal_id is None:
            logger.warning(f"[Fusion] Paper not found in Neo4j — id='{paper_id}'")
            return empty

        # ── Step 2: Get all outgoing relationships ─────────────────────────────
        ctx = dict(empty)
        with driver.session() as s:
            rels = s.run(
                """
                MATCH (p)-[r]->(n)
                WHERE id(p) = $iid
                RETURN type(r)                           AS rel,
                       labels(n)[0]                      AS label,
                       coalesce(n.name, n.title, '')     AS name
                """,
                iid=internal_id,
            ).data()

        REL_MAP = {
            "AUTHORED_BY":    "authors",
            "USES":           "models",
            "SOLVES":         "tasks",
            "TAGGED":         "keywords",
            "BELONGS_TO":     "domain",
            "PRESENTED_AT":   "venue",
            "EVALUATED_BY":   "metrics",
        }

        for row in rels:
            name = (row.get("name") or "").strip()
            rel  = row.get("rel", "")
            if not name:
                continue
            key = REL_MAP.get(rel)
            if key == "domain":
                ctx["domain"] = name
            elif key == "venue":
                ctx["venue"] = name
            elif key and isinstance(ctx.get(key), list):
                if name not in ctx[key]:
                    ctx[key].append(name)

        logger.info(
            f"[Fusion] Graph — "
            f"authors={len(ctx['authors'])} "
            f"models={len(ctx['models'])} "
            f"keywords={len(ctx['keywords'])} "
            f"domain='{ctx['domain']}'"
        )

        # ── Step 3: Keyword entity search for query terms ──────────────────────
        STOP = {"what","who","when","where","how","is","are","the","a","an",
                "of","in","to","for","and","or","by","this","that","does"}
        kws  = [
            w.strip("?.,!").lower()
            for w in query.split()
            if len(w.strip("?.,!")) > 3 and w.lower() not in STOP
        ][:6]

        if kws:
            with driver.session() as s:
                for kw in kws:
                    rows = s.run(
                        """
                        MATCH (n)
                        WHERE (n:Keyword OR n:Algorithm OR n:Dataset
                               OR n:Task OR n:Model OR n:Satellite)
                          AND toLower(coalesce(n.name,'')) CONTAINS toLower($kw)
                        OPTIONAL MATCH (p2:Paper)-[:TAGGED|USES|SOLVES]->(n)
                        RETURN n.name       AS entity,
                               labels(n)[0] AS type,
                               collect(DISTINCT p2.title)[..2] AS papers
                        LIMIT 5
                        """,
                        kw=kw,
                    ).data()
                    for row in rows:
                        entity = (row.get("entity") or "").strip()
                        if not entity:
                            continue
                        papers = [p for p in row.get("papers", []) if p]
                        fact   = f"{row.get('type','Entity')}: {entity}"
                        if papers:
                            fact += f" (in: {', '.join(papers[:2])})"
                        if fact not in ctx["related_facts"]:
                            ctx["related_facts"].append(fact)

        return ctx

    except Exception as e:
        logger.warning(f"[Fusion] Neo4j failed (non-fatal): {e}")
        return empty


# ══════════════════════════════════════════════════════════════════════════════
# CONVERSATION HISTORY
# ══════════════════════════════════════════════════════════════════════════════

def _format_history(history: list[dict]) -> str:
    if not history:
        return ""
    recent = history[-MAX_HISTORY_TURNS:]
    lines  = [
        "=== CONVERSATION HISTORY ===",
        "Use this to resolve references like 'it', 'this', 'the model'.\n",
    ]
    total = 0
    for turn in recent:
        q     = turn.get("query", "")
        a     = (turn.get("answer", "") or "")[:250] + "..."
        entry = f"[Turn {turn.get('turn','?')}]\nUser: {q}\nAssistant: {a}\n"
        if total + len(entry) > MAX_HISTORY_CHARS:
            break
        lines.append(entry)
        total += len(entry)
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# PROMPT BUILDER
# ══════════════════════════════════════════════════════════════════════════════

def _build_prompt(
    query:    str,
    chunks:   list[dict],
    graph:    dict,
    image_kws:list[str],
    history:  list[dict],
) -> str:
    parts = []

    # 1. History
    hist = _format_history(history)
    if hist:
        parts.append(hist)

    # 2. Qdrant chunks
    if chunks:
        parts.append("=== DOCUMENT EXCERPTS ===")
        for i, c in enumerate(chunks[:5], 1):
            p    = c.get("payload", c)
            text = (p.get("text") or "")[:600]
            parts.append(
                f"[{i}] {p.get('section','?')} | p.{p.get('page_num','?')} "
                f"| score={c.get('score',0):.3f}\n{text}"
            )

    # 3. Graph context — ALWAYS included when paper loaded
    graph_lines = []
    if graph.get("authors"):
        graph_lines.append(f"Authors     : {', '.join(graph['authors'][:5])}")
    if graph.get("domain"):
        graph_lines.append(f"Domain      : {graph['domain']}")
    if graph.get("models"):
        graph_lines.append(f"Models      : {', '.join(graph['models'][:5])}")
    if graph.get("datasets"):
        graph_lines.append(f"Datasets    : {', '.join(graph['datasets'][:5])}")
    if graph.get("tasks"):
        graph_lines.append(f"Tasks       : {', '.join(graph['tasks'][:5])}")
    if graph.get("keywords"):
        graph_lines.append(f"Keywords    : {', '.join(graph['keywords'][:10])}")
    if graph.get("venue"):
        graph_lines.append(f"Venue       : {graph['venue']}")
    if graph.get("related_facts"):
        graph_lines.append("Related facts:")
        for f in graph["related_facts"][:5]:
            graph_lines.append(f"  • {f}")

    parts.append("=== KNOWLEDGE GRAPH CONTEXT ===")
    if graph_lines:
        parts.extend(graph_lines)
    else:
        parts.append(
            "No entities found for this paper in Neo4j yet. "
            "Re-ingest the paper to populate the graph."
        )

    # 4. Image keywords
    if image_kws:
        parts.append(f"=== IMAGE KEYWORDS ===\n{', '.join(image_kws)}")

    # 5. Question
    parts.append(f"=== CURRENT QUESTION ===\n{query}")

    return "\n\n".join(parts)


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ══════════════════════════════════════════════════════════════════════════════

def fuse(
    query:                str,
    qdrant_chunks:        list[dict],
    paper_node_id:        str | None  = None,
    image_keywords:       list[str]   = None,
    conversation_history: list[dict]  = None,
) -> FusedContext:
    """
    Merge Qdrant chunks + Neo4j graph + history into one structured prompt.
    Graph is always fetched when paper_node_id is provided.
    """
    image_keywords       = image_keywords       or []
    conversation_history = conversation_history or []

    graph = _get_graph_context(query, paper_node_id)

    prompt = _build_prompt(
        query, qdrant_chunks, graph, image_keywords, conversation_history
    )

    chunk_dicts = []
    for c in qdrant_chunks:
        p = c.get("payload", c)
        chunk_dicts.append({
            "score":   c.get("score", 0.0),
            "text":    p.get("text", ""),
            "section": p.get("section", ""),
            "page_num":p.get("page_num", "?"),
            "title":   p.get("title", ""),
            "payload": p,
        })

    logger.info(
        f"[Fusion] prompt={len(prompt)} chars  "
        f"chunks={len(chunk_dicts)}  "
        f"graph_authors={len(graph['authors'])}  "
        f"graph_kws={len(graph['keywords'])}"
    )

    return FusedContext(
        qdrant_chunks=       qdrant_chunks,
        graph_entities=      graph,
        image_keywords=      image_keywords,
        conversation_history=conversation_history,
        prompt_text=         prompt,
        chunk_dicts=         chunk_dicts,
    )