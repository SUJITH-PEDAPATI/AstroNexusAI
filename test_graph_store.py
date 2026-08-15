"""
AstroNexus AI — Topic Graph Store
====================================
Maintains per-topic subgraphs in Neo4j for every user query.

Schema added (single database, isolated subgraphs):

    (:Topic {topic_id, label, created_at, query_count})
        -[:HAS_ENTITY]->  (:Entity {name, type})
        -[:HAS_KEYWORD]-> (:Keyword {name})
    (:QuerySession {session_id})-[:BELONGS_TO]->(:Topic)

Topic detection:
    - Extract keywords from the query.
    - Compute a stable topic_id from a canonical label (sorted top-3 keywords).
    - MERGE the Topic node — same topic reused on follow-up queries.
    - All entities/keywords are linked to the topic, not to a global pool.

Graph retrieval is filtered by topic_id so unrelated topics never mix.

Call from any query handler:
    from backend.agents.topic_graph_store import store_query_topic
    result = store_query_topic(query="What are black holes?")
    # result.topic_id, result.topic_label, result.entities, ...
"""
from __future__ import annotations

import hashlib
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)


# ── Simple stop-word list ─────────────────────────────────────────────────────
_STOP = {
    "what", "where", "when", "who", "how", "why", "which", "the", "is", "are",
    "was", "were", "be", "been", "being", "have", "has", "had", "does", "do",
    "did", "will", "would", "shall", "should", "may", "might", "can", "could",
    "not", "and", "or", "but", "if", "then", "than", "that", "this", "these",
    "those", "a", "an", "of", "in", "to", "for", "on", "at", "by", "from",
    "with", "about", "between", "into", "through", "during", "before",
    "after", "me", "my", "it", "its", "your", "their", "our",
    "explain", "tell", "describe", "define", "give", "show", "find", "list",
    "example", "examples", "some", "any", "all", "each", "every", "more",
    "most", "very", "just", "also", "here", "there",
}


def _extract_keywords(text: str) -> list[str]:
    """
    Extract meaningful keywords from a query using the existing keyword
    extractor if available, falling back to simple tokenisation.
    """
    try:
        from backend.voice.keyword_extractor import extract
        result = extract(text, use_llm_fallback=False)
        kws = result.keywords + [e.name.lower() for e in result.entities]
        filtered = [k for k in kws if k and len(k) > 2 and k.lower() not in _STOP]
        if filtered:
            return filtered[:12]
    except Exception:
        pass

    # Fallback: tokenise the query text
    tokens = [
        t.strip("?.,!;:\"'()[]{}").lower()
        for t in text.split()
    ]
    return [t for t in tokens if len(t) > 3 and t not in _STOP][:12]


def _make_topic_id(keywords: list[str]) -> tuple[str, str]:
    """
    Derive a stable (topic_id, topic_label) from the top-3 keywords.
    Same set of top keywords → same topic_id → topic is reused.

    Returns:
        (topic_id: str, topic_label: str)
    """
    top = sorted(set(kw.lower() for kw in keywords[:6]))[:3]
    if not top:
        fallback_id = str(uuid.uuid4())
        return fallback_id, "general"
    label     = " / ".join(top)
    stable    = hashlib.sha256(label.encode()).hexdigest()[:16]
    topic_id  = f"topic_{stable}"
    return topic_id, label


# ── Result type ───────────────────────────────────────────────────────────────

@dataclass
class TopicResult:
    topic_id:    str
    topic_label: str
    keywords:    list[str]      = field(default_factory=list)
    entities:    list[str]      = field(default_factory=list)
    is_new:      bool           = True
    error:       Optional[str]  = None


# ── Neo4j write helpers ───────────────────────────────────────────────────────

def _run(cypher: str, **params) -> None:
    """Execute a non-fatal Cypher write."""
    try:
        from backend.graph.neo4j_client import _get_driver
        driver = _get_driver()
        with driver.session() as s:
            s.run(cypher, **params)
    except Exception as e:
        logger.warning(f"[TopicGraph] Cypher failed: {e} | query: {cypher[:60]}")


def _ensure_constraints() -> None:
    """Create Topic constraint once — safe to call multiple times."""
    try:
        from backend.graph.neo4j_client import _get_driver
        driver = _get_driver()
        with driver.session() as s:
            s.run(
                "CREATE CONSTRAINT topic_id_unique IF NOT EXISTS "
                "FOR (t:Topic) REQUIRE t.topic_id IS UNIQUE"
            )
    except Exception:
        pass   # constraint may already exist


_constraints_created = False


def store_query_topic(
    query:      str,
    session_id: Optional[str]  = None,
    paper_ids:  list[str]      = (),
) -> TopicResult:
    """
    Detect or create the topic for a query, then write the topic subgraph.

    Steps:
        1. Extract keywords from the query.
        2. Derive stable topic_id from top-3 keywords.
        3. MERGE Topic node (creates if new, updates if existing).
        4. Write entities/keywords as children of the topic.
        5. Link QuerySession → Topic.
        6. If paper_ids provided, link Topic → Papers.
        7. Return TopicResult with topic_id, label, entities, is_new flag.

    Args:
        query:      The user query string.
        session_id: Optional query session ID (auto-generated if None).
        paper_ids:  Paper IDs returned by Qdrant retrieval for this query.

    Returns:
        TopicResult
    """
    global _constraints_created
    if not _constraints_created:
        _ensure_constraints()
        _constraints_created = True

    keywords = _extract_keywords(query)
    topic_id, topic_label = _make_topic_id(keywords)
    session_id = session_id or str(uuid.uuid4())[:12]
    ts = datetime.now(timezone.utc).isoformat()

    logger.info(
        f"[TopicGraph] query='{query[:50]}' "
        f"→ topic='{topic_label}' id={topic_id} "
        f"kws={keywords[:5]}"
    )

    # ── Check if topic is new ─────────────────────────────────────────────────
    is_new = True
    try:
        from backend.graph.neo4j_client import _get_driver
        driver = _get_driver()
        with driver.session() as s:
            row = s.run(
                "MATCH (t:Topic {topic_id:$tid}) RETURN t.query_count AS c",
                tid=topic_id,
            ).single()
            if row:
                is_new = False
    except Exception:
        pass

    # ── MERGE Topic node ──────────────────────────────────────────────────────
    _run(
        """
        MERGE (t:Topic {topic_id: $tid})
        SET t.label       = $label,
            t.updated_at  = $ts,
            t.query_count = coalesce(t.query_count, 0) + 1,
            t.created_at  = coalesce(t.created_at, $ts)
        """,
        tid=topic_id, label=topic_label, ts=ts,
    )

    # ── MERGE Keyword nodes and link to Topic ─────────────────────────────────
    for kw in keywords:
        kw_clean = kw.lower().strip()
        if not kw_clean or len(kw_clean) < 3:
            continue
        _run(
            """
            MERGE (k:Keyword {name: $kw})
            SET k.query_count = coalesce(k.query_count, 0) + 1,
                k.last_queried = $ts
            WITH k
            MATCH (t:Topic {topic_id: $tid})
            MERGE (t)-[:HAS_KEYWORD]->(k)
            """,
            kw=kw_clean, ts=ts, tid=topic_id,
        )

    # ── MERGE Entity nodes from query tokens and link to Topic ────────────────
    entities_written: list[str] = []
    try:
        from backend.voice.keyword_extractor import extract
        result = extract(query, use_llm_fallback=False)
        for ent in result.entities:
            name = ent.name.strip()
            if not name or len(name) < 2:
                continue
            _run(
                """
                MERGE (e:Entity {name: $name})
                SET e.type       = $etype,
                    e.last_seen  = $ts,
                    e.topic_id   = $tid
                WITH e
                MATCH (t:Topic {topic_id: $tid})
                MERGE (t)-[:HAS_ENTITY]->(e)
                """,
                name=name, etype=ent.label or "Entity", ts=ts, tid=topic_id,
            )
            entities_written.append(name)
    except Exception as e:
        logger.debug(f"[TopicGraph] Entity extraction skipped: {e}")

    # ── Link QuerySession → Topic ─────────────────────────────────────────────
    _run(
        """
        MERGE (qs:QuerySession {session_id: $sid})
        SET qs.last_query = $query, qs.last_seen = $ts
        WITH qs
        MATCH (t:Topic {topic_id: $tid})
        MERGE (qs)-[:BELONGS_TO]->(t)
        """,
        sid=session_id, query=query[:200], ts=ts, tid=topic_id,
    )

    # ── Link Topic → Papers (from Qdrant retrieval) ───────────────────────────
    for pid in paper_ids:
        _run(
            """
            MATCH (t:Topic {topic_id: $tid})
            MATCH (p:Paper  {paper_id: $pid})
            MERGE (t)-[:DISCUSSED_IN]->(p)
            """,
            tid=topic_id, pid=pid,
        )

    logger.info(
        f"[TopicGraph] {'NEW' if is_new else 'UPDATED'} topic={topic_id} "
        f"label='{topic_label}' "
        f"keywords={len(keywords)} entities={len(entities_written)}"
    )

    return TopicResult(
        topic_id=    topic_id,
        topic_label= topic_label,
        keywords=    keywords,
        entities=    entities_written,
        is_new=      is_new,
    )


def get_topic_graph(topic_id: str) -> list[dict]:
    """
    Return all nodes and relationships for a specific topic.
    Strictly filtered — never returns nodes from other topics.

    Returns:
        list of {topic, topic_id, relation, target_type, target_name}
    """
    try:
        from backend.graph.neo4j_client import _get_driver
        driver = _get_driver()
        with driver.session() as s:
            rows = s.run(
                """
                MATCH (t:Topic {topic_id: $tid})-[r]->(n)
                RETURN t.label      AS topic,
                       t.topic_id   AS topic_id,
                       type(r)      AS relation,
                       labels(n)[0] AS target_type,
                       n.name       AS target_name
                ORDER BY type(r), n.name
                """,
                tid=topic_id,
            ).data()
        return rows
    except Exception as e:
        logger.warning(f"[TopicGraph] get_topic_graph failed: {e}")
        return []


def find_topic_by_query(query: str) -> Optional[str]:
    """
    Given a query, return the topic_id it maps to (without writing anything).
    Useful for retrieval-only graph lookups.
    """
    keywords  = _extract_keywords(query)
    topic_id, _ = _make_topic_id(keywords)
    return topic_id


def list_topics(limit: int = 50) -> list[dict]:
    """
    Return all topics sorted by query count.
    Useful for graph visualization and analytics.
    """
    try:
        from backend.graph.neo4j_client import _get_driver
        driver = _get_driver()
        with driver.session() as s:
            return s.run(
                """
                MATCH (t:Topic)
                OPTIONAL MATCH (t)-[:HAS_KEYWORD]->(k:Keyword)
                OPTIONAL MATCH (t)-[:HAS_ENTITY]->(e:Entity)
                RETURN t.topic_id   AS topic_id,
                       t.label      AS label,
                       t.query_count AS query_count,
                       t.created_at  AS created_at,
                       count(DISTINCT k) AS keyword_count,
                       count(DISTINCT e) AS entity_count
                ORDER BY t.query_count DESC
                LIMIT $limit
                """,
                limit=limit,
            ).data()
    except Exception as e:
        logger.warning(f"[TopicGraph] list_topics failed: {e}")
        return []