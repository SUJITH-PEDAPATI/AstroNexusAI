"""
Fix 2: QueryKeywordStore — Session.run() multiple values error.

Root cause:
    s.run("MERGE (qs:QuerySession {session_id: $sid})", query=query, sid=sid)
    Neo4j driver sees 'query' as BOTH a positional arg (the Cypher string)
    AND a keyword arg. This crashes.

Fix: never use 'query' as a parameter name in Neo4j calls.
    Rename to 'q_text', 'user_query', 'last_q' — anything except 'query'.
"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime

logger = logging.getLogger(__name__)


def store_query_keywords(
    query:      str,
    keywords:   list[str],
    session_id: str | None = None,
    query_type: str | None = None,
) -> int:
    if not keywords:
        return 0

    session_id = session_id or str(uuid.uuid4())[:8]

    try:
        from backend.graph.neo4j_client import _get_driver
        driver = _get_driver()
        count  = 0

        with driver.session() as s:
            # FIX: renamed 'query' param to 'last_q' — avoids Neo4j driver conflict
            s.run(
                """
                MERGE (qs:QuerySession {session_id: $sid})
                SET qs.last_query  = $last_q,
                    qs.query_type  = $qtype,
                    qs.last_seen   = $ts,
                    qs.query_count = coalesce(qs.query_count, 0) + 1
                """,
                sid=    session_id,
                last_q= query[:200],   # renamed from 'query' to 'last_q'
                qtype=  query_type or "unknown",
                ts=     datetime.utcnow().isoformat(),
            )

            for kw in keywords:
                if not kw or len(kw.strip()) < 3:
                    continue

                s.run(
                    """
                    MERGE (k:Keyword {name: $kw})
                    SET k.query_count  = coalesce(k.query_count, 0) + 1,
                        k.last_queried = $ts,
                        k.source       = 'user_query'
                    """,
                    kw=kw.lower().strip(),
                    ts=datetime.utcnow().isoformat(),
                )

                s.run(
                    """
                    MATCH (qs:QuerySession {session_id: $sid})
                    MATCH (k:Keyword {name: $kw})
                    MERGE (qs)-[:ASKED_ABOUT]->(k)
                    """,
                    sid=session_id,
                    kw=kw.lower().strip(),
                )
                count += 1

        logger.info(f"[QueryKeywordStore] Stored {count} keywords session={session_id}")
        return count

    except Exception as e:
        logger.warning(f"[QueryKeywordStore] Failed (non-fatal): {e}")
        return 0


def get_top_query_keywords(limit: int = 20) -> list[dict]:
    try:
        from backend.graph.neo4j_client import _get_driver
        driver = _get_driver()
        with driver.session() as s:
            rows = s.run(
                """
                MATCH (k:Keyword)
                WHERE k.source = 'user_query' AND k.query_count IS NOT NULL
                RETURN k.name AS keyword, k.query_count AS count
                ORDER BY k.query_count DESC LIMIT $lim
                """,
                lim=limit,
            ).data()
        return rows
    except Exception as e:
        logger.warning(f"[QueryKeywordStore] Analytics failed: {e}")
        return []