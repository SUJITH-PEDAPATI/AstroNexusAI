"""
AstroNexus AI — Graph Agent

Handles knowledge graph queries via Neo4j.
Uses the existing neo4j_client.py function-based API directly —
no class import needed.

Fix applied:
    WRONG: from backend.graph.neo4j_client import Neo4jClient
    RIGHT: from backend.graph.neo4j_client import _get_driver, get_graph_stats
"""
from __future__ import annotations

import logging

from backend.agents.state import AgentState

logger = logging.getLogger(__name__)


def graph_agent_node(state: AgentState) -> AgentState:
    """
    LangGraph node: knowledge graph Q&A via Neo4j.

    Query flow:
        1. Detect the topic_id for this query (stable hash of top keywords).
        2. Retrieve the topic subgraph (filtered by topic_id).
        3. Fall back to keyword search on Paper/Model/Dataset nodes.
        4. Format relationships as a natural language context string.
        5. Gracefully handle offline / empty database.
    """
    query = state.get("query", "")
    logger.info(f"[GraphAgent] Query: {query[:60]}")

    try:
        from backend.graph.neo4j_client import _get_driver, get_graph_stats
        from backend.agents.topic_graph_store import (
            find_topic_by_query,
            get_topic_graph,
            store_query_topic,
        )

        driver = _get_driver()

        # ── Step 1: get or create topic subgraph ─────────────────────────────
        topic_id = find_topic_by_query(query)
        topic_rows = get_topic_graph(topic_id)

        # Ensure topic is written (creates if new) — non-fatal
        try:
            topic_result = store_query_topic(query=query)
            topic_id = topic_result.topic_id
            logger.info(
                f"[GraphAgent] topic='{topic_result.topic_label}' "
                f"id={topic_id} new={topic_result.is_new}"
            )
            # Refresh topic rows after potential write
            topic_rows = get_topic_graph(topic_id)
        except Exception as _te:
            logger.debug(f"[GraphAgent] topic store skipped: {_te}")

        # ── Step 2: format topic subgraph ─────────────────────────────────────
        results = []
        if topic_rows:
            lines = [f"Topic graph for '{topic_id}':\n"]
            seen = set()
            for row in topic_rows[:20]:
                key = f"{row.get('relation')}:{row.get('target_name')}"
                if key in seen:
                    continue
                seen.add(key)
                lines.append(
                    f"  [{row.get('relation')}] → "
                    f"{row.get('target_type','Node')}: {row.get('target_name')}"
                )
            results = topic_rows
            answer = "\n".join(lines)
            logger.info(f"[GraphAgent] Topic graph: {len(topic_rows)} rows")
        else:
            # ── Step 3: fallback — keyword search across Papers/Models ─────────
            STOP_WORDS = {
                "who", "what", "when", "where", "how", "the", "is",
                "are", "was", "did", "does", "a", "an", "of", "in",
                "to", "for", "and", "or", "by", "at", "on",
                "explain", "define", "describe",
            }
            keywords = [
                w.strip("?.,!\"'").lower()
                for w in query.split()
                if len(w.strip("?.,!\"'")) > 3
                and w.strip("?.,!\"'").lower() not in STOP_WORDS
            ][:5]
            logger.info(f"[GraphAgent] Fallback keyword search: {keywords}")

            fb_rows = []
            with driver.session() as session:
                for kw in keywords:
                    rows = session.run(
                        """
                        MATCH (n)
                        WHERE (n:Paper OR n:Model OR n:Dataset OR n:Author
                               OR n:Entity OR n:Keyword)
                          AND toLower(coalesce(n.name, n.title, ''))
                              CONTAINS toLower($kw)
                        OPTIONAL MATCH (n)-[r]->(m)
                        RETURN labels(n)[0]  AS type,
                               coalesce(n.name, n.title, '') AS name,
                               type(r)       AS relation,
                               coalesce(m.name, m.title, '') AS related
                        LIMIT 8
                        """,
                        kw=kw,
                    ).data()
                    fb_rows.extend(rows)

            if fb_rows:
                lines = ["Knowledge graph results (keyword search):\n"]
                seen = set()
                for row in fb_rows[:15]:
                    key = f"{row.get('name')}:{row.get('related')}"
                    if key in seen:
                        continue
                    seen.add(key)
                    if row.get("relation") and row.get("related"):
                        lines.append(
                            f"  {row.get('type','Node')}: {row.get('name')} "
                            f"→ [{row.get('relation')}] → {row.get('related')}"
                        )
                    else:
                        lines.append(f"  {row.get('type','Node')}: {row.get('name')}")
                answer  = "\n".join(lines)
                results = fb_rows
            else:
                stats  = get_graph_stats()
                answer = (
                    f"No graph data found for this query.\n"
                    f"Graph contains: {stats.get('nodes', {})}.\n"
                    f"topic_id={topic_id}"
                )

            logger.info(f"[GraphAgent] Fallback found {len(fb_rows)} results")

        graph_context = answer

    except ImportError as e:
        # neo4j_client missing or broken — clear error
        logger.error(f"[GraphAgent] Import error: {e}")
        answer        = f"Graph agent import error: {e}"
        graph_context = ""

    except Exception as e:
        # Neo4j offline, auth failure, Cypher error — graceful fallback
        logger.warning(f"[GraphAgent] Neo4j unavailable: {e}")
        answer = (
            f"Knowledge graph is currently offline or empty.\n"
            f"Make sure Neo4j is running: docker start astronexus-neo4j\n"
            f"Error: {e}"
        )
        graph_context = ""

    logger.info(f"[GraphAgent] Answer: {answer[:100]}...")
    return {**state, "graph_context": graph_context, "final_answer": answer}