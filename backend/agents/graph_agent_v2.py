"""
AstroNexus AI — Graph Agent (domain-aware)

Updated to query Domain and Tag nodes seeded by seed_graph.py
and linked by graph_builder.py during paper ingestion.
"""
from __future__ import annotations

import logging

from backend.agents.state import AgentState

logger = logging.getLogger(__name__)


def graph_agent_node(state: AgentState) -> AgentState:
    """
    LangGraph node: domain-aware knowledge graph Q&A.

    Query strategy:
        1. Keyword search on Paper/Model/Dataset/Author nodes
        2. Domain search — find papers in matching domain
        3. Tag search — find papers tagged with matching keywords
        4. Combine results into structured answer
    """
    query = state.get("query", "")
    logger.info(f"[GraphAgent] Query: {query[:60]}")

    STOP_WORDS = {
        "who", "what", "when", "where", "how", "the", "is", "are",
        "was", "did", "does", "a", "an", "of", "in", "to", "for",
        "and", "or", "by", "at", "on", "authored", "written",
    }

    keywords = [
        w.strip("?.,!\"'").lower()
        for w in query.split()
        if len(w.strip("?.,!\"'")) > 3
        and w.strip("?.,!\"'").lower() not in STOP_WORDS
    ][:5]

    logger.info(f"[GraphAgent] Keywords: {keywords}")

    try:
        from backend.graph.neo4j_client import _get_driver, get_graph_stats

        driver  = _get_driver()
        results = []

        with driver.session() as session:

            # ── Query 1: Entity search (papers, models, datasets, authors) ─────
            for kw in keywords:
                rows = session.run(
                    """
                    MATCH (n)
                    WHERE (n:Paper OR n:Model OR n:Dataset OR n:Author)
                      AND toLower(coalesce(n.name, n.title, ''))
                          CONTAINS toLower($kw)
                    OPTIONAL MATCH (n)-[r]->(m)
                    RETURN
                        labels(n)[0]                         AS type,
                        coalesce(n.name, n.title, '')        AS name,
                        type(r)                              AS relation,
                        coalesce(m.name, m.title, '')        AS related
                    LIMIT 8
                    """,
                    kw=kw,
                ).data()
                results.extend(rows)

            # ── Query 2: Domain search ─────────────────────────────────────────
            for kw in keywords:
                rows = session.run(
                    """
                    MATCH (p:Paper)-[:BELONGS_TO]->(d:Domain)
                    WHERE toLower(d.name) CONTAINS toLower($kw)
                       OR toLower(d.key)  CONTAINS toLower($kw)
                    RETURN
                        'Paper'   AS type,
                        coalesce(p.title, p.name, '') AS name,
                        'BELONGS_TO' AS relation,
                        d.name    AS related
                    LIMIT 5
                    """,
                    kw=kw,
                ).data()
                results.extend(rows)

            # ── Query 3: Tag search ────────────────────────────────────────────
            for kw in keywords:
                rows = session.run(
                    """
                    MATCH (p:Paper)-[:TAGGED]->(t:Tag)
                    WHERE toLower(t.name) CONTAINS toLower($kw)
                    RETURN
                        'Paper'   AS type,
                        coalesce(p.title, p.name, '') AS name,
                        'TAGGED'  AS relation,
                        t.name    AS related
                    LIMIT 5
                    """,
                    kw=kw,
                ).data()
                results.extend(rows)

        if not results:
            stats  = get_graph_stats()
            answer = (
                f"No results found in the knowledge graph for: {', '.join(keywords)}\n"
                f"Graph stats: {stats.get('nodes', {})}\n\n"
                f"Tip: Ingest related papers via POST /upload to populate the graph."
            )
        else:
            # Deduplicate and format
            lines = [f"Knowledge graph results for '{query[:50]}':\n"]
            seen  = set()

            for row in results[:20]:
                name    = row.get("name", "").strip()
                related = row.get("related", "").strip()
                rel     = row.get("relation", "")
                rtype   = row.get("type", "Node")

                if not name:
                    continue

                key = f"{name}:{related}:{rel}"
                if key in seen:
                    continue
                seen.add(key)

                if rel and related:
                    lines.append(
                        f"  {rtype}: {name} -> [{rel}] -> {related}"
                    )
                else:
                    lines.append(f"  {rtype}: {name}")

            answer = "\n".join(lines)
            logger.info(f"[GraphAgent] Found {len(results)} graph results")

        graph_context = answer

    except ImportError as e:
        logger.error(f"[GraphAgent] Import error: {e}")
        answer        = f"Graph agent import error: {e}"
        graph_context = ""

    except Exception as e:
        logger.warning(f"[GraphAgent] Neo4j unavailable: {e}")
        answer = (
            f"Knowledge graph offline or error: {e}\n"
            f"Start Neo4j: docker start astronexus-neo4j"
        )
        graph_context = ""

    logger.info(f"[GraphAgent] Answer: {answer[:120]}...")
    return {**state, "graph_context": graph_context, "final_answer": answer}