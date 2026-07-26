"""
AstroNexus AI — Neo4j Client

Handles connection and Cypher execution for writing extracted entities,
relationships, and querying graph statistics.
"""
from __future__ import annotations

import logging
import os
from typing import Optional, Any

logger = logging.getLogger(__name__)

NEO4J_URI      = os.environ.get("NEO4J_URI",      "bolt://localhost:7687")
NEO4J_USER     = os.environ.get("NEO4J_USER",     "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "astronexus123")

_driver = None


def _get_driver():
    global _driver
    if _driver is not None:
        return _driver

    try:
        from neo4j import GraphDatabase
        _driver = GraphDatabase.driver(
            NEO4J_URI,
            auth=(NEO4J_USER, NEO4J_PASSWORD),
        )
        logger.info(f"[Neo4j] Connected to {NEO4J_URI}")
        return _driver
    except Exception as e:
        logger.error(f"[Neo4j] Connection failed: {e}")
        raise


def _run(cypher: str, **params) -> list[dict]:
    """Execute Cypher write/query statement."""
    driver = _get_driver()
    with driver.session() as session:
        result = session.run(cypher, **params)
        return result.data()


def _run_read(cypher: str, **params) -> list[dict]:
    """Read-only Cypher statement execution helper."""
    return _run(cypher, **params)


def create_constraints() -> None:
    """Create uniqueness constraints for key node types."""
    constraints = [
        "CREATE CONSTRAINT paper_id IF NOT EXISTS FOR (p:Paper) REQUIRE p.node_id IS UNIQUE",
        "CREATE CONSTRAINT author_name IF NOT EXISTS FOR (a:Author) REQUIRE a.name IS UNIQUE",
        "CREATE CONSTRAINT model_name IF NOT EXISTS FOR (m:Model) REQUIRE m.name IS UNIQUE",
        "CREATE CONSTRAINT dataset_name IF NOT EXISTS FOR (d:Dataset) REQUIRE d.name IS UNIQUE",
        "CREATE CONSTRAINT task_name IF NOT EXISTS FOR (t:Task) REQUIRE t.name IS UNIQUE",
    ]
    for cypher in constraints:
        try:
            _run(cypher)
        except Exception as e:
            logger.debug(f"[Neo4j] Constraint warning: {e}")


def write_extraction(extraction: Any) -> None:
    """
    Write an ExtractionResult instance into Neo4j graph.
    """
    if hasattr(extraction, "paper") and extraction.paper:
        p = extraction.paper
        _run(
            "MERGE (p:Paper {node_id: $node_id}) "
            "SET p.title=$title, p.year=$year, p.doi=$doi, p.abstract=$abstract",
            node_id=p.node_id,
            title=p.title,
            year=p.year,
            doi=p.doi,
            abstract=p.abstract,
        )

    for author in getattr(extraction, "authors", []):
        _run(
            "MERGE (a:Author {name: $name}) "
            "WITH a MATCH (p:Paper {node_id: $pid}) "
            "MERGE (p)-[:AUTHORED_BY]->(a)",
            name=author.name,
            pid=extraction.paper.node_id,
        )

    for model in getattr(extraction, "models", []):
        _run(
            "MERGE (m:Model {name: $name}) "
            "WITH m MATCH (p:Paper {node_id: $pid}) "
            "MERGE (p)-[:USES]->(m)",
            name=model.name,
            pid=extraction.paper.node_id,
        )

    for dataset in getattr(extraction, "datasets", []):
        _run(
            "MERGE (d:Dataset {name: $name}) "
            "WITH d MATCH (p:Paper {node_id: $pid}) "
            "MERGE (p)-[:USES]->(d)",
            name=dataset.name,
            pid=extraction.paper.node_id,
        )

    for task in getattr(extraction, "tasks", []):
        _run(
            "MERGE (t:Task {name: $name}) "
            "WITH t MATCH (p:Paper {node_id: $pid}) "
            "MERGE (p)-[:SOLVES]->(t)",
            name=task.name,
            pid=extraction.paper.node_id,
        )


class Neo4jClient:
    """Class wrapper for Neo4j operations."""

    def __init__(self) -> None:
        self.driver = _get_driver()

    def query(self, cypher: str, **params) -> list[dict]:
        with self.driver.session() as session:
            result = session.run(cypher, **params)
            return result.data()


# ══════════════════════════════════════════════════════════════════════════════
# NEW NODE WRITERS
# ══════════════════════════════════════════════════════════════════════════════

def write_institution(name: str, country: str = "") -> None:
    """MERGE an Institution node."""
    _run_read(
        "MERGE (i:Institution {name: $name}) "
        "SET i.country = $country",
        name=name, country=country,
    )


def write_algorithm(
    name:   str,
    algo_type: str = "",
    domain: str = "",
) -> None:
    """MERGE an Algorithm node."""
    _run_read(
        "MERGE (a:Algorithm {name: $name}) "
        "SET a.type=$type, a.domain=$domain",
        name=name, type=algo_type, domain=domain,
    )


def write_metric(name: str, task: str = "") -> None:
    """MERGE a Metric node."""
    _run_read(
        "MERGE (m:Metric {name: $name}) "
        "SET m.task = $task",
        name=name, task=task,
    )


def write_conference_or_journal(
    name:        str,
    venue_type:  str = "Conference",
    domain:      str = "",
) -> None:
    """MERGE a Conference or Journal node based on venue_type."""
    if venue_type == "Journal":
        _run_read(
            "MERGE (j:Journal {name: $name}) SET j.domain=$domain",
            name=name, domain=domain,
        )
    else:
        _run_read(
            "MERGE (c:Conference {name: $name}) SET c.domain=$domain",
            name=name, domain=domain,
        )


def write_keyword(name: str, domain: str = "") -> None:
    """MERGE a Keyword node."""
    _run_read(
        "MERGE (k:Keyword {name: $name}) "
        "SET k.domain = $domain",
        name=name, domain=domain,
    )


def link_paper_to_ontology(
    paper_node_id:  str,
    algorithms:     list[str] = None,
    metrics:        list[str] = None,
    keywords:       list[str] = None,
    institutions:   list[str] = None,
    venue_name:     Optional[str] = None,
    venue_type:     str = "Conference",
    satellites:     list[str] = None,
) -> dict[str, int]:
    counts: dict[str, int] = {}

    if algorithms:
        for algo in algorithms:
            _run_read(
                "MATCH (p:Paper {node_id:$pid}) "
                "MATCH (a:Algorithm {name:$aname}) "
                "MERGE (p)-[:USES]->(a)",
                pid=paper_node_id, aname=algo,
            )
        counts["USES_algorithm"] = len(algorithms)

    if metrics:
        for metric in metrics:
            _run_read(
                "MATCH (p:Paper {node_id:$pid}) "
                "MATCH (m:Metric {name:$mname}) "
                "MERGE (p)-[:EVALUATED_BY]->(m)",
                pid=paper_node_id, mname=metric,
            )
        counts["EVALUATED_BY"] = len(metrics)

    if keywords:
        for kw in keywords:
            _run_read(
                "MATCH (p:Paper {node_id:$pid}) "
                "MATCH (k:Keyword {name:$kname}) "
                "MERGE (p)-[:TAGGED]->(k)",
                pid=paper_node_id, kname=kw,
            )
        counts["TAGGED_keyword"] = len(keywords)

    if institutions:
        for inst in institutions:
            _run_read(
                "MATCH (p:Paper {node_id:$pid}) "
                "MATCH (a:Author)-[:AUTHORED_BY]-(p) "
                "MATCH (i:Institution {name:$iname}) "
                "MERGE (a)-[:AFFILIATED_WITH]->(i)",
                pid=paper_node_id, iname=inst,
            )
        counts["AFFILIATED_WITH"] = len(institutions)

    if venue_name:
        label = "Journal" if venue_type == "Journal" else "Conference"
        _run_read(
            f"MATCH (p:Paper {{node_id:$pid}}) "
            f"MATCH (v:{label} {{name:$vname}}) "
            f"MERGE (p)-[:PRESENTED_AT]->(v)",
            pid=paper_node_id, vname=venue_name,
        )
        counts["PRESENTED_AT"] = 1

    if satellites:
        for sat in satellites:
            _run_read(
                "MATCH (p:Paper {node_id:$pid}) "
                "MATCH (s:Satellite {name:$sname}) "
                "MERGE (p)-[:USES]->(s)",
                pid=paper_node_id, sname=sat,
            )
        counts["USES_satellite"] = len(satellites)

    logger.info(
        f"[Neo4j] Ontology links for {paper_node_id[:12]}...: {counts}"
    )
    return counts


def get_paper_graph_context(paper_node_id: str) -> dict:
    rows = _run_read(
        """
        MATCH (p:Paper {node_id: $pid})
        OPTIONAL MATCH (p)-[:AUTHORED_BY]->(au:Author)
        OPTIONAL MATCH (au)-[:AFFILIATED_WITH]->(inst:Institution)
        OPTIONAL MATCH (p)-[:USES]->(m:Model)
        OPTIONAL MATCH (p)-[:USES]->(a:Algorithm)
        OPTIONAL MATCH (p)-[:USES]->(d:Dataset)
        OPTIONAL MATCH (p)-[:SOLVES]->(t:Task)
        OPTIONAL MATCH (p)-[:EVALUATED_BY]->(met:Metric)
        OPTIONAL MATCH (p)-[:PRESENTED_AT]->(v)
        OPTIONAL MATCH (p)-[:BELONGS_TO]->(dom:Domain)
        OPTIONAL MATCH (p)-[:TAGGED]->(kw:Keyword)
        OPTIONAL MATCH (p)-[:USES]->(sat:Satellite)
        RETURN
            p.title          AS title,
            p.year           AS year,
            p.doi            AS doi,
            collect(DISTINCT au.name)   AS authors,
            collect(DISTINCT inst.name) AS institutions,
            collect(DISTINCT m.name)    AS models,
            collect(DISTINCT a.name)    AS algorithms,
            collect(DISTINCT d.name)    AS datasets,
            collect(DISTINCT t.name)    AS tasks,
            collect(DISTINCT met.name)  AS metrics,
            v.name                      AS venue,
            dom.name                    AS domain,
            collect(DISTINCT kw.name)   AS keywords,
            collect(DISTINCT sat.name)  AS satellites
        """,
        pid=paper_node_id,
    )

    if not rows:
        return {}

    row = rows[0]
    return {
        "title":        row.get("title", ""),
        "year":         row.get("year", ""),
        "doi":          row.get("doi", ""),
        "authors":      [a for a in row.get("authors", [])       if a],
        "institutions": [i for i in row.get("institutions", [])  if i],
        "models":       [m for m in row.get("models", [])        if m],
        "algorithms":   [a for a in row.get("algorithms", [])    if a],
        "datasets":     [d for d in row.get("datasets", [])      if d],
        "tasks":        [t for t in row.get("tasks", [])         if t],
        "metrics":      [m for m in row.get("metrics", [])       if m],
        "venue":        row.get("venue", ""),
        "domain":       row.get("domain", ""),
        "keywords":     [k for k in row.get("keywords", [])      if k],
        "satellites":   [s for s in row.get("satellites", [])    if s],
    }


def get_entity_neighbours(
    entity_name: str,
    depth:       int = 1,
    limit:       int = 20,
) -> list[dict]:
    rows = _run_read(
        f"""
        MATCH (n {{name: $name}})-[r*1..{depth}]->(m)
        RETURN
            n.name           AS source,
            [rel in r | type(rel)] AS relations,
            m.name           AS target,
            labels(m)[0]     AS target_type
        LIMIT $limit
        """,
        name=entity_name, limit=limit,
    )
    return [
        {
            "source":      row.get("source"),
            "relations":   row.get("relations", []),
            "target":      row.get("target"),
            "target_type": row.get("target_type"),
        }
        for row in rows
    ]


def get_graph_stats() -> dict:
    node_types = [
        "Paper", "Author", "Model", "Dataset", "Task", "Venue",
        "Domain", "Tag", "Keyword",
        "Institution", "Satellite", "Sensor", "Mission",
        "Algorithm", "Metric", "LossFunction", "Conference", "Journal",
    ]

    node_counts: dict[str, int] = {}
    for ntype in node_types:
        rows = _run_read(f"MATCH (n:{ntype}) RETURN count(n) AS n")
        node_counts[ntype] = rows[0]["n"] if rows else 0

    rel_rows = _run_read(
        "MATCH ()-[r]->() "
        "RETURN type(r) AS t, count(r) AS n "
        "ORDER BY n DESC"
    )
    rel_counts = {row["t"]: row["n"] for row in rel_rows}

    return {
        "nodes":         {k: v for k, v in node_counts.items() if v > 0},
        "relations":     rel_counts,
        "total_nodes":   sum(node_counts.values()),
        "total_relations": sum(rel_counts.values()),
    }


def query_paper_graph(query_str: str) -> list[dict]:
    """
    Search paper nodes and relationships in the graph.
    Convenience alias for graph agents.
    """
    return _run_read(
        """
        MATCH (n)
        WHERE (n:Paper OR n:Model OR n:Dataset OR n:Author)
          AND toLower(coalesce(n.name, n.title, '')) CONTAINS toLower($q)
        OPTIONAL MATCH (n)-[r]->(m)
        RETURN
            labels(n)[0]                  AS type,
            coalesce(n.name, n.title, '') AS name,
            type(r)                       AS relation,
            coalesce(m.name, m.title, '') AS related
        LIMIT 10
        """,
        q=query_str,
    )