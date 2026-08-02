from __future__ import annotations

import logging
import os

from backend.graph.models import (
    PaperNode, AuthorNode, ModelNode,
    DatasetNode, TaskNode, VenueNode,
    GraphRelation, ExtractionResult,
)

logger = logging.getLogger(__name__)

# Module-level driver singleton
_driver = None


def _get_driver():
    """
    Lazily initialize the Neo4j driver.

    Reads connection details from environment variables:
        NEO4J_URI      = bolt://localhost:7687
        NEO4J_USER     = neo4j
        NEO4J_PASSWORD = astronexus123

    Install: pip install neo4j
    """
    global _driver

    if _driver is not None:
        return _driver

    try:
        from neo4j import GraphDatabase
    except ImportError as e:
        raise ImportError("Install required package: pip install neo4j") from e

    uri      = os.environ.get("NEO4J_URI",      "bolt://localhost:7687")
    user     = os.environ.get("NEO4J_USER",     "neo4j")
    password = os.environ.get("NEO4J_PASSWORD", "astronexus123")

    logger.info(f"[Neo4j] Connecting to {uri}")
    _driver = GraphDatabase.driver(uri, auth=(user, password))
    _driver.verify_connectivity()
    logger.info("[Neo4j] Connected.")
    return _driver


def close() -> None:
    """Close the Neo4j driver. Call on app shutdown."""
    global _driver
    if _driver:
        _driver.close()
        _driver = None
        logger.info("[Neo4j] Connection closed.")


def create_constraints() -> None:
    """
    Create uniqueness constraints for all node types.
    Ensures no duplicate nodes are created on repeated ingestion.
    Safe to call multiple times — skips if constraints already exist.
    """
    constraints = [
        "CREATE CONSTRAINT paper_id_unique   IF NOT EXISTS FOR (p:Paper)   REQUIRE p.node_id IS UNIQUE",
        "CREATE CONSTRAINT author_id_unique  IF NOT EXISTS FOR (a:Author)  REQUIRE a.node_id IS UNIQUE",
        "CREATE CONSTRAINT model_id_unique   IF NOT EXISTS FOR (m:Model)   REQUIRE m.node_id IS UNIQUE",
        "CREATE CONSTRAINT dataset_id_unique IF NOT EXISTS FOR (d:Dataset) REQUIRE d.node_id IS UNIQUE",
        "CREATE CONSTRAINT task_id_unique    IF NOT EXISTS FOR (t:Task)    REQUIRE t.node_id IS UNIQUE",
        "CREATE CONSTRAINT venue_id_unique   IF NOT EXISTS FOR (v:Venue)   REQUIRE v.node_id IS UNIQUE",
    ]

    driver = _get_driver()
    with driver.session() as session:
        for cypher in constraints:
            session.run(cypher)
    logger.info("[Neo4j] Constraints created.")


# ── Node upsert helpers ────────────────────────────────────────────────────────
# MERGE on node_id ensures idempotent writes — safe to re-run.

def upsert_paper(paper: PaperNode) -> None:
    cypher = """
    MERGE (p:Paper {node_id: $node_id})
    SET p.title       = $title,
        p.year        = $year,
        p.doi         = $doi,
        p.arxiv_id    = $arxiv_id,
        p.abstract    = $abstract,
        p.source_file = $source_file,
        p.paper_id    = $paper_id
    """
    _run(cypher, **paper.model_dump())


def upsert_author(author: AuthorNode) -> None:
    cypher = """
    MERGE (a:Author {node_id: $node_id})
    SET a.name = $name
    """
    _run(cypher, **author.model_dump())


def upsert_model(model: ModelNode) -> None:
    cypher = """
    MERGE (m:Model {node_id: $node_id})
    SET m.name        = $name,
        m.description = $description
    """
    _run(cypher, **model.model_dump())


def upsert_dataset(dataset: DatasetNode) -> None:
    cypher = """
    MERGE (d:Dataset {node_id: $node_id})
    SET d.name        = $name,
        d.description = $description
    """
    _run(cypher, **dataset.model_dump())


def upsert_task(task: TaskNode) -> None:
    cypher = """
    MERGE (t:Task {node_id: $node_id})
    SET t.name = $name
    """
    _run(cypher, **task.model_dump())


def upsert_venue(venue: VenueNode) -> None:
    cypher = """
    MERGE (v:Venue {node_id: $node_id})
    SET v.name = $name,
        v.year = $year
    """
    _run(cypher, **venue.model_dump())


def upsert_relation(relation: GraphRelation) -> None:
    """
    Create a relationship between two existing nodes.
    Uses dynamic relationship type via APOC-free approach.
    """
    cypher = f"""
    MATCH (a {{node_id: $from_id}})
    MATCH (b {{node_id: $to_id}})
    MERGE (a)-[r:{relation.relation_type.value}]->(b)
    SET r += $properties
    """
    _run(cypher,
         from_id=relation.from_id,
         to_id=relation.to_id,
         properties=relation.properties)


def write_extraction(result: ExtractionResult) -> None:
    """
    Write a full ExtractionResult to Neo4j in one call.
    Upserts all nodes then all relations.

    Args:
        result: Output from entity_extractor.py
    """
    logger.info(f"[Neo4j] Writing graph for paper: '{result.paper.title}'")

    # Upsert all nodes
    upsert_paper(result.paper)

    for author  in result.authors:  upsert_author(author)
    for model   in result.models:   upsert_model(model)
    for dataset in result.datasets: upsert_dataset(dataset)
    for task    in result.tasks:    upsert_task(task)
    for venue   in result.venues:   upsert_venue(venue)

    # Upsert all relations
    for relation in result.relations:
        upsert_relation(relation)

    logger.info(
        f"[Neo4j] Written — "
        f"{len(result.authors)} authors, "
        f"{len(result.models)} models, "
        f"{len(result.datasets)} datasets, "
        f"{len(result.tasks)} tasks, "
        f"{len(result.relations)} relations"
    )


def get_graph_stats() -> dict:
    """Return node and relation counts — useful for verifying writes."""
    cypher = """
    MATCH (n)
    RETURN labels(n)[0] AS label, count(n) AS count
    ORDER BY count DESC
    """
    driver = _get_driver()
    with driver.session() as session:
        result = session.run(cypher)
        node_counts = {row["label"]: row["count"] for row in result}

    cypher_rels = "MATCH ()-[r]->() RETURN type(r) AS type, count(r) AS count ORDER BY count DESC"
    with driver.session() as session:
        result = session.run(cypher_rels)
        rel_counts = {row["type"]: row["count"] for row in result}

    return {"nodes": node_counts, "relations": rel_counts}


def query_paper_graph(paper_title: str) -> list[dict]:
    """
    Return the full subgraph for a paper — all connected nodes and relations.
    Useful for graph visualization in the frontend.
    """
    cypher = """
    MATCH (p:Paper)
    WHERE toLower(p.title) CONTAINS toLower($title)
    MATCH (p)-[r]->(n)
    RETURN p.title AS paper,
           type(r) AS relation,
           labels(n)[0] AS target_type,
           n.name AS target_name
    """
    driver = _get_driver()
    with driver.session() as session:
        result = session.run(cypher, title=paper_title)
        return [dict(row) for row in result]


# ── Internal helper ────────────────────────────────────────────────────────────

def _run(cypher: str, **params) -> None:
    """Execute a write Cypher query."""
    driver = _get_driver()
    with driver.session() as session:
        session.run(cypher, **params)

"""
AstroNexus AI — Neo4j Client Extension

ADD THESE FUNCTIONS to the bottom of backend/graph/neo4j_client.py
Do NOT remove or modify any existing functions.

New functions added:
    write_institution()
    write_algorithm()
    write_metric()
    write_conference_or_journal()
    write_keyword()
    link_paper_to_ontology()
    get_paper_graph_context()
    get_entity_neighbours()
    _run()                        (internal helper)
"""
from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)


# ── Internal helper ────────────────────────────────────────────────────────────

def _run(cypher: str, **params) -> list[dict]:
    """
    Run a Cypher statement and return results as list of dicts.
    Uses the existing _get_driver() so connection is shared.
    """
    from backend.graph.neo4j_client import _get_driver
    driver = _get_driver()
    with driver.session() as s:
        result = s.run(cypher, **params)
        return result.data()


# ══════════════════════════════════════════════════════════════════════════════
# NEW NODE WRITERS
# ══════════════════════════════════════════════════════════════════════════════

def write_institution(name: str, country: str = "") -> None:
    """MERGE an Institution node."""
    _run(
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
    _run(
        "MERGE (a:Algorithm {name: $name}) "
        "SET a.type=$type, a.domain=$domain",
        name=name, type=algo_type, domain=domain,
    )


def write_metric(name: str, task: str = "") -> None:
    """MERGE a Metric node."""
    _run(
        "MERGE (m:Metric {name: $name}) "
        "SET m.task = $task",
        name=name, task=task,
    )


def write_conference_or_journal(
    name:        str,
    venue_type:  str = "Conference",   # "Conference" or "Journal"
    domain:      str = "",
) -> None:
    """MERGE a Conference or Journal node based on venue_type."""
    if venue_type == "Journal":
        _run(
            "MERGE (j:Journal {name: $name}) SET j.domain=$domain",
            name=name, domain=domain,
        )
    else:
        _run(
            "MERGE (c:Conference {name: $name}) SET c.domain=$domain",
            name=name, domain=domain,
        )


def write_keyword(name: str, domain: str = "") -> None:
    """MERGE a Keyword node."""
    _run(
        "MERGE (k:Keyword {name: $name}) "
        "SET k.domain = $domain",
        name=name, domain=domain,
    )


# ══════════════════════════════════════════════════════════════════════════════
# PAPER → ONTOLOGY RELATIONSHIP WRITERS
# ══════════════════════════════════════════════════════════════════════════════

def link_paper_to_ontology(
    paper_node_id:  str,
    algorithms:     list[str] = None,
    metrics:        list[str] = None,
    keywords:       list[str] = None,
    institutions:   list[str] = None,
    venue_name:     Optional[str] = None,
    venue_type:     str = "Conference",   # "Conference" or "Journal"
    satellites:     list[str] = None,
) -> dict[str, int]:
    """
    Link an ingested paper to pre-existing ontology nodes.

    All lookups use MATCH (not MERGE) so we only create edges to
    entities that already exist in the ontology — no phantom nodes.

    Returns dict of relationship counts per type.
    """
    counts: dict[str, int] = {}

    # Paper → Algorithm (USES)
    if algorithms:
        for algo in algorithms:
            _run(
                "MATCH (p:Paper {node_id:$pid}) "
                "MATCH (a:Algorithm {name:$aname}) "
                "MERGE (p)-[:USES]->(a)",
                pid=paper_node_id, aname=algo,
            )
        counts["USES_algorithm"] = len(algorithms)

    # Paper → Metric (EVALUATED_BY)
    if metrics:
        for metric in metrics:
            _run(
                "MATCH (p:Paper {node_id:$pid}) "
                "MATCH (m:Metric {name:$mname}) "
                "MERGE (p)-[:EVALUATED_BY]->(m)",
                pid=paper_node_id, mname=metric,
            )
        counts["EVALUATED_BY"] = len(metrics)

    # Paper → Keyword (TAGGED)
    if keywords:
        for kw in keywords:
            _run(
                "MATCH (p:Paper {node_id:$pid}) "
                "MATCH (k:Keyword {name:$kname}) "
                "MERGE (p)-[:TAGGED]->(k)",
                pid=paper_node_id, kname=kw,
            )
        counts["TAGGED_keyword"] = len(keywords)

    # Author → Institution (AFFILIATED_WITH)
    if institutions:
        for inst in institutions:
            _run(
                "MATCH (p:Paper {node_id:$pid}) "
                "MATCH (a:Author)-[:AUTHORED_BY]-(p) "
                "MATCH (i:Institution {name:$iname}) "
                "MERGE (a)-[:AFFILIATED_WITH]->(i)",
                pid=paper_node_id, iname=inst,
            )
        counts["AFFILIATED_WITH"] = len(institutions)

    # Paper → Conference or Journal (PRESENTED_AT)
    if venue_name:
        label = "Journal" if venue_type == "Journal" else "Conference"
        _run(
            f"MATCH (p:Paper {{node_id:$pid}}) "
            f"MATCH (v:{label} {{name:$vname}}) "
            f"MERGE (p)-[:PRESENTED_AT]->(v)",
            pid=paper_node_id, vname=venue_name,
        )
        counts["PRESENTED_AT"] = 1

    # Paper → Satellite (USES)
    if satellites:
        for sat in satellites:
            _run(
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


# ══════════════════════════════════════════════════════════════════════════════
# GRAPH CONTEXT RETRIEVAL  (used by knowledge_fusion.py)
# ══════════════════════════════════════════════════════════════════════════════
"""
Fix 1: get_paper_graph_context() — use correct property name
Fix 2: get_entity_neighbours() — simplify
Fix 3: get_graph_stats() — robustify

APPEND these to backend/graph/neo4j_client.py
replacing any existing get_paper_graph_context definition.
"""

def get_paper_graph_context(paper_node_id: str) -> dict:
    """
    Retrieve graph context for a paper.
    Tries both paper_id and node_id — handles both naming conventions.
    Also discovers actual relationship types dynamically.
    """
    driver = _get_driver()

    # Step 1: Find the paper by either property
    paper = None
    with driver.session() as s:
        for prop in ["paper_id", "node_id", "id"]:
            row = s.run(
                f"MATCH (p:Paper {{{prop}: $pid}}) "
                "RETURN p.title AS title, p.year AS year, "
                "p.doi AS doi, id(p) AS internal_id "
                "LIMIT 1",
                pid=paper_node_id,
            ).single()
            if row:
                paper = dict(row)
                paper["prop_used"] = prop
                break

    if not paper:
        return {}

    internal_id = paper["internal_id"]

    # Step 2: Get all relationships from this paper dynamically
    with driver.session() as s:
        rels = s.run(
            """
            MATCH (p:Paper)-[r]->(n)
            WHERE id(p) = $iid
            RETURN type(r) AS rel, labels(n)[0] AS label,
                   n.name AS name, n.title AS title
            """,
            iid=internal_id,
        ).data()

    # Step 3: Organise by relationship type
    context = {
        "title":      paper.get("title", ""),
        "year":       paper.get("year", ""),
        "doi":        paper.get("doi", ""),
        "authors":    [],
        "institutions":[],
        "models":     [],
        "algorithms": [],
        "datasets":   [],
        "tasks":      [],
        "metrics":    [],
        "venue":      "",
        "domain":     "",
        "keywords":   [],
        "satellites": [],
    }

    REL_MAP = {
        "AUTHORED_BY":   "authors",
        "USES":          "models",
        "SOLVES":        "tasks",
        "BELONGS_TO":    "domain",
        "TAGGED":        "keywords",
        "EVALUATED_BY":  "metrics",
        "PRESENTED_AT":  "venue",
        "AFFILIATED_WITH":"institutions",
    }

    for row in rels:
        name = row.get("name") or row.get("title") or ""
        rel  = row.get("rel", "")
        if not name:
            continue
        key = REL_MAP.get(rel)
        if key == "domain":
            context["domain"] = name
        elif key == "venue":
            context["venue"] = name
        elif key and isinstance(context.get(key), list):
            if name not in context[key]:
                context[key].append(name)

    return context
def get_graph_stats() -> dict:
    """
    Return node and relationship counts for the entire graph.
    Extended version — includes new ontology node types.
    (Replaces existing get_graph_stats if you paste over it,
     or add as get_full_graph_stats() to keep backward compat.)
    """
    node_types = [
        "Paper", "Author", "Model", "Dataset", "Task", "Venue",
        "Domain", "Tag", "Keyword",
        "Institution", "Satellite", "Sensor", "Mission",
        "Algorithm", "Metric", "LossFunction", "Conference", "Journal",
    ]

    node_counts: dict[str, int] = {}
    for ntype in node_types:
        rows = _run(f"MATCH (n:{ntype}) RETURN count(n) AS n")
        node_counts[ntype] = rows[0]["n"] if rows else 0

    rel_rows = _run(
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