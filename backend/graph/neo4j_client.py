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