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
        "CREATE CONSTRAINT paper_node_id_unique  IF NOT EXISTS FOR (p:Paper)   REQUIRE p.node_id  IS UNIQUE",
        "CREATE CONSTRAINT paper_paper_id_unique IF NOT EXISTS FOR (p:Paper)   REQUIRE p.paper_id IS UNIQUE",
        "CREATE CONSTRAINT author_id_unique      IF NOT EXISTS FOR (a:Author)  REQUIRE a.node_id  IS UNIQUE",
        "CREATE CONSTRAINT model_id_unique       IF NOT EXISTS FOR (m:Model)   REQUIRE m.node_id  IS UNIQUE",
        "CREATE CONSTRAINT dataset_id_unique     IF NOT EXISTS FOR (d:Dataset) REQUIRE d.node_id  IS UNIQUE",
        "CREATE CONSTRAINT task_id_unique        IF NOT EXISTS FOR (t:Task)    REQUIRE t.node_id  IS UNIQUE",
        "CREATE CONSTRAINT venue_id_unique       IF NOT EXISTS FOR (v:Venue)   REQUIRE v.node_id  IS UNIQUE",
        "CREATE CONSTRAINT keyword_name_unique   IF NOT EXISTS FOR (k:Keyword) REQUIRE k.name     IS UNIQUE",
        "CREATE CONSTRAINT entity_name_unique    IF NOT EXISTS FOR (e:Entity)  REQUIRE e.name     IS UNIQUE",
        "CREATE CONSTRAINT query_id_unique       IF NOT EXISTS FOR (q:Query)   REQUIRE q.query_id IS UNIQUE",
        "CREATE CONSTRAINT voice_id_unique       IF NOT EXISTS FOR (v:VoiceInput) REQUIRE v.voice_id IS UNIQUE",
    ]

    driver = _get_driver()
    with driver.session() as session:
        for cypher in constraints:
            session.run(cypher)
    logger.info("[Neo4j] Constraints created.")


# ── Node upsert helpers ────────────────────────────────────────────────────────
# MERGE on node_id ensures idempotent writes — safe to re-run.

def upsert_paper(paper: PaperNode) -> None:
    """
    Upsert a Paper node, MERGEing on paper_id for isolation.
    paper_id is the stable identifier that flows from PDF → Qdrant → Neo4j.
    node_id is kept for backward compatibility with relation lookups.
    """
    cypher = """
    MERGE (p:Paper {paper_id: $paper_id})
    SET p.node_id     = $node_id,
        p.title       = $title,
        p.year        = $year,
        p.doi         = $doi,
        p.arxiv_id    = $arxiv_id,
        p.abstract    = $abstract,
        p.source_file = $source_file,
        p.filename    = $source_file
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
    Searches by paper_id first (exact), then falls back to title substring.
    """
    cypher = """
    MATCH (p:Paper)
    WHERE p.paper_id = $title OR toLower(p.title) CONTAINS toLower($title)
    MATCH (p)-[r]->(n)
    RETURN p.title    AS paper,
           p.paper_id AS paper_id,
           type(r)    AS relation,
           labels(n)[0] AS target_type,
           n.name     AS target_name
    """
    driver = _get_driver()
    with driver.session() as session:
        result = session.run(cypher, title=paper_title)
        return [dict(row) for row in result]


def query_paper_graph_by_id(paper_id: str) -> list[dict]:
    """
    Return the full subgraph for a SPECIFIC paper identified by paper_id.
    Only returns nodes and relationships belonging to this paper.
    Never mixes data from other papers.
    """
    cypher = """
    MATCH (p:Paper {paper_id: $pid})
    MATCH (p)-[r]->(n)
    RETURN p.title    AS paper,
           p.paper_id AS paper_id,
           type(r)    AS relation,
           labels(n)[0] AS target_type,
           n.name     AS target_name,
           n.node_id  AS target_id
    ORDER BY type(r), n.name
    """
    driver = _get_driver()
    with driver.session() as session:
        result = session.run(cypher, pid=paper_id)
        return [dict(row) for row in result]


def upsert_query_node(
    query_id:   str,
    text:       str,
    paper_ids:  list[str],
    entity_names: list[str] | None = None,
    timestamp:  str | None = None,
) -> None:
    """
    Create a Query node and link it to every relevant Paper.
    Also links to any Entity nodes whose names appear in the query.

    Args:
        query_id:     Unique ID for this query (uuid)
        text:         The user's query text
        paper_ids:    IDs of papers retrieved for this query
        entity_names: Entity names mentioned in the query
        timestamp:    ISO timestamp (defaults to now)
    """
    import datetime as _dt
    ts = timestamp or _dt.datetime.now(_dt.timezone.utc).isoformat()

    # Create Query node
    _run(
        """
        MERGE (q:Query {query_id: $qid})
        SET q.text      = $text,
            q.timestamp = $ts
        """,
        qid=text[:500], text=text, ts=ts,
    )

    # Link Query → Paper (all papers retrieved for this query)
    for pid in paper_ids:
        _run(
            """
            MATCH (q:Query {query_id: $qid})
            MATCH (p:Paper {paper_id: $pid})
            MERGE (q)-[:RELATED_TO]->(p)
            """,
            qid=text[:500], pid=pid,
        )

    # Link Query → Entity (entities mentioned in the query)
    for name in (entity_names or []):
        if not name.strip():
            continue
        _run(
            """
            MERGE (e:Entity {name: $name})
            WITH e
            MATCH (q:Query {query_id: $qid})
            MERGE (q)-[:MENTIONS]->(e)
            """,
            name=name.strip(), qid=text[:500],
        )

    logger.info(
        f"[Neo4j] Query node written — papers={len(paper_ids)} "
        f"entities={len(entity_names or [])}"
    )


def upsert_voice_input_node(
    voice_id:    str,
    text:        str,
    paper_ids:   list[str],
    entity_names: list[str] | None = None,
    timestamp:   str | None = None,
) -> None:
    """
    Create a VoiceInput node and link it to relevant Papers and Entities.

    Args:
        voice_id:     Unique ID for this voice session
        text:         The transcribed voice text
        paper_ids:    IDs of papers relevant to this voice query
        entity_names: Entities extracted from the voice transcript
        timestamp:    ISO timestamp
    """
    import datetime as _dt
    ts = timestamp or _dt.datetime.now(_dt.timezone.utc).isoformat()

    # Create VoiceInput node
    _run(
        """
        MERGE (v:VoiceInput {voice_id: $vid})
        SET v.text      = $text,
            v.timestamp = $ts
        """,
        vid=voice_id, text=text[:2000], ts=ts,
    )

    # Link VoiceInput → Paper
    for pid in paper_ids:
        _run(
            """
            MATCH (v:VoiceInput {voice_id: $vid})
            MATCH (p:Paper {paper_id: $pid})
            MERGE (v)-[:RELATED_TO]->(p)
            """,
            vid=voice_id, pid=pid,
        )

    # Link VoiceInput → Entity
    for name in (entity_names or []):
        if not name.strip():
            continue
        _run(
            """
            MERGE (e:Entity {name: $name})
            WITH e
            MATCH (v:VoiceInput {voice_id: $vid})
            MERGE (v)-[:MENTIONS]->(e)
            """,
            name=name.strip(), vid=voice_id,
        )

    logger.info(
        f"[Neo4j] VoiceInput node written — papers={len(paper_ids)} "
        f"entities={len(entity_names or [])}"
    )


def upsert_entity_for_paper(paper_id: str, entity_name: str, entity_type: str = "Entity") -> None:
    """
    Create an Entity node and link it explicitly to a specific Paper.
    The same entity can exist in multiple papers; each paper retains its own
    HAS_ENTITY relationship — they are never mixed.

    Args:
        paper_id:    The paper this entity belongs to
        entity_name: The entity name (e.g. "BERT", "WMT-2014")
        entity_type: The entity type label (e.g. "Model", "Dataset")
    """
    _run(
        f"""
        MERGE (e:Entity {{name: $name}})
        SET e.type = $etype
        WITH e
        MATCH (p:Paper {{paper_id: $pid}})
        MERGE (p)-[:HAS_ENTITY]->(e)
        """,
        name=entity_name.strip(),
        etype=entity_type,
        pid=paper_id,
    )


def upsert_keyword_for_paper(paper_id: str, keyword: str) -> None:
    """
    Create a Keyword node and link it explicitly to a specific Paper.

    Args:
        paper_id: The paper this keyword belongs to
        keyword:  The keyword string
    """
    _run(
        """
        MERGE (k:Keyword {name: $kw})
        WITH k
        MATCH (p:Paper {paper_id: $pid})
        MERGE (p)-[:HAS_KEYWORD]->(k)
        """,
        kw=keyword.lower().strip(),
        pid=paper_id,
    )


def list_papers() -> list[dict]:
    """Return all ingested papers with their paper_id, title, and stats."""
    cypher = """
    MATCH (p:Paper)
    OPTIONAL MATCH (p)-[:HAS_ENTITY]->(e:Entity)
    OPTIONAL MATCH (p)-[:HAS_KEYWORD]->(k:Keyword)
    OPTIONAL MATCH (p)-[:TAGGED]->(kt:Keyword)
    RETURN p.paper_id AS paper_id,
           p.title    AS title,
           p.filename AS filename,
           p.year     AS year,
           count(DISTINCT e) AS entity_count,
           count(DISTINCT k) + count(DISTINCT kt) AS keyword_count
    ORDER BY p.title
    """
    driver = _get_driver()
    with driver.session() as session:
        return [dict(row) for row in session.run(cypher)]


# ── Internal helper ────────────────────────────────────────────────────────────

def _run(cypher: str, **params) -> None:
    """Execute a write Cypher query."""
    driver = _get_driver()
    with driver.session() as session:
        session.run(cypher, **params)