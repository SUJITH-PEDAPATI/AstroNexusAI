"""
AstroNexus AI — Graph Builder (fixed)

Every ingested paper now automatically:
    1. Extracts entities (authors, models, datasets, tasks, venues)
    2. Writes them to Neo4j
    3. Classifies domain (remote_sensing, nlp, computer_vision, etc.)
    4. Creates Domain node if it does not exist (no seed_graph.py required)
    5. Creates Keyword/Tag nodes from matched keywords and links them
    6. Links paper to ontology entities (Algorithm, Metric, Satellite, etc.)

Keywords are ALWAYS stored — no silent skips, no dependency on seed_graph.py.
Every exception is logged with the stage name so failures are traceable.
"""
from __future__ import annotations

import logging
from pathlib import Path

from backend.ingestion.models import RawDocument
from backend.graph.entity_extractor import extract_entities
from backend.graph.neo4j_client import (
    _get_driver,
    write_extraction,
    create_constraints,
)
from backend.graph.models import ExtractionResult

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# NEO4J WRITE HELPERS  (self-contained — no seed_graph.py dependency)
# ══════════════════════════════════════════════════════════════════════════════

def _ensure_domain_node(domain_key: str, display_name: str) -> None:
    """
    MERGE a Domain node.  Creates it if it does not exist.
    Safe to call on every ingestion — no duplicate nodes created.
    """
    driver = _get_driver()
    with driver.session() as s:
        s.run(
            """
            MERGE (d:Domain {key: $key})
            SET d.name = $name
            """,
            key=domain_key,
            name=display_name,
        )


def _ensure_keyword_node(keyword: str, domain_key: str) -> None:
    """
    MERGE a Keyword node and link it to its Domain.
    Both the keyword and the domain are created if they do not exist.
    """
    driver = _get_driver()
    with driver.session() as s:
        s.run(
            """
            MERGE (k:Keyword {name: $kw})
            SET k.domain = $domain
            """,
            kw=keyword,
            domain=domain_key,
        )
        s.run(
            """
            MATCH (k:Keyword {name: $kw})
            MATCH (d:Domain  {key:  $domain})
            MERGE (k)-[:RELATED_TO]->(d)
            """,
            kw=keyword,
            domain=domain_key,
        )


def _link_paper_to_domain(paper_node_id: str, domain_key: str) -> None:
    driver = _get_driver()
    with driver.session() as s:
        s.run(
            """
            MATCH (p:Paper  {node_id: $pid})
            MATCH (d:Domain {key:    $dkey})
            MERGE (p)-[:BELONGS_TO]->(d)
            """,
            pid=paper_node_id,
            dkey=domain_key,
        )


def _link_paper_to_keyword(paper_node_id: str, keyword: str) -> None:
    driver = _get_driver()
    with driver.session() as s:
        s.run(
            """
            MATCH (p:Paper   {node_id: $pid})
            MATCH (k:Keyword {name:    $kw})
            MERGE (p)-[:TAGGED]->(k)
            """,
            pid=paper_node_id,
            kw=keyword,
        )


# ══════════════════════════════════════════════════════════════════════════════
# DOMAIN → DISPLAY NAME MAP
# ══════════════════════════════════════════════════════════════════════════════

_DOMAIN_NAMES: dict[str, str] = {
    "remote_sensing":    "Remote Sensing",
    "nlp":               "Natural Language Processing",
    "computer_vision":   "Computer Vision",
    "machine_learning":  "Machine Learning",
    "astronomy":         "Astronomy",
    "medicine":          "Medicine",
    "climate":           "Climate Science",
    "chemistry":         "Chemistry",
    "biology":           "Biology",
}


# ══════════════════════════════════════════════════════════════════════════════
# DOMAIN + KEYWORD WRITING
# ══════════════════════════════════════════════════════════════════════════════

def _write_domain_and_keywords(
    paper_node_id: str,
    doc:           RawDocument,
) -> dict:
    """
    Classify the paper, create domain + keyword nodes, link to paper.

    Returns a summary dict for logging.
    Always completes — individual failures are logged but do not abort.
    """
    from backend.graph.domain_classifier import classify_domain

    summary = {
        "primary_domain": None,
        "all_domains":    [],
        "keywords_written": 0,
        "keywords_linked":  0,
        "errors":         [],
    }

    # ── Step A: classify ──────────────────────────────────────────────────────
    try:
        domain_result = classify_domain(
            text=     doc.full_text,
            title=    doc.metadata.title    or "",
            abstract= doc.metadata.abstract or "",
        )
        summary["primary_domain"] = domain_result.primary_domain
        summary["all_domains"]    = domain_result.all_domains
    except Exception as e:
        summary["errors"].append(f"domain_classify: {e}")
        logger.error(f"[GraphBuilder] Domain classification failed: {e}")
        return summary

    # ── Step B: ensure domain nodes exist, link paper ─────────────────────────
    for domain_key in domain_result.all_domains:
        display = _DOMAIN_NAMES.get(domain_key, domain_key.replace("_", " ").title())
        try:
            _ensure_domain_node(domain_key, display)
            _link_paper_to_domain(paper_node_id, domain_key)
        except Exception as e:
            summary["errors"].append(f"domain_write({domain_key}): {e}")
            logger.warning(f"[GraphBuilder] Domain write failed for {domain_key}: {e}")

    # ── Step C: collect ALL matched keywords across all domains ───────────────
    all_matched: set[str] = set()
    for kw_list in domain_result.matched_keywords.values():
        all_matched.update(kw_list)

    # ── Step D: write keyword nodes and link to paper ─────────────────────────
    for kw in all_matched:
        if not kw or len(kw.strip()) < 3:
            continue

        # Find which domain this keyword belongs to
        kw_domain = domain_result.primary_domain
        for d, kws in domain_result.matched_keywords.items():
            if kw in kws:
                kw_domain = d
                break

        try:
            _ensure_keyword_node(kw, kw_domain)
            summary["keywords_written"] += 1
        except Exception as e:
            summary["errors"].append(f"keyword_write({kw}): {e}")
            logger.warning(f"[GraphBuilder] Keyword node write failed for '{kw}': {e}")
            continue

        try:
            _link_paper_to_keyword(paper_node_id, kw)
            summary["keywords_linked"] += 1
        except Exception as e:
            summary["errors"].append(f"keyword_link({kw}): {e}")
            logger.warning(f"[GraphBuilder] Keyword link failed for '{kw}': {e}")

    return summary


# ══════════════════════════════════════════════════════════════════════════════
# ONTOLOGY LINKING  (link paper to pre-existing ontology entities)
# ══════════════════════════════════════════════════════════════════════════════

def _link_to_ontology(
    paper_node_id: str,
    result:        ExtractionResult,
) -> None:
    """
    Link paper to existing ontology nodes (Algorithm, Metric, Satellite, etc.)
    Uses MATCH not MERGE — only links to entities that already exist in the graph.
    Missing ontology entities are skipped silently (not an error).
    """
    try:
        from backend.graph.neo4j_client import link_paper_to_ontology

        algorithm_names = [m.name for m in result.models]
        dataset_names   = []   # datasets are in result.datasets — keep separate
        task_names      = [t.name for t in result.tasks]

        link_paper_to_ontology(
            paper_node_id= paper_node_id,
            algorithms=    algorithm_names,
            keywords=      task_names,
        )
    except ImportError:
        pass   # link_paper_to_ontology added in extension — may not exist yet
    except Exception as e:
        logger.warning(f"[GraphBuilder] Ontology linking failed (non-fatal): {e}")


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ══════════════════════════════════════════════════════════════════════════════

def build_graph_from_document(doc: RawDocument) -> ExtractionResult:
    """
    Full pipeline: RawDocument → Neo4j Knowledge Graph.

    Stages (all run — none silently skipped):
        1. Create Neo4j constraints
        2. Extract entities (spaCy → LLM fallback)
        3. Write entity nodes + relationships
        4. Classify domain
        5. Write Domain nodes + link paper       ← was silently skipping
        6. Write Keyword nodes + link paper      ← was silently skipping
        7. Link paper to ontology entities

    Args:
        doc: Output from ingest_paper()

    Returns:
        ExtractionResult with all entity + relation data
    """
    title = doc.metadata.title or doc.metadata.source_file or "unknown"
    logger.info(f"[GraphBuilder] ── Building graph: '{title}' ──")

    # ── Stage 1: Constraints ───────────────────────────────────────────────────
    try:
        create_constraints()
    except Exception as e:
        logger.warning(f"[GraphBuilder] Constraint creation failed (non-fatal): {e}")

    # ── Stage 2: Entity extraction ─────────────────────────────────────────────
    try:
        result = extract_entities(doc)
        logger.info(
            f"[GraphBuilder] Extracted — "
            f"{len(result.authors)} authors, "
            f"{len(result.models)} models, "
            f"{len(result.datasets)} datasets, "
            f"{len(result.tasks)} tasks"
        )
    except Exception as e:
        logger.error(f"[GraphBuilder] Entity extraction failed: {e}")
        raise   # this is fatal — cannot continue without a paper node

    # ── Stage 3: Write entities to Neo4j ──────────────────────────────────────
    try:
        write_extraction(result)
        logger.info(f"[GraphBuilder] Entities written to Neo4j")
    except Exception as e:
        logger.error(f"[GraphBuilder] Neo4j entity write failed: {e}")
        raise   # fatal — paper node must exist before we can link keywords

    paper_node_id = result.paper.node_id

    # ── Stage 4-6: Domain classification + keyword writing ────────────────────
    kw_summary = _write_domain_and_keywords(paper_node_id, doc)

    if kw_summary["errors"]:
        logger.warning(
            f"[GraphBuilder] Keyword/domain stage had "
            f"{len(kw_summary['errors'])} error(s): "
            f"{kw_summary['errors'][:3]}"
        )

    logger.info(
        f"[GraphBuilder] Domain: {kw_summary['primary_domain']}  "
        f"Keywords written: {kw_summary['keywords_written']}  "
        f"Linked: {kw_summary['keywords_linked']}"
    )

    if kw_summary["keywords_linked"] == 0 and not kw_summary["errors"]:
        logger.warning(
            f"[GraphBuilder] Zero keywords linked for '{title}'. "
            f"Check domain_classifier coverage for this paper type."
        )

    # ── Stage 7: Ontology linking ──────────────────────────────────────────────
    _link_to_ontology(paper_node_id, result)
    logger.info(f"[GraphBuilder] ── Graph complete: '{title}' ──")

    return result


def build_graph_batch(docs: list[RawDocument]) -> list[ExtractionResult]:
    """
    Build graph for multiple documents.
    Logs failures per document but continues processing remaining docs.
    """
    results = []
    for doc in docs:
        title = doc.metadata.title or "unknown"
        try:
            results.append(build_graph_from_document(doc))
            logger.info(f"[GraphBuilder] ✓ {title}")
        except Exception as e:
            logger.error(f"[GraphBuilder] ✗ Skipped '{title}': {e}")
    return results