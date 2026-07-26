"""
AstroNexus AI — Graph Builder (updated)

Adds domain classification and keyword tagging to the existing
entity extraction pipeline.

After extraction:
    Paper → Domain    (BELONGS_TO)
    Paper → Tag       (TAGGED)  ← keywords matched in the paper text
"""
from __future__ import annotations

import logging
from pathlib import Path

from backend.ingestion.models import RawDocument
from backend.graph.entity_extractor import extract_entities
from backend.graph.neo4j_client import (
    write_extraction,
    create_constraints,
)
from backend.graph.models import ExtractionResult

logger = logging.getLogger(__name__)


def _write_domain_and_tags(
    paper_node_id: str,
    domain_result,
) -> None:
    """
    Write domain classification and keyword tags to Neo4j.

    Creates:
        (Paper)-[:BELONGS_TO]->(Domain)
        (Paper)-[:TAGGED]->(Tag)   ← only tags that exist from seed_graph.py
    """
    from backend.graph.neo4j_client import _run

    # Link paper to its primary domain
    _run(
        """
        MATCH (p:Paper {node_id: $paper_id})
        MATCH (d:Domain {key: $domain_key})
        MERGE (p)-[:BELONGS_TO]->(d)
        """,
        paper_id=   paper_node_id,
        domain_key= domain_result.primary_domain,
    )

    # Link paper to all matched keyword tags
    all_matched = []
    for kws in domain_result.matched_keywords.values():
        all_matched.extend(kws)

    for kw in set(all_matched):
        _run(
            """
            MATCH (p:Paper {node_id: $paper_id})
            MATCH (t:Tag {name: $kw})
            MERGE (p)-[:TAGGED]->(t)
            """,
            paper_id= paper_node_id,
            kw=       kw,
        )

    logger.info(
        f"[GraphBuilder] Domain: {domain_result.primary_domain}  "
        f"Tags linked: {len(set(all_matched))}"
    )


def build_graph_from_document(doc: RawDocument) -> ExtractionResult:
    """
    Main orchestrator: RawDocument → Neo4j Knowledge Graph.

    Pipeline:
        1. Ensure Neo4j constraints exist
        2. Extract entities (spaCy → LLM fallback)
        3. Write all nodes and relations to Neo4j
        4. Classify paper domain (keyword matching)
        5. Write domain and keyword tag links

    Args:
        doc: Output from ingest_paper()

    Returns:
        ExtractionResult (for inspection/testing)
    """
    logger.info(f"[GraphBuilder] Building graph: '{doc.metadata.title}'")

    # Step 1: Ensure constraints
    create_constraints()

    # Step 2: Extract entities
    result = extract_entities(doc)

    # Step 3: Write entities to Neo4j
    write_extraction(result)

    # Step 4: Classify domain
    try:
        from backend.graph.domain_classifier import classify_domain

        domain_result = classify_domain(
            text=     doc.full_text,
            title=    doc.metadata.title or "",
            abstract= doc.metadata.abstract or "",
        )

        logger.info(
            f"[GraphBuilder] Domain classification: "
            f"{domain_result.primary_domain}  "
            f"(all: {domain_result.all_domains})"
        )

        # Step 5: Write domain + tags
        _write_domain_and_tags(result.paper.node_id, domain_result)

    except Exception as e:
        # Non-fatal — entity extraction already succeeded
        logger.warning(f"[GraphBuilder] Domain tagging failed (non-fatal): {e}")

    logger.info("[GraphBuilder] Graph built successfully.")
    return result


def build_graph_batch(docs: list[RawDocument]) -> list[ExtractionResult]:
    """
    Build graph for multiple documents, skipping failures.
    """
    results = []
    for doc in docs:
        try:
            results.append(build_graph_from_document(doc))
        except Exception as e:
            logger.warning(f"[GraphBuilder] Skipped '{doc.metadata.title}': {e}")
    return results