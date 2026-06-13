from __future__ import annotations

import logging

from backend.ingestion.models import RawDocument
from backend.graph.entity_extractor import extract_entities
from backend.graph.neo4j_client import write_extraction, create_constraints
from backend.graph.models import ExtractionResult

logger = logging.getLogger(__name__)


def build_graph_from_document(doc: RawDocument) -> ExtractionResult:
    """
    Main orchestrator: RawDocument → Neo4j Knowledge Graph.

    Pipeline:
        1. Ensure Neo4j constraints exist
        2. Extract entities (spaCy → LLM fallback)
        3. Write all nodes and relations to Neo4j

    Args:
        doc: Output from ingest_paper()

    Returns:
        ExtractionResult (for inspection/testing)
    """
    logger.info(f"[GraphBuilder] Building graph for: '{doc.metadata.title}'")

    # Step 1: Ensure constraints (idempotent)
    create_constraints()

    # Step 2: Extract entities
    result = extract_entities(doc)

    # Step 3: Write to Neo4j
    write_extraction(result)

    logger.info(f"[GraphBuilder] Graph built successfully.")
    return result


def build_graph_batch(docs: list[RawDocument]) -> list[ExtractionResult]:
    """
    Build graph for multiple documents, skipping failures with a warning.

    Args:
        docs: List of RawDocuments from ingest_batch()

    Returns:
        list[ExtractionResult]
    """
    results = []
    for doc in docs:
        try:
            results.append(build_graph_from_document(doc))
        except Exception as e:
            logger.warning(f"[GraphBuilder] Skipped '{doc.metadata.title}': {e}")
    return results