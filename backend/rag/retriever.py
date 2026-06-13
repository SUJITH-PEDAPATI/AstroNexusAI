from __future__ import annotations

import logging
from dataclasses import dataclass

from backend.embeddings.embedder import embed_query
from backend.rag.vector_store import _get_client, COLLECTION_NAME

logger = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────────────────────────────────────
DEFAULT_TOP_K         = 5      # number of chunks to retrieve
DEFAULT_SCORE_THRESHOLD = 0.05  # minimum cosine similarity score (0-1)


@dataclass
class RetrievedChunk:
    """A single retrieval result returned from Qdrant."""
    chunk_id:    str
    paper_id:    str
    score:       float          # cosine similarity (higher = more relevant)
    text:        str
    section:     str | None
    page_num:    int | None
    title:       str | None
    authors:     list[str]
    doi:         str | None
    year:        int | None
    source_file: str | None


def retrieve(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    score_threshold: float = DEFAULT_SCORE_THRESHOLD,
    filter_paper_id: str | None = None,
    filter_section: str | None = None,
    filter_year_from: int | None = None,
) -> list[RetrievedChunk]:
    """
    Semantic search over the Qdrant 'papers' collection.

    Steps:
        1. Embed the query using the same BGE-M3 model (with query instruction)
        2. Run Qdrant vector search with optional metadata filters
        3. Return results above the score threshold as RetrievedChunk objects

    Args:
        query:            Natural language search query
        top_k:            Max number of results to return (default 5)
        score_threshold:  Minimum similarity score to include (default 0.55)
        filter_paper_id:  Only return chunks from this specific paper
        filter_section:   Only return chunks from this section (e.g. "Methods")
        filter_year_from: Only return chunks from papers published >= this year

    Returns:
        list[RetrievedChunk] sorted by score descending
    """
    from qdrant_client.models import Filter, FieldCondition, MatchValue, Range

    logger.info(f"[Retriever] Query: '{query[:80]}'")

    # ── Step 1: Embed query ────────────────────────────────────────────────────
    query_vector = embed_query(query)
    logger.info(f"[Retriever] Query embedded — dim={len(query_vector)}")

    # ── Step 2: Build optional filters ────────────────────────────────────────
    conditions = []

    if filter_paper_id:
        conditions.append(
            FieldCondition(key="paper_id", match=MatchValue(value=filter_paper_id))
        )

    if filter_section:
        conditions.append(
            FieldCondition(key="section", match=MatchValue(value=filter_section))
        )

    if filter_year_from:
        conditions.append(
            FieldCondition(key="year", range=Range(gte=filter_year_from))
        )

    qdrant_filter = Filter(must=conditions) if conditions else None

    # ── Step 3: Vector search ─────────────────────────────────────────────────
    client = _get_client()

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k,
        score_threshold=score_threshold,
        query_filter=qdrant_filter,
        with_payload=True,
        with_vectors=False,
    )
    
    logger.info(f"[Retriever] {len(results.points)} results above threshold {score_threshold}")

    # ── Step 4: Map to RetrievedChunk ─────────────────────────────────────────
    retrieved: list[RetrievedChunk] = []

    for hit in results.points:
        p = hit.payload or {}
        retrieved.append(
            RetrievedChunk(
                chunk_id=p.get("chunk_id", str(hit.id)),
                paper_id=p.get("paper_id", ""),
                score=round(hit.score, 6),
                text=p.get("text", ""),
                section=p.get("section"),
                page_num=p.get("page_num"),
                title=p.get("title"),
                authors=p.get("authors", []),
                doi=p.get("doi"),
                year=p.get("year"),
                source_file=p.get("source_file"),
            )
        )

    return retrieved


def retrieve_for_rag(query: str, top_k: int = DEFAULT_TOP_K) -> str:
    """
    Convenience wrapper that returns retrieved chunks formatted
    as a single context string ready to inject into an LLM prompt.

    Used by the ResearchAgent in Week 7.

    Format:
        [1] (score: 0.82) — "Attention Is All You Need" (2017), Methods, p.3
        <chunk text>

        [2] (score: 0.79) — ...
    """
    chunks = retrieve(query, top_k=top_k)

    if not chunks:
        return "No relevant context found in the knowledge base."

    parts: list[str] = []
    for i, c in enumerate(chunks, start=1):
        header = (
            f"[{i}] (score: {c.score:.4f}) — "
            f"\"{c.title or 'Unknown'}\" ({c.year or 'n/a'}), "
            f"{c.section or 'Unknown section'}, p.{c.page_num or '?'}"
        )
        parts.append(f"{header}\n{c.text}")

    return "\n\n".join(parts)