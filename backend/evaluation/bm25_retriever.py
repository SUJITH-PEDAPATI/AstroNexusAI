"""
AstroNexus AI — BM25 Retriever

Provides keyword-based retrieval as a baseline for comparison
against the existing dense retriever (Qdrant + BGE-M3).

Returns RetrievedChunk objects — identical structure to the dense
retriever — so the evaluation pipeline works without any changes.

BM25 index is built at runtime from chunks stored in Qdrant.
This avoids maintaining a separate index and ensures both retrievers
operate on exactly the same corpus.

Install: pip install rank-bm25
"""
from __future__ import annotations

import logging
import re
import string
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class BM25Index:
    """
    In-memory BM25 index built from Qdrant chunk payloads.

    Attributes:
        paper_id:   The paper this index covers (None = all papers)
        chunks:     Raw chunk text in index order
        payloads:   Full Qdrant payload per chunk
        bm25:       rank_bm25.BM25Okapi instance
    """
    paper_id: Optional[str]
    chunks:   list[str]         = field(default_factory=list)
    payloads: list[dict]        = field(default_factory=list)
    bm25:     object            = field(default=None)   # BM25Okapi


def _tokenize(text: str) -> list[str]:
    """
    Simple scientific text tokenizer.

    Lowercases, removes punctuation except hyphens (model names like
    GPT-4, BERT-large need to stay intact), and splits on whitespace.
    """
    text = text.lower()
    # Keep hyphens — important for model/dataset names
    text = re.sub(r"[^\w\s\-]", " ", text)
    tokens = text.split()
    # Remove very short tokens (single chars, numbers under 2 digits)
    return [t for t in tokens if len(t) > 1]


def build_bm25_index(
    paper_id: Optional[str] = None,
    collection: str          = "papers",
) -> BM25Index:
    """
    Build a BM25 index from chunks stored in Qdrant.

    Fetches all chunks for the given paper_id (or all chunks if None)
    and builds a BM25Okapi index over their text.

    Args:
        paper_id:   Filter to chunks from this paper only.
                    Pass None to index the entire collection.
        collection: Qdrant collection name.

    Returns:
        BM25Index ready for querying.
    """
    try:
        from rank_bm25 import BM25Okapi
    except ImportError as e:
        raise ImportError("Install rank-bm25: pip install rank-bm25") from e

    from qdrant_client import QdrantClient
    from qdrant_client.models import Filter, FieldCondition, MatchValue

    client = QdrantClient(host="localhost", port=6333)

    # Build Qdrant filter
    scroll_filter = None
    if paper_id:
        scroll_filter = Filter(
            must=[FieldCondition(key="paper_id", match=MatchValue(value=paper_id))]
        )

    # Scroll all matching chunks (Qdrant paginates at 1000)
    all_chunks:   list[str]  = []
    all_payloads: list[dict] = []
    offset = None

    while True:
        results, next_offset = client.scroll(
            collection_name= collection,
            scroll_filter=   scroll_filter,
            limit=           1000,
            offset=          offset,
            with_payload=    True,
            with_vectors=    False,
        )

        for point in results:
            payload = point.payload or {}
            text    = payload.get("text", "").strip()
            if text:
                all_chunks.append(text)
                all_payloads.append(payload)

        if next_offset is None:
            break
        offset = next_offset

    if not all_chunks:
        logger.warning(
            f"[BM25] No chunks found for paper_id='{paper_id}' "
            f"in collection '{collection}'"
        )

    logger.info(f"[BM25] Building index over {len(all_chunks)} chunks")

    tokenized = [_tokenize(chunk) for chunk in all_chunks]
    bm25      = BM25Okapi(tokenized)

    return BM25Index(
        paper_id= paper_id,
        chunks=   all_chunks,
        payloads= all_payloads,
        bm25=     bm25,
    )


def bm25_retrieve(
    index:           BM25Index,
    query:           str,
    top_k:           int = 5,
) -> list:
    """
    Query a BM25Index and return top-k results as RetrievedChunk objects.

    Returns the same RetrievedChunk dataclass used by the dense retriever,
    so the evaluation pipeline requires zero changes to handle BM25 results.

    BM25 scores are NOT cosine similarities — they are unnormalised term
    frequency scores. We normalise them to [0, 1] by dividing by the
    maximum score in the result set so they can be compared meaningfully.

    Args:
        index:  BM25Index built by build_bm25_index()
        query:  Natural language question
        top_k:  Number of results to return

    Returns:
        list[RetrievedChunk] sorted by BM25 score descending
    """
    from backend.rag.retriever import RetrievedChunk

    if not index.chunks or index.bm25 is None:
        return []

    query_tokens = _tokenize(query)
    if not query_tokens:
        return []

    # BM25 scores for all chunks
    scores = index.bm25.get_scores(query_tokens)

    # Get top-k indices sorted by score descending
    import numpy as np
    top_indices = np.argsort(scores)[::-1][:top_k]

    # Normalise scores to [0, 1]
    max_score = float(scores[top_indices[0]]) if len(top_indices) > 0 else 1.0
    if max_score == 0.0:
        max_score = 1.0  # avoid division by zero for zero-score results

    results = []
    for idx in top_indices:
        score   = float(scores[idx])
        if score <= 0.0:
            continue   # skip zero-score chunks (no query term overlap)

        payload = index.payloads[idx]
        results.append(RetrievedChunk(
            chunk_id=    payload.get("chunk_id",    str(idx)),
            paper_id=    payload.get("paper_id",    ""),
            score=       round(score / max_score, 6),   # normalised
            text=        payload.get("text",        ""),
            section=     payload.get("section"),
            page_num=    payload.get("page_num"),
            title=       payload.get("title"),
            authors=     payload.get("authors",     []),
            doi=         payload.get("doi"),
            year=        payload.get("year"),
            source_file= payload.get("source_file"),
        ))

    return results