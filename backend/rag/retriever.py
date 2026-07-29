"""
AstroNexus AI — Retriever (Fixed RRF scoring bug)

Bug: RRF reranking was overwriting cosine similarity scores with
     tiny RRF weights (0.016). Abstention threshold saw 0.016 and
     refused to answer even when retrieval was correct.

Fix: Use RRF only for ORDERING. Keep original cosine score.
"""
from __future__ import annotations

import logging, os, re
from dataclasses import dataclass

logger = logging.getLogger(__name__)

QDRANT_HOST     = os.environ.get("QDRANT_HOST",       "localhost")
QDRANT_PORT     = int(os.environ.get("QDRANT_PORT",   "6333"))
COLLECTION_NAME = os.environ.get("QDRANT_COLLECTION", "papers")
RRF_K           = 60


@dataclass
class RetrievedChunk:
    chunk_id:     str
    text:         str
    score:        float   # ← always cosine similarity (0-1)
    section:      str
    page_num:     str | int
    title:        str
    paper_id:     str


def _embed_query(query: str) -> list[float]:
    from backend.ingestion.chunking import IngestedChunk
    from backend.embeddings import embed_chunks

    chunk = IngestedChunk(
        chunk_id=    "query_embed",
        paper_id=    "query",
        chunk_index= 0,
        text=        query,
        section=     "query",
        page_num=    0,
        metadata=    {},
    )

    result   = embed_chunks([chunk])
    embedded = result[0]

    if hasattr(embedded, "vector"):    return list(embedded.vector)
    if hasattr(embedded, "embedding"): return list(embedded.embedding)
    return list(embedded)


def _qdrant_search(vector: list[float], paper_id: str | None, top_k: int) -> list:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Filter, FieldCondition, MatchValue

    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    filt   = Filter(must=[FieldCondition(
        key="paper_id", match=MatchValue(value=paper_id)
    )]) if paper_id else None

    try:
        r = client.query_points(
            collection_name=COLLECTION_NAME,
            query=vector, query_filter=filt,
            limit=top_k, with_payload=True,
        )
        return r.points if hasattr(r, "points") else r
    except Exception:
        return client.search(
            collection_name=COLLECTION_NAME,
            query_vector=vector, query_filter=filt,
            limit=top_k, with_payload=True,
        )


def _bm25_rerank(query: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    """
    Reorder chunks using BM25 + RRF fusion.
    IMPORTANT: preserve original cosine score — only change ORDER.
    """
    if len(chunks) <= 1:
        return chunks
    try:
        from rank_bm25 import BM25Okapi
        def tok(t): return re.findall(r'\b[a-zA-Z0-9]+\b', t.lower())

        bm25        = BM25Okapi([tok(c.text) for c in chunks])
        bm25_scores = bm25.get_scores(tok(query))

        # Dense rank (from Qdrant order)
        dense_rank = {c.chunk_id: i for i, c in enumerate(chunks)}

        # BM25 rank
        bm25_order = sorted(range(len(chunks)), key=lambda i: bm25_scores[i], reverse=True)
        bm25_rank  = {chunks[i].chunk_id: r for r, i in enumerate(bm25_order)}

        # RRF score — used ONLY for ordering
        rrf = {
            cid: 0.6/(RRF_K + dense_rank[cid] + 1) + 0.4/(RRF_K + bm25_rank.get(cid, 99) + 1)
            for cid in dense_rank
        }

        # Sort by RRF but KEEP original cosine score
        ranked = sorted(chunks, key=lambda c: rrf[c.chunk_id], reverse=True)
        # Do NOT overwrite c.score — cosine similarity is preserved

        return ranked

    except Exception as e:
        logger.warning(f"[Retriever] BM25 failed: {e}")
        return chunks


def retrieve(
    query:    str,
    top_k:    int        = 5,
    paper_id: str | None = None,
) -> list[RetrievedChunk]:
    """
    Embed query → Qdrant search → BM25 reorder → return top-k.
    Scores are cosine similarities (0.0 to 1.0).
    """
    try:
        vector = _embed_query(query)
        logger.info(f"[Retriever] Embedded dim={len(vector)} norm={sum(x*x for x in vector)**0.5:.3f}")
    except Exception as e:
        logger.error(f"[Retriever] Embed failed: {e}")
        return []

    try:
        raw = _qdrant_search(vector, paper_id, top_k * 3)
        logger.info(f"[Retriever] Qdrant returned {len(raw)} results")
    except Exception as e:
        logger.error(f"[Retriever] Qdrant failed: {e}")
        return []

    if not raw:
        logger.warning("[Retriever] Zero results — check paper_id filter")
        return []

    chunks = [
        RetrievedChunk(
            chunk_id= str(r.id),
            text=     (r.payload or {}).get("text", ""),
            score=    r.score,    # ← cosine similarity from Qdrant
            section=  (r.payload or {}).get("section", "Unknown"),
            page_num= (r.payload or {}).get("page_num", "?"),
            title=    (r.payload or {}).get("title", ""),
            paper_id= (r.payload or {}).get("paper_id", ""),
        )
        for r in raw
    ]

    # Log scores before reranking
    logger.info(f"[Retriever] Cosine scores: {[round(c.score,3) for c in chunks[:5]]}")

    # Reorder by BM25+RRF — scores unchanged
    final = _bm25_rerank(query, chunks)[:top_k]
    top   = final[0].score if final else 0

    logger.info(f"[Retriever] Final {len(final)} chunks top_cosine={top:.3f}")
    return final


def retrieve_for_rag(query: str, top_k: int = 5, paper_id: str | None = None) -> str:
    chunks = retrieve(query, top_k=top_k, paper_id=paper_id)
    return "\n\n".join(
        f"[{c.section} p.{c.page_num} score={c.score:.3f}]\n{c.text}"
        for c in chunks
    )