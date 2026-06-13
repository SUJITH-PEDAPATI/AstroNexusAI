from __future__ import annotations

import logging
import os

from backend.embeddings.models import EmbeddedChunk

logger = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────────────────────────────────────
QDRANT_HOST       = os.environ.get("QDRANT_HOST", "localhost")
QDRANT_PORT       = int(os.environ.get("QDRANT_PORT", 6333))
COLLECTION_NAME   = "papers"
VECTOR_DIM        = 1024
DISTANCE          = "Cosine"
UPSERT_BATCH_SIZE = 64

# Module-level singleton
_client = None


def _get_client():
    """
    Lazily initialize Qdrant client.
    Connects to local Qdrant instance at localhost:6333.

    Install: pip install qdrant-client
    """
    global _client

    if _client is not None:
        return _client

    try:
        from qdrant_client import QdrantClient
    except ImportError as e:
        raise ImportError(
            "Install required package: pip install qdrant-client"
        ) from e

    logger.info(f"[VectorStore] Connecting to Qdrant at {QDRANT_HOST}:{QDRANT_PORT}")
    _client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    logger.info("[VectorStore] Connected.")
    return _client


def ensure_collection() -> None:
    """
    Create the 'papers' collection if it doesn't already exist.

    Collection config:
        - Vector size : 1024  (BGE-M3 output dim)
        - Distance    : Cosine
        - HNSW index  : default params (ef=128, m=16)
    """
    from qdrant_client.models import Distance, VectorParams

    client = _get_client()
    existing = [c.name for c in client.get_collections().collections]

    if COLLECTION_NAME in existing:
        logger.info(f"[VectorStore] Collection '{COLLECTION_NAME}' already exists — skipping creation.")
        return

    logger.info(f"[VectorStore] Creating collection '{COLLECTION_NAME}'...")
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=VECTOR_DIM,
            distance=Distance.COSINE,
        ),
    )
    logger.info(f"[VectorStore] Collection '{COLLECTION_NAME}' created.")


def upsert_chunks(embedded_chunks: list[EmbeddedChunk]) -> int:
    """
    Upsert a list of EmbeddedChunks into the Qdrant 'papers' collection.

    Dedup check: if this paper_id already exists in the collection,
    the upsert is skipped entirely. Call delete_collection() to re-ingest.

    Each point stored in Qdrant:
        id      : UUID string (chunk_id)
        vector  : 1024-dim normalized float list
        payload : {chunk_id, paper_id, chunk_index, text,
                   section, page_num, title, authors,
                   doi, year, source_file}

    Args:
        embedded_chunks: Output from embed_chunks()

    Returns:
        Number of points successfully upserted
    """
    from qdrant_client.models import (
        PointStruct,
        Filter,
        FieldCondition,
        MatchValue,
    )

    if not embedded_chunks:
        logger.warning("[VectorStore] No chunks to upsert.")
        return 0

    client = _get_client()
    ensure_collection()

    # ── Dedup check: skip if paper already ingested ────────────────────────────
    # Placed OUTSIDE the batch loop — only needs to run once per paper
    paper_id = embedded_chunks[0].paper_id

    existing = client.scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter=Filter(
            must=[
                FieldCondition(
                    key="paper_id",
                    match=MatchValue(value=paper_id),
                )
            ]
        ),
        limit=1,
        with_payload=False,
        with_vectors=False,
    )[0]

    if existing:
        logger.warning(
            f"[VectorStore] paper_id '{paper_id}' already exists in collection "
            f"— skipping upsert. Call delete_collection() to re-ingest."
        )
        return 0

    # ── Batch upsert ───────────────────────────────────────────────────────────
    total         = len(embedded_chunks)
    upserted      = 0
    total_batches = (total + UPSERT_BATCH_SIZE - 1) // UPSERT_BATCH_SIZE

    logger.info(f"[VectorStore] Upserting {total} points into '{COLLECTION_NAME}'...")

    for batch_start in range(0, total, UPSERT_BATCH_SIZE):
        batch     = embedded_chunks[batch_start : batch_start + UPSERT_BATCH_SIZE]
        batch_num = batch_start // UPSERT_BATCH_SIZE + 1

        points = [
            PointStruct(
                id=ec.chunk_id,
                vector=ec.vector,
                payload=ec.qdrant_payload(),
            )
            for ec in batch
        ]

        client.upsert(
            collection_name=COLLECTION_NAME,
            points=points,
            wait=True,
        )

        upserted += len(batch)
        logger.info(
            f"[VectorStore] Batch {batch_num}/{total_batches} — "
            f"{upserted}/{total} points upserted"
        )

    logger.info(f"[VectorStore] Upsert complete — {upserted} points in '{COLLECTION_NAME}'")
    return upserted


def get_collection_info() -> dict:
    """
    Return stats about the 'papers' collection.
    Useful for verifying upsert worked correctly.
    """
    client = _get_client()

    try:
        info = client.get_collection(COLLECTION_NAME)
        return {
            "collection":   COLLECTION_NAME,
            "total_points": info.points_count,
            "vector_size":  info.config.params.vectors.size,
            "distance":     info.config.params.vectors.distance.name,
            "status":       info.status.name,
        }
    except Exception as e:
        return {"error": str(e)}


def delete_collection() -> None:
    """
    Drop the 'papers' collection entirely.
    Use during development to reset the vector store before re-ingesting.
    """
    client = _get_client()
    existing = [c.name for c in client.get_collections().collections]

    if COLLECTION_NAME in existing:
        logger.warning(f"[VectorStore] Deleting collection '{COLLECTION_NAME}'...")
        client.delete_collection(COLLECTION_NAME)
        logger.info(f"[VectorStore] Deleted.")
    else:
        logger.info(f"[VectorStore] Collection '{COLLECTION_NAME}' does not exist.")