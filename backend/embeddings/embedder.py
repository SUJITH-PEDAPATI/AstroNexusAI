from __future__ import annotations

import hashlib
import logging
import os
import time

from backend.ingestion.models import IngestedChunk
from backend.embeddings.models import EmbeddedChunk

logger = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────────────────────────────────────
# MODEL_NAME   = "Qwen/Qwen3-Embedding-0.6B"
# MODEL_NAME = "Qwen/Qwen3-Embedding-4B"
MODEL_NAME = "BAAI/bge-m3"
# VECTOR_DIM   = 2560
VECTOR_DIM = 1024
BATCH_SIZE   = 32
MAX_RETRIES  = 3
RETRY_DELAY  = 20    # seconds between retries on 503 (model cold start)

# Qwen3-Embedding instruction prefixes
DOCUMENT_INSTRUCTION = (
    "Instruct: Given a scientific research paper passage, "
    "retrieve relevant passages that answer scientific questions\nPassage: "
)
QUERY_INSTRUCTION = (
    "Instruct: Given a scientific research question, "
    "retrieve relevant passages from research papers\nQuery: "
)


_vector_cache: dict[str, list[float]] = {}
_model = None

def _get_hf_token() -> str:
    token = os.environ.get("HF_API_TOKEN") or os.environ.get("HUGGINGFACE_API_KEY")
    if not token:
        raise EnvironmentError(
            "HuggingFace API token not found.\n"
            "Add to your .env file: HF_API_TOKEN=hf_xxxxxxxxxxxx\n"
            "Get your token at: https://huggingface.co/settings/tokens"
        )
    return token


def _get_model():
    global _model

    """ Loading the model Globally if found"""

    if _model is not None:
        return _model

    try:
        from langchain_huggingface import HuggingFaceEndpointEmbeddings
    except ImportError as e:
        raise ImportError(
            "Install required package: pip install langchain-huggingface"
        ) from e

    token = _get_hf_token()
    logger.info(f"[Embedder] Connecting to HF Inference API — model: {MODEL_NAME}")

    _model = HuggingFaceEndpointEmbeddings(
        model=MODEL_NAME,
        huggingfacehub_api_token=token,
    )

    logger.info("[Embedder] HuggingFaceEndpointEmbeddings ready.")
    return _model


def _normalize(vector: list[float]) -> list[float]:
    """L2-normalize a vector for cosine similarity in Qdrant."""
    norm = sum(x * x for x in vector) ** 0.5
    return [x / norm for x in vector] if norm > 0 else vector


def _chunk_hash(text: str) -> str:
    """SHA-256 fingerprint for dedup cache."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _embed_with_retry(model, texts: list[str]) -> list[list[float]]:
    """
    Call model.embed_documents() with retry logic.
    Handles 503 (model cold start on free tier) automatically.
    """
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return model.embed_documents(texts)
        except Exception as e:
            error_str = str(e).lower()
            if "503" in error_str or "loading" in error_str:
                logger.warning(
                    f"[Embedder] Model loading (503) — "
                    f"attempt {attempt}/{MAX_RETRIES}. "
                    f"Waiting {RETRY_DELAY}s..."
                )
                time.sleep(RETRY_DELAY)
            elif "429" in error_str or "rate" in error_str:
                logger.warning("[Embedder] Rate limited — waiting 60s...")
                time.sleep(60)
            else:
                raise

    raise RuntimeError(
        f"HF Inference API failed after {MAX_RETRIES} attempts. "
        "Try again in a minute — model may still be warming up."
    )


def embed_chunks(
    chunks: list[IngestedChunk],
    batch_size: int = BATCH_SIZE,
    use_cache: bool = True,
) -> list[EmbeddedChunk]:
    """
    Embed a list of IngestedChunks using Qwen3-Embedding-0.6B
    via LangChain HuggingFaceEndpointEmbeddings (HF Inference API).

    Steps:
        1. Prepend document instruction to each chunk text
        2. Skip chunks already in the in-process cache
        3. Embed uncached chunks in batches via HF API
        4. L2-normalize all vectors
        5. Return list[EmbeddedChunk] in same order as input

    Args:
        chunks:     Output from chunk_document()
        batch_size: Texts per API call (default 32)
        use_cache:  Skip re-embedding identical text this session

    Returns:
        list[EmbeddedChunk]
    """
    if not chunks:
        return []

    model = _get_model()

    # ── Identify what needs embedding ─────────────────────────────────────────
    to_embed_indices: list[int] = []
    to_embed_texts:   list[str] = []

    for i, chunk in enumerate(chunks):
        h = _chunk_hash(chunk.text)
        if use_cache and h in _vector_cache:
            continue
        to_embed_indices.append(i)
        to_embed_texts.append(chunk.text)

    cached_count = len(chunks) - len(to_embed_indices)
    if cached_count:
        logger.info(f"[Embedder] {cached_count} chunks served from cache")

    # ── Batch embed via HF API ─────────────────────────────────────────────────
    if to_embed_texts:
        total_batches = (len(to_embed_texts) + batch_size - 1) // batch_size
        logger.info(
            f"[Embedder] Embedding {len(to_embed_texts)} chunks "
            f"in {total_batches} batch(es) via HF API..."
        )

        all_vectors: list[list[float]] = []

        for batch_start in range(0, len(to_embed_texts), batch_size):
            batch_texts = to_embed_texts[batch_start : batch_start + batch_size]
            batch_num   = batch_start // batch_size + 1

            logger.info(
                f"[Embedder] Batch {batch_num}/{total_batches} "
                f"({len(batch_texts)} chunks)"
            )

            raw_vectors = _embed_with_retry(model, batch_texts)
            all_vectors.extend(raw_vectors)

        # Normalize and store in cache
        for idx, vec in zip(to_embed_indices, all_vectors):
            h = _chunk_hash(chunks[idx].text)
            _vector_cache[h] = _normalize(vec)

        logger.info(f"[Embedder] Embedding complete — {len(all_vectors)} vectors produced")

    # ── Assemble EmbeddedChunk list ────────────────────────────────────────────
    embedded: list[EmbeddedChunk] = []
    for chunk in chunks:
        h = _chunk_hash(chunk.text)
        vector = _vector_cache[h]
        embedded.append(
            EmbeddedChunk(
                chunk=chunk,
                vector=vector,
                vector_dim=len(vector),
                embedding_model=MODEL_NAME,
            )
        )

    logger.info(f"[Embedder] Produced {len(embedded)} EmbeddedChunks")
    return embedded


def embed_query(query_text: str) -> list[float]:
    """
    Embed a single query string for retrieval.
    Uses query instruction — called by retriever in Step 5.

    Args:
        query_text: The user's search query

    Returns:
        Normalized dense vector
    """
    model = _get_model()
    instructed = query_text

    logger.info(f"[Embedder] Embedding query: '{query_text[:60]}'")

    # embed_query is LangChain's single-text embedding method
    vector = model.embed_query(instructed)
    return _normalize(vector)