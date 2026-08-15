from __future__ import annotations

import hashlib
import logging
import os
import time
from pathlib import Path

from backend.ingestion.models import IngestedChunk
from backend.embeddings.models import EmbeddedChunk

logger = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────────────────────────────────────
MODEL_NAME  = "BAAI/bge-m3"
VECTOR_DIM  = 1024
BATCH_SIZE  = 32
MAX_RETRIES = 3
RETRY_DELAY = 20    # seconds between retries on 503 (model cold start)

# When True (default), use local sentence-transformers instead of HF Inference API.
# Set USE_LOCAL_EMBEDDING=false in .env to revert to remote HF endpoint.
USE_LOCAL_EMBEDDING: bool = (
    os.environ.get("USE_LOCAL_EMBEDDING", "true").lower() != "false"
)

# Local model cache directory (relative to project root)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
LOCAL_MODEL_CACHE = str(_PROJECT_ROOT / "models")

# Instruction prefixes (kept for future Qwen3 use)
DOCUMENT_INSTRUCTION = (
    "Instruct: Given a scientific research paper passage, "
    "retrieve relevant passages that answer scientific questions\nPassage: "
)
QUERY_INSTRUCTION = (
    "Instruct: Given a scientific research question, "
    "retrieve relevant passages from research papers\nQuery: "
)


_vector_cache: dict[str, list[float]] = {}
_model       = None   # remote HF endpoint model
_local_model = None   # local SentenceTransformer model

def _get_hf_token() -> str:
    # Load .env explicitly — this module may be imported before any entry
    # point (evaluate.py, main.py, etc.) has had a chance to call load_dotenv.
    try:
        from dotenv import load_dotenv as _load_dotenv
        from pathlib import Path as _Path
        # Walk up from this file to the project root (.env lives there)
        _here = _Path(__file__).resolve()
        for _parent in [_here.parent, _here.parent.parent, _here.parent.parent.parent]:
            _env = _parent / ".env"
            if _env.exists():
                _load_dotenv(_env, override=False)  # override=False: don't clobber shell vars
                break
    except ImportError:
        pass  # python-dotenv not installed; rely on shell environment

    token = os.environ.get("HF_API_TOKEN") or os.environ.get("HUGGINGFACE_API_KEY")
    if not token:
        raise EnvironmentError(
            "HuggingFace API token not found.\n"
            "Add to your .env file: HF_API_TOKEN=hf_xxxxxxxxxxxx\n"
            "Get your token at: https://huggingface.co/settings/tokens"
        )
    return token


def _get_local_model():
    """
    Load BAAI/bge-m3 from local model cache via sentence-transformers.
    This is the preferred path for evaluation — no HF API quota consumed.
    """
    global _local_model
    if _local_model is not None:
        return _local_model

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        raise ImportError(
            "Install required package: pip install sentence-transformers"
        ) from e

    logger.info(
        f"[Embedder] Loading local model '{MODEL_NAME}' "
        f"from cache: {LOCAL_MODEL_CACHE}"
    )
    _local_model = SentenceTransformer(
        MODEL_NAME,
        cache_folder=LOCAL_MODEL_CACHE,
    )

    # Dimension assertion — catches model/collection mismatches early
    _probe = _local_model.encode("probe", normalize_embeddings=True)
    actual_dim = len(_probe)
    if actual_dim != VECTOR_DIM:
        raise RuntimeError(
            f"[Embedder] Dimension mismatch: model produces {actual_dim}-dim vectors "
            f"but VECTOR_DIM={VECTOR_DIM}. "
            f"Check MODEL_NAME and re-index Qdrant if necessary."
        )
    logger.info(
        f"[Embedder] Local model ready — dim={actual_dim}, "
        f"matches Qdrant collection ({VECTOR_DIM}-dim)."
    )
    return _local_model


def _get_model():
    """Load the remote HF Inference API endpoint model (singleton)."""
    global _model
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
    Handles 503/504 (model cold start / gateway timeout) automatically.
    """
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return model.embed_documents(texts)
        except Exception as e:
            error_str = str(e).lower()
            if any(code in error_str for code in ["503", "504", "502", "timeout", "loading"]):
                logger.warning(
                    f"[Embedder] HF API timeout/loading issue ({e}) — "
                    f"attempt {attempt}/{MAX_RETRIES}. "
                    f"Waiting {RETRY_DELAY}s..."
                )
                time.sleep(RETRY_DELAY)
            elif "429" in error_str or "rate" in error_str:
                logger.warning("[Embedder] Rate limited — waiting 30s...")
                time.sleep(30)
            else:
                if attempt == MAX_RETRIES:
                    raise
                logger.warning(f"[Embedder] Attempt {attempt} failed ({e}), retrying in {RETRY_DELAY}s...")
                time.sleep(RETRY_DELAY)

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
    Embed a list of IngestedChunks using BAAI/bge-m3.

    When USE_LOCAL_EMBEDDING=true (default), uses local sentence-transformers
    — no HF API quota consumed. Set USE_LOCAL_EMBEDDING=false in .env to
    revert to the remote HF Inference API endpoint.

    Steps:
        1. Skip chunks already in the in-process cache
        2. Embed uncached chunks in batches
        3. L2-normalize all vectors
        4. Return list[EmbeddedChunk] in same order as input

    Args:
        chunks:     Output from chunk_document()
        batch_size: Texts per call (default 32)
        use_cache:  Skip re-embedding identical text this session

    Returns:
        list[EmbeddedChunk]
    """
    if not chunks:
        return []

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

    # ── Batch embed ────────────────────────────────────────────────────────────
    if to_embed_texts:
        total_batches = (len(to_embed_texts) + batch_size - 1) // batch_size
        backend_name  = "local sentence-transformers" if USE_LOCAL_EMBEDDING else "HF Inference API"
        logger.info(
            f"[Embedder] Embedding {len(to_embed_texts)} chunks "
            f"in {total_batches} batch(es) via {backend_name}..."
        )

        all_vectors: list[list[float]] = []

        if USE_LOCAL_EMBEDDING:
            local_model = _get_local_model()
            for batch_start in range(0, len(to_embed_texts), batch_size):
                batch_texts = to_embed_texts[batch_start : batch_start + batch_size]
                batch_num   = batch_start // batch_size + 1
                logger.info(
                    f"[Embedder] Batch {batch_num}/{total_batches} "
                    f"({len(batch_texts)} chunks) — local"
                )
                vecs = local_model.encode(
                    batch_texts,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )
                all_vectors.extend(vecs.tolist())
        else:
            model = _get_model()
            for batch_start in range(0, len(to_embed_texts), batch_size):
                batch_texts = to_embed_texts[batch_start : batch_start + batch_size]
                batch_num   = batch_start // batch_size + 1
                logger.info(
                    f"[Embedder] Batch {batch_num}/{total_batches} "
                    f"({len(batch_texts)} chunks) — HF API"
                )
                raw_vectors = _embed_with_retry(model, batch_texts)
                all_vectors.extend(raw_vectors)

        # Normalize and store in cache
        for idx, vec in zip(to_embed_indices, all_vectors):
            h = _chunk_hash(chunks[idx].text)
            _vector_cache[h] = _normalize(list(vec)) if not isinstance(vec, list) else _normalize(vec)

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
    Called by retriever._embed_query() and ablation._eval_one_retrieval().

    When USE_LOCAL_EMBEDDING=true (default): uses local SentenceTransformer.
    When USE_LOCAL_EMBEDDING=false: uses remote HF Inference API with retry.

    Args:
        query_text: The user's search query

    Returns:
        Normalized 1024-dim dense vector (matches Qdrant 'papers' collection)
    """
    logger.info(f"[Embedder] Embedding query (backend={'local' if USE_LOCAL_EMBEDDING else 'HF API'}): '{query_text[:60]}'")

    if USE_LOCAL_EMBEDDING:
        local_model = _get_local_model()
        vec = local_model.encode(query_text, normalize_embeddings=True)
        vector = vec.tolist() if hasattr(vec, "tolist") else list(vec)
        logger.info(f"[Embedder] Query embedding dim={len(vector)}")
        return vector
    else:
        model = _get_model()
        # Use _embed_with_retry to handle 504 Gateway Timeouts
        vectors = _embed_with_retry(model, [query_text])
        return _normalize(vectors[0])