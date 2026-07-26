"""
AstroNexus AI — DINOv2 Feature Extractor

Architecture:
    DINOv2        → 768-dim embedding  → Qdrant (similarity search)
    clip_classifier → zero-shot label  → classification result

Fix applied:
    CLIP classification moved to clip_classifier.py which calls
    model.get_image_features(pixel_values=...) correctly — returns
    a plain tensor, not BaseModelOutputWithPooling.
"""
from __future__ import annotations

import hashlib
import logging
import traceback
from pathlib import Path

from backend.vision.models import DINOv2Result
from backend.vision.image_loader import load_image

logger = logging.getLogger(__name__)

DINOV2_MODEL = "facebook/dinov2-base"
VECTOR_DIM   = 768
QDRANT_HOST  = "localhost"
QDRANT_PORT  = 6333
COLLECTION   = "satellite_images"

_dinov2_processor = None
_dinov2_model     = None


def _cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def _load_dinov2():
    global _dinov2_processor, _dinov2_model
    if _dinov2_model is not None:
        return _dinov2_processor, _dinov2_model

    from transformers import AutoImageProcessor, AutoModel
    import torch

    device = "cuda" if _cuda_available() else "cpu"
    logger.info(f"[DINOv2] Loading {DINOV2_MODEL} on {device.upper()}...")
    _dinov2_processor = AutoImageProcessor.from_pretrained(DINOV2_MODEL)
    _dinov2_model     = AutoModel.from_pretrained(DINOV2_MODEL).to(device)
    _dinov2_model.eval()
    logger.info("[DINOv2] Ready.")
    return _dinov2_processor, _dinov2_model


def _normalize(vector: list[float]) -> list[float]:
    norm = sum(x * x for x in vector) ** 0.5
    return [x / norm for x in vector] if norm > 0 else vector


def extract_features(
    file_path:          str | Path,
    extract_patches:    bool = False,
    run_classification: bool = True,
) -> DINOv2Result:
    """
    Extract DINOv2 features from a satellite image.

    DINOv2 → 768-dim CLS token embedding (used for Qdrant + similarity).
    CLIP    → zero-shot classification (separate model, separate call).
    """
    import torch

    path = Path(file_path)
    logger.info(f"[DINOv2] Processing: {path.name}")

    pil_image, _ = load_image(path)
    processor, model = _load_dinov2()
    device = next(model.parameters()).device

    inputs = processor(images=pil_image, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    # CLS token → image-level representation  shape: (1, 768)
    cls_token   = outputs.last_hidden_state[:, 0, :]
    feature_vec = _normalize(cls_token[0].cpu().float().tolist())
    logger.info(f"[DINOv2] Feature dim: {len(feature_vec)}")

    # Patch embeddings (optional)
    patch_features = []
    if extract_patches:
        patch_tensor   = outputs.last_hidden_state[:, 1:, :]
        patch_features = [
            _normalize(p.cpu().float().tolist())
            for p in patch_tensor[0]
        ]
        logger.info(f"[DINOv2] Patch embeddings: {len(patch_features)}")

    # CLIP classification — imported from standalone module
    classification, confidence = None, None
    if run_classification:
        try:
            from backend.vision.clip_classifier import classify
            classification, confidence = classify(pil_image)
        except Exception:
            # Print full traceback — never silent
            logger.error(
                f"[DINOv2] CLIP classification raised:\n"
                f"{traceback.format_exc()}"
            )
            classification = "classification_error"
            confidence     = 0.0

    return DINOv2Result(
        file_path=                 str(path),
        feature_vector=            feature_vec,
        patch_features=            patch_features,
        vector_dim=                len(feature_vec),
        model_name=                DINOV2_MODEL,
        classification=            classification,
        classification_confidence= confidence,
    )


def store_in_qdrant(result: DINOv2Result) -> bool:
    """
    Upsert DINOv2 feature vector into Qdrant 'satellite_images' collection.

    Prints the full traceback on every failure step — no silent False returns.
    """
    # ── Import ─────────────────────────────────────────────────────────────────
    try:
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams, PointStruct
    except ImportError:
        logger.error(
            "[Qdrant] qdrant-client not installed.\n"
            "  Fix: pip install qdrant-client"
        )
        return False

    # ── Connect ────────────────────────────────────────────────────────────────
    try:
        client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        client.get_collections()   # lightweight connectivity check
        logger.info(f"[Qdrant] Connected to {QDRANT_HOST}:{QDRANT_PORT}")
    except Exception:
        logger.error(
            f"[Qdrant] Connection failed.\n"
            f"  Is Qdrant running? Start with:\n"
            f"  docker run -p 6333:6333 qdrant/qdrant\n\n"
            f"{traceback.format_exc()}"
        )
        return False

    # ── Create collection ──────────────────────────────────────────────────────
    try:
        existing = [c.name for c in client.get_collections().collections]
        if COLLECTION not in existing:
            logger.info(f"[Qdrant] Creating collection '{COLLECTION}'...")
            client.create_collection(
                collection_name= COLLECTION,
                vectors_config=  VectorParams(
                    size=     VECTOR_DIM,
                    distance= Distance.COSINE,
                ),
            )
            logger.info(f"[Qdrant] Collection '{COLLECTION}' created.")
        else:
            logger.info(f"[Qdrant] Collection '{COLLECTION}' exists.")
    except Exception:
        logger.error(
            f"[Qdrant] Collection setup failed:\n{traceback.format_exc()}"
        )
        return False

    # ── Build point ────────────────────────────────────────────────────────────
    try:
        hex_hash = hashlib.md5(result.file_path.encode("utf-8")).hexdigest()
        point_id = int(hex_hash[:15], 16)

        # Ensure vector is a plain Python list of floats
        vector = [float(x) for x in result.feature_vector]

        point = PointStruct(
            id=      point_id,
            vector=  vector,
            payload= {
                "file_path":      result.file_path,
                "file_name":      Path(result.file_path).name,
                "classification": result.classification,
                "confidence":     result.classification_confidence,
                "model":          result.model_name,
                "vector_dim":     result.vector_dim,
            },
        )
        logger.info(f"[Qdrant] Point ID: {point_id}  vector_len: {len(vector)}")
    except Exception:
        logger.error(
            f"[Qdrant] Failed to build PointStruct:\n{traceback.format_exc()}"
        )
        return False

    # ── Upsert ─────────────────────────────────────────────────────────────────
    try:
        client.upsert(
            collection_name= COLLECTION,
            points=          [point],
            wait=            True,
        )
    except Exception:
        logger.error(
            f"[Qdrant] Upsert failed:\n{traceback.format_exc()}"
        )
        return False

    # ── Verify ─────────────────────────────────────────────────────────────────
    try:
        info = client.get_collection(COLLECTION)
        logger.info(
            f"[Qdrant] ✓ Stored. "
            f"Collection '{COLLECTION}' total points: {info.points_count}"
        )
        return True
    except Exception:
        logger.error(
            f"[Qdrant] Post-upsert verification failed:\n{traceback.format_exc()}"
        )
        return False


def find_similar_images(
    query_image_path: str | Path,
    top_k:            int = 5,
) -> list[dict]:
    """Find visually similar satellite images via Qdrant cosine search."""
    try:
        from qdrant_client import QdrantClient
    except ImportError:
        logger.error("[Similarity] pip install qdrant-client")
        return []

    try:
        result = extract_features(query_image_path, run_classification=False)
    except Exception:
        logger.error(
            f"[Similarity] Feature extraction failed:\n{traceback.format_exc()}"
        )
        return []

    try:
        client   = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        existing = [c.name for c in client.get_collections().collections]

        if COLLECTION not in existing:
            logger.warning(
                f"[Similarity] Collection '{COLLECTION}' not found. "
                "Index images first."
            )
            return []

        info = client.get_collection(COLLECTION)
        if info.points_count == 0:
            logger.warning(
                f"[Similarity] Collection '{COLLECTION}' is empty."
            )
            return []

        hits = client.search(
            collection_name= COLLECTION,
            query_vector=    [float(x) for x in result.feature_vector],
            limit=           top_k,
            with_payload=    True,
            with_vectors=    False,
        )

        return [
            {
                "rank":           i + 1,
                "file_name":      h.payload.get("file_name", "unknown"),
                "file_path":      h.payload.get("file_path", ""),
                "classification": h.payload.get("classification", "unknown"),
                "confidence":     h.payload.get("confidence", 0.0),
                "score":          round(h.score, 6),
            }
            for i, h in enumerate(hits)
        ]

    except Exception:
        logger.error(
            f"[Similarity] Search failed:\n{traceback.format_exc()}"
        )
        return []