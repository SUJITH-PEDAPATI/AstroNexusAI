"""
AstroNexus AI — DINOv2 Feature Extractor

Architecture:
    DINOv2  → 768-dim image embedding  → Qdrant (similarity search)
    CLIP    → zero-shot classification → label + confidence score

Both models are kept separate and serve different purposes.
"""
from __future__ import annotations

import hashlib
import logging
import traceback
from pathlib import Path
from typing import Optional

from backend.vision.models import DINOv2Result
from backend.vision.image_loader import load_image

logger = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────────────────────────────────────
DINOV2_MODEL = "facebook/dinov2-base"
CLIP_MODEL   = "openai/clip-vit-base-patch32"
VECTOR_DIM   = 768
QDRANT_HOST  = "localhost"
QDRANT_PORT  = 6333
COLLECTION   = "satellite_images"

# Descriptive prompts improve CLIP accuracy over bare class names
SATELLITE_CLASSES = [
    "satellite image of dense urban area with buildings and roads",
    "satellite image of forest or woodland with trees",
    "satellite image of agricultural farmland with crops or fields",
    "satellite image of river lake or ocean water body",
    "satellite image of desert or barren dry land",
    "satellite image of snow covered mountain or glacier",
    "satellite image of residential neighborhood with houses",
    "satellite image of industrial area with factories",
    "satellite image of coastal beach or shoreline",
    "satellite image of wetland or marsh",
]

CLASS_LABELS = [
    "urban", "forest", "agricultural", "water body",
    "desert", "snow/ice", "residential", "industrial",
    "coastal", "wetland",
]

# Module-level singletons — loaded once, reused
_dinov2_processor = None
_dinov2_model     = None
_clip_processor   = None
_clip_model       = None


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


def _load_clip():
    global _clip_processor, _clip_model
    if _clip_model is not None:
        return _clip_processor, _clip_model

    from transformers import CLIPProcessor, CLIPModel

    logger.info(f"[CLIP] Loading {CLIP_MODEL}...")
    _clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL)
    _clip_model     = CLIPModel.from_pretrained(CLIP_MODEL)
    _clip_model.eval()
    logger.info("[CLIP] Ready.")
    return _clip_processor, _clip_model


def _normalize(vector: list[float]) -> list[float]:
    """L2-normalize for cosine similarity."""
    norm = sum(x * x for x in vector) ** 0.5
    return [x / norm for x in vector] if norm > 0 else vector


# ══════════════════════════════════════════════════════════════════════════════
# CLASSIFICATION
# ══════════════════════════════════════════════════════════════════════════════

def classify_with_clip(pil_image) -> tuple[str, float]:
    """
    Zero-shot land-use classification using CLIP.

    Uses CLIP's image encoder — completely separate from DINOv2.
    Computes cosine similarity between image embedding and text
    label embeddings in CLIP's shared multimodal space.

    Returns:
        (label, confidence)  e.g. ("urban", 0.87)
        ("classification_error", 0.0) if CLIP fails — never "unknown"
    """
    import torch
    import torch.nn.functional as F

    # ── Step 1: Load CLIP ──────────────────────────────────────────────────────
    try:
        processor, model = _load_clip()
    except Exception as e:
        # Print full traceback so the caller can see what went wrong
        logger.error(f"[CLIP] Model loading failed:\n{traceback.format_exc()}")
        return "classification_error", 0.0

    # ── Step 2: Encode image with CLIP's image encoder ─────────────────────────
    try:
        image_inputs = processor(
            images=        pil_image,
            return_tensors="pt",
        )
        with torch.no_grad():
            image_features = model.get_image_features(**image_inputs)
        image_features = F.normalize(image_features, dim=-1)

    except Exception as e:
        logger.error(f"[CLIP] Image encoding failed:\n{traceback.format_exc()}")
        return "classification_error", 0.0

    # ── Step 3: Encode class label texts ──────────────────────────────────────
    try:
        text_inputs = processor(
            text=          SATELLITE_CLASSES,
            return_tensors="pt",
            padding=       True,
            truncation=    True,
        )
        with torch.no_grad():
            text_features = model.get_text_features(**text_inputs)
        text_features = F.normalize(text_features, dim=-1)

    except Exception as e:
        logger.error(f"[CLIP] Text encoding failed:\n{traceback.format_exc()}")
        return "classification_error", 0.0

    # ── Step 4: Cosine similarity → softmax → top class ───────────────────────
    try:
        # image_features: (1, 512)  text_features: (N, 512)
        # Result: (1, N) similarity scores
        similarities = (image_features @ text_features.T).squeeze(0)  # (N,)
        probs        = similarities.softmax(dim=-1)                    # (N,)

        best_idx   = int(probs.argmax().item())
        confidence = float(probs[best_idx].item())
        label      = CLASS_LABELS[best_idx]

        # Log full distribution for debugging
        logger.info(f"[CLIP] Classification result: '{label}' ({confidence:.4f})")
        top3_vals, top3_idx = probs.topk(3)
        for v, i in zip(top3_vals.tolist(), top3_idx.tolist()):
            logger.info(f"[CLIP]   {CLASS_LABELS[i]:<15} {v:.4f}")

        return label, round(confidence, 6)

    except Exception as e:
        logger.error(f"[CLIP] Scoring failed:\n{traceback.format_exc()}")
        return "classification_error", 0.0


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════

def extract_features(
    file_path:          str | Path,
    extract_patches:    bool = False,
    run_classification: bool = True,
) -> DINOv2Result:
    """
    Extract DINOv2 features from a satellite image.

    DINOv2 produces the 768-dim embedding used for Qdrant storage
    and similarity search. CLIP runs separately for classification.

    Args:
        file_path:          Path to image (JPG / PNG / GeoTIFF)
        extract_patches:    Extract per-patch embeddings (196 patches)
        run_classification: Run CLIP zero-shot classification

    Returns:
        DINOv2Result with feature vector and optional classification
    """
    import torch

    path = Path(file_path)
    logger.info(f"[DINOv2] Processing: {path.name}")

    # Load image — errors propagate to caller
    pil_image, _ = load_image(path)

    # DINOv2 forward pass
    processor, model = _load_dinov2()
    device = next(model.parameters()).device

    inputs = processor(images=pil_image, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    # CLS token = image-level representation  shape: (1, 768)
    cls_token   = outputs.last_hidden_state[:, 0, :]
    feature_vec = _normalize(cls_token[0].cpu().float().tolist())

    logger.info(f"[DINOv2] Feature dim: {len(feature_vec)}")

    # Optional patch embeddings  shape: (196, 768)
    patch_features = []
    if extract_patches:
        patch_tensor   = outputs.last_hidden_state[:, 1:, :]
        patch_features = [
            _normalize(p.cpu().float().tolist())
            for p in patch_tensor[0]
        ]
        logger.info(f"[DINOv2] Patch embeddings: {len(patch_features)}")

    # CLIP classification — separate model, separate encoder
    classification, confidence = None, None
    if run_classification:
        classification, confidence = classify_with_clip(pil_image)

    return DINOv2Result(
        file_path=                 str(path),
        feature_vector=            feature_vec,
        patch_features=            patch_features,
        vector_dim=                len(feature_vec),
        model_name=                DINOV2_MODEL,
        classification=            classification,
        classification_confidence= confidence,
    )


# ══════════════════════════════════════════════════════════════════════════════
# QDRANT STORAGE
# ══════════════════════════════════════════════════════════════════════════════

def store_in_qdrant(result: DINOv2Result) -> bool:
    """
    Upsert DINOv2 feature vector into Qdrant 'satellite_images' collection.

    Point ID is a deterministic integer from the file path MD5 hash
    so re-indexing the same image is idempotent (upsert, not duplicate).

    Prints the full exception traceback on failure — never silently
    returns False without explaining why.

    Returns:
        True  — insertion confirmed via collection point count
        False — insertion failed (full error printed to log)
    """
    try:
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams, PointStruct
    except ImportError:
        logger.error(
            "[Qdrant] qdrant-client not installed.\n"
            "  Run: pip install qdrant-client"
        )
        return False

    # ── Connect ────────────────────────────────────────────────────────────────
    try:
        client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        # Verify connection with a lightweight call
        client.get_collections()
        logger.info(f"[Qdrant] Connected to {QDRANT_HOST}:{QDRANT_PORT}")
    except Exception as e:
        logger.error(
            f"[Qdrant] Connection failed — is Qdrant running?\n"
            f"  Start with: docker run -p 6333:6333 qdrant/qdrant\n"
            f"  Error: {e}\n{traceback.format_exc()}"
        )
        return False

    # ── Create collection if missing ───────────────────────────────────────────
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
            logger.info(f"[Qdrant] Collection '{COLLECTION}' already exists.")
    except Exception as e:
        logger.error(
            f"[Qdrant] Collection creation failed:\n{traceback.format_exc()}"
        )
        return False

    # ── Build point ────────────────────────────────────────────────────────────
    try:
        # Deterministic int ID from file path hash (15 hex digits = safe int64)
        hex_hash = hashlib.md5(result.file_path.encode("utf-8")).hexdigest()
        point_id = int(hex_hash[:15], 16)

        point = PointStruct(
            id=      point_id,
            vector=  result.feature_vector,
            payload= {
                "file_path":      result.file_path,
                "file_name":      Path(result.file_path).name,
                "classification": result.classification,
                "confidence":     result.classification_confidence,
                "model":          result.model_name,
                "vector_dim":     result.vector_dim,
            },
        )
    except Exception as e:
        logger.error(
            f"[Qdrant] Failed to build PointStruct:\n{traceback.format_exc()}"
        )
        return False

    # ── Upsert ─────────────────────────────────────────────────────────────────
    try:
        client.upsert(
            collection_name= COLLECTION,
            points=          [point],
            wait=            True,    # block until indexed
        )
    except Exception as e:
        logger.error(
            f"[Qdrant] Upsert failed:\n{traceback.format_exc()}"
        )
        return False

    # ── Verify ─────────────────────────────────────────────────────────────────
    try:
        info         = client.get_collection(COLLECTION)
        total_points = info.points_count
        logger.info(
            f"[Qdrant] ✓ Stored in '{COLLECTION}' — "
            f"total points: {total_points}"
        )
        return True
    except Exception as e:
        logger.error(
            f"[Qdrant] Verification failed after upsert:\n{traceback.format_exc()}"
        )
        # Upsert may still have succeeded even if verification fails
        return False


# ══════════════════════════════════════════════════════════════════════════════
# SIMILARITY SEARCH
# ══════════════════════════════════════════════════════════════════════════════

def find_similar_images(
    query_image_path: str | Path,
    top_k:            int = 5,
) -> list[dict]:
    """
    Find satellite images visually similar to a query image.

    Steps:
        1. Extract DINOv2 features from query image
        2. Search Qdrant 'satellite_images' collection
        3. Return ranked results

    Args:
        query_image_path: Path to query satellite image
        top_k:            Number of similar images to return

    Returns:
        list of dicts:
            rank, file_name, classification, confidence, score
    """
    try:
        from qdrant_client import QdrantClient
    except ImportError:
        logger.error("[Similarity] qdrant-client not installed.")
        return []

    logger.info(f"[Similarity] Query: {Path(query_image_path).name}")

    # Extract query features (no classification needed for similarity)
    try:
        result = extract_features(
            query_image_path,
            run_classification=False,
        )
    except Exception as e:
        logger.error(
            f"[Similarity] Feature extraction failed:\n{traceback.format_exc()}"
        )
        return []

    # Search Qdrant
    try:
        client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

        # Check collection exists and has points
        existing = [c.name for c in client.get_collections().collections]
        if COLLECTION not in existing:
            logger.warning(
                f"[Similarity] Collection '{COLLECTION}' not found. "
                f"Index images first using store_in_qdrant()."
            )
            return []

        info = client.get_collection(COLLECTION)
        if info.points_count == 0:
            logger.warning(
                f"[Similarity] Collection '{COLLECTION}' is empty. "
                f"Index images first."
            )
            return []

        hits = client.search(
            collection_name= COLLECTION,
            query_vector=    result.feature_vector,
            limit=           top_k,
            with_payload=    True,
            with_vectors=    False,
        )

        results = [
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

        if results:
            logger.info(
                f"[Similarity] Found {len(results)} results "
                f"(top score: {results[0]['score']:.4f})"
            )
        else:
            logger.info("[Similarity] No results found.")

        return results

    except Exception as e:
        logger.error(
            f"[Similarity] Qdrant search failed:\n{traceback.format_exc()}"
        )
        return []