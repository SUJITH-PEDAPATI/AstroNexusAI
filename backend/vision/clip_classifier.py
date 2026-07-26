"""
AstroNexus AI — CLIP Zero-Shot Classifier

Standalone module. Called by dinov2_extractor.py for land-use classification.
DINOv2 and CLIP are kept fully separate.

Root cause fix:
    CLIPModel has two ways to get image features:

    WAY 1 (WRONG for classification):
        out = model.vision_model(pixel_values=x)
        → returns BaseModelOutputWithPooling (a dataclass)
        → out.shape raises AttributeError

    WAY 2 (CORRECT):
        out = model.get_image_features(pixel_values=x)
        → returns a plain torch.Tensor of shape (batch, projection_dim)
        → out.shape works fine

This file uses WAY 2 exclusively.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"

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

_processor = None
_model     = None


def _load():
    global _processor, _model
    if _model is not None:
        return _processor, _model
    from transformers import CLIPProcessor, CLIPModel
    logger.info(f"[CLIP] Loading {CLIP_MODEL_NAME}...")
    _processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
    _model     = CLIPModel.from_pretrained(CLIP_MODEL_NAME)
    _model.eval()
    logger.info("[CLIP] Ready.")
    return _processor, _model


def classify(pil_image) -> tuple[str, float]:
    """
    Zero-shot land-use classification using CLIP.

    Returns:
        (label, confidence)  e.g. ("urban", 0.87)
    """
    import torch
    import torch.nn.functional as F

    processor, model = _load()

    # ── Encode image ───────────────────────────────────────────────────────────
    image_inputs = processor(images=pil_image, return_tensors="pt")

    # get_image_features() returns Tensor(batch, 512) — NOT BaseModelOutputWithPooling
    with torch.no_grad():
        image_features = model.get_image_features(
            pixel_values=image_inputs["pixel_values"]
        )

    # Guard: crash loudly if the return type is wrong
    if not isinstance(image_features, torch.Tensor):
        raise TypeError(
            f"model.get_image_features() returned {type(image_features)}, "
            f"expected torch.Tensor. "
            f"Upgrade transformers: pip install --upgrade transformers"
        )

    image_features = F.normalize(image_features, dim=-1)   # (1, 512)

    # ── Encode texts ───────────────────────────────────────────────────────────
    text_inputs = processor(
        text=          SATELLITE_CLASSES,
        return_tensors="pt",
        padding=       True,
        truncation=    True,
        max_length=    77,
    )
    with torch.no_grad():
        text_features = model.get_text_features(
            input_ids=      text_inputs["input_ids"],
            attention_mask= text_inputs["attention_mask"],
        )

    text_features = F.normalize(text_features, dim=-1)     # (N, 512)

    # ── Similarity → softmax → top class ──────────────────────────────────────
    # (1, 512) @ (512, N) → (1, N) → squeeze → (N,)
    similarities = (image_features @ text_features.T).squeeze(0)
    probs        = similarities.softmax(dim=-1)

    best_idx   = int(probs.argmax().item())
    confidence = float(probs[best_idx].item())
    label      = CLASS_LABELS[best_idx]

    # Log top-3
    logger.info(f"[CLIP] Result: '{label}' ({confidence:.4f})")
    top3_vals, top3_idx = probs.topk(min(3, len(CLASS_LABELS)))
    for v, i in zip(top3_vals.tolist(), top3_idx.tolist()):
        logger.info(f"[CLIP]   {CLASS_LABELS[i]:<15} {v:.4f}")

    return label, round(confidence, 6)