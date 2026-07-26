"""
AstroNexus AI — SAM2 Segmentation (Fixed)

Root cause of empty masks:
    SAM2ImagePredictor requires explicit point/box prompts.
    Without prompts it returns nothing.
    Fix: use SAM2AutomaticMaskGenerator which segments the whole image.

Install:
    pip install segment-anything-2
    # or
    pip install git+https://github.com/facebookresearch/sam2.git
"""
from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

HF_MODEL_ID  = "facebook/sam2-hiera-base-plus"
OUTPUT_DIR   = Path("output")
VIZ_FILENAME = "segmented.png"

# Lower thresholds = more masks retained
# SAM2 defaults are often too strict for satellite imagery
MIN_MASK_AREA       = 200    # pixels — drop tiny noise masks
IOU_THRESHOLD       = 0.7    # predicted IoU filter
STABILITY_THRESHOLD = 0.75   # stability score filter

_generator = None   # SAM2AutomaticMaskGenerator singleton
_device    = None


# ══════════════════════════════════════════════════════════════════════════════
# DEVICE
# ══════════════════════════════════════════════════════════════════════════════

def _get_device() -> str:
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


# ══════════════════════════════════════════════════════════════════════════════
# SINGLETON LOADER — SAM2AutomaticMaskGenerator only
# ══════════════════════════════════════════════════════════════════════════════

def _load_generator():
    """
    Load SAM2AutomaticMaskGenerator — once per process.

    SAM2AutomaticMaskGenerator runs a dense grid of prompts internally
    and returns all masks without requiring caller-supplied points/boxes.
    This is the correct API for unsupervised segmentation.

    Thresholds are relaxed below SAM2 defaults so satellite imagery
    (which has large uniform regions) is not entirely filtered out.
    """
    global _generator, _device

    if _generator is not None:
        return _generator

    _device = _get_device()
    logger.info(f"[SAM2] Loading SAM2AutomaticMaskGenerator on {_device.upper()}...")

    try:
        import torch
        from sam2.build_sam import build_sam2
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

        # Build model from HuggingFace Hub checkpoint
        sam2_model = build_sam2(
            config_file= "configs/sam2.1/sam2.1_hiera_b+.yaml",
            ckpt_path=   HF_MODEL_ID,
            device=      _device,
            apply_postprocessing=False,
        )

        _generator = SAM2AutomaticMaskGenerator(
            model=                   sam2_model,
            points_per_side=         16,      # 16×16 = 256 prompt points (default 32×32 is slow)
            points_per_batch=        64,
            pred_iou_thresh=         IOU_THRESHOLD,
            stability_score_thresh=  STABILITY_THRESHOLD,
            stability_score_offset=  1.0,
            box_nms_thresh=          0.7,
            crop_n_layers=           0,
            min_mask_region_area=    MIN_MASK_AREA,
        )

        logger.info("[SAM2] SAM2AutomaticMaskGenerator ready.")
        return _generator

    except Exception as e:
        raise RuntimeError(
            f"SAM2AutomaticMaskGenerator failed to load.\n"
            f"Install: pip install segment-anything-2\n"
            f"Error: {e}"
        ) from e


# ══════════════════════════════════════════════════════════════════════════════
# SEGMENTATION
# ══════════════════════════════════════════════════════════════════════════════

def _run_automatic_segmentation(image_rgb: np.ndarray) -> list[dict]:
    """
    Run SAM2AutomaticMaskGenerator on an RGB uint8 numpy array.

    SAM2 requires:
        - dtype: uint8
        - shape: (H, W, 3)
        - channel order: RGB (not BGR)

    Returns list of mask dicts from SAM2, each containing:
        segmentation    np.ndarray bool (H, W)
        area            int
        bbox            [x, y, w, h]
        predicted_iou   float
        stability_score float
        point_coords    list
        crop_box        list
    """
    import torch

    assert image_rgb.dtype == np.uint8,   f"Expected uint8, got {image_rgb.dtype}"
    assert image_rgb.ndim  == 3,          f"Expected (H,W,3), got {image_rgb.shape}"
    assert image_rgb.shape[2] == 3,       f"Expected 3 channels, got {image_rgb.shape[2]}"

    logger.info(
        f"[SAM2] Input shape={image_rgb.shape}  "
        f"dtype={image_rgb.dtype}  "
        f"device={_device}"
    )

    generator = _load_generator()

    with torch.inference_mode():
        masks = generator.generate(image_rgb)

    logger.info(f"[SAM2] Raw masks from generator: {len(masks)}")

    # Log per-mask debug info for first 5
    for i, m in enumerate(masks[:5]):
        logger.info(
            f"[SAM2]   mask[{i}] area={m.get('area',0)}  "
            f"iou={m.get('predicted_iou',0):.3f}  "
            f"stability={m.get('stability_score',0):.3f}"
        )

    return masks


# ══════════════════════════════════════════════════════════════════════════════
# VISUALISATION
# ══════════════════════════════════════════════════════════════════════════════

def _save_visualization(
    pil_image: Image.Image,
    masks:     list[dict],
    out_path:  Path,
) -> None:
    """Draw each mask as a semi-transparent random-colour overlay."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    base    = pil_image.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    arr     = np.array(overlay)

    rng = random.Random(42)   # deterministic colours

    for mask_data in masks:
        seg = mask_data.get("segmentation")
        if seg is None:
            continue
        r = rng.randint(40, 255)
        g = rng.randint(40, 255)
        b = rng.randint(40, 255)
        arr[seg] = [r, g, b, 140]

    overlay   = Image.fromarray(arr, "RGBA")
    composed  = Image.alpha_composite(base, overlay).convert("RGB")
    composed.save(str(out_path))
    logger.info(f"[SAM2] Visualization → {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ══════════════════════════════════════════════════════════════════════════════

def segment_image(
    source:   Union[str, Path, Image.Image],
    save_viz: bool = True,
    viz_path: Path = OUTPUT_DIR / VIZ_FILENAME,
) -> dict:
    """
    Run SAM2 automatic segmentation on a satellite image.

    Args:
        source:   File path (str/Path) or PIL Image
        save_viz: Save coloured overlay to viz_path
        viz_path: Output path for visualization PNG

    Returns:
        {
            "mask_count": int,
            "segments": [
                {
                    "id":              int,
                    "area":            int,
                    "bbox":            [x, y, w, h],
                    "stability_score": float,
                    "predicted_iou":   float
                },
                ...
            ]
        }
    """
    # ── Load image → RGB uint8 numpy ───────────────────────────────────────────
    if isinstance(source, (str, Path)):
        from backend.vision.image_loader import load_image
        pil_image, _ = load_image(Path(source))
    else:
        pil_image = source

    pil_image  = pil_image.convert("RGB")
    image_rgb  = np.array(pil_image, dtype=np.uint8)

    logger.info(
        f"[SAM2] Image prepared: shape={image_rgb.shape} "
        f"dtype={image_rgb.dtype} "
        f"min={image_rgb.min()} max={image_rgb.max()}"
    )

    # ── Run automatic segmentation ─────────────────────────────────────────────
    raw_masks = _run_automatic_segmentation(image_rgb)

    logger.info(f"[SAM2] Masks after generator (before any filtering): {len(raw_masks)}")

    # Sort by area descending (largest regions first)
    raw_masks = sorted(raw_masks, key=lambda m: m.get("area", 0), reverse=True)

    # ── Build metadata — no raw arrays ────────────────────────────────────────
    segments = [
        {
            "id":              i,
            "area":            int(m.get("area", 0)),
            "bbox":            [int(v) for v in m.get("bbox", [0, 0, 0, 0])],
            "stability_score": round(float(m.get("stability_score", 0.0)), 4),
            "predicted_iou":   round(float(m.get("predicted_iou",   0.0)), 4),
        }
        for i, m in enumerate(raw_masks)
    ]

    logger.info(f"[SAM2] Final segment count: {len(segments)}")

    # ── Visualization ──────────────────────────────────────────────────────────
    if save_viz and raw_masks:
        try:
            _save_visualization(pil_image, raw_masks, viz_path)
        except Exception as e:
            logger.warning(f"[SAM2] Visualization failed (non-fatal): {e}")

    return {
        "mask_count": len(segments),
        "segments":   segments,
    }