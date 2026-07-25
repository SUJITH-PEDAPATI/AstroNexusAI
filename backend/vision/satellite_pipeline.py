"""
AstroNexus AI — Satellite Analysis Pipeline (Week 4: DINOv2 Fixed)

Run from project root:

    # Analyze a single image
    python -m backend.vision.satellite_pipeline <path_to_image>

    # Find similar images
    python -m backend.vision.satellite_pipeline <path_to_image> --similar

Supports: JPG, PNG, GeoTIFF
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def analyze_satellite_image(
    file_path:          str | Path,
    store_in_qdrant:    bool = True,
    extract_patches:    bool = False,
    run_classification: bool = True,
) -> "SatelliteAnalysisResult":
    """
    Full DINOv2 satellite analysis pipeline.

        Stage 1 — Load image + extract metadata
        Stage 2 — DINOv2 feature extraction (768-dim)
        Stage 3 — CLIP zero-shot classification
        Stage 4 — Qdrant upsert (verified)

    Florence-2 captioning and SAM2 segmentation are added in Week 5.
    """
    from backend.vision.models import SatelliteAnalysisResult
    from backend.vision.image_loader import load_image
    from backend.vision.dinov2_extractor import (
        extract_features, store_in_qdrant as _store
    )

    path   = Path(file_path)
    stages = []
    errors = []

    logger.info(f"[SatellitePipeline] Starting: {path.name}")

    # ── Stage 1: Load image ────────────────────────────────────────────────────
    try:
        pil_image, metadata = load_image(path)
        stages.append("image_loaded")
        logger.info(
            f"[SatellitePipeline] Loaded: "
            f"{metadata.width}x{metadata.height} {metadata.format}"
        )
    except Exception as e:
        msg = f"Image loading failed: {e}"
        logger.error(f"[SatellitePipeline] {msg}")
        from backend.vision.models import SatelliteAnalysisResult
        return SatelliteAnalysisResult(
            file_path=str(path), metadata=None, errors=[msg]
        )

    # ── Stage 2+3: DINOv2 + CLIP classification ────────────────────────────────
    dinov2_result = None
    try:
        dinov2_result = extract_features(
            file_path=          path,
            extract_patches=    extract_patches,
            run_classification= run_classification,
        )
        stages.append("dinov2_features")
        if dinov2_result.classification:
            stages.append("clip_classification")
            logger.info(
                f"[SatellitePipeline] Class: {dinov2_result.classification} "
                f"({dinov2_result.classification_confidence:.4f})"
            )
    except Exception as e:
        msg = f"DINOv2/CLIP failed: {e}"
        logger.error(f"[SatellitePipeline] {msg}")
        errors.append(msg)

    # ── Stage 4: Qdrant storage ────────────────────────────────────────────────
    if store_in_qdrant and dinov2_result is not None:
        try:
            stored = _store(dinov2_result)
            if stored:
                stages.append("qdrant_stored")
                logger.info("[SatellitePipeline] ✓ Qdrant storage confirmed")
            else:
                errors.append("Qdrant storage returned False — check logs")
        except Exception as e:
            msg = f"Qdrant storage failed: {e}"
            logger.error(f"[SatellitePipeline] {msg}")
            errors.append(msg)

    return SatelliteAnalysisResult(
        file_path=       str(path),
        metadata=        metadata,
        dinov2=          dinov2_result,
        pipeline_stages= stages,
        errors=          errors,
    )


def run_test(file_path: str, find_similar: bool = False) -> None:
    """CLI test runner — analyze image and optionally run similarity search."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    path = Path(file_path)
    if not path.exists():
        print(f"\n  [Error] File not found: {path}")
        return

    print(f"\n{'='*60}")
    print("ASTRONEXUS — SATELLITE ANALYSIS  (DINOv2 + CLIP)")
    print(f"{'='*60}")

    result = analyze_satellite_image(path)

    # ── Terminal output ────────────────────────────────────────────────────────
    print(f"\n  METADATA")
    if result.metadata:
        print(f"    Size       : {result.metadata.width}x{result.metadata.height}")
        print(f"    Format     : {result.metadata.format}")
        print(f"    File size  : {result.metadata.file_size_kb:.1f} KB")

    print(f"\n  DINOV2 FEATURES")
    if result.dinov2:
        vec = result.dinov2.feature_vector
        preview = ", ".join(f"{v:+.4f}" for v in vec[:6])
        norm    = sum(x * x for x in vec) ** 0.5
        print(f"    Vector dim  : {result.dinov2.vector_dim}")
        print(f"    Preview     : [{preview}, ...]")
        print(f"    L2 norm     : {norm:.6f}  {'✓ normalized' if abs(norm-1.0) < 1e-3 else '⚠'}")

    print(f"\n  CLASSIFICATION  (CLIP zero-shot)")
    if result.dinov2 and result.dinov2.classification:
        print(f"    Class      : {result.dinov2.classification}")
        print(f"    Confidence : {result.dinov2.classification_confidence:.4f}")
    else:
        print(f"    Status     : classification_pending")

    print(f"\n  PIPELINE STAGES")
    for i, stage in enumerate(result.pipeline_stages, 1):
        print(f"    {i}. {stage}")

    if result.errors:
        print(f"\n  ERRORS")
        for e in result.errors:
            print(f"    ⚠ {e}")

    # ── Similarity search ──────────────────────────────────────────────────────
    if find_similar:
        print(f"\n  SIMILARITY SEARCH  (top-5)")
        from backend.vision.dinov2_extractor import find_similar_images
        similar = find_similar_images(path, top_k=5)
        if not similar:
            print("    No indexed images to compare against.")
            print("    Index more images first, then re-run with --similar")
        else:
            for item in similar:
                print(
                    f"    [{item['rank']}] score={item['score']:.4f}  "
                    f"{item['file_name']}  ({item['classification']})"
                )

    # ── Save JSON output ───────────────────────────────────────────────────────
    out_dir  = Path("output")
    out_dir.mkdir(exist_ok=True)
    out_file = out_dir / f"satellite_{path.stem}_dinov2.json"

    output = {
        "file_path":       result.file_path,
        "pipeline_stages": result.pipeline_stages,
        "errors":          result.errors,
    }
    if result.metadata:
        output["metadata"] = {
            "width":   result.metadata.width,
            "height":  result.metadata.height,
            "format":  result.metadata.format,
            "size_kb": result.metadata.file_size_kb,
        }
    if result.dinov2:
        output["dinov2"] = {
            "vector_dim":       result.dinov2.vector_dim,
            "vector_preview":   result.dinov2.feature_vector[:8],
            "l2_norm":          round(sum(x*x for x in result.dinov2.feature_vector)**0.5, 6),
            "classification":   result.dinov2.classification,
            "confidence":       result.dinov2.classification_confidence,
            "model":            result.dinov2.model_name,
            "classifier_model": "openai/clip-vit-base-patch32",
        }

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\n  ✓ Saved → {out_file}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    import sys
    args = sys.argv[1:]
    if not args:
        print("Usage: python -m backend.vision.satellite_pipeline <image_path> [--similar]")
        sys.exit(1)
    file_path   = args[0]
    find_similar = "--similar" in args
    run_test(file_path, find_similar=find_similar)