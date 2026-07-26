"""
AstroNexus AI — Satellite Analysis Pipeline
Version: 3.0 — Week 5 complete (DINOv2 + Gemini + SAM2)

Run from project root:
    python -m backend.vision.satellite_pipeline <image>
    python -m backend.vision.satellite_pipeline <image> --no-sam2
    python -m backend.vision.satellite_pipeline <image> --similar
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

PIPELINE_VERSION = "3.0-week5"


def analyze_satellite_image(
    file_path:       str | Path,
    run_florence:    bool = True,
    run_sam2:        bool = True,
    extract_patches: bool = False,
) -> "SatelliteAnalysisResult":
    """
    Full satellite image analysis pipeline.

    Stages:
        image_loaded         — PIL image + metadata
        dinov2_features      — 768-dim DINOv2 embedding
        florence2_caption    — Gemini Vision caption + bounding boxes
        florence2_detection  — (included in florence2_caption stage)
        sam2_segmentation    — SAM2 automatic mask generation

    Returns SatelliteAnalysisResult.
    """
    from backend.vision.models import SatelliteAnalysisResult
    from backend.vision.image_loader import load_image
    from backend.vision.dinov2_extractor import extract_features

    path   = Path(file_path)
    stages = []
    errors = []

    logger.info(f"[Pipeline v{PIPELINE_VERSION}] Starting: {path.name}")

    # ── Stage 1: Load image ────────────────────────────────────────────────────
    try:
        _, metadata = load_image(path)
        stages.append("image_loaded")
        logger.info(
            f"[Pipeline] Loaded: {metadata.width}x{metadata.height} {metadata.format}"
        )
    except Exception as e:
        errors.append(f"Image loading failed: {e}")
        return SatelliteAnalysisResult(
            file_path=str(path), metadata=None, errors=errors
        )

    # ── Stage 2: DINOv2 features ───────────────────────────────────────────────
    dinov2_result = None
    try:
        dinov2_result = extract_features(
            path,
            extract_patches=    extract_patches,
            run_classification= False,
        )
        stages.append("dinov2_features")
        logger.info(f"[Pipeline] DINOv2 dim={dinov2_result.vector_dim}")
    except Exception as e:
        errors.append(f"DINOv2 failed: {e}")
        logger.error(f"[Pipeline] DINOv2 error: {e}")

    # ── Stage 3: Gemini captioning + detection ─────────────────────────────────
    florence_result = None
    if run_florence:
        try:
            from backend.vision.florence2_captioner import caption_image
            logger.info("[Pipeline] Running Gemini captioner...")
            florence_result = caption_image(path)
            stages.append("florence2_caption")
            if florence_result.bounding_boxes:
                stages.append("florence2_detection")
            logger.info(f"[Pipeline] Caption: {florence_result.caption[:80]}")
        except Exception as e:
            errors.append(f"Gemini captioning failed: {e}")
            logger.error(f"[Pipeline] Gemini error: {e}")

    # ── Stage 4: SAM2 segmentation ─────────────────────────────────────────────
    sam2_result = None
    if run_sam2:
        try:
            from backend.vision.sam2_segmentor import segment_image
            logger.info("[Pipeline] Running SAM2...")
            sam2_result = segment_image(path)
            stages.append("sam2_segmentation")
            logger.info(
                f"[Pipeline] SAM2: {sam2_result.total_segments} segments"
            )
        except Exception as e:
            errors.append(f"SAM2 failed: {e}")
            logger.error(f"[Pipeline] SAM2 error: {e}")

    return SatelliteAnalysisResult(
        file_path=       str(path),
        metadata=        metadata,
        dinov2=          dinov2_result,
        florence2=       florence_result,
        sam2=            sam2_result,
        pipeline_stages= stages,
        errors=          errors,
    )


def run_test(
    file_path:    str,
    find_similar: bool = False,
    skip_sam2:    bool = False,
) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    path = Path(file_path)
    if not path.exists():
        print(f"\n  [Error] File not found: {path}")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"ASTRONEXUS SATELLITE PIPELINE  v{PIPELINE_VERSION}")
    print(f"{'='*60}")

    result = analyze_satellite_image(
        path,
        run_florence= not find_similar,
        run_sam2=     not skip_sam2 and not find_similar,
    )

    # ── Metadata ───────────────────────────────────────────────────────────────
    print(f"\n  [1] METADATA")
    if result.metadata:
        print(f"      Size   : {result.metadata.width}x{result.metadata.height}")
        print(f"      Format : {result.metadata.format}")

    # ── DINOv2 ─────────────────────────────────────────────────────────────────
    print(f"\n  [2] DINOV2")
    if result.dinov2:
        vec  = result.dinov2.feature_vector
        norm = sum(x*x for x in vec)**0.5
        print(f"      Dim  : {result.dinov2.vector_dim}")
        print(f"      Norm : {norm:.6f}  {'✓' if abs(norm-1.0)<1e-3 else '⚠'}")

    # ── Gemini ─────────────────────────────────────────────────────────────────
    print(f"\n  [3] GEMINI CAPTIONING")
    if result.florence2:
        print(f"      Caption  : {result.florence2.caption}")
        if result.florence2.detailed_caption:
            print(f"      Detailed : {str(result.florence2.detailed_caption)[:120]}...")
        print(f"      Objects  : {len(result.florence2.bounding_boxes)} detected")
        for b in result.florence2.bounding_boxes[:5]:
            print(f"        {b['label']:<22} {b['bbox']}")
    else:
        print("      Status : not run or failed")

    # ── SAM2 ───────────────────────────────────────────────────────────────────
    print(f"\n  [4] SAM2 SEGMENTATION")
    if result.sam2:
        print(f"      Masks : {result.sam2.total_segments}")
        for seg in result.sam2.masks[:5]:
            print(
                f"        [{seg['segment_id']}] {seg['label']:<20} "
                f"area={seg['area_pct']}%  "
                f"bbox={seg['bbox']}"
            )
        print(f"      Segmentation complete")
    else:
        print("      Status : not run or failed")

    # ── Similarity search ──────────────────────────────────────────────────────
    if find_similar and result.dinov2:
        print(f"\n  [5] SIMILARITY SEARCH")
        from backend.vision.dinov2_extractor import find_similar_images
        similar = find_similar_images(path, top_k=5)
        if not similar:
            print("      No indexed images found")
        else:
            for item in similar:
                print(f"      [{item['rank']}] {item['score']:.4f}  {item['file_name']}")

    # ── Pipeline stages ─────────────────────────────────────────────────────────
    print(f"\n  PIPELINE STAGES")
    for i, s in enumerate(result.pipeline_stages, 1):
        print(f"      {i}. {s}")

    if result.errors:
        print(f"\n  ERRORS")
        for e in result.errors:
            print(f"      ⚠ {e}")

    # ── Save JSON ──────────────────────────────────────────────────────────────
    out_dir  = Path("output")
    out_dir.mkdir(exist_ok=True)
    out_file = out_dir / f"satellite_{path.stem}_full.json"

    output: dict = {
        "pipeline_version": PIPELINE_VERSION,
        "file_path":        str(path),
        "pipeline_stages":  result.pipeline_stages,
        "errors":           result.errors,
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
            "vector_dim":     result.dinov2.vector_dim,
            "vector_preview": result.dinov2.feature_vector[:8],
            "l2_norm":        round(
                sum(x*x for x in result.dinov2.feature_vector)**0.5, 6
            ),
            "model": result.dinov2.model_name,
        }

    if result.florence2:
        output["florence2"] = {
            "caption":          result.florence2.caption,
            "detailed_caption": result.florence2.detailed_caption,
            "bounding_boxes":   result.florence2.bounding_boxes,
            "model":            result.florence2.model_name,
        }

    if result.sam2:
        output["sam2"] = {
            "total_segments": result.sam2.total_segments,
            "masks":          result.sam2.masks,
            "model":          result.sam2.model_name,
        }

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\n  ✓ JSON saved  → {out_file}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        print(
            "Usage:\n"
            "  python -m backend.vision.satellite_pipeline <image>\n"
            "  python -m backend.vision.satellite_pipeline <image> --no-sam2\n"
            "  python -m backend.vision.satellite_pipeline <image> --similar"
        )
        sys.exit(1)

    run_test(
        file_path=    args[0],
        find_similar= "--similar" in args,
        skip_sam2=    "--no-sam2" in args,
    )