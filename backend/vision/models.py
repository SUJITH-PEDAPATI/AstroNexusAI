"""
AstroNexus AI — Vision Models
Pydantic models for satellite image analysis results.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class ImageMetadata:
    """Basic metadata extracted from a satellite image."""
    file_path:   str
    file_name:   str
    width:       int
    height:      int
    channels:    int
    file_size_kb: float
    format:      str   # JPG, PNG, TIFF


@dataclass
class DINOv2Result:
    """Output from DINOv2 feature extraction."""
    file_path:        str
    feature_vector:   list[float]          # 768-dim CLS token embedding
    patch_features:   list[list[float]]    # per-patch embeddings (optional)
    vector_dim:       int
    model_name:       str
    classification:   Optional[str] = None         # predicted land-use class
    classification_confidence: Optional[float] = None
    similar_images:   list[dict] = field(default_factory=list)  # for similarity search


@dataclass
class Florence2Result:
    """Output from Florence-2 captioning and grounding."""
    file_path:        str
    caption:          str                          # image caption
    detailed_caption: Optional[str] = None        # detailed description
    bounding_boxes:   list[dict] = field(default_factory=list)  # [{label, bbox, confidence}]
    model_name:       str = "microsoft/Florence-2-base"


@dataclass
class SAM2Result:
    """Output from SAM2 segmentation."""
    file_path:        str
    masks:            list[dict] = field(default_factory=list)  # [{label, area_pct, mask_path}]
    total_segments:   int = 0
    model_name:       str = "facebook/sam2-hiera-base-plus"


@dataclass
class SatelliteAnalysisResult:
    """Combined result from the full satellite analysis pipeline."""
    file_path:        str
    metadata:         ImageMetadata
    dinov2:           Optional[DINOv2Result]   = None
    florence2:        Optional[Florence2Result] = None
    sam2:             Optional[SAM2Result]      = None
    pipeline_stages:  list[str] = field(default_factory=list)
    errors:           list[str] = field(default_factory=list)