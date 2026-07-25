"""
AstroNexus AI — Satellite Image Loader

Loads and preprocesses satellite images for the vision pipeline.
Supports: JPG, PNG, GeoTIFF (.tif/.tiff)

Install: pip install Pillow rasterio numpy
"""
from __future__ import annotations

import logging
from pathlib import Path

from backend.vision.models import ImageMetadata

logger = logging.getLogger(__name__)

SUPPORTED_FORMATS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".geotiff"}


def load_image(file_path: str | Path) -> tuple:
    """
    Load a satellite image and return (PIL.Image, ImageMetadata).

    Handles regular images via Pillow and GeoTIFF via rasterio.
    GeoTIFF files are converted to RGB PIL Images for downstream models.

    Returns:
        (pil_image, ImageMetadata)
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_FORMATS:
        raise ValueError(f"Unsupported format: {suffix}. Supported: {SUPPORTED_FORMATS}")

    file_size_kb = path.stat().st_size / 1024

    # ── GeoTIFF via rasterio ───────────────────────────────────────────────────
    if suffix in {".tif", ".tiff", ".geotiff"}:
        return _load_geotiff(path, file_size_kb)

    # ── Standard image via Pillow ──────────────────────────────────────────────
    return _load_standard(path, file_size_kb)


def _load_standard(path: Path, file_size_kb: float) -> tuple:
    try:
        from PIL import Image
    except ImportError as e:
        raise ImportError("Install Pillow: pip install Pillow") from e

    img = Image.open(path).convert("RGB")
    w, h = img.size

    metadata = ImageMetadata(
        file_path=    str(path),
        file_name=    path.name,
        width=        w,
        height=       h,
        channels=     3,
        file_size_kb= round(file_size_kb, 2),
        format=       path.suffix.lstrip(".").upper(),
    )

    logger.info(f"[ImageLoader] Loaded {path.name} — {w}x{h} RGB")
    return img, metadata


def _load_geotiff(path: Path, file_size_kb: float) -> tuple:
    try:
        import rasterio
        import numpy as np
        from PIL import Image
    except ImportError as e:
        raise ImportError(
            "Install rasterio and numpy: pip install rasterio numpy"
        ) from e

    with rasterio.open(str(path)) as src:
        # Read first 3 bands as RGB (standard for multispectral imagery)
        n_bands = min(src.count, 3)
        bands   = src.read(list(range(1, n_bands + 1)))   # (bands, H, W)
        w, h    = src.width, src.height

    # Normalize to 0-255 uint8
    import numpy as np
    arr = np.transpose(bands, (1, 2, 0))   # (H, W, bands)
    arr = arr.astype(float)

    # Per-channel min-max normalization
    for c in range(arr.shape[2]):
        ch_min, ch_max = arr[:, :, c].min(), arr[:, :, c].max()
        if ch_max > ch_min:
            arr[:, :, c] = (arr[:, :, c] - ch_min) / (ch_max - ch_min) * 255

    arr = arr.astype(np.uint8)
    if arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)   # grayscale → RGB

    img = Image.fromarray(arr, mode="RGB")

    metadata = ImageMetadata(
        file_path=    str(path),
        file_name=    path.name,
        width=        w,
        height=       h,
        channels=     n_bands,
        file_size_kb= round(file_size_kb, 2),
        format=       "GeoTIFF",
    )

    logger.info(f"[ImageLoader] Loaded GeoTIFF {path.name} — {w}x{h}, {n_bands} bands")
    return img, metadata


def preprocess_for_dinov2(pil_image) -> "torch.Tensor":
    """
    Preprocess a PIL image for DINOv2 input.
    Returns a (1, 3, 224, 224) float32 tensor normalized to ImageNet stats.
    """
    try:
        from torchvision import transforms
        import torch
    except ImportError as e:
        raise ImportError("Install torchvision: pip install torchvision") from e

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],   # ImageNet mean
            std= [0.229, 0.224, 0.225],   # ImageNet std
        ),
    ])
    return transform(pil_image).unsqueeze(0)   # (1, 3, 224, 224)