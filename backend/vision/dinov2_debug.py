"""
AstroNexus AI — DINOv2 Debug Script

Runs CLIP classification and Qdrant insertion with ZERO exception handling.
Python will print the full traceback, file, and line number on any failure.

Run from project root:
    python -m backend.vision.dinov2_debug <path_to_image>
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")


def debug_clip(pil_image):
    print("\n" + "="*60)
    print("DEBUG: CLIP CLASSIFICATION")
    print("="*60)

    import torch
    import torch.nn.functional as F
    from transformers import CLIPProcessor, CLIPModel

    CLIP_MODEL = "openai/clip-vit-base-patch32"

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

    print(f"  Loading CLIP: {CLIP_MODEL}")
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL)
    model     = CLIPModel.from_pretrained(CLIP_MODEL)
    model.eval()
    print("  CLIP loaded OK")

    print("  Encoding image...")
    image_inputs    = processor(images=pil_image, return_tensors="pt")
    print(f"  image_inputs keys: {list(image_inputs.keys())}")
    with torch.no_grad():
        image_features = model.get_image_features(**image_inputs)
    print(f"  image_features shape: {image_features.shape}")
    image_features  = F.normalize(image_features, dim=-1)

    print("  Encoding texts...")
    text_inputs = processor(
        text=          SATELLITE_CLASSES,
        return_tensors="pt",
        padding=       True,
        truncation=    True,
    )
    print(f"  text_inputs keys: {list(text_inputs.keys())}")
    with torch.no_grad():
        text_features = model.get_text_features(**text_inputs)
    print(f"  text_features shape: {text_features.shape}")
    text_features = F.normalize(text_features, dim=-1)

    print("  Computing similarities...")
    similarities = (image_features @ text_features.T).squeeze(0)
    print(f"  similarities shape: {similarities.shape}")
    probs        = similarities.softmax(dim=-1)

    best_idx   = int(probs.argmax().item())
    confidence = float(probs[best_idx].item())
    label      = CLASS_LABELS[best_idx]

    print(f"\n  RESULT: '{label}' confidence={confidence:.4f}")
    print("\n  Top-3:")
    top3_vals, top3_idx = probs.topk(3)
    for v, i in zip(top3_vals.tolist(), top3_idx.tolist()):
        print(f"    {CLASS_LABELS[i]:<15} {v:.4f}")

    return label, confidence


def debug_qdrant(feature_vector: list[float], file_path: str):
    print("\n" + "="*60)
    print("DEBUG: QDRANT INSERTION")
    print("="*60)

    import hashlib
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams, PointStruct

    COLLECTION = "satellite_images"
    VECTOR_DIM = 768

    print("  Connecting to Qdrant at localhost:6333...")
    client = QdrantClient(host="localhost", port=6333)
    collections = client.get_collections()
    print(f"  Connected. Existing collections: {[c.name for c in collections.collections]}")

    existing = [c.name for c in collections.collections]
    if COLLECTION not in existing:
        print(f"  Creating collection '{COLLECTION}'...")
        client.create_collection(
            collection_name= COLLECTION,
            vectors_config=  VectorParams(
                size=     VECTOR_DIM,
                distance= Distance.COSINE,
            ),
        )
        print(f"  Collection '{COLLECTION}' created.")
    else:
        print(f"  Collection '{COLLECTION}' already exists.")

    hex_hash = hashlib.md5(file_path.encode("utf-8")).hexdigest()
    point_id = int(hex_hash[:15], 16)
    print(f"  Point ID: {point_id}")
    print(f"  Vector length: {len(feature_vector)}")
    print(f"  Vector type: {type(feature_vector[0])}")

    point = PointStruct(
        id=      point_id,
        vector=  feature_vector,
        payload= {
            "file_path":      file_path,
            "file_name":      Path(file_path).name,
        },
    )

    print("  Upserting point...")
    client.upsert(
        collection_name= COLLECTION,
        points=          [point],
        wait=            True,
    )

    info = client.get_collection(COLLECTION)
    print(f"  ✓ Upsert confirmed. Total points: {info.points_count}")
    return True


def run(file_path: str):
    from backend.vision.image_loader import load_image
    from backend.vision.dinov2_extractor import extract_features

    path = Path(file_path)
    print(f"\nFile: {path}")

    # Step 1: Load image
    print("\n" + "="*60)
    print("DEBUG: IMAGE LOADING")
    print("="*60)
    pil_image, metadata = load_image(path)
    print(f"  Size   : {metadata.width}x{metadata.height}")
    print(f"  Format : {metadata.format}")

    # Step 2: DINOv2 features (known working)
    print("\n" + "="*60)
    print("DEBUG: DINOV2 FEATURES")
    print("="*60)
    result = extract_features(path, run_classification=False)
    print(f"  Vector dim : {result.vector_dim}")
    norm = sum(x*x for x in result.feature_vector)**0.5
    print(f"  L2 norm    : {norm:.6f}")

    # Step 3: CLIP — NO exception handling — raw traceback on failure
    label, confidence = debug_clip(pil_image)

    # Step 4: Qdrant — NO exception handling — raw traceback on failure
    debug_qdrant(result.feature_vector, str(path))

    print("\n" + "="*60)
    print("ALL DEBUG STEPS PASSED")
    print("="*60)
    print(f"  Classification : {label} ({confidence:.4f})")
    print(f"  Qdrant         : stored successfully")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m backend.vision.dinov2_debug <path_to_image>")
        sys.exit(1)
    run(sys.argv[1])