"""
AstroNexus AI — Vision Captioner (Gemini) with Neo4j keyword writing

Every image upload now:
    1. Generates caption + detailed description via Gemini Vision
    2. Extracts keywords from the caption
    3. Writes keywords to Neo4j and links to an ImageSession node
    4. Stores DINOv2 vector in Qdrant (satellite_images collection)

This means image keywords become queryable in the graph,
and the Research Agent can use them when answering questions.
"""
from __future__ import annotations

import base64
import io
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Optional

from backend.vision.models import Florence2Result
from backend.vision.image_loader import load_image

logger = logging.getLogger(__name__)

MODEL_NAME   = "gemini-2.0-flash"
MAX_IMG_SIZE = 1024

_client = None


# ══════════════════════════════════════════════════════════════════════════════
# GEMINI CLIENT
# ══════════════════════════════════════════════════════════════════════════════

def _get_client():
    global _client
    if _client is not None:
        return _client

    try:
        import google.generativeai as genai
    except ImportError as e:
        raise ImportError("pip install google-generativeai") from e

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise EnvironmentError("GEMINI_API_KEY not set in .env")

    genai.configure(api_key=api_key)
    _client = genai.GenerativeModel(MODEL_NAME)
    logger.info(f"[Captioner] Gemini ready: {MODEL_NAME}")
    return _client


# ══════════════════════════════════════════════════════════════════════════════
# IMAGE PREP
# ══════════════════════════════════════════════════════════════════════════════

def _prepare_image(pil_image):
    from PIL import Image
    w, h = pil_image.size
    if max(w, h) > MAX_IMG_SIZE:
        scale = MAX_IMG_SIZE / max(w, h)
        pil_image = pil_image.resize((int(w*scale), int(h*scale)), Image.LANCZOS)
        logger.info(f"[Captioner] Resized {w}x{h} → {pil_image.size}")
    return pil_image


# ══════════════════════════════════════════════════════════════════════════════
# KEYWORD EXTRACTION FROM CAPTION
# ══════════════════════════════════════════════════════════════════════════════

# Domain-specific terms to extract from captions
_LAND_TYPES     = ["urban", "forest", "agricultural", "water", "desert", "snow",
                   "wetland", "coastal", "residential", "industrial", "vegetation",
                   "grassland", "barren", "cropland", "river", "lake", "ocean"]
_HAZARD_TYPES   = ["flood", "fire", "wildfire", "deforestation", "drought",
                   "landslide", "erosion", "pollution", "oil spill", "storm"]
_INFRA_TYPES    = ["road", "highway", "bridge", "building", "airport", "port",
                   "railway", "power plant", "dam", "reservoir"]
_SENSOR_HINTS   = ["multispectral", "sar", "hyperspectral", "optical", "thermal",
                   "lidar", "radar", "infrared", "panchromatic"]

_ALL_DOMAIN_KEYWORDS = set(
    _LAND_TYPES + _HAZARD_TYPES + _INFRA_TYPES + _SENSOR_HINTS
)


def _extract_keywords_from_caption(caption: str, detailed: str) -> list[str]:
    """
    Extract domain-relevant keywords from Gemini caption text.
    Returns deduplicated list of matched keywords.
    """
    combined = f"{caption} {detailed}".lower()
    found    = []
    for kw in _ALL_DOMAIN_KEYWORDS:
        if re.search(r'\b' + re.escape(kw) + r'\b', combined):
            found.append(kw)
    return sorted(set(found))


# ══════════════════════════════════════════════════════════════════════════════
# NEO4J KEYWORD WRITING FOR IMAGES
# ══════════════════════════════════════════════════════════════════════════════

def _write_image_keywords_to_neo4j(
    image_path: str,
    keywords:   list[str],
    caption:    str,
) -> None:
    """
    Write image session + extracted keywords to Neo4j.

    Creates:
        (:ImageSession {path, caption}) -[:HAS_KEYWORD]-> (:Keyword {name})

    Keywords are MERGED so existing keyword nodes are reused.
    """
    if not keywords:
        logger.info("[Captioner] No keywords to write to Neo4j")
        return

    try:
        from backend.graph.neo4j_client import _get_driver

        driver = _get_driver()
        with driver.session() as s:

            # Create ImageSession node
            s.run(
                """
                MERGE (img:ImageSession {path: $path})
                SET img.caption   = $caption,
                    img.timestamp = timestamp()
                """,
                path=    str(image_path),
                caption= caption[:500],
            )

            # Write each keyword and link to ImageSession
            for kw in keywords:
                s.run(
                    """
                    MERGE (k:Keyword {name: $kw})
                    SET k.domain = 'remote_sensing'
                    WITH k
                    MATCH (img:ImageSession {path: $path})
                    MERGE (img)-[:HAS_KEYWORD]->(k)
                    """,
                    kw=  kw,
                    path=str(image_path),
                )

            # Link to Remote Sensing domain
            s.run(
                """
                MERGE (d:Domain {key: 'remote_sensing'})
                SET d.name = 'Remote Sensing'
                WITH d
                MATCH (img:ImageSession {path: $path})
                MERGE (img)-[:BELONGS_TO]->(d)
                """,
                path=str(image_path),
            )

        logger.info(
            f"[Captioner] Neo4j: wrote {len(keywords)} keywords "
            f"for image {Path(image_path).name}"
        )

    except Exception as e:
        logger.warning(f"[Captioner] Neo4j keyword write failed (non-fatal): {e}")


# ══════════════════════════════════════════════════════════════════════════════
# PROMPTS
# ══════════════════════════════════════════════════════════════════════════════

_COMBINED_PROMPT = """\
This is a satellite image. Analyze it and respond in exactly this format:

SHORT: <one precise sentence describing the main content>
DETAILED: <3-5 sentences covering terrain type, land use, visible structures, \
vegetation, water bodies, urban features, environmental conditions, \
and any notable characteristics>

Be specific about what is actually visible."""

_DETECTION_PROMPT = """\
This is a satellite image ({w}x{h} pixels).

Identify the main distinct regions, objects, or land-use areas visible.
Return a JSON array only:

[
  {{"label": "region name", "bbox_2d": [x1, y1, x2, y2]}},
  ...
]

Rules:
- Identify 3-8 meaningful regions
- Use specific labels (e.g. "residential blocks", "river", "highway", "forest patch")
- Boxes within image bounds (0-{w} for x, 0-{h} for y)
- Return only the JSON array"""


# ══════════════════════════════════════════════════════════════════════════════
# GEMINI INFERENCE
# ══════════════════════════════════════════════════════════════════════════════

def _run_prompt(pil_image, prompt: str) -> str:
    client = _get_client()
    t0     = time.perf_counter()
    response = client.generate_content(
        [prompt, pil_image],
        generation_config={"temperature": 0.2, "max_output_tokens": 512},
    )
    logger.info(f"[Captioner] Gemini: {(time.perf_counter()-t0)*1000:.0f}ms")
    return response.text.strip()


# ══════════════════════════════════════════════════════════════════════════════
# PARSING
# ══════════════════════════════════════════════════════════════════════════════

def _parse_captions(raw: str) -> tuple[str, str]:
    caption = ""
    detailed = ""
    collecting = None

    for line in raw.split("\n"):
        s = line.strip()
        if s.upper().startswith("SHORT:"):
            caption    = s[6:].strip()
            collecting = None
        elif s.upper().startswith("DETAILED:"):
            detailed   = s[9:].strip()
            collecting = "detailed"
        elif collecting == "detailed" and s:
            detailed += " " + s

    if not caption and not detailed:
        sentences = [s.strip() for s in raw.split(".") if s.strip()]
        caption   = sentences[0] + "." if sentences else raw[:200]
        detailed  = ". ".join(sentences[1:4]) if len(sentences) > 1 else ""

    return caption.strip(), detailed.strip()


def _parse_bounding_boxes(raw: str, w: int, h: int) -> list[dict]:
    boxes = []
    try:
        raw   = re.sub(r"```(?:json)?", "", raw).strip()
        match = re.search(r'\[.*\]', raw, re.DOTALL)
        if not match:
            return boxes
        data = json.loads(match.group())
        for item in data:
            label = item.get("label", "region")
            bbox  = item.get("bbox_2d", item.get("bbox", []))
            if len(bbox) == 4:
                x1, y1, x2, y2 = [float(v) for v in bbox]
                x1, x2 = max(0, min(x1, w)), max(0, min(x2, w))
                y1, y2 = max(0, min(y1, h)), max(0, min(y2, h))
                if x2 > x1 and y2 > y1:
                    boxes.append({
                        "label": str(label),
                        "bbox":  [round(x1,1), round(y1,1),
                                  round(x2,1), round(y2,1)],
                    })
    except Exception as e:
        logger.warning(f"[Captioner] Bbox parse warning: {e}")
    return boxes


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ══════════════════════════════════════════════════════════════════════════════

def caption_image(
    file_path:      str | Path,
    run_detection:  bool = True,
    run_detailed:   bool = True,
    write_to_graph: bool = True,    # ← NEW: auto-write keywords to Neo4j
) -> Florence2Result:
    """
    Caption a satellite image via Gemini Vision.

    Now also:
        - Extracts keywords from the caption
        - Writes keywords to Neo4j (ImageSession → Keyword nodes)
        - Keywords become queryable in graph + usable by Research Agent

    Returns Florence2Result (unchanged public API).
    """
    t_total = time.perf_counter()
    path    = Path(file_path)
    logger.info(f"[Captioner] ── Starting: {path.name} ──")

    pil_image, _ = load_image(path)
    pil_image    = _prepare_image(pil_image)
    w, h         = pil_image.width, pil_image.height

    # Caption + detailed
    cap_raw = _run_prompt(pil_image, _COMBINED_PROMPT)
    caption, detailed_caption = _parse_captions(cap_raw)
    logger.info(f"[Captioner] Caption: {caption}")

    # Detection
    bounding_boxes = []
    if run_detection:
        det_raw        = _run_prompt(pil_image, _DETECTION_PROMPT.format(w=w, h=h))
        bounding_boxes = _parse_bounding_boxes(det_raw, w, h)
        logger.info(f"[Captioner] {len(bounding_boxes)} boxes detected")

    # ── Extract keywords from caption ──────────────────────────────────────────
    image_keywords = _extract_keywords_from_caption(
        caption,
        detailed_caption or "",
    )
    logger.info(f"[Captioner] Keywords extracted: {image_keywords}")

    # ── Write keywords to Neo4j ────────────────────────────────────────────────
    if write_to_graph and image_keywords:
        _write_image_keywords_to_neo4j(
            image_path= str(path),
            keywords=   image_keywords,
            caption=    caption,
        )

    logger.info(
        f"[Captioner] ── Total: {(time.perf_counter()-t_total)*1000:.0f}ms ──"
    )

    result = Florence2Result(
        file_path=        str(path),
        caption=          caption,
        detailed_caption= detailed_caption if run_detailed else None,
        bounding_boxes=   bounding_boxes,
        model_name=       MODEL_NAME,
    )

    # Attach keywords to result for downstream use (Knowledge Fusion)
    result.image_keywords = image_keywords   # type: ignore[attr-defined]

    return result