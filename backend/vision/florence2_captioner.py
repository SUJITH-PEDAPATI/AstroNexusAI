"""
AstroNexus AI — Vision Captioner (Google Gemini API)

Sends the actual satellite image to Gemini Vision for:
    - Short caption
    - Detailed description
    - Bounding box detection

Model: gemini-2.0-flash (free tier, generous rate limits)
Install: pip install google-generativeai

Set in .env:
    GEMINI_API_KEY=your_key_here

Get key free at: https://aistudio.google.com

Public API: identical to Florence-2 — Florence2Result returned.
SAM2 and satellite_pipeline.py unchanged.
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

from backend.vision.models import Florence2Result
from backend.vision.image_loader import load_image

logger = logging.getLogger(__name__)

MODEL_NAME   = "gemini-flash-latest"  # Stable alias → always the newest Flash available for your tier
MAX_IMG_SIZE = 1024   # Gemini handles up to 3072px but 1024 is fast + accurate

_client = None


# ══════════════════════════════════════════════════════════════════════════════
# CLIENT
# ══════════════════════════════════════════════════════════════════════════════

def _get_client():
    """
    Build and cache a google-genai Client.

    Uses the NEW google-genai SDK (pip install google-genai).
    The old google-generativeai package is deprecated/EOL — do NOT use it.
    """
    global _client
    if _client is not None:
        return _client

    # Load .env so GEMINI_API_KEY is available regardless of how the script is invoked
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass  # python-dotenv not installed; rely on env vars already set

    try:
        from google import genai  # google-genai (NOT google-generativeai)
    except ImportError as e:
        raise ImportError(
            "Install the Google GenAI SDK: pip install google-genai\n"
            "(Do NOT use google-generativeai — it is deprecated and EOL)"
        ) from e

    api_key = (
        os.environ.get("GEMINI_API_KEY") or
        os.environ.get("GOOGLE_API_KEY")
    )
    if not api_key:
        raise EnvironmentError(
            "GEMINI_API_KEY not found in environment.\n"
            "Add to .env: GEMINI_API_KEY=your_key_here\n"
            "Get free key at: https://aistudio.google.com"
        )

    _client = genai.Client(api_key=api_key)
    logger.info(f"[Captioner] Gemini client ready — model: {MODEL_NAME}")
    return _client


# ══════════════════════════════════════════════════════════════════════════════
# IMAGE ENCODING
# ══════════════════════════════════════════════════════════════════════════════

def _prepare_image(pil_image):
    """
    Resize to MAX_IMG_SIZE and return a Gemini-compatible image part.
    Gemini accepts PIL images directly via google.generativeai.
    """
    from PIL import Image

    w, h = pil_image.size
    if max(w, h) > MAX_IMG_SIZE:
        scale    = MAX_IMG_SIZE / max(w, h)
        new_w    = int(w * scale)
        new_h    = int(h * scale)
        pil_image = pil_image.resize((new_w, new_h), Image.LANCZOS)
        logger.info(f"[Captioner] Resized {w}x{h} → {new_w}x{new_h}")

    return pil_image


# ══════════════════════════════════════════════════════════════════════════════
# PROMPTS
# ══════════════════════════════════════════════════════════════════════════════

_COMBINED_PROMPT = """\
This is a satellite image. Analyze it carefully and respond in exactly this format:

SHORT: <one precise sentence describing the main content>
DETAILED: <2-4 sentences covering terrain type, land use, visible structures, \
vegetation, water bodies, urban features, or any notable characteristics>

Be specific about what is actually visible. Do not guess."""

_DETECTION_PROMPT = """\
This is a satellite image ({w}x{h} pixels).

Identify the main distinct regions, objects, or land-use areas visible.
Return a JSON array only — no explanation:

[
  {{"label": "region name", "bbox_2d": [x1, y1, x2, y2]}},
  ...
]

Rules:
- x1,y1 = top-left corner in pixels
- x2,y2 = bottom-right corner in pixels  
- Identify 3-8 meaningful regions
- Use specific labels (e.g. "residential blocks", "river", "highway", "forest")
- Boxes must be within image bounds (0-{w} for x, 0-{h} for y)"""


# ══════════════════════════════════════════════════════════════════════════════
# INFERENCE
# ══════════════════════════════════════════════════════════════════════════════

def _run_prompt(pil_image, prompt: str) -> str:
    """
    Send image + prompt to Gemini and return response text.

    Uses the new google-genai SDK:
        client.models.generate_content(model=..., contents=[...])
    """
    from google.genai import types
    import io

    client = _get_client()

    # Encode PIL image → bytes for the new SDK
    buf = io.BytesIO()
    pil_image.save(buf, format="JPEG", quality=90)
    image_bytes = buf.getvalue()

    t0       = time.perf_counter()
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=[
            types.Part(text=prompt),
            types.Part(
                inline_data=types.Blob(data=image_bytes, mime_type="image/jpeg")
            ),
        ],
        config=types.GenerateContentConfig(
            temperature=0.2,
            max_output_tokens=512,
        ),
    )
    elapsed = (time.perf_counter() - t0) * 1000
    logger.info(f"[Captioner] Gemini inference: {elapsed:.0f}ms")

    return response.text.strip()


# ══════════════════════════════════════════════════════════════════════════════
# PARSING
# ══════════════════════════════════════════════════════════════════════════════

def _parse_captions(raw: str) -> tuple[str, str]:
    """Parse SHORT / DETAILED from Gemini response."""
    caption          = ""
    detailed_caption = ""
    collecting       = None

    for line in raw.split("\n"):
        stripped = line.strip()
        if stripped.upper().startswith("SHORT:"):
            caption    = stripped[6:].strip()
            collecting = None
        elif stripped.upper().startswith("DETAILED:"):
            detailed_caption = stripped[9:].strip()
            collecting       = "detailed"
        elif collecting == "detailed" and stripped:
            detailed_caption += " " + stripped

    # Fallback if Gemini didn't follow the format
    if not caption and not detailed_caption:
        sentences = [s.strip() for s in raw.split(".") if s.strip()]
        caption          = sentences[0] + "." if sentences else raw[:200]
        detailed_caption = ". ".join(sentences[1:4]) if len(sentences) > 1 else ""

    return caption.strip(), detailed_caption.strip()


def _parse_bounding_boxes(raw: str, w: int, h: int) -> list[dict]:
    """Parse JSON bounding boxes from Gemini detection response."""
    boxes = []
    try:
        # Strip markdown code fences if present
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
                x1 = max(0.0, min(x1, w))
                y1 = max(0.0, min(y1, h))
                x2 = max(0.0, min(x2, w))
                y2 = max(0.0, min(y2, h))
                if x2 > x1 and y2 > y1:
                    boxes.append({
                        "label": str(label),
                        "bbox":  [round(x1, 1), round(y1, 1),
                                  round(x2, 1), round(y2, 1)],
                    })
    except Exception as e:
        logger.warning(f"[Captioner] Bbox parse warning: {e}")
    return boxes


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ══════════════════════════════════════════════════════════════════════════════

def caption_image(
    file_path:     str | Path,
    run_detection: bool = True,
    run_detailed:  bool = True,
) -> Florence2Result:
    """
    Generate captions and bounding boxes by sending the satellite image
    directly to Google Gemini Vision API.

    Pipeline:
        1. Load image + resize
        2. Send image + caption prompt → parse SHORT / DETAILED
        3. Send image + detection prompt → parse bounding boxes

    The actual image pixels are sent to Gemini in both calls.

    Returns Florence2Result — identical to Florence-2 API.
    SAM2 and satellite_pipeline.py unchanged.
    """
    t_total = time.perf_counter()
    path    = Path(file_path)
    logger.info(f"[Captioner] ── Starting: {path.name} ──")

    # ── Load + prepare ─────────────────────────────────────────────────────────
    t0           = time.perf_counter()
    pil_image, _ = load_image(path)
    pil_image    = _prepare_image(pil_image)
    w, h         = pil_image.width, pil_image.height
    logger.info(
        f"[Captioner] Load+resize: {(time.perf_counter()-t0)*1000:.0f}ms  "
        f"{w}x{h}"
    )

    # ── Caption + detailed ─────────────────────────────────────────────────────
    t0       = time.perf_counter()
    cap_raw  = _run_prompt(pil_image, _COMBINED_PROMPT)
    logger.info(
        f"[Captioner] Caption call: {(time.perf_counter()-t0)*1000:.0f}ms"
    )

    caption, detailed_caption = _parse_captions(cap_raw)
    logger.info(f"[Captioner] Caption: {caption}")
    if detailed_caption:
        logger.info(f"[Captioner] Detailed: {detailed_caption[:100]}...")

    # ── Detection ──────────────────────────────────────────────────────────────
    bounding_boxes = []
    if run_detection:
        t0      = time.perf_counter()
        det_raw = _run_prompt(
            pil_image,
            _DETECTION_PROMPT.format(w=w, h=h),
        )
        bounding_boxes = _parse_bounding_boxes(det_raw, w, h)
        logger.info(
            f"[Captioner] Detection call: {(time.perf_counter()-t0)*1000:.0f}ms  "
            f"{len(bounding_boxes)} boxes"
        )
        for b in bounding_boxes[:5]:
            logger.info(f"[Captioner]   {b['label']}: {b['bbox']}")

    logger.info(
        f"[Captioner] ── Total: {(time.perf_counter()-t_total)*1000:.0f}ms ──"
    )

    return Florence2Result(
        file_path=        str(path),
        caption=          caption,
        detailed_caption= detailed_caption if run_detailed else None,
        bounding_boxes=   bounding_boxes,
        model_name=       MODEL_NAME,
    )