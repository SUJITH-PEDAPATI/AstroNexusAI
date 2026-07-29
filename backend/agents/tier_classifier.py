"""
AstroNexus AI — Tier Classifier

Classifies every query into one of three tiers:

    TIER 1 — Scientific query + paper uploaded
              → Full RAG + Neo4j + Ollama + Gemini + Evaluator

    TIER 2 — Scientific query, no paper
              → Keywords → APIs → Neo4j → Gemini (real data)

    TIER 3 — General / off-topic query
              → Ollama + Gemini + scope note

Classification is fast — keyword matching first, LLM only for ambiguous.
"""
from __future__ import annotations

import re
import logging

logger = logging.getLogger(__name__)

# ── Scientific domain keywords ────────────────────────────────────────────────
SCIENCE_KEYWORDS = {
    # Astronomy / Space
    "star", "galaxy", "black hole", "nebula", "quasar", "pulsar", "exoplanet",
    "dark matter", "dark energy", "cosmology", "universe", "telescope", "orbit",
    "nasa", "esa", "isro", "jaxa", "satellite", "spacecraft", "mission",
    "hubble", "jwst", "james webb", "kepler", "tess", "voyager", "cassini",
    "astrophysics", "spectroscopy", "photometry", "redshift", "parallax",

    # Remote Sensing / Earth Observation
    "remote sensing", "earth observation", "sentinel", "landsat", "modis",
    "sar", "lidar", "radar", "multispectral", "hyperspectral", "ndvi",
    "land cover", "land use", "flood", "wildfire", "deforestation",
    "geospatial", "gis", "geotiff", "raster",

    # AI / ML / Research
    "machine learning", "deep learning", "neural network", "transformer",
    "bert", "gpt", "llm", "cnn", "resnet", "attention", "embedding",
    "classification", "segmentation", "detection", "dataset", "benchmark",
    "paper", "research", "model", "algorithm", "accuracy", "loss",
    "training", "inference", "fine-tuning", "rag", "retrieval",

    # Climate / Environment
    "climate", "weather", "temperature", "precipitation", "atmosphere",
    "carbon", "emission", "global warming", "sea level", "glacier",
    "air quality", "pollution", "ozone",

    # Physics / Math
    "quantum", "relativity", "gravity", "particle", "photon", "wavelength",
    "frequency", "spectrum", "energy", "mass", "velocity", "acceleration",
}

STOP_WORDS = {
    "what", "who", "when", "where", "how", "is", "are", "was", "will",
    "the", "a", "an", "of", "in", "to", "for", "and", "or", "by",
    "tell", "me", "give", "show", "find", "can", "about", "does",
    "please", "could", "would", "should", "explain", "describe",
}


def extract_keywords(query: str) -> list[str]:
    """Extract meaningful keywords from query."""
    words    = re.findall(r'\b[a-zA-Z]{3,}\b', query.lower())
    keywords = [w for w in words if w not in STOP_WORDS]

    # Multi-word phrases
    phrases = [
        "machine learning", "deep learning", "neural network",
        "remote sensing", "earth observation", "dark matter", "dark energy",
        "black hole", "climate change", "global warming", "air quality",
        "land cover", "land use", "sea level", "space mission",
    ]
    query_lower = query.lower()
    for phrase in phrases:
        if phrase in query_lower:
            keywords.insert(0, phrase)

    return list(dict.fromkeys(keywords))


def is_scientific(keywords: list[str]) -> bool:
    """Check if any keyword is science-related."""
    for kw in keywords:
        if kw in SCIENCE_KEYWORDS:
            return True
        for sci_kw in SCIENCE_KEYWORDS:
            if len(kw) > 4 and (sci_kw in kw or kw in sci_kw):
                return True
    return False


def classify_tier(
    query:        str,
    paper_loaded: bool = False,
    paper_id:     str | None = None,
) -> dict:
    """
    Classify query into Tier 1, 2, or 3.

    Returns:
        {
            "tier":       1 | 2 | 3,
            "label":      "research" | "science_api" | "general",
            "keywords":   list[str],
            "is_science": bool,
            "reason":     str,
        }
    """
    keywords   = extract_keywords(query)
    scientific = is_scientific(keywords)

    if scientific and (paper_loaded or paper_id):
        return {
            "tier":       1,
            "label":      "research",
            "keywords":   keywords,
            "is_science": True,
            "reason":     "scientific query + paper available → full RAG pipeline",
        }

    elif scientific:
        return {
            "tier":       2,
            "label":      "science_api",
            "keywords":   keywords,
            "is_science": True,
            "reason":     "scientific query, no paper → real-time API + Gemini",
        }

    else:
        return {
            "tier":       3,
            "label":      "general",
            "keywords":   keywords,
            "is_science": False,
            "reason":     "general query → Ollama + Gemini + scope note",
        }