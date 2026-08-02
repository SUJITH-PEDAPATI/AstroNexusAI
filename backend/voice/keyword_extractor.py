"""
AstroNexus AI — Keyword & Entity Extractor
============================================
Extracts meaningful scientific keywords and named entities from
the voice transcript before it enters the RAG + Knowledge Graph pipeline.

Priority strategy (fastest first, most accurate last):
    1. Pattern matching   — deterministic, zero latency, high precision
                           for known space/astronomy/AI terms
    2. spaCy NER          — if en_core_web_sm is installed
                           (pip install spacy && python -m spacy download en_core_web_sm)
    3. Existing extractor — delegates to backend.agents.tier_classifier
                           (already used by the query router)
    4. LLM fallback       — Ollama/Gemini, only if 1-3 return nothing

Returns:
    KeywordResult with:
        keywords   — filtered word-level terms
        entities   — named entities (name, label) from NER
        scientific — multi-word scientific phrases
        all_terms  — deduplicated union of all three

Design for future extensibility:
    • Add a new strategy by implementing _Strategy and adding to STRATEGIES list
    • WhisperX word-level timestamps can be merged here for click-to-seek
    • Confidence scores can be attached per entity for graph edge weighting

Dependencies:
    Required:  (none beyond stdlib + existing project deps)
    Optional:  spacy, en_core_web_sm
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import NamedTuple

logger = logging.getLogger(__name__)

# ── Stop words ────────────────────────────────────────────────────────────────

_STOP = frozenset({
    "a","an","the","and","or","but","if","in","on","at","to","for","of",
    "with","by","from","as","is","was","are","were","be","been","being",
    "have","has","had","do","does","did","will","would","could","should",
    "may","might","shall","can","need","dare","ought","used","about","above",
    "after","again","against","all","am","any","because","before","between",
    "both","during","each","few","further","here","how","i","into","it","its",
    "just","more","most","my","no","not","now","only","other","our","out",
    "over","own","same","so","some","such","than","that","their","them",
    "then","there","these","they","this","those","through","too","under",
    "up","very","we","what","when","where","which","while","who","why",
    "you","your","also","yet","still","just","even","well","got","get",
    "make","go","going","said","say","say","tell","think","know","see","come",
    "look","use","find","give","take","want","seem","feel","try","leave","call",
})

# ── Scientific domain patterns ────────────────────────────────────────────────

_MULTI_WORD = [
    # Space missions & telescopes
    "james webb space telescope", "hubble space telescope",
    "origins space telescope", "chandra x-ray observatory",
    "spitzer space telescope", "kepler space telescope",
    "transiting exoplanet survey satellite",
    "international space station",
    # Earth observation
    "remote sensing", "earth observation", "synthetic aperture radar",
    "land use", "land cover", "sea level rise", "climate change",
    "global warming", "air quality index", "ndvi", "multispectral imaging",
    "hyperspectral imaging",
    # Astronomy
    "dark matter", "dark energy", "black hole", "neutron star",
    "white dwarf", "planetary nebula", "globular cluster",
    "gravitational wave", "cosmic microwave background",
    "event horizon", "accretion disk", "stellar nucleosynthesis",
    "redshift", "blueshift", "doppler effect",
    # Exoplanets & astrobiology
    "exoplanet", "exoplanets", "habitable zone", "goldilocks zone",
    "biosignature", "biosignatures", "transit photometry",
    "radial velocity", "direct imaging",
    # AI & ML
    "machine learning", "deep learning", "neural network",
    "transformer model", "large language model", "reinforcement learning",
    "convolutional neural network", "generative adversarial network",
    "retrieval augmented generation", "knowledge graph",
    "natural language processing", "computer vision",
]

_SINGLE = {
    # Missions / satellites
    "jwst", "hubble", "iss", "artemis", "perseverance", "curiosity",
    "cassini", "voyager", "pioneer", "new horizons", "dawn", "osiris-rex",
    "juno", "insight", "maven", "opportunity", "spirit", "ingenuity",
    "sentinel", "landsat", "goes", "modis", "aster",
    # Organizations
    "nasa", "esa", "isro", "jaxa", "roscosmos", "spacex", "blue origin",
    "boeing", "northrop", "lockheed",
    # Celestial objects
    "milky way", "andromeda", "proxima", "centauri", "orion",
    "crab nebula", "pillars of creation", "horsehead nebula",
    "sombrero galaxy", "whirlpool galaxy",
    "mercury", "venus", "mars", "jupiter", "saturn",
    "uranus", "neptune", "pluto", "ceres", "eris",
    # Science terms
    "photometry", "spectroscopy", "interferometry", "astrometry",
    "parallax", "parsec", "lightyear", "astronomical unit",
    "luminosity", "magnitude", "albedo", "eccentricity",
    "inclination", "perihelion", "aphelion", "barycenter",
    "electromagnetic", "infrared", "ultraviolet", "x-ray", "gamma-ray",
    "radio wave", "microwave", "visible light",
    "astrophysics", "cosmology", "astrochemistry", "astrobiology",
    "heliophysics", "planetology",
}

# NER label → human-readable category
_NER_KEEP = {"ORG", "PERSON", "GPE", "LOC", "PRODUCT", "EVENT", "WORK_OF_ART", "NORP"}

# ── Result types ──────────────────────────────────────────────────────────────

class Entity(NamedTuple):
    name:  str    # exact surface form from text
    label: str    # spaCy NER label or "PATTERN"

@dataclass
class KeywordResult:
    """All extracted terms, ready for Neo4j and RAG."""
    keywords:   list[str]    = field(default_factory=list)  # single-token keywords
    entities:   list[Entity] = field(default_factory=list)  # named entities
    scientific: list[str]    = field(default_factory=list)  # multi-word science terms
    all_terms:  list[str]    = field(default_factory=list)  # deduplicated union
    method:     str          = "pattern"                     # strategy used

    def is_empty(self) -> bool:
        return not self.all_terms

    def summary(self) -> str:
        return (
            f"keywords={len(self.keywords)} "
            f"entities={len(self.entities)} "
            f"scientific={len(self.scientific)}"
        )


# ── Strategy 1 — Pattern matching (always runs) ───────────────────────────────

def _extract_patterns(text: str) -> tuple[list[str], list[str], list[Entity]]:
    """
    Fast, deterministic keyword extraction using compiled patterns.
    No external dependencies.

    Returns:
        (keywords, scientific_phrases, pattern_entities)
    """
    tl = text.lower()

    # Multi-word scientific phrases
    scientific = [p for p in _MULTI_WORD if p in tl]

    # Single-word / short domain terms
    pattern_entities = [
        Entity(name=term.title(), label="PATTERN")
        for term in _SINGLE
        if term.lower() in tl
    ]

    # General keyword extraction — reuse existing tier_classifier if available
    keywords: list[str] = []
    try:
        from backend.agents.tier_classifier import extract_keywords
        keywords = extract_keywords(text)
    except Exception:
        # Fallback: simple tokenisation + stop-word filter
        tokens = re.findall(r'\b[a-zA-Z][a-zA-Z\-]{2,}\b', text)
        seen: dict[str, None] = {}
        for tok in tokens:
            t = tok.lower()
            if t not in _STOP and t not in seen:
                seen[t] = None
                keywords.append(t)

    return keywords, scientific, pattern_entities


# ── Strategy 2 — spaCy NER (optional, higher accuracy) ───────────────────────

def _extract_spacy(text: str) -> list[Entity]:
    """
    Run spaCy en_core_web_sm NER.
    Returns empty list if spaCy or the model is not installed — non-fatal.
    """
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(text)
        entities = [
            Entity(name=ent.text, label=ent.label_)
            for ent in doc.ents
            if ent.label_ in _NER_KEEP and len(ent.text) > 2
        ]
        logger.debug(f"[KeywordExtractor/spaCy] {len(entities)} entities")
        return entities
    except Exception as e:
        logger.debug(f"[KeywordExtractor/spaCy] unavailable: {e}")
        return []


# ── Strategy 3 — LLM fallback (last resort) ──────────────────────────────────

def _extract_llm(text: str) -> list[str]:
    """
    Ask Ollama to extract entities when pattern + NER yield nothing.
    Non-fatal: returns empty list on any failure.
    """
    try:
        import json
        import urllib.request
        import os

        base  = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
        model = os.environ.get("OLLAMA_MODEL",    "qwen3:4b")

        prompt = (
            "Extract all important named entities, scientific terms, missions, "
            "organizations, and technical keywords from this text. "
            "Return ONLY a JSON array of strings, no explanation.\n\n"
            f"Text: {text[:400]}"
        )
        payload = json.dumps({
            "model": model, "prompt": prompt,
            "stream": False, "options": {"temperature": 0.0, "num_predict": 200},
        }).encode()

        req = urllib.request.Request(
            f"{base}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            raw    = json.loads(resp.read()).get("response", "")
            # Extract JSON array from model output
            m = re.search(r'\[.*?\]', raw, re.DOTALL)
            if m:
                items = json.loads(m.group())
                return [str(i).strip() for i in items if isinstance(i, str)]
    except Exception as e:
        logger.debug(f"[KeywordExtractor/LLM] failed: {e}")
    return []


# ── Public API ────────────────────────────────────────────────────────────────

def extract(text: str, use_llm_fallback: bool = True) -> KeywordResult:
    """
    Extract keywords and entities from a transcript.

    Runs strategies in order: pattern → spaCy → LLM fallback.

    Args:
        text:             The transcript text
        use_llm_fallback: Call LLM if pattern + spaCy yield nothing

    Returns:
        KeywordResult with all extracted terms
    """
    if not text or not text.strip():
        return KeywordResult()

    logger.info(f"[KeywordExtractor] Extracting from: {text[:60]!r}")

    # ── Strategy 1: pattern ───────────────────────────────────────────────────
    keywords, scientific, pattern_entities = _extract_patterns(text)
    method = "pattern"

    # ── Strategy 2: spaCy NER ────────────────────────────────────────────────
    spacy_entities = _extract_spacy(text)
    if spacy_entities:
        method = "spacy+pattern"

    # Merge entities from both sources; deduplicate by normalised name
    all_entities: list[Entity] = []
    seen_names: set[str] = set()
    for ent in pattern_entities + spacy_entities:
        key = ent.name.lower().strip()
        if key not in seen_names and len(key) > 2:
            seen_names.add(key)
            all_entities.append(ent)

    # ── Strategy 3: LLM fallback ──────────────────────────────────────────────
    llm_extras: list[str] = []
    if use_llm_fallback and not keywords and not scientific and not all_entities:
        llm_extras = _extract_llm(text)
        if llm_extras:
            method = "llm"
            logger.info(f"[KeywordExtractor] LLM found {len(llm_extras)} terms")

    # ── Build unified all_terms list ──────────────────────────────────────────
    seen: set[str] = set()
    all_terms: list[str] = []
    for term in (
        [e.name for e in all_entities]
        + scientific
        + keywords
        + llm_extras
    ):
        t = term.strip()
        k = t.lower()
        if k and k not in seen and k not in _STOP and len(k) > 2:
            seen.add(k)
            all_terms.append(t)

    result = KeywordResult(
        keywords=   keywords,
        entities=   all_entities,
        scientific= scientific,
        all_terms=  all_terms,
        method=     method,
    )
    logger.info(f"[KeywordExtractor] {result.summary()} via {method}")
    return result