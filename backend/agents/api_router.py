"""
AstroNexus AI — API Router (Intelligence Layer)

Decides WHICH external APIs to call for a given query.
Does NOT call every API for every keyword — uses relevance scoring and caching.

Decision pipeline:
    1. Extract domain keywords from query
    2. Lookup Neo4j: which APIs are linked to those keywords?
    3. Score relevance (keyword match strength + query intent alignment)
    4. Check cache (same query within TTL → skip API call)
    5. Return ranked list of APIs to call, with extracted params

Only APIs scoring above RELEVANCE_THRESHOLD are called.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

RELEVANCE_THRESHOLD = 0.35   # minimum score to trigger an API call
CACHE_TTL_SECONDS   = 300    # 5 minutes — same query reuses cached result
MAX_APIS_PER_QUERY  = 2      # never call more than 2 APIs per query

# Simple in-memory cache: {query_hash: (timestamp, result)}
_cache: dict[str, tuple[float, list]] = {}


@dataclass
class APICandidate:
    """A scored API candidate for a given query."""
    name:         str
    endpoint:     str
    method:       str
    description:  str
    score:        float
    matched_keywords: list[str]
    extracted_params: dict     = field(default_factory=dict)
    requires_key: bool         = False


# ══════════════════════════════════════════════════════════════════════════════
# KEYWORD EXTRACTOR
# ══════════════════════════════════════════════════════════════════════════════

_STOP_WORDS = {
    "what", "who", "when", "where", "how", "is", "are", "was", "will",
    "the", "a", "an", "of", "in", "to", "for", "and", "or", "by",
    "about", "tell", "me", "give", "show", "find", "get", "can", "you",
    "next", "last", "some", "this", "that", "it", "its", "there",
}

_LOCATION_RE = re.compile(
    r'\b(?:in|at|near|over|around)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b'
)
_YEAR_RE     = re.compile(r'\b(20\d{2})\b')
_NUM_RE      = re.compile(r'\b(\d+)\s*(?:year|month|day|week)s?\b')


def extract_query_keywords(query: str) -> dict:
    """
    Extract structured information from query for API parameter building.

    Returns:
        {
            "keywords":  list[str],   content keywords
            "location":  str | None,  detected location name
            "year":      str | None,  detected year
            "timespan":  int | None,  detected number of days/years
        }
    """
    words = query.lower().split()
    keywords = [
        w.strip("?.,!\"'")
        for w in words
        if len(w.strip("?.,!\"'")) > 3
        and w.strip("?.,!\"'") not in _STOP_WORDS
    ]

    # Multi-word phrase matching
    query_lower = query.lower()
    phrases = [
        "climate change", "global warming", "air quality", "wildfire",
        "flood detection", "land cover", "earth observation", "sea level",
        "carbon emission", "temperature anomaly", "satellite imagery",
        "active fire", "forest fire", "precipitation", "sea surface",
    ]
    for phrase in phrases:
        if phrase in query_lower and phrase.split()[0] not in keywords:
            keywords.insert(0, phrase)

    location = None
    loc_match = _LOCATION_RE.search(query)
    if loc_match:
        location = loc_match.group(1)

    year = None
    year_match = _YEAR_RE.search(query)
    if year_match:
        year = year_match.group(1)

    timespan = None
    num_match = _NUM_RE.search(query)
    if num_match:
        timespan = int(num_match.group(1))

    return {
        "keywords": list(dict.fromkeys(keywords)),   # dedup preserving order
        "location": location,
        "year":     year,
        "timespan": timespan,
    }


# ══════════════════════════════════════════════════════════════════════════════
# NEO4J API LOOKUP
# ══════════════════════════════════════════════════════════════════════════════

def _lookup_apis_from_neo4j(keywords: list[str]) -> list[dict]:
    """
    Query Neo4j for ExternalAPI nodes linked to the given keywords.
    Returns raw API records with matched keyword counts.
    """
    if not keywords:
        return []

    try:
        from backend.graph.neo4j_client import _get_driver
        driver = _get_driver()

        results: dict[str, dict] = {}

        with driver.session() as s:
            for kw in keywords[:8]:   # limit to avoid slow queries
                rows = s.run(
                    """
                    MATCH (k:Keyword)-[:TRIGGERS]->(a:ExternalAPI)
                    WHERE toLower(k.name) CONTAINS toLower($kw)
                      AND a.enabled = true
                    RETURN
                        a.name         AS name,
                        a.endpoint     AS endpoint,
                        a.method       AS method,
                        a.description  AS description,
                        a.params       AS params,
                        a.requires_key AS requires_key,
                        k.name         AS matched_kw
                    LIMIT 10
                    """,
                    kw=kw,
                ).data()

                for row in rows:
                    api_name = row["name"]
                    if api_name not in results:
                        results[api_name] = {**row, "matched_keywords": []}
                    results[api_name]["matched_keywords"].append(row["matched_kw"])

        return list(results.values())

    except Exception as e:
        logger.warning(f"[APIRouter] Neo4j lookup failed: {e}")
        return []


# ══════════════════════════════════════════════════════════════════════════════
# RELEVANCE SCORER
# ══════════════════════════════════════════════════════════════════════════════

def _score_api(
    api:              dict,
    query_keywords:   list[str],
    query_info:       dict,
) -> float:
    """
    Score how relevant an API is for this query.

    Factors:
        - Number of matched keywords (primary signal)
        - Location present + API needs location (bonus)
        - Timespan present + API supports it (bonus)
        - API requires key but none set (penalty)

    Returns score 0.0–1.0
    """
    matched = api.get("matched_keywords", [])
    n_matched = len(matched)

    if n_matched == 0:
        return 0.0

    # Base score: matched keywords / total query keywords
    base = min(n_matched / max(len(query_keywords), 1), 1.0)

    # Bonus: location detected and this API can use it
    params_str = api.get("params", "")
    if query_info.get("location") and "latitude" in params_str:
        base += 0.15

    # Bonus: timespan detected
    if query_info.get("timespan") and (
        "forecast_days" in params_str or "day_range" in params_str
    ):
        base += 0.10

    # Penalty: requires API key that isn't set
    if api.get("requires_key"):
        import os
        api_name = api.get("name", "")
        env_key  = f"{api_name.upper()}_API_KEY"
        if not os.environ.get(env_key):
            base -= 0.20

    return round(min(max(base, 0.0), 1.0), 3)


# ══════════════════════════════════════════════════════════════════════════════
# PARAMETER EXTRACTOR
# ══════════════════════════════════════════════════════════════════════════════

# Default coordinates for location-agnostic queries
DEFAULT_LAT = 28.6139   # New Delhi (adjust per deployment region)
DEFAULT_LON = 77.2090

_GEOCODE_CACHE: dict[str, tuple[float, float]] = {}

def _extract_params(api: dict, query_info: dict) -> dict:
    """
    Build API call parameters from query context.
    Uses sensible defaults when explicit values are missing.
    """
    name      = api.get("name", "")
    params    = {}
    location  = query_info.get("location")
    timespan  = query_info.get("timespan")

    # Geocode location if present
    lat, lon = DEFAULT_LAT, DEFAULT_LON
    if location:
        if location in _GEOCODE_CACHE:
            lat, lon = _GEOCODE_CACHE[location]
        else:
            try:
                import urllib.request, urllib.parse
                url = (
                    "https://nominatim.openstreetmap.org/search?"
                    + urllib.parse.urlencode({"q": location, "format": "json", "limit": 1})
                )
                req = urllib.request.Request(
                    url, headers={"User-Agent": "AstroNexusAI/1.0"}
                )
                with urllib.request.urlopen(req, timeout=5) as resp:
                    data = json.loads(resp.read())
                if data:
                    lat = float(data[0]["lat"])
                    lon = float(data[0]["lon"])
                    _GEOCODE_CACHE[location] = (lat, lon)
            except Exception:
                pass   # use defaults

    if name == "open_meteo":
        forecast_days = min(timespan or 7, 16)
        params = {
            "latitude":     lat,
            "longitude":    lon,
            "daily":        "temperature_2m_max,temperature_2m_min,precipitation_sum,windspeed_10m_max",
            "timezone":     "auto",
            "forecast_days":forecast_days,
        }

    elif name == "nasa_power":
        params = {
            "parameters": "T2M,PRECTOTCORR,WS2M,ALLSKY_SFC_SW_DWN",
            "community":  "RE",
            "longitude":  lon,
            "latitude":   lat,
            "format":     "JSON",
            "start":      "20240101",
            "end":        "20241231",
        }

    elif name == "nasa_firms":
        params = {
            "source":    "VIIRS_SNPP_NRT",
            "area":      f"{lon-5:.1f},{lat-5:.1f},{lon+5:.1f},{lat+5:.1f}",
            "day_range": min(timespan or 7, 10),
        }

    elif name == "semantic_scholar":
        keywords = query_info.get("keywords", [])
        params = {
            "query":  " ".join(keywords[:4]),
            "limit":  5,
            "fields": "title,authors,year,abstract,citationCount",
        }

    elif name == "arxiv":
        keywords = query_info.get("keywords", [])
        params = {
            "search_query": f"all:{' '.join(keywords[:3])}",
            "max_results":  3,
            "sortBy":       "submittedDate",
            "sortOrder":    "descending",
        }

    elif name == "openstreetmap_nominatim":
        params = {
            "q":      location or " ".join(query_info.get("keywords", [])[:3]),
            "format": "json",
            "limit":  3,
        }

    elif name == "open_aq":
        params = {
            "city":       location or "",
            "parameter":  "pm25,no2,o3",
            "limit":      5,
            "date_from":  "2024-01-01",
        }

    elif name == "noaa_climate":
        params = {
            "datasetid": "GHCND",
            "startdate": f"{query_info.get('year', '2023')}-01-01",
            "enddate":   f"{query_info.get('year', '2023')}-12-31",
            "limit":     10,
        }

    return params


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ══════════════════════════════════════════════════════════════════════════════

def select_apis(query: str) -> list[APICandidate]:
    """
    Intelligence layer: decide which APIs to call for this query.

    Args:
        query: User query string

    Returns:
        Ranked list of APICandidate (may be empty if no API is relevant).
        Maximum MAX_APIS_PER_QUERY candidates returned.
        Only candidates with score >= RELEVANCE_THRESHOLD included.
    """
    # Cache check
    query_hash = hashlib.md5(query.lower().encode()).hexdigest()[:12]
    now = time.time()
    if query_hash in _cache:
        cached_time, cached_result = _cache[query_hash]
        if now - cached_time < CACHE_TTL_SECONDS:
            logger.info(f"[APIRouter] Cache hit for query hash {query_hash}")
            return cached_result

    # Extract structured info from query
    query_info = extract_query_keywords(query)
    keywords   = query_info["keywords"]

    logger.info(f"[APIRouter] Keywords: {keywords[:6]}  location={query_info['location']}")

    if not keywords:
        return []

    # Neo4j lookup
    api_records = _lookup_apis_from_neo4j(keywords)
    if not api_records:
        logger.info("[APIRouter] No matching APIs in registry")
        return []

    # Score and filter
    candidates = []
    for api in api_records:
        score = _score_api(api, keywords, query_info)
        if score < RELEVANCE_THRESHOLD:
            logger.info(
                f"[APIRouter] {api['name']:<25} score={score:.3f} "
                f"→ below threshold, skipping"
            )
            continue

        params = _extract_params(api, query_info)
        candidates.append(APICandidate(
            name=             api["name"],
            endpoint=         api.get("endpoint", ""),
            method=           api.get("method", "GET"),
            description=      api.get("description", ""),
            score=            score,
            matched_keywords= api.get("matched_keywords", []),
            extracted_params= params,
            requires_key=     api.get("requires_key", False),
        ))
        logger.info(
            f"[APIRouter] {api['name']:<25} score={score:.3f} "
            f"matched={api.get('matched_keywords', [])} → SELECTED"
        )

    # Sort by score, cap at MAX_APIS_PER_QUERY
    candidates.sort(key=lambda c: c.score, reverse=True)
    candidates = candidates[:MAX_APIS_PER_QUERY]

    # Cache result
    _cache[query_hash] = (now, candidates)

    logger.info(f"[APIRouter] Selected {len(candidates)} API(s) to call")
    return candidates


def clear_cache() -> None:
    """Clear the API result cache."""
    _cache.clear()
    logger.info("[APIRouter] Cache cleared")