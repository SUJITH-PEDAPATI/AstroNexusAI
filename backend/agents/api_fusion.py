"""
AstroNexus AI — API Fusion Layer

Calls selected external APIs in parallel, formats results,
and merges them into the Knowledge Fusion context before LLM generation.

Flow:
    APIRouter.select_apis(query)   → list of APICandidate
    APIFusionLayer.call(candidates) → list of APIResult
    APIFusionLayer.format(results)  → structured context string
    → merged into KnowledgeFusionLayer prompt

Supports:
    open_meteo            (weather/climate forecast)
    nasa_power            (NASA satellite meteorological data)
    nasa_firms            (wildfire detection)
    semantic_scholar      (scientific paper search)
    arxiv                 (preprint search)
    openstreetmap_nominatim (geocoding)
    open_aq               (air quality)
    noaa_climate          (historical climate)
"""
from __future__ import annotations

import json
import logging
import time
import urllib.parse
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor, TimeoutError, as_completed
from dataclasses import dataclass, field
from typing import Optional

from backend.agents.api_router import APICandidate

logger = logging.getLogger(__name__)

API_TIMEOUT     = 10    # seconds per API call
MAX_WORKERS     = 4     # parallel API calls


@dataclass
class APIResult:
    """Result from one external API call."""
    api_name:   str
    success:    bool
    data:       dict | list | None = None
    summary:    str                = ""
    error:      str                = ""
    latency_ms: float              = 0.0


# ══════════════════════════════════════════════════════════════════════════════
# INDIVIDUAL API CALLERS
# ══════════════════════════════════════════════════════════════════════════════

def _get(url: str, params: dict, headers: dict = None) -> dict | list:
    """HTTP GET with timeout."""
    full_url = url + "?" + urllib.parse.urlencode(params)
    req      = urllib.request.Request(
        full_url,
        headers={**(headers or {}), "User-Agent": "AstroNexusAI/1.0"},
    )
    with urllib.request.urlopen(req, timeout=API_TIMEOUT) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _call_open_meteo(params: dict) -> tuple[dict, str]:
    data = _get("https://api.open-meteo.com/v1/forecast", params)

    daily   = data.get("daily", {})
    dates   = daily.get("time", [])
    t_max   = daily.get("temperature_2m_max", [])
    t_min   = daily.get("temperature_2m_min", [])
    precip  = daily.get("precipitation_sum", [])

    lines = [f"Weather forecast ({len(dates)} days):"]
    for i, date in enumerate(dates[:7]):
        tmax = f"{t_max[i]:.1f}°C" if i < len(t_max) else "?"
        tmin = f"{t_min[i]:.1f}°C" if i < len(t_min) else "?"
        prec = f"{precip[i]:.1f}mm" if i < len(precip) else "?"
        lines.append(f"  {date}: High {tmax}, Low {tmin}, Precip {prec}")

    return data, "\n".join(lines)


def _call_nasa_power(params: dict) -> tuple[dict, str]:
    data     = _get("https://power.larc.nasa.gov/api/temporal/daily/point", params)
    features = data.get("features", [{}])
    props    = features[0].get("properties", {}) if features else {}
    pars     = props.get("parameter", {})

    lines = ["NASA POWER satellite data:"]
    if "T2M" in pars:
        vals = list(pars["T2M"].values())
        avg  = sum(vals) / len(vals) if vals else 0
        lines.append(f"  Avg temperature (2m): {avg:.1f}°C")
    if "PRECTOTCORR" in pars:
        vals  = list(pars["PRECTOTCORR"].values())
        total = sum(vals)
        lines.append(f"  Total precipitation: {total:.1f}mm")

    return data, "\n".join(lines)


def _call_nasa_firms(params: dict) -> tuple[dict | str, str]:
    import os
    api_key = os.environ.get("NASA_FIRMS_API_KEY", "")
    url     = f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{api_key}/VIIRS_SNPP_NRT"
    area    = params.get("area", "")
    days    = params.get("day_range", 7)

    req = urllib.request.Request(
        f"{url}/{area}/{days}",
        headers={"User-Agent": "AstroNexusAI/1.0"},
    )
    with urllib.request.urlopen(req, timeout=API_TIMEOUT) as resp:
        csv_data = resp.read().decode("utf-8")

    lines    = csv_data.strip().split("\n")
    n_fires  = max(len(lines) - 1, 0)
    summary  = f"NASA FIRMS: {n_fires} active fire detections in the specified region."
    return {"raw_csv_lines": n_fires}, summary


def _call_semantic_scholar(params: dict) -> tuple[dict, str]:
    data    = _get("https://api.semanticscholar.org/graph/v1/paper/search", params)
    papers  = data.get("data", [])

    lines = [f"Semantic Scholar: {len(papers)} related papers found:"]
    for p in papers[:4]:
        authors = [a.get("name", "") for a in p.get("authors", [])[:2]]
        lines.append(
            f"  [{p.get('year', '?')}] {p.get('title', 'Unknown')} "
            f"— {', '.join(authors)} "
            f"(citations: {p.get('citationCount', 0)})"
        )
        if p.get("abstract"):
            lines.append(f"      {p['abstract'][:120]}...")

    return data, "\n".join(lines)


def _call_arxiv(params: dict) -> tuple[dict, str]:
    import xml.etree.ElementTree as ET

    url      = "http://export.arxiv.org/api/query?" + urllib.parse.urlencode(params)
    req      = urllib.request.Request(url, headers={"User-Agent": "AstroNexusAI/1.0"})
    with urllib.request.urlopen(req, timeout=API_TIMEOUT) as resp:
        xml_data = resp.read()

    root    = ET.fromstring(xml_data)
    ns      = {"atom": "http://www.w3.org/2005/Atom"}
    entries = root.findall("atom:entry", ns)

    lines = [f"arXiv: {len(entries)} recent preprints:"]
    for e in entries[:3]:
        title   = e.findtext("atom:title", namespaces=ns, default="").strip()
        summary = e.findtext("atom:summary", namespaces=ns, default="").strip()[:100]
        pub     = e.findtext("atom:published", namespaces=ns, default="")[:10]
        lines.append(f"  [{pub}] {title}")
        lines.append(f"      {summary}...")

    return {"count": len(entries)}, "\n".join(lines)


def _call_open_aq(params: dict) -> tuple[dict, str]:
    data     = _get("https://api.openaq.org/v2/measurements", params)
    results  = data.get("results", [])

    lines = [f"OpenAQ air quality ({len(results)} readings):"]
    by_param: dict[str, list] = {}
    for r in results:
        param = r.get("parameter", "?")
        val   = r.get("value", 0)
        by_param.setdefault(param, []).append(val)

    for param, vals in by_param.items():
        avg = sum(vals) / len(vals) if vals else 0
        lines.append(f"  {param}: avg={avg:.2f} (n={len(vals)})")

    return data, "\n".join(lines)


def _call_nominatim(params: dict) -> tuple[dict, str]:
    data  = _get("https://nominatim.openstreetmap.org/search", params)
    lines = [f"Location data ({len(data)} results):"]
    for r in data[:2]:
        lines.append(
            f"  {r.get('display_name', '?')} "
            f"[{r.get('lat', '?')}, {r.get('lon', '?')}]"
        )
    return data, "\n".join(lines)


# ── Dispatcher ─────────────────────────────────────────────────────────────────
_CALLERS = {
    "open_meteo":               _call_open_meteo,
    "nasa_power":               _call_nasa_power,
    "nasa_firms":               _call_nasa_firms,
    "semantic_scholar":         _call_semantic_scholar,
    "arxiv":                    _call_arxiv,
    "open_aq":                  _call_open_aq,
    "openstreetmap_nominatim":  _call_nominatim,
}


def _call_one(candidate: APICandidate) -> APIResult:
    """Call a single API and return a structured result."""
    t0 = time.perf_counter()
    try:
        caller = _CALLERS.get(candidate.name)
        if caller is None:
            return APIResult(
                api_name=candidate.name,
                success=False,
                error=f"No caller implemented for {candidate.name}",
            )

        data, summary = caller(candidate.extracted_params)
        latency = (time.perf_counter() - t0) * 1000

        logger.info(
            f"[APIFusion] {candidate.name:<25} "
            f"OK  {latency:.0f}ms"
        )
        return APIResult(
            api_name=   candidate.name,
            success=    True,
            data=       data,
            summary=    summary,
            latency_ms= latency,
        )

    except urllib.error.HTTPError as e:
        logger.warning(f"[APIFusion] {candidate.name} HTTP {e.code}: {e.reason}")
        return APIResult(api_name=candidate.name, success=False,
                         error=f"HTTP {e.code}: {e.reason}")
    except Exception as e:
        logger.warning(f"[APIFusion] {candidate.name} failed: {e}")
        return APIResult(api_name=candidate.name, success=False, error=str(e))


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ══════════════════════════════════════════════════════════════════════════════

def call_apis(candidates: list[APICandidate]) -> list[APIResult]:
    """
    Call all selected APIs in parallel.

    Args:
        candidates: Output of APIRouter.select_apis()

    Returns:
        List of APIResult (one per candidate, success or failure)
    """
    if not candidates:
        return []

    logger.info(f"[APIFusion] Calling {len(candidates)} API(s) in parallel...")

    results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(_call_one, c): c for c in candidates}
        for future in as_completed(futures, timeout=API_TIMEOUT + 5):
            try:
                results.append(future.result())
            except Exception as e:
                cand = futures[future]
                results.append(
                    APIResult(api_name=cand.name, success=False, error=str(e))
                )

    return results


def format_for_prompt(results: list[APIResult]) -> str:
    """
    Format API results as a structured section for the LLM prompt.
    Only includes successful results.
    """
    successful = [r for r in results if r.success and r.summary]
    if not successful:
        return ""

    lines = ["=== REAL-TIME EXTERNAL DATA ==="]
    for result in successful:
        lines.append(f"\n[{result.api_name.replace('_', ' ').upper()}]")
        lines.append(result.summary)
        lines.append(f"(Source: {result.api_name}, latency: {result.latency_ms:.0f}ms)")

    return "\n".join(lines)