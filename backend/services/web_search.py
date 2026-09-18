"""
AstroNexus AI — Web Search Service
====================================
Modular, provider-swappable web search layer.

Provider priority (auto-detected from environment):
    1. Serper (SERPER_API_KEY)        — highest quality, astronomy-aware
    2. Brave  (BRAVE_API_KEY)         — privacy-focused, good for science
    3. SerpAPI (SERPAPI_KEY)          — reliable, broad coverage
    4. DuckDuckGo (no key required)   — free fallback, always available

Set ONE env var to activate a paid provider.
If none are set, DuckDuckGo is used automatically.

Usage:
    from backend.services.web_search import web_search_service
    results = web_search_service.search("JWST latest discoveries")
    # → [WebResult(title, url, snippet, source, score), ...]
"""
from __future__ import annotations

import json
import logging
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Optional

logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────

SEARCH_TIMEOUT     = 8      # seconds per request
MAX_RESULTS        = 6      # results returned to the pipeline
CACHE_TTL          = 300    # 5 min cache for identical queries
MAX_RETRIES        = 2

# Trusted domains get a score boost during ranking
TRUSTED_DOMAINS: dict[str, float] = {
    "nasa.gov": 1.0, "esa.int": 1.0, "arxiv.org": 0.95,
    "nature.com": 0.9, "science.org": 0.9,
    "aas.org": 0.9, "iopscience.iop.org": 0.88,
    "nist.gov": 0.85, "noaa.gov": 0.85,
    "space.com": 0.75, "skyandtelescope.org": 0.75,
    "wikipedia.org": 0.65, "bbc.com": 0.7, "reuters.com": 0.7,
}

# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class WebResult:
    title:   str
    url:     str
    snippet: str
    source:  str       # human-readable domain, e.g. "nasa.gov"
    score:   float = 0.5      # relevance/trust score 0–1
    date:    Optional[str] = None   # publication date if available


# ── Simple in-memory cache ────────────────────────────────────────────────────

_cache: dict[str, tuple[float, list[WebResult]]] = {}


def _cached(key: str) -> list[WebResult] | None:
    if key in _cache:
        ts, results = _cache[key]
        if time.time() - ts < CACHE_TTL:
            logger.debug(f"[WebSearch] Cache hit: {key[:40]}")
            return results
    return None


def _cache_set(key: str, results: list[WebResult]) -> None:
    _cache[key] = (time.time(), results)


# ── Domain trust scorer ───────────────────────────────────────────────────────

def _domain_score(url: str) -> float:
    for domain, score in TRUSTED_DOMAINS.items():
        if domain in url:
            return score
    return 0.5


def _extract_source(url: str) -> str:
    try:
        host = urllib.parse.urlparse(url).netloc
        return host.replace("www.", "")
    except Exception:
        return url[:40]


# ── Provider: DuckDuckGo (no API key — always works) ─────────────────────────

def _ddg_search(query: str, n: int) -> list[WebResult]:
    """
    DuckDuckGo HTML search — no API key required.

    Uses the DDG HTML endpoint which returns real web results, unlike the
    Instant Answer API (api.duckduckgo.com/?format=json) which only
    returns results for well-known entities and is empty for most queries.
    """
    import html
    import re as _re

    # DDG HTML search endpoint
    params = {"q": query, "kl": "us-en"}
    url = "https://html.duckduckgo.com/html/?" + urllib.parse.urlencode(params)

    try:
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0.0.0 Safari/537.36"
                ),
                "Accept": "text/html,application/xhtml+xml",
                "Accept-Language": "en-US,en;q=0.9",
            },
        )
        with urllib.request.urlopen(req, timeout=SEARCH_TIMEOUT) as resp:
            raw_html = resp.read().decode("utf-8", errors="replace")
    except Exception as e:
        logger.warning(f"[WebSearch/DDG] Request failed: {e}")
        return []

    results: list[WebResult] = []

    # Parse result blocks: <div class="result__body"> ... </div>
    # Extract title, URL and snippet from each block
    blocks = _re.findall(
        r'<a[^>]+class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>.*?'
        r'<a[^>]+class="result__snippet"[^>]*>(.*?)</a>',
        raw_html, _re.DOTALL
    )

    for href, title_html, snippet_html in blocks[:n]:
        # DDG wraps URLs in a redirect — extract the real URL
        real_url = href
        uddg_match = _re.search(r"uddg=([^&]+)", href)
        if uddg_match:
            try:
                real_url = urllib.parse.unquote(uddg_match.group(1))
            except Exception:
                real_url = href

        title   = html.unescape(_re.sub(r"<[^>]+>", "", title_html)).strip()
        snippet = html.unescape(_re.sub(r"<[^>]+>", "", snippet_html)).strip()

        if not real_url.startswith("http") or not title:
            continue

        src = _extract_source(real_url)
        results.append(WebResult(
            title=   title[:120],
            url=     real_url,
            snippet= snippet[:400],
            source=  src,
            score=   _domain_score(real_url),
        ))

    logger.info(f"[WebSearch/DDG] {len(results)} results for '{query[:50]}'")

    # Fallback to Instant Answer API for entity queries (returns 0 blocks)
    if not results:
        ia_params = {"q": query, "format": "json", "no_redirect": "1", "no_html": "1"}
        ia_url = "https://api.duckduckgo.com/?" + urllib.parse.urlencode(ia_params)
        try:
            ia_req = urllib.request.Request(
                ia_url, headers={"User-Agent": "AstroNexusAI/2.0"}
            )
            with urllib.request.urlopen(ia_req, timeout=SEARCH_TIMEOUT) as resp:
                data = json.loads(resp.read())
            if data.get("AbstractText") and data.get("AbstractURL"):
                src = _extract_source(data["AbstractURL"])
                results.append(WebResult(
                    title=   data.get("Heading", "Result"),
                    url=     data["AbstractURL"],
                    snippet= data["AbstractText"][:500],
                    source=  src,
                    score=   _domain_score(data["AbstractURL"]),
                ))
            for rel in data.get("RelatedTopics", [])[:n - len(results)]:
                if isinstance(rel, dict) and rel.get("FirstURL"):
                    results.append(WebResult(
                        title=   rel.get("Text", "")[:80],
                        url=     rel["FirstURL"],
                        snippet= rel.get("Text", "")[:300],
                        source=  _extract_source(rel["FirstURL"]),
                        score=   _domain_score(rel["FirstURL"]),
                    ))
        except Exception:
            pass

    return results[:n]

    # Related topics
    for topic in data.get("RelatedTopics", [])[:n]:
        if isinstance(topic, dict) and topic.get("FirstURL"):
            snippet = topic.get("Text", "")
            url2    = topic["FirstURL"]
            src     = _extract_source(url2)
            results.append(WebResult(
                title=   snippet[:60] + "…" if len(snippet) > 60 else snippet,
                url=     url2,
                snippet= snippet[:400],
                source=  src,
                score=   _domain_score(url2),
            ))

    return results[:n]


# ── Provider: Serper (serper.dev) ─────────────────────────────────────────────

def _serper_search(query: str, n: int, api_key: str) -> list[WebResult]:
    payload = json.dumps({"q": query, "num": n, "gl": "us", "hl": "en"}).encode()
    req = urllib.request.Request(
        "https://google.serper.dev/search",
        data=payload,
        headers={
            "X-API-KEY":    api_key,
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=SEARCH_TIMEOUT) as resp:
            data = json.loads(resp.read())
    except Exception as e:
        logger.warning(f"[WebSearch/Serper] Request failed: {e}")
        return []

    results: list[WebResult] = []
    for item in data.get("organic", [])[:n]:
        url = item.get("link", "")
        results.append(WebResult(
            title=   item.get("title",   ""),
            url=     url,
            snippet= item.get("snippet", ""),
            source=  _extract_source(url),
            score=   _domain_score(url),
            date=    item.get("date"),
        ))
    return results


# ── Provider: Brave Search ───────────────────────────────────────────────────

def _brave_search(query: str, n: int, api_key: str) -> list[WebResult]:
    params = {"q": query, "count": str(n), "search_lang": "en"}
    req = urllib.request.Request(
        "https://api.search.brave.com/res/v1/web/search?" + urllib.parse.urlencode(params),
        headers={
            "Accept":           "application/json",
            "Accept-Encoding":  "gzip",
            "X-Subscription-Token": api_key,
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=SEARCH_TIMEOUT) as resp:
            data = json.loads(resp.read())
    except Exception as e:
        logger.warning(f"[WebSearch/Brave] Request failed: {e}")
        return []

    results: list[WebResult] = []
    for item in data.get("web", {}).get("results", [])[:n]:
        url = item.get("url", "")
        results.append(WebResult(
            title=   item.get("title",       ""),
            url=     url,
            snippet= item.get("description", ""),
            source=  _extract_source(url),
            score=   _domain_score(url),
            date=    item.get("age"),
        ))
    return results


# ── Provider: SerpAPI ─────────────────────────────────────────────────────────

def _serpapi_search(query: str, n: int, api_key: str) -> list[WebResult]:
    params = {
        "engine":  "google",
        "q":       query,
        "num":     str(n),
        "api_key": api_key,
    }
    url = "https://serpapi.com/search?" + urllib.parse.urlencode(params)
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=SEARCH_TIMEOUT) as resp:
            data = json.loads(resp.read())
    except Exception as e:
        logger.warning(f"[WebSearch/SerpAPI] Request failed: {e}")
        return []

    results: list[WebResult] = []
    for item in data.get("organic_results", [])[:n]:
        url2 = item.get("link", "")
        results.append(WebResult(
            title=   item.get("title",   ""),
            url=     url2,
            snippet= item.get("snippet", ""),
            source=  _extract_source(url2),
            score=   _domain_score(url2),
            date=    item.get("date"),
        ))
    return results


# ── Main service class ────────────────────────────────────────────────────────

class WebSearchService:
    """
    Provider-agnostic web search.

    Auto-selects the best available provider.
    Falls back gracefully if the primary fails.
    Results are cached and ranked by domain trust.
    """

    def __init__(self) -> None:
        self._serper_key  = os.environ.get("SERPER_API_KEY",  "")
        self._brave_key   = os.environ.get("BRAVE_API_KEY",   "")
        self._serpapi_key = os.environ.get("SERPAPI_KEY",     "")

        if self._serper_key:
            self._provider = "serper"
        elif self._brave_key:
            self._provider = "brave"
        elif self._serpapi_key:
            self._provider = "serpapi"
        else:
            self._provider = "ddg"

        logger.info(f"[WebSearch] Provider: {self._provider}")

    @property
    def provider(self) -> str:
        return self._provider

    def search(
        self,
        query:    str,
        n:        int  = MAX_RESULTS,
        retries:  int  = MAX_RETRIES,
    ) -> list[WebResult]:
        """
        Search the web.

        Args:
            query:   Search query string
            n:       Maximum number of results
            retries: Retry attempts on failure

        Returns:
            List of WebResult sorted by score (highest first).
            Empty list on total failure — never raises.
        """
        cache_key = f"{self._provider}:{query}:{n}"
        cached = _cached(cache_key)
        if cached is not None:
            return cached

        results: list[WebResult] = []

        for attempt in range(retries + 1):
            try:
                if self._provider == "serper":
                    results = _serper_search(query, n, self._serper_key)
                elif self._provider == "brave":
                    results = _brave_search(query, n, self._brave_key)
                elif self._provider == "serpapi":
                    results = _serpapi_search(query, n, self._serpapi_key)
                else:
                    results = _ddg_search(query, n)

                if results:
                    break

            except Exception as e:
                logger.warning(f"[WebSearch] Attempt {attempt+1} failed: {e}")
                if attempt < retries:
                    time.sleep(0.5 * (attempt + 1))

        # Fallback to DDG if paid provider returned nothing
        if not results and self._provider != "ddg":
            logger.info("[WebSearch] Primary provider empty — trying DDG fallback")
            results = _ddg_search(query, n)

        # Sort by trust score
        results.sort(key=lambda r: r.score, reverse=True)

        _cache_set(cache_key, results)
        logger.info(f"[WebSearch] '{query[:50]}' → {len(results)} results via {self._provider}")
        return results

    def format_context(self, results: list[WebResult]) -> str:
        """
        Format results into a context string for the LLM.

        Each result is clearly labelled with its source URL
        so the LLM can produce accurate citations.
        """
        if not results:
            return ""
        lines = ["=== WEB SEARCH RESULTS ==="]
        for i, r in enumerate(results, 1):
            lines.append(
                f"\n[WEB {i}] {r.title}\n"
                f"Source: {r.source}  ({r.url})\n"
                f"{r.snippet}\n"
                + (f"Date: {r.date}\n" if r.date else "")
            )
        return "\n".join(lines)


# ── Module-level singleton ────────────────────────────────────────────────────

web_search_service = WebSearchService()