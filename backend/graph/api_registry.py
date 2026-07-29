"""
AstroNexus AI — External API Registry

Seeds Neo4j with ExternalAPI nodes linked to Domain and Keyword nodes.
Adding a new API = add one entry to EXTERNAL_APIS dict. Zero code changes.

Run once (safe to re-run — uses MERGE):
    python -m backend.graph.api_registry

Node type: ExternalAPI
    name          unique identifier
    endpoint      base URL or function name
    description   what this API provides
    method        GET | POST | INTERNAL
    params        JSON string of required params
    rate_limit    requests per minute
    requires_key  True if API key needed
    enabled       True/False (disable without deleting)

Relationships:
    (Domain)-[:HAS_API]->(ExternalAPI)
    (Keyword)-[:TRIGGERS]->(ExternalAPI)
"""
from __future__ import annotations

import json
import logging
import sys

logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════════════
# API REGISTRY
# Add new APIs here. Zero code changes needed elsewhere.
# ══════════════════════════════════════════════════════════════════════════════

EXTERNAL_APIS: dict[str, dict] = {

    # ── Climate / Weather ──────────────────────────────────────────────────────
    "open_meteo": {
        "description": "Free weather forecast and historical climate data. No API key required.",
        "endpoint":    "https://api.open-meteo.com/v1/forecast",
        "method":      "GET",
        "params":      {"latitude": "float", "longitude": "float",
                        "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum",
                        "forecast_days": "int (1-16)"},
        "rate_limit":  10000,
        "requires_key":False,
        "enabled":     True,
        "domains":     ["climate", "remote_sensing"],
        "keywords":    ["climate", "weather", "temperature", "precipitation",
                        "forecast", "rainfall", "humidity", "wind",
                        "climate change", "global warming"],
    },

    "noaa_climate": {
        "description": "NOAA Climate Data Online — historical climate observations.",
        "endpoint":    "https://www.ncdc.noaa.gov/cdo-web/api/v2/data",
        "method":      "GET",
        "params":      {"datasetid": "GHCND", "startdate": "str", "enddate": "str",
                        "stationid": "str", "limit": "int"},
        "rate_limit":  1000,
        "requires_key":True,
        "enabled":     True,
        "domains":     ["climate"],
        "keywords":    ["climate data", "historical climate", "temperature anomaly",
                        "precipitation record", "climate record", "weather station"],
    },

    # ── Satellite / Earth Observation ─────────────────────────────────────────
    "nasa_power": {
        "description": "NASA POWER — solar and meteorological data from satellite observations.",
        "endpoint":    "https://power.larc.nasa.gov/api/temporal/daily/point",
        "method":      "GET",
        "params":      {"parameters": "T2M,PRECTOTCORR,WS2M",
                        "community": "RE", "longitude": "float",
                        "latitude": "float", "format": "JSON"},
        "rate_limit":  30,
        "requires_key":False,
        "enabled":     True,
        "domains":     ["climate", "remote_sensing"],
        "keywords":    ["nasa", "solar radiation", "meteorological", "satellite data",
                        "earth observation", "remote sensing data"],
    },

    "nasa_firms": {
        "description": "NASA FIRMS — active wildfire and fire detection from MODIS/VIIRS.",
        "endpoint":    "https://firms.modaps.eosdis.nasa.gov/api/area/csv",
        "method":      "GET",
        "params":      {"source": "VIIRS_SNPP_NRT", "area": "str",
                        "day_range": "int (1-10)", "date": "YYYY-MM-DD"},
        "rate_limit":  100,
        "requires_key":True,
        "enabled":     True,
        "domains":     ["remote_sensing", "climate"],
        "keywords":    ["wildfire", "fire detection", "fire", "burnt area",
                        "forest fire", "modis fire", "viirs", "active fire"],
    },

    "usgs_landsat": {
        "description": "USGS EarthExplorer — Landsat satellite imagery search and download.",
        "endpoint":    "https://m2m.cr.usgs.gov/api/api/json/stable/scene-search",
        "method":      "POST",
        "params":      {"datasetName": "landsat_ot_c2_l2", "maxResults": "int",
                        "startDate": "str", "endDate": "str"},
        "rate_limit":  100,
        "requires_key":True,
        "enabled":     True,
        "domains":     ["remote_sensing"],
        "keywords":    ["landsat", "satellite imagery", "land surface", "land cover",
                        "earth observation", "multispectral"],
    },

    # ── Scientific Literature ──────────────────────────────────────────────────
    "semantic_scholar": {
        "description": "Semantic Scholar — search 200M+ scientific papers by keyword or topic.",
        "endpoint":    "https://api.semanticscholar.org/graph/v1/paper/search",
        "method":      "GET",
        "params":      {"query": "str", "limit": "int (1-100)",
                        "fields": "title,authors,year,abstract,citationCount"},
        "rate_limit":  100,
        "requires_key":False,
        "enabled":     True,
        "domains":     ["nlp", "computer_vision", "machine_learning",
                        "remote_sensing", "astronomy", "medicine"],
        "keywords":    ["paper", "research", "publication", "study", "literature",
                        "recent paper", "latest research", "who published",
                        "citation", "survey"],
    },

    "arxiv": {
        "description": "arXiv — search preprints in physics, math, CS, and related fields.",
        "endpoint":    "http://export.arxiv.org/api/query",
        "method":      "GET",
        "params":      {"search_query": "str", "max_results": "int",
                        "sortBy": "submittedDate", "sortOrder": "descending"},
        "rate_limit":  300,
        "requires_key":False,
        "enabled":     True,
        "domains":     ["nlp", "computer_vision", "machine_learning", "remote_sensing"],
        "keywords":    ["arxiv", "preprint", "recent model", "new paper",
                        "latest model", "state of the art", "2024", "2025"],
    },

    # ── Geospatial ────────────────────────────────────────────────────────────
    "openstreetmap_nominatim": {
        "description": "OpenStreetMap Nominatim — geocoding and reverse geocoding.",
        "endpoint":    "https://nominatim.openstreetmap.org/search",
        "method":      "GET",
        "params":      {"q": "str", "format": "json", "limit": "int"},
        "rate_limit":  60,
        "requires_key":False,
        "enabled":     True,
        "domains":     ["remote_sensing"],
        "keywords":    ["location", "coordinates", "latitude", "longitude",
                        "where is", "city", "country", "region", "place"],
    },

    # ── Air Quality ────────────────────────────────────────────────────────────
    "open_aq": {
        "description": "OpenAQ — real-time and historical air quality measurements worldwide.",
        "endpoint":    "https://api.openaq.org/v2/measurements",
        "method":      "GET",
        "params":      {"city": "str", "parameter": "pm25,no2,o3,co",
                        "limit": "int", "date_from": "str"},
        "rate_limit":  200,
        "requires_key":False,
        "enabled":     True,
        "domains":     ["climate", "remote_sensing"],
        "keywords":    ["air quality", "pollution", "pm2.5", "particulate matter",
                        "aqi", "air pollution", "smog", "no2", "ozone"],
    },
}


# ══════════════════════════════════════════════════════════════════════════════
# LOADER
# ══════════════════════════════════════════════════════════════════════════════

class APIRegistryLoader:

    def __init__(self):
        from backend.graph.neo4j_client import _get_driver
        self._driver = _get_driver()

    def _create_constraints(self) -> None:
        with self._driver.session() as s:
            s.run(
                "CREATE CONSTRAINT api_name_unique IF NOT EXISTS "
                "FOR (a:ExternalAPI) REQUIRE a.name IS UNIQUE"
            )

    def load_all(self) -> dict[str, int]:
        self._create_constraints()
        counts = {"apis": 0, "domain_links": 0, "keyword_links": 0}

        with self._driver.session() as s:
            for api_name, api_data in EXTERNAL_APIS.items():
                if not api_data.get("enabled", True):
                    continue

                # Create ExternalAPI node
                s.run(
                    """
                    MERGE (a:ExternalAPI {name: $name})
                    SET a.description  = $desc,
                        a.endpoint     = $endpoint,
                        a.method       = $method,
                        a.params       = $params,
                        a.rate_limit   = $rate_limit,
                        a.requires_key = $requires_key,
                        a.enabled      = $enabled
                    """,
                    name=        api_name,
                    desc=        api_data["description"],
                    endpoint=    api_data["endpoint"],
                    method=      api_data["method"],
                    params=      json.dumps(api_data["params"]),
                    rate_limit=  api_data["rate_limit"],
                    requires_key=api_data.get("requires_key", False),
                    enabled=     api_data.get("enabled", True),
                )
                counts["apis"] += 1

                # Link to Domain nodes
                for domain_key in api_data.get("domains", []):
                    s.run(
                        """
                        MERGE (d:Domain {key: $dkey})
                        SET d.name = $dname
                        WITH d
                        MATCH (a:ExternalAPI {name: $aname})
                        MERGE (d)-[:HAS_API]->(a)
                        """,
                        dkey=  domain_key,
                        dname= domain_key.replace("_", " ").title(),
                        aname= api_name,
                    )
                    counts["domain_links"] += 1

                # Link to Keyword nodes
                for kw in api_data.get("keywords", []):
                    s.run(
                        """
                        MERGE (k:Keyword {name: $kw})
                        WITH k
                        MATCH (a:ExternalAPI {name: $aname})
                        MERGE (k)-[:TRIGGERS]->(a)
                        """,
                        kw=    kw,
                        aname= api_name,
                    )
                    counts["keyword_links"] += 1

                logger.info(
                    f"[APIRegistry] {api_name:<25} "
                    f"domains={len(api_data['domains'])} "
                    f"keywords={len(api_data['keywords'])}"
                )

        return counts

    def verify(self) -> None:
        with self._driver.session() as s:
            apis = s.run(
                "MATCH (a:ExternalAPI) RETURN a.name AS name, "
                "a.enabled AS enabled ORDER BY name"
            ).data()
            links = s.run(
                "MATCH (k:Keyword)-[:TRIGGERS]->(a:ExternalAPI) "
                "RETURN count(*) AS n"
            ).single()

        print(f"\n  ExternalAPI nodes  : {len(apis)}")
        for a in apis:
            print(f"    {'✓' if a['enabled'] else '–'}  {a['name']}")
        print(f"  Keyword→API links  : {links['n']}")


def run() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    print("\n" + "="*55)
    print("ASTRONEXUS — EXTERNAL API REGISTRY LOADER")
    print("="*55)
    try:
        loader = APIRegistryLoader()
        counts = loader.load_all()
        print(f"\n  APIs loaded        : {counts['apis']}")
        print(f"  Domain links       : {counts['domain_links']}")
        print(f"  Keyword links      : {counts['keyword_links']}")
        loader.verify()
        print("\n✓ API registry loaded.")
        print("  Run: python -m backend.graph.api_registry")
    except Exception as e:
        print(f"\n✗ Failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    run()