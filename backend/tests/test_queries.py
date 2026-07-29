"""
AstroNexus AI — 35 Test Queries
Run: python -m backend.tests.test_queries
"""
from __future__ import annotations
import json, logging, sys, time
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logging.basicConfig(level=logging.WARNING)

TESTS = [
    # ── API Intelligence Layer (should fetch real data) ─────────────────────
    {"id":"A01","q":"What is the climate in next 5 years?",                    "expect":"general",   "api":["open_meteo","nasa_power"]},
    {"id":"A02","q":"Is there any active wildfire in California right now?",   "expect":"general",   "api":["nasa_firms"]},
    {"id":"A03","q":"What is the air quality in Delhi today?",                 "expect":"general",   "api":["open_aq"]},
    {"id":"A04","q":"Show me recent papers on flood detection using SAR",       "expect":"general",   "api":["semantic_scholar","arxiv"]},
    {"id":"A05","q":"What is the rainfall forecast for Mumbai next 7 days?",   "expect":"general",   "api":["open_meteo"]},
    {"id":"A06","q":"Find latest arxiv papers on Vision Transformer 2024",     "expect":"general",   "api":["arxiv"]},
    {"id":"A07","q":"NASA satellite data for temperature in Chennai",          "expect":"general",   "api":["nasa_power"]},
    {"id":"A08","q":"What is the weather like in London this week?",           "expect":"general",   "api":["open_meteo"]},

    # ── Knowledge Graph queries ──────────────────────────────────────────────
    {"id":"G01","q":"Who authored Attention Is All You Need?",                 "expect":"graph",     "api":[]},
    {"id":"G02","q":"Which satellites are part of the Copernicus mission?",    "expect":"graph",     "api":[]},
    {"id":"G03","q":"What sensor does Sentinel-1 carry?",                      "expect":"graph",     "api":[]},
    {"id":"G04","q":"Which algorithms are used for semantic segmentation?",    "expect":"graph",     "api":[]},
    {"id":"G05","q":"What papers in the graph use DINOv2?",                    "expect":"graph",     "api":[]},
    {"id":"G06","q":"Which missions does NASA operate?",                       "expect":"graph",     "api":[]},
    {"id":"G07","q":"List all keywords tagged under remote sensing domain",    "expect":"graph",     "api":[]},

    # ── Research / RAG queries ───────────────────────────────────────────────
    {"id":"R01","q":"What is the main contribution of this paper?",            "expect":"research",  "api":[]},
    {"id":"R02","q":"What dataset was used for training?",                     "expect":"research",  "api":[]},
    {"id":"R03","q":"What loss function was used in this model?",              "expect":"research",  "api":[]},
    {"id":"R04","q":"What are the limitations mentioned by the authors?",      "expect":"research",  "api":[]},
    {"id":"R05","q":"What baseline models were compared in the experiments?",  "expect":"research",  "api":[]},
    {"id":"R06","q":"Summarize the proposed methodology in simple terms",      "expect":"research",  "api":[]},
    {"id":"R07","q":"What evaluation metrics were reported in the results?",   "expect":"research",  "api":[]},

    # ── General knowledge (no API triggered) ────────────────────────────────
    {"id":"K01","q":"What is reinforcement learning?",                         "expect":"general",   "api":[]},
    {"id":"K02","q":"Explain the difference between CNN and Transformer",      "expect":"general",   "api":[]},
    {"id":"K03","q":"What is overfitting in machine learning?",                "expect":"general",   "api":[]},
    {"id":"K04","q":"Who is the CEO of Google?",                               "expect":"general",   "api":[]},
    {"id":"K05","q":"What is the capital of France?",                          "expect":"general",   "api":[]},

    # ── Edge / Adversarial ───────────────────────────────────────────────────
    {"id":"E01","q":"What does Section 99.9 say about the results?",          "expect":"research",  "api":[]},
    {"id":"E02","q":"Tell me everything about this paper",                     "expect":"research",  "api":[]},
    {"id":"E03","q":"What is the model?",                                      "expect":"general",   "api":[]},
    {"id":"E04","q":"???",                                                     "expect":"general",   "api":[]},
    {"id":"E05","q":"wht is attntion mechnism in transformrs",                "expect":"general",   "api":[]},

    # ── Cross-domain (interesting routing challenges) ─────────────────────
    {"id":"X01","q":"What papers use Sentinel-2 for flood detection?",         "expect":"graph",     "api":[]},
    {"id":"X02","q":"Latest research on wildfire detection using satellite",   "expect":"general",   "api":["nasa_firms","arxiv"]},
    {"id":"X03","q":"Climate change impact on crop monitoring papers",         "expect":"general",   "api":["open_meteo","semantic_scholar"]},
]

GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
RESET  = "\033[0m"


def run():
    from backend.agents.router import classify_query
    from backend.agents.api_router import select_apis

    print(f"\n{'═'*70}")
    print("  ASTRONEXUS AI — 35 TEST QUERIES")
    print(f"{'═'*70}")

    results = []
    for t in TESTS:
        qid    = t["id"]
        query  = t["q"]
        expect = t["expect"]
        exp_apis = t.get("api", [])

        # Route
        state      = {"metadata": {"paper_loaded": qid.startswith("R")}}
        got_route  = classify_query(query, state)
        route_ok   = got_route == expect

        # API selection
        try:
            candidates = select_apis(query)
            got_apis   = [c.name for c in candidates]
        except Exception as e:
            got_apis = [f"ERROR:{e}"]

        api_ok = True   # API check is informational — not pass/fail

        status = f"{GREEN}✓{RESET}" if route_ok else f"{RED}✗{RESET}"
        api_str = f"{CYAN}{got_apis}{RESET}" if got_apis else "—"

        print(
            f"\n  {status} [{qid}] {query[:55]}"
            + ("..." if len(query) > 55 else "")
        )
        print(
            f"       Route  : {got_route:<12} "
            f"{'✓' if route_ok else f'expected {expect}'}"
        )
        if got_apis:
            print(f"       APIs   : {got_apis}")
        if exp_apis:
            print(f"       Expect : {exp_apis}")

        results.append({
            "id":       qid,
            "query":    query,
            "route_ok": route_ok,
            "route":    got_route,
            "apis":     got_apis,
        })

    passed = sum(r["route_ok"] for r in results)
    total  = len(results)

    print(f"\n{'═'*70}")
    print(f"  ROUTING: {passed}/{total} correct  ({100*passed//total}%)")

    api_triggered = [r for r in results if r["apis"]]
    print(f"  APIS triggered on {len(api_triggered)}/{total} queries")
    print(f"{'═'*70}\n")

    Path("output").mkdir(exist_ok=True)
    with open("output/test_queries_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("  Saved → output/test_queries_results.json\n")


if __name__ == "__main__":
    run()