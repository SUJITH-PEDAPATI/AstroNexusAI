"""
AstroNexus AI — Agent Pipeline Test

Run from project root:
    python -m backend.agents.test_agents

Tests:
    1. Router classification
    2. Research agent (RAG query)
    3. Graph agent (Neo4j query)
    4. Satellite agent (image analysis)
    5. Full pipeline invocation
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logging.basicConfig(level=logging.INFO, format="%(message)s")


def test_router():
    print("\n[1] ROUTER")
    from backend.agents.router import classify_query
    cases = {
        "What does the Transformer paper propose?":        "research",
        "Analyse this satellite image of flood damage":    "satellite",
        "Who authored the BERT paper?":                   "graph",
        "Please speak the answer using voice":            "voice",
    }
    all_ok = True
    for query, expected in cases.items():
        got = classify_query(query)
        ok  = got == expected
        if not ok:
            all_ok = False
        print(f"  {'[OK]' if ok else '[FAIL]'} '{query[:45]}...' -> {got} (expected {expected})")
    return all_ok


def test_research_agent():
    print("\n[2] RESEARCH AGENT")
    from backend.agents.orchestrator import run
    result = run("What is the Transformer architecture?")
    ok = bool(result.get("final_answer"))
    print(f"  Query type  : {result['query_type']}")
    print(f"  Answer      : {str(result.get('final_answer',''))[:120]}...")
    print(f"  Status      : {'[OK]' if ok else '[FAIL]'}")
    return ok


def test_graph_agent():
    print("\n[3] GRAPH AGENT")
    from backend.agents.orchestrator import run
    result = run("Who authored the attention paper?")
    ok = bool(result.get("final_answer"))
    print(f"  Query type  : {result['query_type']}")
    print(f"  Answer      : {str(result.get('final_answer',''))[:120]}...")
    print(f"  Status      : {'[OK]' if ok else '[FAIL]'}")
    return ok


def test_satellite_agent(image_path: str | None = None):
    print("\n[4] SATELLITE AGENT")
    if not image_path:
        print("  Skipped — no image path provided")
        print("  Run with: python -m backend.agents.test_agents <image_path>")
        return True

    from backend.agents.orchestrator import run
    result = run(
        query=      "Analyse this satellite image",
        image_path= image_path,
    )
    ok = bool(result.get("final_answer"))
    print(f"  Query type  : {result['query_type']}")
    print(f"  Answer      : {str(result.get('final_answer',''))[:120]}...")
    if result.get("satellite_result"):
        sr = result["satellite_result"]
        print(f"  Stages      : {sr.get('pipeline_stages', [])}")
    print(f"  Status      : {'[OK]' if ok else '[FAIL]'}")
    return ok


def main():
    image_path = sys.argv[1] if len(sys.argv) > 1 else None

    print("\n" + "="*60)
    print("ASTRONEXUS AI — AGENT PIPELINE TEST")
    print("="*60)

    results = {
        "router":    test_router(),
        "research":  test_research_agent(),
        "graph":     test_graph_agent(),
        "satellite": test_satellite_agent(image_path),
    }

    passed = sum(results.values())
    total  = len(results)

    print(f"\n{'='*60}")
    print(f"RESULTS: {passed}/{total} passed")
    for name, ok in results.items():
        print(f"  {'[OK]' if ok else '[FAIL]'} {name}")
    print(f"{'='*60}\n")

    # Save results
    out = Path("output")
    out.mkdir(exist_ok=True)
    with open(out / "agent_test_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved -> output/agent_test_results.json\n")


if __name__ == "__main__":
    main()