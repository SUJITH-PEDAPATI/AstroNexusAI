"""
AstroNexus AI — Full Answer Test
Runs the complete pipeline and prints actual answers.

Run:
    python -m backend.tests.test_answers
    python -m backend.tests.test_answers --category api
    python -m backend.tests.test_answers --id A01
"""
from __future__ import annotations

import sys
import logging
import json
import time
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logging.basicConfig(level=logging.WARNING)

GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BLUE   = "\033[94m"
RESET  = "\033[0m"
BOLD   = "\033[1m"

TESTS = [
    # ── API — real-time data ──────────────────────────────────────────────────
    {"id":"A01", "cat":"api",      "q":"What is the climate forecast for next 7 days in Delhi?"},
    {"id":"A02", "cat":"api",      "q":"Is there any active wildfire detected recently?"},
    {"id":"A03", "cat":"api",      "q":"What is the air quality in Mumbai today?"},
    {"id":"A04", "cat":"api",      "q":"Find recent papers on flood detection using SAR imagery"},
    {"id":"A05", "cat":"api",      "q":"What is the rainfall forecast for Chennai next week?"},

    # ── General knowledge ─────────────────────────────────────────────────────
    {"id":"K01", "cat":"general",  "q":"What is reinforcement learning?"},
    {"id":"K02", "cat":"general",  "q":"Explain the difference between CNN and Transformer"},
    {"id":"K03", "cat":"general",  "q":"What is overfitting in machine learning?"},

    # ── Knowledge Graph ───────────────────────────────────────────────────────
    {"id":"G01", "cat":"graph",    "q":"Who authored Attention Is All You Need?"},
    {"id":"G02", "cat":"graph",    "q":"Which satellites are part of the Copernicus mission?"},
    {"id":"G03", "cat":"graph",    "q":"What sensor does Sentinel-1 carry?"},
    {"id":"G04", "cat":"graph",    "q":"Which algorithms are used for semantic segmentation?"},

    # ── Research / RAG ────────────────────────────────────────────────────────
    {"id":"R01", "cat":"research", "q":"What is the main contribution of the uploaded paper?"},
    {"id":"R02", "cat":"research", "q":"What dataset was used for training?"},
    {"id":"R03", "cat":"research", "q":"What are the limitations mentioned by the authors?"},

    # ── Edge cases ────────────────────────────────────────────────────────────
    {"id":"E01", "cat":"edge",     "q":"What does Section 99.9 say about the results?"},
    {"id":"E02", "cat":"edge",     "q":"wht is attntion mechnism in transformrs"},
    {"id":"E03", "cat":"edge",     "q":"What is the model?"},
]


def run_query(q: str, cat: str) -> dict:
    """Run through full orchestrator pipeline."""
    from backend.agents.orchestrator import run

    paper_loaded = cat == "research"

    t0     = time.perf_counter()
    result = run(
        query=        q,
        paper_loaded= paper_loaded,
    )
    elapsed = time.perf_counter() - t0

    meta = result.get("metadata") or {}
    evl  = meta.get("evaluation") or {}

    return {
        "answer":      result.get("final_answer", ""),
        "route":       result.get("query_type", "?"),
        "confidence":  evl.get("confidence", "—"),
        "grounding":   evl.get("grounding_score", 0.0),
        "reliable":    evl.get("is_reliable", False),
        "warnings":    evl.get("warnings", []),
        "citations":   evl.get("citations", []),
        "apis_called": meta.get("apis_called", []),
        "ollama":      meta.get("ollama_used", False),
        "gemini":      meta.get("gemini_used", False),
        "elapsed":     round(elapsed, 2),
        "error":       result.get("error", ""),
    }


def print_result(test: dict, result: dict) -> None:
    qid   = test["id"]
    query = test["q"]
    cat   = test["cat"]

    print(f"\n{'─'*65}")
    print(f"{BOLD}[{qid}] {cat.upper()}{RESET}")
    print(f"{BLUE}Q:{RESET} {query}")
    print(f"{'─'*65}")

    # Routing
    route = result["route"]
    print(f"  {CYAN}Route{RESET}      : {route}")

    # APIs called
    if result["apis_called"]:
        print(f"  {CYAN}APIs used{RESET}  : {result['apis_called']}")

    # LLMs used
    llms = []
    if result["ollama"]: llms.append("Ollama")
    if result["gemini"]: llms.append("Gemini")
    if llms:
        print(f"  {CYAN}LLMs{RESET}       : {' → '.join(llms)}")

    # Evaluation
    conf      = result["confidence"]
    conf_color = GREEN if conf == "HIGH" else YELLOW if conf == "MEDIUM" else RED
    reliable  = result["reliable"]

    print(f"  {CYAN}Confidence{RESET} : {conf_color}{conf}{RESET}  "
          f"grounding={result['grounding']:.2f}  "
          f"reliable={GREEN+'YES'+RESET if reliable else RED+'NO'+RESET}")

    # Warnings
    if result["warnings"]:
        for w in result["warnings"][:2]:
            print(f"  {YELLOW}⚠ {w[:80]}{RESET}")

    # Citations
    if result["citations"]:
        print(f"  {CYAN}Citations{RESET}  :")
        for c in result["citations"][:2]:
            print(f"    [{c.get('section','?')} p.{c.get('page','?')}] "
                  f"score={c.get('score',0):.3f}")

    # Error
    if result["error"]:
        print(f"  {RED}Error{RESET}      : {result['error'][:80]}")

    # Answer
    print(f"\n  {BOLD}Answer:{RESET}")
    answer = result["answer"] or "(no answer)"
    # Print answer with word-wrap at 65 chars
    words  = answer.split()
    line   = "  "
    for word in words:
        if len(line) + len(word) > 67:
            print(line)
            line = "  " + word + " "
        else:
            line += word + " "
    if line.strip():
        print(line)

    print(f"\n  {CYAN}⏱ {result['elapsed']}s{RESET}")


def main():
    args     = sys.argv[1:]
    cat_filter = None
    id_filter  = None

    for i, a in enumerate(args):
        if a == "--category" and i+1 < len(args):
            cat_filter = args[i+1]
        if a == "--id" and i+1 < len(args):
            id_filter = args[i+1]

    tests = TESTS
    if cat_filter:
        tests = [t for t in tests if t["cat"] == cat_filter]
    if id_filter:
        tests = [t for t in tests if t["id"] == id_filter]

    print(f"\n{'═'*65}")
    print(f"{BOLD}  ASTRONEXUS AI — FULL ANSWER TEST{RESET}")
    print(f"  Running {len(tests)} queries through complete pipeline")
    print(f"{'═'*65}")

    all_results = []
    for test in tests:
        try:
            result = run_query(test["q"], test["cat"])
        except Exception as e:
            result = {
                "answer": "", "route": "ERROR", "confidence": "LOW",
                "grounding": 0.0, "reliable": False, "warnings": [],
                "citations": [], "apis_called": [], "ollama": False,
                "gemini": False, "elapsed": 0.0, "error": str(e),
            }

        print_result(test, result)
        all_results.append({"test": test, "result": result})

    # Summary
    print(f"\n{'═'*65}")
    print(f"{BOLD}  SUMMARY{RESET}")
    print(f"{'═'*65}")

    total     = len(all_results)
    reliable  = sum(1 for r in all_results if r["result"]["reliable"])
    api_used  = sum(1 for r in all_results if r["result"]["apis_called"])
    high_conf = sum(1 for r in all_results if r["result"]["confidence"] == "HIGH")
    avg_time  = sum(r["result"]["elapsed"] for r in all_results) / total

    print(f"  Total queries    : {total}")
    print(f"  Reliable answers : {GREEN}{reliable}/{total}{RESET}")
    print(f"  High confidence  : {GREEN}{high_conf}/{total}{RESET}")
    print(f"  API data used    : {CYAN}{api_used}/{total}{RESET}")
    print(f"  Avg latency      : {avg_time:.2f}s")
    print(f"{'═'*65}\n")

    # Save
    Path("output").mkdir(exist_ok=True)
    with open("output/answer_test_results.json", "w") as f:
        json.dump(
            [{"id": r["test"]["id"], "q": r["test"]["q"],
              "route": r["result"]["route"],
              "confidence": r["result"]["confidence"],
              "reliable": r["result"]["reliable"],
              "apis": r["result"]["apis_called"],
              "answer_preview": (r["result"]["answer"] or "")[:200]}
             for r in all_results],
            f, indent=2,
        )
    print("  Saved → output/answer_test_results.json\n")


if __name__ == "__main__":
    main()