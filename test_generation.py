"""
AstroNexusAI -- Generation Quality Test
========================================
Tests the full pipeline end-to-end:
  Retrieval (BGE-M3 + Qdrant) -> Gemini generation -> Metrics

Metrics computed per answer:
  Token F1      -- word overlap with reference answer
  Fact Coverage -- fraction of required facts mentioned
  Grounding     -- how many generated tokens appear in retrieved text
  Latency       -- retrieval_ms + generation_ms

Run from project root:
    .\\venv\\Scripts\\python test_generation.py

Optional flags:
    --limit N     test only first N questions (default: 5)
    --judge       also run Ollama LLM judge (default: off)
    --paper AION-1|KnowledgeGraph  filter by paper
    --save        save full results to generation_test_results.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# -- Path setup ---------------------------------------------------------------
_ROOT    = Path(__file__).resolve().parent          # project root
_BACKEND = _ROOT / "backend"
for _p in [str(_ROOT), str(_BACKEND)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from dotenv import load_dotenv
load_dotenv(_ROOT / ".env")

# -- Imports ------------------------------------------------------------------
from backend.rag.retriever import retrieve
from backend.agents.orchestrator import run as orchestrator_run
from backend.evaluation.metrics import token_f1, fact_coverage, grounding_score

# ── Dataset & paper map ──────────────────────────────────────────────────────
DATASET = _BACKEND / "evaluation" / "dataset.jsonl"

PAPER_ID_MAP = {
    "AION-1":         "d9009cdf0b8e9e48e39a1de55089379c",
    "KnowledgeGraph": "94712bcc659c4176f1dee812054444fa",
}

SEP = "-" * 70


def load_questions(limit: int | None, paper_filter: str | None) -> list[dict]:
    qs = []
    with open(DATASET, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            q = json.loads(line)
            if paper_filter and q.get("paper_label") != paper_filter:
                continue
            qs.append(q)
    return qs[:limit] if limit else qs


def run_one(q: dict) -> dict:
    """Run retrieval + generation for one question. Returns metrics dict."""
    query    = q["question"]
    paper_id = PAPER_ID_MAP.get(q.get("paper_label", ""))
    ref      = q.get("reference_answer", "")
    facts    = q.get("required_facts", [])

    # -- Retrieval ------------------------------------------------------------
    t0     = time.perf_counter()
    chunks = retrieve(query=query, top_k=5,
                      filter_paper_id=paper_id, score_threshold=0.0)
    ret_ms = (time.perf_counter() - t0) * 1000
    texts  = [c.text for c in chunks]
    scores = [round(c.score, 4) for c in chunks]

    # -- Generation (Ollama draft -> Gemini refine) ---------------------------
    t1 = time.perf_counter()
    result = orchestrator_run(
        query=        query,
        paper_loaded= bool(paper_id),
        paper_id=     paper_id,
    )
    gen_ms = (time.perf_counter() - t1) * 1000
    answer = result.get("final_answer", "")

    # -- Metrics --------------------------------------------------------------
    f1  = token_f1(answer, ref)
    fc  = fact_coverage(answer, facts)
    gnd = grounding_score(answer, texts)

    return {
        "id":         q.get("id", "?"),
        "paper":      q.get("paper_label", "?"),
        "difficulty": q.get("difficulty", "?"),
        "question":   query,
        "answer":     answer,
        "reference":  ref,
        "facts":      facts,
        "ret_scores": scores,
        "n_chunks":   len(chunks),
        "token_f1":   round(f1, 4),
        "fact_cov":   round(fc, 4),
        "grounding":  round(gnd, 4),
        "ret_ms":     round(ret_ms),
        "gen_ms":     round(gen_ms),
        "total_ms":   round(ret_ms + gen_ms),
    }


def print_result(r: dict, idx: int, total: int) -> None:
    print(f"\n{SEP}")
    print(f"[{idx}/{total}] {r['id']}  |  paper={r['paper']}  |  {r['difficulty'].upper()}")
    print(SEP)
    print(f"Q: {r['question']}")
    print()
    print(f"ANSWER  ({r['gen_ms']}ms):")
    # Word-wrap at 78 chars
    words, line_buf = r["answer"].split(), ""
    for w in words:
        if len(line_buf) + len(w) + 1 > 78:
            print(f"  {line_buf}")
            line_buf = w
        else:
            line_buf = f"{line_buf} {w}".strip()
    if line_buf:
        print(f"  {line_buf}")
    print()
    ref_preview = r["reference"][:120] + ("..." if len(r["reference"]) > 120 else "")
    print(f"REFERENCE: {ref_preview}")
    print()
    print(f"RETRIEVAL : {r['n_chunks']} chunks | "
          f"top scores {r['ret_scores'][:3]} | {r['ret_ms']}ms")
    print()
    f1_flag  = "GOOD" if r["token_f1"]  > 0.3 else "LOW"
    gnd_flag = "GROUNDED" if r["grounding"] > 0.3 else "UNGROUNDED"
    print("METRICS:")
    print(f"  Token F1      : {r['token_f1']:.4f}  [{f1_flag}]")
    print(f"  Fact Coverage : {r['fact_cov']:.4f}  required={r['facts']}")
    print(f"  Grounding     : {r['grounding']:.4f}  [{gnd_flag}]")
    print(f"  Total latency : {r['total_ms']}ms  "
          f"(retrieval={r['ret_ms']}ms + generation={r['gen_ms']}ms)")
    if "judge_score" in r:
        js = r["judge_score"]
        print(f"  LLM Judge     : {js}/3  ({r.get('judge_reason','')[:80]})")


def print_summary(results: list[dict]) -> None:
    n = len(results)
    if n == 0:
        print("\nNo results to summarise.")
        return

    avg = lambda k: sum(r[k] for r in results) / n  # noqa: E731

    print(f"\n{'='*70}")
    print(f"GENERATION EVALUATION SUMMARY  ({n} questions)")
    print(f"{'='*70}")
    print(f"  Avg Token F1      : {avg('token_f1'):.4f}")
    print(f"  Avg Fact Coverage : {avg('fact_cov'):.4f}")
    print(f"  Avg Grounding     : {avg('grounding'):.4f}")
    print(f"  Avg Latency       : {avg('total_ms'):.0f}ms  "
          f"(ret={avg('ret_ms'):.0f}ms  gen={avg('gen_ms'):.0f}ms)")

    if "judge_score" in results[0]:
        scored = [r for r in results if r.get("judge_score") is not None]
        if scored:
            avg_j = sum(r["judge_score"] for r in scored) / len(scored)
            print(f"  Avg Judge Score   : {avg_j:.2f}/3  ({len(scored)} scored)")

    print(f"{'='*70}")

    # Per-difficulty breakdown
    for diff in ["easy", "medium", "hard"]:
        sub = [r for r in results if r["difficulty"] == diff]
        if not sub:
            continue
        print(f"\n  {diff.upper()} ({len(sub)} questions):")
        print(f"    TokenF1={sum(r['token_f1'] for r in sub)/len(sub):.4f}  "
              f"FactCov={sum(r['fact_cov'] for r in sub)/len(sub):.4f}  "
              f"Grounding={sum(r['grounding'] for r in sub)/len(sub):.4f}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="AstroNexusAI Generation Quality Test",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--limit", type=int, default=5,
                        help="Number of questions to test (default: 5)")
    parser.add_argument("--paper", default=None,
                        choices=["AION-1", "KnowledgeGraph"],
                        help="Restrict to one paper")
    parser.add_argument("--judge", action="store_true",
                        help="Run Ollama qwen3:4b LLM judge after each answer")
    parser.add_argument("--save", action="store_true",
                        help="Save results to generation_test_results.json")
    args = parser.parse_args()

    print(f"\n{'='*70}")
    print("AstroNexusAI -- Generation Quality Test")
    print(f"  LLM pipeline : Ollama qwen3:4b (draft) + Gemini 3.1-flash-lite (refine)")
    print(f"  Embedding    : BAAI/bge-m3 local")
    print(f"  Questions    : {args.limit or 'all'}")
    print(f"  Paper filter : {args.paper or 'all'}")
    print(f"  LLM Judge    : {'ON  (Ollama qwen3:4b)' if args.judge else 'OFF'}")
    print(f"{'='*70}")

    qs = load_questions(args.limit, args.paper)
    print(f"\nLoaded {len(qs)} questions from dataset.jsonl")

    results = []
    for i, q in enumerate(qs, 1):
        print(f"\n>>> [{i}/{len(qs)}] Running {q.get('id','?')} ...", flush=True)
        try:
            r = run_one(q)

            if args.judge and q.get("reference_answer"):
                try:
                    from backend.evaluation.judge import judge as llm_judge
                    score, reason = llm_judge(
                        question=         q["question"],
                        answer=           r["answer"],
                        reference_answer= q["reference_answer"],
                        required_facts=   q.get("required_facts", []),
                    )
                    r["judge_score"]  = score
                    r["judge_reason"] = reason
                except Exception as je:
                    r["judge_score"]  = None
                    r["judge_reason"] = f"Judge failed: {je}"

            results.append(r)
            print_result(r, i, len(qs))

        except Exception as e:
            import traceback
            print(f"  [ERROR] {e}")
            traceback.print_exc()

    print_summary(results)

    if args.save:
        out = _ROOT / "generation_test_results.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Results saved to: {out}")


if __name__ == "__main__":
    main()
