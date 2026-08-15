"""
AstroNexusAI — End-to-End Evaluation
======================================
Evaluates the COMPLETE AstroNexusAI pipeline exactly as the frontend calls it.

Pipeline under test (mirrors POST /chat exactly):
    Question
     → orchestrator.run(query, paper_loaded, paper_id)
         → router_node  (query classification)
         → research_agent_node
             → embed_query  (Ollama qwen3:4b)
             → Qdrant dense retrieval
             → knowledge_fusion.fuse  (Qdrant + Neo4j graph context)
             → Qwen3:4b  CoT draft
             → Gemini 2.0 Flash  refinement
             → AnswerEvaluator  (grounding + confidence + grade)
     → final_answer, query_type, rag_context, grade, citations
     ← returned to evaluation harness

The E2E latency timer surrounds the ENTIRE orchestrator.run() call — not
just retrieval or Gemini separately.  It measures exactly what a user waits
for when they send a question.

Metrics (all from existing evaluation.metrics implementations):
    - Token F1          (lexical overlap vs reference answer)
    - Fact Coverage     (required key-facts present in answer)
    - Grounding         (answer sentences supported by retrieved context)
    - Hallucination %   (grounding < 0.40)
    - Answer Relevance  (judge score 0–3 via Ollama, scaled 0–1)
    - E2E Latency       (mean / median / P95, milliseconds)
    - Success / Failure rate

Does NOT call Gemini separately — uses Gemini only as part of the real pipeline.
Does NOT modify or re-run Qdrant retrieval outside the pipeline.
Does NOT overwrite existing evaluation results.

Usage:
    python -m evaluation.e2e_eval --limit 5          # smoke test
    python -m evaluation.e2e_eval --limit 30         # standard (default)
    python -m evaluation.e2e_eval --limit 35         # full dataset
    python -m evaluation.e2e_eval --no-judge --limit 5
    python -m evaluation.e2e_eval --paper AION-1
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

_PROJECT = Path(__file__).resolve().parent.parent
if str(_PROJECT) not in sys.path:
    sys.path.insert(0, str(_PROJECT))

# Reuse existing metric implementations — no duplication
from evaluation.metrics import token_f1, fact_coverage, grounding_score
from evaluation.judge   import judge as llm_judge

RESULTS_DIR = _PROJECT / "evaluation" / "results" / "end_to_end"
DATASET     = _PROJECT / "evaluation" / "dataset.jsonl"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ── Dataset helpers ───────────────────────────────────────────────────────────

def load_qs(limit: int, paper_filter: Optional[str] = None) -> list[dict]:
    qs = []
    with open(DATASET) as f:
        for line in f:
            if not line.strip():
                continue
            q = json.loads(line)
            if paper_filter and q["paper_label"] not in (paper_filter, "cross"):
                continue
            qs.append(q)
    return qs[:limit]


def get_paper_id_map() -> dict[str, str]:
    """Resolve {label: paper_id} by scanning Qdrant payloads."""
    from backend.rag.vector_store import _get_client, COLLECTION_NAME
    client = _get_client()
    results, _ = client.scroll(
        collection_name=COLLECTION_NAME, limit=2000,
        with_payload=True, with_vectors=False,
    )
    papers: dict[str, str] = {}
    for r in results:
        p   = r.payload or {}
        pid = p.get("paper_id", "")
        if pid and pid not in papers:
            papers[pid] = p.get("title", "").lower()

    mapping: dict[str, str] = {}
    for pid, title in papers.items():
        if   "aion"         in title: mapping["AION-1"]         = pid
        elif "astrom"       in title: mapping["AstroM3"]        = pid
        elif "knowledge"    in title: mapping["KnowledgeGraph"] = pid
    return mapping


# ── Pipeline runner ───────────────────────────────────────────────────────────

def run_e2e(
    questions:    list[dict],
    paper_id_map: dict[str, str],
    use_judge:    bool = True,
) -> list[dict]:
    """
    Run the complete AstroNexusAI pipeline for every question.

    The E2E timer wraps the ENTIRE orchestrator.run() call.
    Nothing inside the pipeline is bypassed or short-circuited.

    Stages timed inside a single measurement:
        router_node → research_agent (embed → Qdrant → KG → Qwen3 → Gemini → Evaluator)
    """
    from backend.agents.orchestrator import run as pipeline_run

    rows: list[dict] = []
    n = len(questions)

    for i, q in enumerate(questions, 1):
        label    = q["paper_label"]
        paper_id = paper_id_map.get(label) if label != "cross" else None
        print(f"\n[{i}/{n}] {q['id']} | {q['difficulty']} | {q['question'][:65]}...")

        # ── E2E timer starts HERE — surrounds the complete pipeline ───────────
        t_start = time.perf_counter()

        try:
            result = pipeline_run(
                query=        q["question"],
                paper_loaded= bool(paper_id),
                paper_id=     paper_id,
            )
            e2e_ms = (time.perf_counter() - t_start) * 1000  # timer ends HERE

            answer     = result.get("final_answer", "") or ""
            query_type = result.get("query_type",   "") or ""
            rag_ctx    = result.get("rag_context",  "") or ""
            error_flag = result.get("error")

        except Exception as exc:
            e2e_ms = (time.perf_counter() - t_start) * 1000
            print(f"  Pipeline FAILED ({e2e_ms:.0f}ms): {exc}")
            rows.append({
                "id":          q["id"],
                "paper_label": label,
                "difficulty":  q["difficulty"],
                "question":    q["question"],
                "error":       str(exc),
                "e2e_ms":      round(e2e_ms, 1),
            })
            continue

        if not answer:
            print(f"  Pipeline returned empty answer ({e2e_ms:.0f}ms)")
            rows.append({
                "id":          q["id"],
                "paper_label": label,
                "difficulty":  q["difficulty"],
                "question":    q["question"],
                "error":       "empty_answer",
                "e2e_ms":      round(e2e_ms, 1),
            })
            continue

        # ── Metrics ────────────────────────────────────────────────────────────
        ref    = q.get("reference_answer", "")
        facts  = q.get("required_facts",   [])

        # Grounding uses rag_context (what the pipeline actually retrieved)
        ctx_chunks = [rag_ctx] if rag_ctx else []

        f1   = token_f1(answer, ref)
        fc   = fact_coverage(answer, facts)
        grd  = grounding_score(answer, ctx_chunks)
        ref_w, gen_w = len(ref.split()), len(answer.split())

        # Answer relevance via Ollama judge (0–3, scaled to 0–1)
        j_score:  Optional[float] = None
        j_reason: Optional[str]   = None
        if use_judge and ref:
            j_score, j_reason = llm_judge(q["question"], answer, ref, facts)

        row = {
            "id":             q["id"],
            "paper_label":    label,
            "difficulty":     q["difficulty"],
            "cross_paper":    q.get("cross_paper", False),
            "question":       q["question"],
            "answer":         answer[:600],
            "reference":      ref[:300],
            "query_type":     query_type,
            # Generation metrics
            "token_f1":       round(f1,  4),
            "fact_coverage":  round(fc,  4),
            "grounding":      round(grd, 4),
            "hallucinated":   grd < 0.40,
            # Relevance
            "judge_score":    j_score,              # 0–3
            "judge_score_01": round(j_score / 3.0, 4) if j_score is not None else None,
            "judge_reason":   j_reason,
            # Length diagnostics
            "answer_words":   gen_w,
            "reference_words":ref_w,
            "length_ratio":   round(gen_w / ref_w, 2) if ref_w else 0.0,
            # Latency — this is the COMPLETE E2E measurement
            "e2e_ms":         round(e2e_ms, 1),
            # Pipeline metadata
            "n_ctx_chars":    len(rag_ctx),
            "pipeline_error": error_flag,
        }
        rows.append(row)

        j_str = f"Rel={j_score:.0f}/3 " if j_score is not None else ""
        print(f"  F1={f1:.3f}  FC={fc:.3f}  Grnd={grd:.3f}  "
              f"{j_str}Halluc={'YES' if grd<0.40 else 'no'}  "
              f"E2E={e2e_ms:.0f}ms")

    return rows


# ── Aggregation ────────────────────────────────────────────────────────────────

def aggregate(rows: list[dict]) -> dict:
    valid  = [r for r in rows if "error" not in r]
    failed = [r for r in rows if "error" in r]

    if not valid:
        return {
            "n_total": len(rows), "n_success": 0, "n_failed": len(failed),
            "success_rate": 0.0, "error": "all questions failed",
        }

    n = len(valid)

    def avg(key: str) -> float:
        vals = [r[key] for r in valid if isinstance(r.get(key), (int, float))]
        return round(statistics.mean(vals), 4) if vals else 0.0

    latencies = sorted(r["e2e_ms"] for r in valid)
    p95_idx   = max(0, int(0.95 * n) - 1)

    halluc_rate = sum(1 for r in valid if r.get("hallucinated")) / n
    js_vals     = [r["judge_score"] for r in valid if r.get("judge_score") is not None]

    agg: dict = {
        # Counts
        "n_total":           len(rows),
        "n_success":         n,
        "n_failed":          len(failed),
        "success_rate":      round(n / len(rows), 4),
        # Generation quality
        "token_f1":          avg("token_f1"),
        "fact_coverage":     avg("fact_coverage"),
        "grounding":         avg("grounding"),
        "hallucination_rate":round(halluc_rate, 4),
        "grounded_pct":      round(1.0 - halluc_rate, 4),
        "length_ratio_mean": avg("length_ratio"),
        # E2E latency
        "e2e_ms_mean":       round(statistics.mean(latencies), 1),
        "e2e_ms_median":     round(statistics.median(latencies), 1),
        "e2e_ms_p95":        round(latencies[p95_idx], 1),
        "e2e_ms_min":        round(latencies[0], 1),
        "e2e_ms_max":        round(latencies[-1], 1),
    }

    if js_vals:
        agg["answer_relevance_mean"] = round(statistics.mean(js_vals) / 3.0, 4)
        agg["judge_score_mean"]      = round(statistics.mean(js_vals), 3)

    # Breakdown by difficulty
    for diff in ("easy", "medium", "hard"):
        sub = [r for r in valid if r.get("difficulty") == diff]
        if sub:
            ns = len(sub)
            agg[f"n_{diff}"]    = ns
            agg[f"f1_{diff}"]   = round(statistics.mean(r["token_f1"]     for r in sub), 4)
            agg[f"fc_{diff}"]   = round(statistics.mean(r["fact_coverage"] for r in sub), 4)
            agg[f"grd_{diff}"]  = round(statistics.mean(r["grounding"]     for r in sub), 4)
            agg[f"lat_{diff}"]  = round(statistics.mean(r["e2e_ms"]        for r in sub), 1)

    return agg


# ── LaTeX output ───────────────────────────────────────────────────────────────

def write_latex(agg: dict, path: Path) -> None:
    n   = agg.get("n_success",      "?")
    f1  = agg.get("token_f1",       "?")
    fc  = agg.get("fact_coverage",  "?")
    grd = agg.get("grounding",      "?")
    hr  = agg.get("hallucination_rate","?")
    rel = agg.get("answer_relevance_mean", "?")
    lat = agg.get("e2e_ms_mean",    "?")
    p95 = agg.get("e2e_ms_p95",     "?")

    lines = [
        r"\begin{table}[t]",
        r"\caption{AstroNexus AI — End-to-End Evaluation (N=" + str(n) + r")}",
        r"\label{tab:e2e}",
        r"\centering",
        r"\renewcommand{\arraystretch}{1.25}",
        r"\begin{tabular}{@{}lcccccc@{}}",
        r"\toprule",
        (r"\textbf{System} & \textbf{Token F1} & \textbf{Fact Cov.}"
         r" & \textbf{Grounding} & \textbf{Halluc.\,$\downarrow$}"
         r" & \textbf{Relevance} & \textbf{E2E (ms)\,$\downarrow$} \\"),
        r"\midrule",
        rf"LLM-only       & [FILL] & [FILL] & [FILL] & [FILL] & [FILL] & [FILL] \\",
        rf"Dense RAG      & [FILL] & [FILL] & [FILL] & [FILL] & [FILL] & [FILL] \\",
        (rf"\textbf{{AstroNexus}} & \textbf{{{f1}}} & \textbf{{{fc}}}"
         rf" & \textbf{{{grd}}} & \textbf{{{hr}}} & \textbf{{{rel}}}"
         rf" & \textbf{{{lat}}} \\"),
        r"\midrule",
        rf"\multicolumn{{7}}{{l}}{{\footnotesize P95 latency = {p95}\,ms}} \\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    path.write_text("\n".join(lines))
    print(f"LaTeX → {path}")


# ── Summary printer ────────────────────────────────────────────────────────────

def print_summary(agg: dict, rows: list[dict], path: Path) -> None:
    failed = [r for r in rows if "error" in r]

    print("\n" + "="*56)
    print("  AstroNexusAI END-TO-END EVALUATION")
    print("="*56)
    print(f"  Questions evaluated  : {agg.get('n_total','?')}")
    print(f"  Successful           : {agg.get('n_success','?')}")
    print(f"  Failed               : {agg.get('n_failed','?')}")
    print(f"  Success rate         : {agg.get('success_rate','?')}")
    if failed:
        for r in failed:
            print(f"    ✗ [{r['id']}] {r.get('error','?')[:80]}")
    print()
    METRICS = [
        ("Answer Relevance (0-1)", "answer_relevance_mean"),
        ("Token F1",               "token_f1"),
        ("Fact Coverage",          "fact_coverage"),
        ("Grounding",              "grounding"),
        ("Hallucination Rate",     "hallucination_rate"),
        ("Grounded %",             "grounded_pct"),
        ("E2E Latency mean (ms)",  "e2e_ms_mean"),
        ("E2E Latency median (ms)","e2e_ms_median"),
        ("E2E Latency P95 (ms)",   "e2e_ms_p95"),
        ("E2E Latency min (ms)",   "e2e_ms_min"),
        ("E2E Latency max (ms)",   "e2e_ms_max"),
    ]
    for label, key in METRICS:
        val = agg.get(key)
        if val is not None:
            print(f"  {label:<30}: {val}")
    for diff in ("easy", "medium", "hard"):
        if f"f1_{diff}" in agg:
            print(f"  [{diff:<6}] F1={agg[f'f1_{diff}']}  "
                  f"FC={agg.get(f'fc_{diff}','?')}  "
                  f"Grnd={agg.get(f'grd_{diff}','?')}  "
                  f"Lat={agg.get(f'lat_{diff}','?')}ms")
    print()
    print(f"  Results saved to: {path}")
    print("="*56)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="AstroNexusAI end-to-end pipeline evaluation"
    )
    parser.add_argument("--limit",    type=int, default=30,
                        help="Number of questions to evaluate (default: 30)")
    parser.add_argument("--no-judge", action="store_true",
                        help="Skip Ollama judge (faster, no answer relevance score)")
    parser.add_argument("--paper",    type=str, default=None,
                        help="Evaluate one paper only: AION-1 | AstroM3 | KnowledgeGraph")
    args = parser.parse_args()

    # Validate Gemini key exists (the pipeline uses it internally)
    gemini_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not gemini_key:
        print(
            "\nERROR: GEMINI_API_KEY is not set.\n"
            "The AstroNexusAI pipeline uses Gemini for answer refinement.\n"
            "Export it before running:\n"
            "  Windows:  set GEMINI_API_KEY=your_key_here\n"
            "  Linux/Mac: export GEMINI_API_KEY=your_key_here\n"
        )
        sys.exit(1)

    qs = load_qs(args.limit, args.paper)
    if not qs:
        print("ERROR: No questions matched the filters.")
        sys.exit(1)

    pid_map = get_paper_id_map()
    if not pid_map:
        print("ERROR: No papers found in Qdrant. Ingest papers first.")
        sys.exit(1)

    print(f"\n{'='*56}")
    print("  AstroNexusAI End-to-End Evaluation")
    print(f"  Questions : {len(qs)}")
    print(f"  Judge     : {'Ollama qwen3:4b' if not args.no_judge else 'disabled'}")
    print(f"  Paper     : {args.paper or 'all'}")
    print(f"  Gemini key: set ({len(gemini_key)} chars)")
    print(f"{'='*56}")
    print()
    print("  Pipeline measured:")
    print("    orchestrator.run()")
    print("    ├── router_node  (query classification)")
    print("    └── research_agent_node")
    print("        ├── embed_query  (Ollama qwen3:4b)")
    print("        ├── Qdrant dense retrieval")
    print("        ├── knowledge_fusion.fuse  (Qdrant + Neo4j)")
    print("        ├── Qwen3:4b  CoT draft")
    print("        ├── Gemini 2.0 Flash  refinement")
    print("        └── AnswerEvaluator  (grounding + grade)")
    print()

    rows = run_e2e(qs, pid_map, use_judge=not args.no_judge)
    agg  = aggregate(rows)

    ts          = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_path = RESULTS_DIR / f"results_e2e_gemini_{ts}.json"
    latex_path  = RESULTS_DIR / "end_to_end_table.tex"

    with open(result_path, "w") as f:
        json.dump({"aggregate": agg, "per_question": rows}, f, indent=2)

    write_latex(agg, latex_path)
    print_summary(agg, rows, result_path)


if __name__ == "__main__":
    main()