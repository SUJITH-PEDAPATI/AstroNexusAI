"""
AstroNexusAI — Generation Evaluation
======================================
Evaluates the RAG generation pipeline using either:
  - orchestrator  (default): the full AstroNexusAI pipeline (Qwen3 + Gemini)
  - gemini:  Qdrant retrieval → context → Gemini 2.0 Flash directly

The two generators are completely independent code paths.
The Ollama-based judge remains the same regardless of generator.

Usage:
    # Orchestrator (existing behaviour, unchanged)
    python -m evaluation.generation_eval
    python -m evaluation.generation_eval --limit 5 --no-judge
    python -m evaluation.generation_eval --paper AION-1

    # Gemini as generator
    python -m evaluation.generation_eval --generator gemini --limit 5 --no-judge
    python -m evaluation.generation_eval --generator gemini --paper AION-1
    python -m evaluation.generation_eval --generator gemini --paper AstroM3
    python -m evaluation.generation_eval --generator gemini --paper KnowledgeGraph
    python -m evaluation.generation_eval --generator gemini

Results are saved to separate files per generator — Gemini results never
overwrite orchestrator results.
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

from evaluation.metrics import token_f1, fact_coverage, grounding_score
from evaluation.judge   import judge as llm_judge

RESULTS_DIR = _PROJECT / "evaluation" / "results" / "generation"
DATASET     = _PROJECT / "evaluation" / "dataset.jsonl"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ── Dataset ────────────────────────────────────────────────────────────────────

def load_qs(limit: Optional[int] = None, paper_filter: Optional[str] = None) -> list[dict]:
    qs = []
    with open(DATASET) as f:
        for line in f:
            if line.strip():
                q = json.loads(line)
                if paper_filter and q["paper_label"] not in (paper_filter, "cross"):
                    continue
                qs.append(q)
    return qs[:limit] if limit else qs


def get_paper_id_map() -> dict[str, str]:
    from backend.rag.vector_store import _get_client, COLLECTION_NAME
    client = _get_client()
    results, _ = client.scroll(collection_name=COLLECTION_NAME, limit=2000,
                               with_payload=True, with_vectors=False)
    papers: dict[str, str] = {}
    for r in results:
        p = r.payload or {}
        pid = p.get("paper_id", "")
        if pid and pid not in papers:
            papers[pid] = p.get("title", "").lower()

    mapping: dict[str, str] = {}
    for pid, title in papers.items():
        if   "aion"         in title: mapping["AION-1"]         = pid
        elif "astrom"       in title: mapping["AstroM3"]        = pid
        elif "knowledge"    in title: mapping["KnowledgeGraph"] = pid
    print(f"Paper map: {mapping}")
    return mapping


# ── Gemini generator ────────────────────────────────────────────────────────────

def _check_gemini_key() -> str:
    """
    Retrieve GEMINI_API_KEY from environment.
    Raises a clear RuntimeError if absent or empty — never falls back silently.
    """
    key = os.getenv("GEMINI_API_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "\n" + "="*60 + "\n"
            "  GEMINI_API_KEY is not set.\n"
            "  Export it before running:\n"
            "    Windows:  set GEMINI_API_KEY=your_key_here\n"
            "    Linux/Mac: export GEMINI_API_KEY=your_key_here\n"
            "  Never hard-code API keys in source files.\n"
            + "="*60
        )
    return key


def _gemini_generate(context_chunks: list[str], question: str, api_key: str) -> str:
    """
    Call Gemini 2.0 Flash directly with the retrieved context.

    Uses the exact same client pattern as research_agent._gemini():
        genai.Client(api_key=...).models.generate_content(
            model="gemini-2.0-flash", ...)

    Raises on any API error — does NOT fall back to Ollama.

    Args:
        context_chunks: text of Qdrant-retrieved chunks
        question:       the evaluation question
        api_key:        GEMINI_API_KEY from environment

    Returns:
        Generated answer string (stripped)

    Raises:
        RuntimeError on API failure
    """
    try:
        from google import genai
        from google.genai import types
    except ImportError as e:
        raise ImportError(
            "google-genai is not installed. Run:\n"
            "  pip install google-genai"
        ) from e

    context_text = "\n\n".join(
        f"[Chunk {i+1}]\n{chunk[:800]}"
        for i, chunk in enumerate(context_chunks)
    ) if context_chunks else "No context retrieved."

    system = (
        "You are a precise scientific assistant. "
        "Answer the question based ONLY on the provided context passages. "
        "Be concise and factual. Do not speculate beyond the evidence."
    )
    prompt = (
        f"CONTEXT:\n{context_text}\n\n"
        f"QUESTION: {question}\n\n"
        f"ANSWER:"
    )

    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=    "gemini-2.0-flash",
            contents= f"{system}\n\n{prompt}",
            config=   types.GenerateContentConfig(
                temperature=     0.1,
                max_output_tokens=1500,
            ),
        )
        answer = response.text.strip() if response.text else ""
        if not answer:
            raise RuntimeError("Gemini returned an empty response")
        return answer

    except Exception as e:
        # Re-raise with context — never swallow or fall back silently
        raise RuntimeError(f"Gemini API error: {e}") from e


# ── Orchestrator generator (existing path, unchanged) ─────────────────────────

def _orchestrator_generate(question: str, paper_id: Optional[str]) -> str:
    """Call the full AstroNexusAI orchestrator pipeline."""
    from backend.agents.orchestrator import run
    result = run(query=question, paper_loaded=bool(paper_id), paper_id=paper_id)
    return result.get("final_answer", "") or ""


# ── Core evaluation loop ────────────────────────────────────────────────────────

def run_generation_eval(
    questions:    list[dict],
    paper_id_map: dict[str, str],
    generator:    str = "orchestrator",   # "orchestrator" | "gemini"
    use_judge:    bool = True,
) -> list[dict]:
    """
    Evaluate every question with the chosen generator.

    Flow (both generators):
        Question → Qdrant retrieval → context → generator → answer → metrics

    The only difference between generators is how the answer is produced.
    Retrieval, metrics, and judge are identical.
    """
    from backend.rag.retriever import retrieve

    # Validate Gemini key once upfront — fail fast before running 35 questions
    gemini_key: Optional[str] = None
    if generator == "gemini":
        gemini_key = _check_gemini_key()   # raises if missing
        print(f"\n  Gemini API key found (length={len(gemini_key)})")
        print("  Model: gemini-2.0-flash")
        print("  Retrieval: AstroNexusAI Qdrant pipeline (unchanged)")

    rows: list[dict] = []

    for i, q in enumerate(questions, 1):
        label    = q["paper_label"]
        paper_id = paper_id_map.get(label) if label != "cross" else None
        print(f"\n[{i}/{len(questions)}] {q['id']} | {label} | {q['question'][:60]}...")

        # ── Retrieval (same for both generators) ──────────────────────────────
        chunk_texts: list[str] = []
        try:
            chunks      = retrieve(q["question"], top_k=5, score_threshold=0.0,
                                   filter_paper_id=paper_id)
            chunk_texts = [c.text for c in chunks]
            print(f"  Retrieved {len(chunks)} chunks")
        except Exception as e:
            print(f"  Retrieval failed: {e}")

        # ── Generation ────────────────────────────────────────────────────────
        t0 = time.perf_counter()
        answer = ""
        gen_error: Optional[str] = None

        try:
            if generator == "gemini":
                answer = _gemini_generate(chunk_texts, q["question"], gemini_key)
            else:
                answer = _orchestrator_generate(q["question"], paper_id)
        except Exception as e:
            gen_error = str(e)
            print(f"  Generation FAILED: {e}")
            rows.append({
                "id":          q["id"],
                "paper_label": label,
                "difficulty":  q["difficulty"],
                "generator":   generator,
                "error":       gen_error,
            })
            continue

        gen_ms = (time.perf_counter() - t0) * 1000

        # ── Metrics ───────────────────────────────────────────────────────────
        ref    = q.get("reference_answer", "")
        f1     = token_f1(answer, ref)
        fc     = fact_coverage(answer, q.get("required_facts", []))
        grd    = grounding_score(answer, chunk_texts)
        ref_w  = len(ref.split())
        gen_w  = len(answer.split())

        # ── Ollama judge (always Ollama regardless of generator) ──────────────
        j_score:  Optional[float] = None
        j_reason: Optional[str]   = None
        if use_judge and ref:
            j_score, j_reason = llm_judge(
                q["question"], answer, ref, q.get("required_facts", [])
            )

        row = {
            "id":              q["id"],
            "paper_label":     label,
            "difficulty":      q["difficulty"],
            "cross_paper":     q.get("cross_paper", False),
            "generator":       generator,
            "question":        q["question"],
            "answer":          answer[:600],
            "reference":       ref[:300],
            "token_f1":        round(f1, 4),
            "fact_coverage":   round(fc, 4),
            "grounding":       round(grd, 4),
            "judge_score":     j_score,
            "judge_reason":    j_reason,
            "answer_words":    gen_w,
            "reference_words": ref_w,
            "length_ratio":    round(gen_w / ref_w, 2) if ref_w else 0.0,
            "gen_ms":          round(gen_ms, 1),
            "hallucinated":    grd < 0.40,
            "n_chunks":        len(chunk_texts),
        }
        rows.append(row)

        j_str = f"Judge={j_score:.0f}/3 " if j_score is not None else ""
        print(f"  F1={f1:.3f}  FC={fc:.3f}  Grnd={grd:.3f}  "
              f"{j_str}Halluc={'YES' if grd<0.40 else 'no'}  "
              f"Words={gen_w}  {gen_ms:.0f}ms")

    return rows


# ── Aggregation ────────────────────────────────────────────────────────────────

def aggregate(rows: list[dict]) -> dict:
    valid = [r for r in rows if "error" not in r]
    if not valid:
        return {"n": 0, "n_errors": len(rows), "error": "all questions failed"}

    def avg(key: str) -> float:
        vals = [r[key] for r in valid if isinstance(r.get(key), (int, float))]
        return round(statistics.mean(vals), 4) if vals else 0.0

    js     = [r["judge_score"] for r in valid if r.get("judge_score") is not None]
    halluc = sum(1 for r in valid if r.get("hallucinated")) / len(valid)

    agg: dict = {
        "generator":          valid[0].get("generator", "orchestrator"),
        "n":                  len(valid),
        "n_errors":           len(rows) - len(valid),
        "n_total":            len(rows),
        "token_f1":           avg("token_f1"),
        "fact_coverage":      avg("fact_coverage"),
        "grounding":          avg("grounding"),
        "hallucination_rate": round(halluc, 4),
        "grounded_pct":       round(1.0 - halluc, 4),
        "length_ratio_mean":  avg("length_ratio"),
        "gen_ms_mean":        avg("gen_ms"),
        "gen_ms_median":      round(
            statistics.median(r["gen_ms"] for r in valid if "gen_ms" in r), 1
        ),
        "gen_ms_p95":         round(
            sorted(r["gen_ms"] for r in valid if "gen_ms" in r)[
                max(0, int(0.95 * len(valid)) - 1)
            ], 1
        ) if valid else 0.0,
    }

    if js:
        agg["judge_score_mean"] = round(statistics.mean(js), 3)
        agg["judge_score_0_1"]  = round(statistics.mean(s / 3.0 for s in js), 4)

    for diff in ("easy", "medium", "hard"):
        sub = [r for r in valid if r.get("difficulty") == diff]
        if sub:
            agg[f"n_{diff}"]   = len(sub)
            agg[f"f1_{diff}"]  = round(statistics.mean(r["token_f1"]     for r in sub), 4)
            agg[f"fc_{diff}"]  = round(statistics.mean(r["fact_coverage"] for r in sub), 4)
            agg[f"grd_{diff}"] = round(statistics.mean(r["grounding"]     for r in sub), 4)

    return agg


# ── LaTeX output ────────────────────────────────────────────────────────────────

def write_latex(agg: dict, path: Path, generator: str) -> None:
    f1  = agg.get("token_f1",           "?")
    fc  = agg.get("fact_coverage",      "?")
    grd = agg.get("grounding",          "?")
    hr  = agg.get("hallucination_rate", "?")
    js  = agg.get("judge_score_0_1",    "?")
    n   = agg.get("n",                  "?")

    gen_label = "Gemini 2.0 Flash (RAG)" if generator == "gemini" else "AstroNexus (Orch.)"

    lines = [
        r"\begin{table}[t]",
        r"\caption{Generation Quality — " + gen_label + r" (N=" + str(n) + r")}",
        r"\label{tab:generation_" + generator + r"}",
        r"\centering",
        r"\renewcommand{\arraystretch}{1.25}",
        r"\begin{tabular}{@{}lccccc@{}}",
        r"\toprule",
        r"\textbf{System} & \textbf{Token F1} & \textbf{Fact Cov.} & "
        r"\textbf{Grounding} & \textbf{Halluc.\,$\downarrow$} & \textbf{Judge/3} \\",
        r"\midrule",
        rf"LLM-only              & [FILL] & [FILL] & [FILL] & [FILL] & [FILL] \\",
        rf"Dense RAG             & [FILL] & [FILL] & [FILL] & [FILL] & [FILL] \\",
        rf"KG-only               & [FILL] & [FILL] & [FILL] & [FILL] & [FILL] \\",
        rf"\textbf{{{gen_label}}} & \textbf{{{f1}}} & \textbf{{{fc}}} & "
        rf"\textbf{{{grd}}} & \textbf{{{hr}}} & \textbf{{{js}}} \\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    path.write_text("\n".join(lines))
    print(f"LaTeX → {path}")


# ── Summary printer ─────────────────────────────────────────────────────────────

def print_summary(agg: dict, rows: list[dict], result_path: Path) -> None:
    valid   = [r for r in rows if "error" not in r]
    failed  = [r for r in rows if "error" in r]
    gen     = agg.get("generator", "orchestrator")

    print("\n" + "="*56)
    print(f"  GENERATION RESULTS — {gen.upper()}")
    print("="*56)
    print(f"  Questions evaluated : {agg.get('n_total','?')}")
    print(f"  Successful          : {len(valid)}")
    print(f"  Failed              : {len(failed)}")
    if failed:
        for r in failed:
            print(f"    ✗ [{r['id']}] {r.get('error','?')[:80]}")
    print()
    METRICS = [
        ("Token F1",           "token_f1"),
        ("Fact Coverage",      "fact_coverage"),
        ("Grounding",          "grounding"),
        ("Hallucination Rate", "hallucination_rate"),
        ("Grounded %",         "grounded_pct"),
        ("Judge Score (0-1)",  "judge_score_0_1"),
        ("Judge Score (0-3)",  "judge_score_mean"),
        ("Latency mean (ms)",  "gen_ms_mean"),
        ("Latency median (ms)","gen_ms_median"),
        ("Latency p95 (ms)",   "gen_ms_p95"),
        ("Length ratio",       "length_ratio_mean"),
    ]
    for label, key in METRICS:
        val = agg.get(key)
        if val is not None:
            print(f"  {label:<28}: {val}")
    for diff in ("easy", "medium", "hard"):
        if f"f1_{diff}" in agg:
            print(f"  F1/{diff:<24}: {agg[f'f1_{diff}']}  "
                  f"FC={agg.get(f'fc_{diff}','?')}  "
                  f"Grnd={agg.get(f'grd_{diff}','?')}")
    print()
    print(f"  Results saved to: {result_path}")
    print("="*56)


# ── Main ────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="AstroNexusAI generation evaluation")
    parser.add_argument(
        "--generator",
        choices=["orchestrator", "gemini"],
        default="orchestrator",
        help="Which generator to evaluate (default: orchestrator)"
    )
    parser.add_argument("--limit",    type=int,  default=None,
                        help="Max questions (e.g. 5 for smoke test)")
    parser.add_argument("--no-judge", action="store_true",
                        help="Skip Ollama judge (faster)")
    parser.add_argument("--paper",    type=str,  default=None,
                        help="Evaluate one paper only: AION-1 | AstroM3 | KnowledgeGraph")
    args = parser.parse_args()

    generator = args.generator
    use_judge = not args.no_judge

    qs = load_qs(args.limit, args.paper)
    if not qs:
        print("ERROR: No questions matched the given filters.")
        sys.exit(1)

    pid_map = get_paper_id_map()
    if not pid_map:
        print("ERROR: No papers found in Qdrant. Ingest papers first.")
        sys.exit(1)

    print(f"\n{'='*56}")
    print(f"  Generator : {generator}")
    print(f"  Questions : {len(qs)}")
    print(f"  Judge     : {'Ollama qwen3:4b' if use_judge else 'disabled'}")
    print(f"  Paper     : {args.paper or 'all'}")
    print(f"{'='*56}")

    rows = run_generation_eval(qs, pid_map, generator=generator, use_judge=use_judge)
    agg  = aggregate(rows)

    # Save results — separate file per generator, never overwrite each other
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = f"results_{generator}_{ts}"
    result_path = RESULTS_DIR / f"{stem}.json"
    latex_path  = RESULTS_DIR / f"generation_table_{generator}.tex"

    with open(result_path, "w") as f:
        json.dump({"aggregate": agg, "rows": rows}, f, indent=2)

    write_latex(agg, latex_path, generator)
    print_summary(agg, rows, result_path)


if __name__ == "__main__":
    main()