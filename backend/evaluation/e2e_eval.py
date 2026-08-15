"""
AstroNexusAI — End-to-End Evaluation
======================================
Runs the full RAG pipeline (retrieval → generation → metrics) against
every question in evaluation/dataset.jsonl.

Pipeline per question:
    Question → Qdrant retrieval → retrieved chunks
             → Gemini generation → answer
             → token_f1 / fact_coverage / grounding
             → (optional) Ollama judge score

Usage:
    # Quick 5-question smoke test — no judge, fast
    python -m evaluation.e2e_eval --limit 5 --no-judge

    # Full 30-question run with Gemini + Ollama judge
    python -m evaluation.e2e_eval --limit 30

    # Override Gemini model
    python -m evaluation.e2e_eval --limit 30 --gemini-model gemini-3.5-flash-lite

    # Filter to one paper only
    python -m evaluation.e2e_eval --paper AION-1 --limit 10

Results are saved to:
    evaluation/results/e2e/results_e2e_<model>_<timestamp>.json
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

# ── Path bootstrap ─────────────────────────────────────────────────────────────
_PROJECT  = Path(__file__).resolve().parent.parent   # → backend/
_ROOT     = _PROJECT.parent                           # → project root

for _p in [str(_PROJECT), str(_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Load .env from project root (GEMINI_API_KEY, QDRANT_HOST, etc.)
try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=_ROOT / ".env", override=False)
except ImportError:
    pass

# ── Local imports ──────────────────────────────────────────────────────────────
from evaluation.metrics import (
    token_f1, fact_coverage, grounding_score,
    RetrievalMetrics, AnswerMetrics, LatencyMetrics,
    aggregate_retrieval, aggregate_answers, aggregate_latency,
    hit_at_k, recall_at_k, precision_at_k, mean_reciprocal_rank, ndcg_at_k,
)
from evaluation.judge import judge as llm_judge

# ── Constants ──────────────────────────────────────────────────────────────────
DATASET     = _PROJECT / "evaluation" / "dataset.jsonl"
RESULTS_DIR = _PROJECT / "evaluation" / "results" / "e2e"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

_DEFAULT_GEMINI_MODEL = "gemini-3.5-flash-lite"
_DEFAULT_TOP_K        = 5


# ── Dataset loader ─────────────────────────────────────────────────────────────

def load_questions(
    limit:        Optional[int] = None,
    paper_filter: Optional[str] = None,
) -> list[dict]:
    """Load questions from dataset.jsonl, with optional paper/limit filters."""
    qs: list[dict] = []
    with open(DATASET, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            q = json.loads(line)
            if paper_filter and q.get("paper_label") not in (paper_filter, "cross"):
                continue
            qs.append(q)
    return qs[:limit] if limit else qs


# ── Paper ID map from Qdrant ──────────────────────────────────────────────────

def get_paper_id_map() -> dict[str, str]:
    """
    Scroll Qdrant to build label -> paper_id mapping.
    e.g. {"AION-1": "d9009cdf...", "AstroM3": "...", "KnowledgeGraph": "..."}
    """
    from backend.rag.vector_store import _get_client, COLLECTION_NAME
    client = _get_client()
    results, _ = client.scroll(
        collection_name=COLLECTION_NAME,
        limit=2000,
        with_payload=True,
        with_vectors=False,
    )

    papers: dict[str, str] = {}
    for r in results:
        p   = r.payload or {}
        pid = p.get("paper_id", "")
        if pid and pid not in papers:
            papers[pid] = p.get("title", "").lower()

    mapping: dict[str, str] = {}
    for pid, title in papers.items():
        if   "aion"      in title: mapping["AION-1"]         = pid
        elif "astrom"    in title: mapping["AstroM3"]        = pid
        elif "knowledge" in title: mapping["KnowledgeGraph"] = pid

    print(f"Paper ID map: {mapping}")
    return mapping


# ── Gemini client (singleton) ─────────────────────────────────────────────────

def _check_gemini_key() -> str:
    key = os.getenv("GEMINI_API_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "\n" + "="*60 + "\n"
            "  GEMINI_API_KEY is not set.\n"
            "  Add it to your .env file:\n"
            "    GEMINI_API_KEY=your_key_here\n"
            + "="*60
        )
    return key


def build_gemini_client(api_key: str):
    """Create the google-genai Client once -- reused for all questions."""
    try:
        from google import genai
    except ImportError as e:
        raise ImportError(
            "google-genai is not installed. Run:\n"
            "  pip install google-genai"
        ) from e
    return genai.Client(api_key=api_key)


def gemini_generate(
    context_chunks: list[str],
    question:       str,
    client,
    model:          str,
) -> str:
    """
    Call Gemini with retrieved context and return the answer string.
    Raises RuntimeError on any API failure -- never falls back silently.
    """
    from google.genai import types

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
        response = client.models.generate_content(
            model=    model,
            contents= f"{system}\n\n{prompt}",
            config=   types.GenerateContentConfig(
                temperature=      0.1,
                max_output_tokens=1500,
            ),
        )
        answer = response.text.strip() if response.text else ""
        if not answer:
            raise RuntimeError("Gemini returned an empty response")
        return answer
    except Exception as e:
        raise RuntimeError(f"Gemini API error: {e}") from e


# ── Retrieval metrics helper ──────────────────────────────────────────────────

def compute_retrieval_metrics(
    retrieved_paper_ids: list[str],
    relevant_paper_ids:  set[str],
) -> RetrievalMetrics:
    return RetrievalMetrics(
        recall_at_1    = recall_at_k(retrieved_paper_ids, relevant_paper_ids, 1),
        recall_at_3    = recall_at_k(retrieved_paper_ids, relevant_paper_ids, 3),
        recall_at_5    = recall_at_k(retrieved_paper_ids, relevant_paper_ids, 5),
        recall_at_10   = recall_at_k(retrieved_paper_ids, relevant_paper_ids, 10),
        precision_at_1 = precision_at_k(retrieved_paper_ids, relevant_paper_ids, 1),
        precision_at_3 = precision_at_k(retrieved_paper_ids, relevant_paper_ids, 3),
        precision_at_5 = precision_at_k(retrieved_paper_ids, relevant_paper_ids, 5),
        mrr            = mean_reciprocal_rank(retrieved_paper_ids, relevant_paper_ids),
        ndcg_at_10     = ndcg_at_k(retrieved_paper_ids, relevant_paper_ids, 10),
        hit_at_1       = hit_at_k(retrieved_paper_ids, relevant_paper_ids, 1),
        hit_at_5       = hit_at_k(retrieved_paper_ids, relevant_paper_ids, 5),
    )


# ── Core evaluation loop ──────────────────────────────────────────────────────

def run_e2e(
    questions:    list[dict],
    paper_id_map: dict[str, str],
    gemini_model: str  = _DEFAULT_GEMINI_MODEL,
    top_k:        int  = _DEFAULT_TOP_K,
    use_judge:    bool = True,
) -> list[dict]:
    """
    For each question:
        1. Retrieve top-k chunks from Qdrant
        2. Generate answer with Gemini
        3. Compute retrieval + answer metrics
        4. (Optional) Ollama judge score

    Failures are recorded per-question with an 'error' field.
    The loop NEVER aborts -- every question produces a row.
    """
    from backend.rag.retriever import retrieve

    # Build Gemini client once before the loop
    gemini_key    = _check_gemini_key()
    gemini_client = build_gemini_client(gemini_key)
    print(f"\n  Gemini client ready -- model: {gemini_model}")
    print(f"  Top-K: {top_k}  |  Judge: {'Ollama' if use_judge else 'disabled'}\n")

    rows: list[dict] = []

    for i, q in enumerate(questions, 1):
        qid   = q["id"]
        label = q.get("paper_label", "cross")
        paper_id: Optional[str] = paper_id_map.get(label) if label != "cross" else None

        print(f"[{i:>2}/{len(questions)}] {qid} | {label} | {q['question'][:55]}...")

        row: dict = {
            "id":           qid,
            "paper_label":  label,
            "difficulty":   q.get("difficulty", ""),
            "cross_paper":  q.get("cross_paper", False),
            "gemini_model": gemini_model,
            "question":     q["question"],
        }

        # ── Step 1: Retrieval ──────────────────────────────────────────────────
        t_ret_start = time.perf_counter()
        chunk_texts:         list[str] = []
        retrieved_paper_ids: list[str] = []

        try:
            chunks = retrieve(
                q["question"],
                top_k=           top_k,
                score_threshold= 0.0,
                filter_paper_id= paper_id,
            )
            chunk_texts         = [c.text     for c in chunks]
            retrieved_paper_ids = [c.paper_id for c in chunks]
            retrieval_ms = (time.perf_counter() - t_ret_start) * 1000
            print(f"         Retrieved {len(chunks)} chunks  ({retrieval_ms:.0f}ms)")
        except Exception as e:
            retrieval_ms = (time.perf_counter() - t_ret_start) * 1000
            print(f"         Retrieval FAILED: {e}")
            row["error"]        = f"retrieval_failed: {e}"
            row["retrieval_ms"] = round(retrieval_ms, 1)
            rows.append(row)
            continue

        # ── Step 2: Retrieval metrics ─────────────────────────────────────────
        relevant_ids: set[str] = {paper_id} if paper_id else set()
        ret_metrics = compute_retrieval_metrics(retrieved_paper_ids, relevant_ids)
        row["retrieval"]    = ret_metrics.to_dict()
        row["retrieval_ms"] = round(retrieval_ms, 1)

        # ── Step 3: Generation ────────────────────────────────────────────────
        t_gen_start = time.perf_counter()
        answer = ""

        try:
            answer        = gemini_generate(chunk_texts, q["question"], gemini_client, gemini_model)
            generation_ms = (time.perf_counter() - t_gen_start) * 1000
            print(f"         Generated {len(answer.split())} words  ({generation_ms:.0f}ms)")
        except Exception as e:
            generation_ms = (time.perf_counter() - t_gen_start) * 1000
            print(f"         Generation FAILED: {e}")
            row["error"]         = f"generation_failed: {e}"
            row["generation_ms"] = round(generation_ms, 1)
            rows.append(row)
            continue

        row["generation_ms"] = round(generation_ms, 1)
        row["total_ms"]      = round(retrieval_ms + generation_ms, 1)

        # ── Step 4: Answer metrics ────────────────────────────────────────────
        ref            = q.get("reference_answer", "")
        required_facts = q.get("required_facts", [])

        f1  = token_f1(answer, ref)
        fc  = fact_coverage(answer, required_facts)
        grd = grounding_score(answer, chunk_texts)
        missing = [fact for fact in required_facts if fact.lower() not in answer.lower()]

        # ── Step 5: Ollama judge (optional) ───────────────────────────────────
        j_score:  Optional[float] = None
        j_reason: Optional[str]   = None
        if use_judge and ref:
            try:
                j_score, j_reason = llm_judge(q["question"], answer, ref, required_facts)
            except Exception as e:
                print(f"         Judge FAILED (non-fatal): {e}")

        ans_metrics = AnswerMetrics(
            token_f1=      f1,
            fact_coverage= fc,
            grounding=     grd,
            judge_score=   j_score,
            judge_reason=  j_reason,
            missing_facts= missing,
        )

        row["answer"]         = answer[:600]
        row["reference"]      = ref[:300]
        row["answer_metrics"] = ans_metrics.to_dict()
        row["hallucinated"]   = grd < 0.40
        row["n_chunks"]       = len(chunk_texts)

        j_str = f"  Judge={j_score:.0f}/3" if j_score is not None else ""
        print(
            f"         F1={f1:.3f}  FC={fc:.3f}  Grnd={grd:.3f}"
            f"{j_str}  Halluc={'YES' if grd < 0.40 else 'no'}"
        )

        rows.append(row)

    return rows


# ── Aggregation ───────────────────────────────────────────────────────────────

def aggregate(rows: list[dict]) -> dict:
    """Compute aggregate retrieval + answer metrics across all valid rows."""
    valid   = [r for r in rows if "error" not in r]
    errored = [r for r in rows if "error" in r]

    if not valid:
        return {
            "n":        0,
            "n_errors": len(errored),
            "n_total":  len(rows),
            "error":    "all questions failed",
        }

    def _rm(d: dict) -> RetrievalMetrics:
        return RetrievalMetrics(
            recall_at_1    = d.get("recall@1",    0.0),
            recall_at_3    = d.get("recall@3",    0.0),
            recall_at_5    = d.get("recall@5",    0.0),
            recall_at_10   = d.get("recall@10",   0.0),
            precision_at_1 = d.get("precision@1", 0.0),
            precision_at_3 = d.get("precision@3", 0.0),
            precision_at_5 = d.get("precision@5", 0.0),
            mrr            = d.get("mrr",         0.0),
            ndcg_at_10     = d.get("ndcg@10",     0.0),
            hit_at_1       = d.get("hit@1",       0.0),
            hit_at_5       = d.get("hit@5",       0.0),
        )

    def _am(d: dict) -> AnswerMetrics:
        return AnswerMetrics(
            token_f1=      d.get("token_f1",      0.0),
            fact_coverage= d.get("fact_coverage",  0.0),
            grounding=     d.get("grounding",      0.0),
            judge_score=   d.get("judge_score"),
        )

    ret_metrics_list = [_rm(r["retrieval"])      for r in valid if "retrieval"      in r]
    ans_metrics_list = [_am(r["answer_metrics"]) for r in valid if "answer_metrics" in r]
    lat_list = [
        LatencyMetrics(
            retrieval_ms=  r.get("retrieval_ms",  0.0),
            generation_ms= r.get("generation_ms", 0.0),
            total_ms=      r.get("total_ms",      0.0),
        )
        for r in valid
    ]

    agg: dict = {
        "gemini_model": valid[0].get("gemini_model", ""),
        "n":            len(valid),
        "n_errors":     len(errored),
        "n_total":      len(rows),
        "retrieval":    aggregate_retrieval(ret_metrics_list),
        "generation":   aggregate_answers(ans_metrics_list),
        "latency":      aggregate_latency(lat_list),
    }

    # Per-difficulty breakdown
    for diff in ("easy", "medium", "hard"):
        sub = [_am(r["answer_metrics"]) for r in valid
               if r.get("difficulty") == diff and "answer_metrics" in r]
        if sub:
            n = len(sub)
            agg[f"generation_{diff}"] = {
                "n":             n,
                "token_f1":      round(sum(m.token_f1      for m in sub) / n, 4),
                "fact_coverage": round(sum(m.fact_coverage for m in sub) / n, 4),
                "grounding":     round(sum(m.grounding     for m in sub) / n, 4),
            }

    return agg


# ── Results printer ───────────────────────────────────────────────────────────

def print_results(agg: dict, rows: list[dict], result_path: Path) -> None:
    valid   = [r for r in rows if "error" not in r]
    errored = [r for r in rows if "error" in r]

    print("\n" + "="*60)
    print("  END-TO-END EVALUATION RESULTS")
    print("="*60)
    print(f"  Model             : {agg.get('gemini_model', '?')}")
    print(f"  Total questions   : {agg.get('n_total', '?')}")
    print(f"  Successful        : {len(valid)}")
    print(f"  Failed            : {len(errored)}")
    if errored:
        for r in errored:
            print(f"    [FAIL] [{r['id']}] {r.get('error','?')[:70]}")

    ret = agg.get("retrieval", {})
    if ret:
        print(f"\n  RETRIEVAL  (n={ret.get('n','?')})")
        print(f"    Hit@1        : {ret.get('hit@1', '?')}")
        print(f"    Hit@5        : {ret.get('hit@5', '?')}")
        print(f"    Recall@5     : {ret.get('recall@5', '?')}")
        print(f"    Precision@5  : {ret.get('precision@5', '?')}")
        print(f"    MRR          : {ret.get('mrr', '?')}")
        print(f"    nDCG@10      : {ret.get('ndcg@10', '?')}")

    gen = agg.get("generation", {})
    if gen:
        print(f"\n  GENERATION  (n={gen.get('n','?')})")
        print(f"    Token F1         : {gen.get('token_f1', '?')}")
        print(f"    Fact Coverage    : {gen.get('fact_coverage', '?')}")
        print(f"    Grounding        : {gen.get('grounding', '?')}")
        print(f"    Correctness      : {gen.get('correctness', '?')}")
        print(f"    Hallucination %  : {gen.get('hallucination_rate', '?')}")
        print(f"    Grounded %       : {gen.get('grounded_pct', '?')}")

    lat = agg.get("latency", {})
    if lat:
        print(f"\n  LATENCY")
        print(f"    Retrieval mean   : {lat.get('retrieval_mean_ms', '?')} ms")
        print(f"    Generation mean  : {lat.get('generation_mean_ms', '?')} ms")
        print(f"    Total mean       : {lat.get('total_mean_ms', '?')} ms")
        print(f"    Total p95        : {lat.get('total_p95_ms', '?')} ms")

    for diff in ("easy", "medium", "hard"):
        key = f"generation_{diff}"
        if key in agg:
            d = agg[key]
            print(
                f"\n  {diff.upper()} (n={d['n']})"
                f"  F1={d['token_f1']}  FC={d['fact_coverage']}  Grnd={d['grounding']}"
            )

    print(f"\n  Results saved to: {result_path}")
    print("="*60)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="AstroNexusAI end-to-end evaluation (retrieval + Gemini generation)"
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Max questions to evaluate (e.g. 5 for a smoke test)",
    )
    parser.add_argument(
        "--paper", type=str, default=None,
        help="Filter to one paper label: AION-1 | AstroM3 | KnowledgeGraph",
    )
    parser.add_argument(
        "--gemini-model", type=str, default=_DEFAULT_GEMINI_MODEL, dest="gemini_model",
        help=f"Gemini model to use (default: {_DEFAULT_GEMINI_MODEL})",
    )
    parser.add_argument(
        "--top-k", type=int, default=_DEFAULT_TOP_K, dest="top_k",
        help=f"Chunks to retrieve per question (default: {_DEFAULT_TOP_K})",
    )
    parser.add_argument(
        "--no-judge", action="store_true",
        help="Skip the Ollama judge (faster -- uses automated metrics only)",
    )
    args = parser.parse_args()

    questions = load_questions(limit=args.limit, paper_filter=args.paper)
    if not questions:
        print("ERROR: No questions matched the given filters.")
        sys.exit(1)

    pid_map = get_paper_id_map()
    if not pid_map:
        print("ERROR: No papers found in Qdrant. Ingest papers first.")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  AstroNexusAI End-to-End Evaluation")
    print(f"{'='*60}")
    print(f"  Gemini model  : {args.gemini_model}")
    print(f"  Top-K         : {args.top_k}")
    print(f"  Questions     : {len(questions)}")
    print(f"  Judge         : {'Ollama (qwen3:4b)' if not args.no_judge else 'disabled'}")
    print(f"  Paper filter  : {args.paper or 'all'}")
    print(f"{'='*60}")

    rows = run_e2e(
        questions=    questions,
        paper_id_map= pid_map,
        gemini_model= args.gemini_model,
        top_k=        args.top_k,
        use_judge=    not args.no_judge,
    )

    agg = aggregate(rows)

    ts         = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_slug = args.gemini_model.replace("/", "-")
    stem       = f"results_e2e_{model_slug}_{ts}"
    result_path = RESULTS_DIR / f"{stem}.json"

    with open(result_path, "w", encoding="utf-8") as f:
        json.dump({"aggregate": agg, "rows": rows}, f, indent=2, ensure_ascii=False)

    print_results(agg, rows, result_path)


if __name__ == "__main__":
    main()
