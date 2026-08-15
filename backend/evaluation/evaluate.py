"""
AstroNexusAI — Full Evaluation Pipeline
=========================================
Measures retrieval + answer quality + latency for the complete
AstroNexusAI pipeline on a set of domain-specific questions.

Usage:
    # From project root (Qdrant + Neo4j + Ollama must be running):
    python -m evaluation.evaluate

    # Discover paper_ids from your Qdrant instance first:
    python -m evaluation.evaluate --discover-papers

    # Retrieval only (no LLM calls, fast):
    python -m evaluation.evaluate --retrieval-only

    # Skip LLM judge (automated metrics only):
    python -m evaluation.evaluate --no-judge

    # Run only specific papers:
    python -m evaluation.evaluate --papers AION-1 KnowledgeGraph

    # Limit questions for a quick smoke test:
    python -m evaluation.evaluate --limit 5

Output:
    evaluation/results/run_<timestamp>.json        complete results
    evaluation/results/run_<timestamp>_summary.txt human-readable summary
    evaluation/results/latex_table.tex             IEEE-ready LaTeX table
    evaluation/results/plots/                      PNG figures

All results are also printed to stdout as they arrive so you can
watch progress in real time.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

# ── Path setup ────────────────────────────────────────────────────────────────
# Add both backend/ and project root so 'from backend.rag.*' imports work
# whether invoked as:
#   python -m evaluation.evaluate          (from backend/)
#   python -m backend.evaluation.evaluate  (from project root)
_PROJECT      = Path(__file__).resolve().parent.parent   # backend/
_PROJECT_ROOT = _PROJECT.parent                           # project root
for _p in [str(_PROJECT), str(_PROJECT_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Windows cp1252 cannot encode Unicode chars — reconfigure to UTF-8
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

try:
    from evaluation.metrics import (
        RetrievalMetrics, AnswerMetrics, LatencyMetrics,
        hit_at_k, recall_at_k, precision_at_k, mean_reciprocal_rank, ndcg_at_k,
        token_f1, fact_coverage, grounding_score,
        aggregate_retrieval, aggregate_answers, aggregate_latency,
    )
except ModuleNotFoundError:
    from backend.evaluation.metrics import (  # type: ignore[no-redef]
        RetrievalMetrics, AnswerMetrics, LatencyMetrics,
        hit_at_k, recall_at_k, precision_at_k, mean_reciprocal_rank, ndcg_at_k,
        token_f1, fact_coverage, grounding_score,
        aggregate_retrieval, aggregate_answers, aggregate_latency,
    )

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("evaluation")

OUT_DIR   = _PROJECT / "evaluation" / "results"
CACHE_DIR = _PROJECT / "evaluation" / "cache"
DATASET   = _PROJECT / "evaluation" / "dataset.jsonl"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class Question:
    id:               str
    paper_label:      str
    question:         str
    required_facts:   list[str]
    reference_answer: str
    expected_sections: list[str]
    difficulty:       str
    cross_paper:      bool

@dataclass
class QuestionResult:
    question_id:     str
    paper_label:     str
    difficulty:      str
    cross_paper:     bool
    query:           str
    retrieved_ids:   list[str]   = field(default_factory=list)
    retrieved_scores: list[float] = field(default_factory=list)
    retrieved_texts: list[str]   = field(default_factory=list)
    answer:          str          = ""
    retrieval:       Optional[dict] = None
    answer_metrics:  Optional[dict] = None
    latency:         Optional[dict] = None
    error:           Optional[str]  = None

# ── Dataset loader ────────────────────────────────────────────────────────────

def load_dataset(path: Path, paper_filter: list[str] | None, limit: int | None) -> list[Question]:
    questions = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            q = Question(
                id=               d["id"],
                paper_label=      d["paper_label"],
                question=         d["question"],
                required_facts=   d.get("required_facts", []),
                reference_answer= d.get("reference_answer", ""),
                expected_sections= d.get("expected_sections", []),
                difficulty=       d.get("difficulty", "medium"),
                cross_paper=      d.get("cross_paper", False),
            )
            if paper_filter and q.paper_label not in paper_filter and q.paper_label != "cross":
                continue
            questions.append(q)
    if limit:
        questions = questions[:limit]
    return questions

# ── Paper ID discovery ────────────────────────────────────────────────────────

def discover_paper_ids() -> dict[str, dict]:
    """
    Query Qdrant and print all paper_ids with their titles and chunk counts.
    Run this first to map paper_label → paper_id before evaluation.
    """
    from backend.rag.vector_store import _get_client, COLLECTION_NAME
    client = _get_client()

    results, _ = client.scroll(
        collection_name=COLLECTION_NAME,
        limit=1000,
        with_payload=True,
        with_vectors=False,
    )
    papers: dict[str, dict] = {}
    for r in results:
        p = r.payload or {}
        pid = p.get("paper_id", "")
        if not pid:
            continue
        if pid not in papers:
            papers[pid] = {
                "paper_id": pid,
                "title":    p.get("title", "Unknown"),
                "authors":  p.get("authors", []),
                "year":     p.get("year"),
                "chunks":   0,
                "sections": set(),
            }
        papers[pid]["chunks"] += 1
        sec = p.get("section")
        if sec:
            papers[pid]["sections"].add(sec)

    print(f"\nFound {len(papers)} papers in Qdrant:\n")
    for info in papers.values():
        print(f"  paper_id : {info['paper_id']}")
        print(f"  title    : {info['title']}")
        print(f"  authors  : {info['authors'][:3]}")
        print(f"  chunks   : {info['chunks']}")
        print(f"  sections : {sorted(info['sections'])}")
        print()
    return papers

# ── Retrieval runner ──────────────────────────────────────────────────────────

def run_retrieval(
    query: str,
    paper_id: str | None,
    top_k: int = 10,
) -> tuple[list, float]:
    """
    Run Qdrant retrieval. Returns (chunks, elapsed_ms).
    """
    from backend.rag.retriever import retrieve
    t0 = time.perf_counter()
    chunks = retrieve(
        query=           query,
        top_k=           top_k,
        score_threshold= 0.0,   # 0 during eval — let k control cut-off
        filter_paper_id= paper_id,
    )
    elapsed = (time.perf_counter() - t0) * 1000
    return chunks, elapsed

# ── Answer runner ─────────────────────────────────────────────────────────────

def run_full_pipeline(
    query:    str,
    paper_id: str | None,
) -> tuple[str, float]:
    """
    Run the real orchestrator pipeline. Returns (answer, elapsed_ms).
    """
    from backend.agents.orchestrator import run
    t0 = time.perf_counter()
    result = run(
        query=        query,
        paper_loaded= bool(paper_id),
        paper_id=     paper_id,
    )
    elapsed = (time.perf_counter() - t0) * 1000
    return result.get("final_answer", ""), elapsed

# ── Main evaluation loop ──────────────────────────────────────────────────────

def evaluate(
    paper_id_map:    dict[str, str],   # paper_label → paper_id in Qdrant
    questions:       list[Question],
    retrieval_only:  bool = False,
    use_judge:       bool = True,
) -> list[QuestionResult]:
    results = []
    n = len(questions)

    for i, q in enumerate(questions, 1):
        print(f"\n[{i}/{n}] {q.id} | {q.difficulty} | {q.question[:70]}...")

        # Resolve paper_id from label
        paper_id = paper_id_map.get(q.paper_label) if q.paper_label != "cross" else None

        result = QuestionResult(
            question_id= q.id,
            paper_label= q.paper_label,
            difficulty=  q.difficulty,
            cross_paper= q.cross_paper,
            query=       q.question,
        )

        # ── Retrieval ─────────────────────────────────────────────────────────
        try:
            chunks, ret_ms = run_retrieval(q.question, paper_id, top_k=10)
            retrieved_ids    = [c.paper_id  for c in chunks]
            retrieved_scores = [c.score     for c in chunks]
            retrieved_texts  = [c.text      for c in chunks]

            result.retrieved_ids    = retrieved_ids
            result.retrieved_scores = retrieved_scores
            result.retrieved_texts  = retrieved_texts

            # Relevant set: any paper_id from the same paper label
            if q.cross_paper:
                # For cross-paper queries, all ingested papers are relevant
                relevant = set(paper_id_map.values())
            else:
                relevant = {paper_id} if paper_id else set()

            rm = RetrievalMetrics(
                recall_at_1=   recall_at_k(retrieved_ids, relevant, 1),
                recall_at_3=   recall_at_k(retrieved_ids, relevant, 3),
                recall_at_5=   recall_at_k(retrieved_ids, relevant, 5),
                recall_at_10=  recall_at_k(retrieved_ids, relevant, 10),
                precision_at_1= precision_at_k(retrieved_ids, relevant, 1),
                precision_at_3= precision_at_k(retrieved_ids, relevant, 3),
                precision_at_5= precision_at_k(retrieved_ids, relevant, 5),
                mrr=           mean_reciprocal_rank(retrieved_ids, relevant),
                ndcg_at_10=    ndcg_at_k(retrieved_ids, relevant, 10),
                hit_at_1=      hit_at_k(retrieved_ids, relevant, 1),
                hit_at_5=      hit_at_k(retrieved_ids, relevant, 5),
            )
            result.retrieval = rm.to_dict()

            print(f"  Retrieval: {len(chunks)} chunks | "
                  f"Recall@5={rm.recall_at_5:.2f} | "
                  f"MRR={rm.mrr:.2f} | "
                  f"{ret_ms:.0f}ms")

        except Exception as e:
            logger.error(f"[{q.id}] Retrieval failed: {e}")
            result.error = f"Retrieval: {e}"
            results.append(result)
            continue

        if retrieval_only:
            results.append(result)
            continue

        # ── Full pipeline (generation) ─────────────────────────────────────────
        try:
            answer, gen_ms = run_full_pipeline(q.question, paper_id)
            result.answer  = answer

            # Answer metrics
            am = AnswerMetrics(
                token_f1=      token_f1(answer, q.reference_answer),
                fact_coverage= fact_coverage(answer, q.required_facts),
                grounding=     grounding_score(answer, retrieved_texts),
                missing_facts= [f for f in q.required_facts if f.lower() not in answer.lower()],
            )

            # LLM judge
            if use_judge and q.reference_answer:
                from evaluation.judge import judge as llm_judge
                j_score, j_reason = llm_judge(
                    question=         q.question,
                    answer=           answer,
                    reference_answer= q.reference_answer,
                    required_facts=   q.required_facts,
                )
                am.judge_score  = j_score
                am.judge_reason = j_reason

            result.answer_metrics = am.to_dict()
            result.latency = asdict(LatencyMetrics(
                retrieval_ms=  ret_ms,
                generation_ms= gen_ms,
                total_ms=      ret_ms + gen_ms,
            ))

            print(f"  Answer:    F1={am.token_f1:.2f} | "
                  f"Facts={am.fact_coverage:.2f} | "
                  f"Grnd={am.grounding:.2f} | "
                  + (f"Judge={am.judge_score:.0f}/3 | " if am.judge_score is not None else "")
                  + f"{gen_ms:.0f}ms gen")

        except Exception as e:
            logger.error(f"[{q.id}] Generation failed: {e}")
            result.error = f"Generation: {e}"

        results.append(result)

    return results

# ── Report generation ─────────────────────────────────────────────────────────

def build_report(results: list[QuestionResult], paper_id_map: dict[str, str]) -> dict:
    valid_ret = [r for r in results if r.retrieval]
    valid_ans = [r for r in results if r.answer_metrics]

    ret_metrics_list  = [RetrievalMetrics(**r.retrieval)   for r in valid_ret]
    ans_metrics_list  = [AnswerMetrics(**{k: v for k, v in r.answer_metrics.items()
                                         if k not in ("correctness",)})
                         for r in valid_ans]
    lat_metrics_list  = [LatencyMetrics(**r.latency)
                         for r in results if r.latency]

    # Breakdown by difficulty
    breakdown = {}
    for diff in ["easy", "medium", "hard"]:
        sub = [r for r in valid_ans if r.difficulty == diff]
        if sub:
            breakdown[diff] = {
                "n": len(sub),
                "token_f1":      round(sum(r.answer_metrics["token_f1"]  for r in sub) / len(sub), 4),
                "fact_coverage": round(sum(r.answer_metrics["fact_coverage"] for r in sub) / len(sub), 4),
                "grounding":     round(sum(r.answer_metrics["grounding"]  for r in sub) / len(sub), 4),
            }

    return {
        "timestamp":     datetime.now().isoformat(),
        "n_total":       len(results),
        "n_errors":      sum(1 for r in results if r.error),
        "paper_id_map":  paper_id_map,
        "retrieval":     aggregate_retrieval(ret_metrics_list),
        "answers":       aggregate_answers(ans_metrics_list) if ans_metrics_list else {},
        "latency":       aggregate_latency(lat_metrics_list) if lat_metrics_list else {},
        "breakdown_by_difficulty": breakdown,
    }


def write_summary(report: dict, path: Path) -> None:
    r  = report["retrieval"]
    a  = report.get("answers", {})
    l  = report.get("latency", {})
    bd = report.get("breakdown_by_difficulty", {})

    lines = [
        "=" * 60,
        "AstroNexusAI Evaluation Summary",
        f"Timestamp : {report['timestamp']}",
        f"Questions : {report['n_total']} total  |  {report['n_errors']} errors",
        "=" * 60,
        "",
        "RETRIEVAL METRICS",
        "-" * 40,
        f"  Recall@1    : {r.get('recall@1', '—')}",
        f"  Recall@3    : {r.get('recall@3', '—')}",
        f"  Recall@5    : {r.get('recall@5', '—')}",
        f"  Recall@10   : {r.get('recall@10','—')}",
        f"  Precision@5 : {r.get('precision@5','—')}",
        f"  MRR         : {r.get('mrr', '—')}",
        f"  nDCG@10     : {r.get('ndcg@10','—')}",
        f"  Hit@1       : {r.get('hit@1', '—')}",
        f"  Hit@5       : {r.get('hit@5', '—')}",
    ]

    if a:
        lines += [
            "",
            "ANSWER QUALITY METRICS",
            "-" * 40,
            f"  Token F1        : {a.get('token_f1', '—')}",
            f"  Fact Coverage   : {a.get('fact_coverage', '—')}",
            f"  Grounding       : {a.get('grounding', '—')}",
            f"  Correctness     : {a.get('correctness', '—')}",
            f"  Hallucination % : {a.get('hallucination_rate', '—')}",
        ]

    if l:
        lines += [
            "",
            "LATENCY",
            "-" * 40,
            f"  Retrieval (mean)  : {l.get('retrieval_mean_ms', '—')} ms",
            f"  Generation (mean) : {l.get('generation_mean_ms', '—')} ms",
            f"  Total (mean)      : {l.get('total_mean_ms', '—')} ms",
            f"  Total (median)    : {l.get('total_median_ms', '—')} ms",
            f"  Total (p95)       : {l.get('total_p95_ms', '—')} ms",
        ]

    if bd:
        lines += ["", "BREAKDOWN BY DIFFICULTY", "-" * 40]
        for diff, m in bd.items():
            lines.append(
                f"  {diff:<8}: n={m['n']}  "
                f"F1={m['token_f1']:.3f}  "
                f"Facts={m['fact_coverage']:.3f}  "
                f"Grnd={m['grounding']:.3f}"
            )

    lines += ["", "=" * 60]
    summary = "\n".join(lines)
    print("\n" + summary)
    path.write_text(summary)


def write_latex(report: dict, path: Path) -> None:
    """
    Write a single IEEE-ready table. Fill in baseline columns manually
    after running the corresponding baseline evaluations.
    """
    r = report["retrieval"]
    a = report.get("answers", {})
    l = report.get("latency", {})

    lines = [
        r"\begin{table*}[t]",
        r"\caption{AstroNexusAI System Evaluation Results}",
        r"\label{tab:eval}",
        r"\centering",
        r"\renewcommand{\arraystretch}{1.25}",
        r"\begin{tabular}{@{}lcccccccc@{}}",
        r"\toprule",
        r"\textbf{Metric} & \textbf{LLM-only} & \textbf{Dense RAG} & \textbf{KG-only} & \textbf{AstroNexus} \\",
        r"\midrule",
        rf"Recall@5         & [FILL] & [FILL] & [FILL] & \textbf{{{r.get('recall@5','?')}}} \\",
        rf"MRR              & [FILL] & [FILL] & [FILL] & \textbf{{{r.get('mrr','?')}}} \\",
        rf"nDCG@10          & [FILL] & [FILL] & [FILL] & \textbf{{{r.get('ndcg@10','?')}}} \\",
        rf"Token F1         & [FILL] & [FILL] & [FILL] & \textbf{{{a.get('token_f1','?')}}} \\",
        rf"Fact Coverage    & [FILL] & [FILL] & [FILL] & \textbf{{{a.get('fact_coverage','?')}}} \\",
        rf"Grounding        & [FILL] & [FILL] & [FILL] & \textbf{{{a.get('grounding','?')}}} \\",
        rf"Hallucination ↓  & [FILL] & [FILL] & [FILL] & \textbf{{{a.get('hallucination_rate','?')}}} \\",
        rf"Latency (ms) ↓   & [FILL] & [FILL] & [FILL] & \textbf{{{l.get('total_mean_ms','?')}}} \\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
    ]
    path.write_text("\n".join(lines))
    print(f"\nLaTeX table written to: {path}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="AstroNexusAI Evaluation Pipeline")
    parser.add_argument("--discover-papers", action="store_true",
                        help="Print all paper_ids from Qdrant and exit")
    parser.add_argument("--retrieval-only",  action="store_true",
                        help="Only run retrieval — skip LLM generation")
    parser.add_argument("--no-judge",        action="store_true",
                        help="Skip LLM judge, use automated metrics only")
    parser.add_argument("--papers",          nargs="+", default=None,
                        help="Only evaluate these paper labels")
    parser.add_argument("--limit",           type=int, default=None,
                        help="Max number of questions to evaluate")
    parser.add_argument("--paper-id-map",    type=str, default=None,
                        help='JSON string mapping label→paper_id, e.g. \'{"AION-1":"abc123"}\'')
    args = parser.parse_args()

    # Discover paper_ids
    if args.discover_papers:
        discover_paper_ids()
        return

    # Build paper_id_map
    # Option 1: pass via CLI   --paper-id-map '{"AION-1":"abc123",...}'
    # Option 2: auto-discover from Qdrant
    if args.paper_id_map:
        paper_id_map = json.loads(args.paper_id_map)
    else:
        print("\nAuto-discovering paper_ids from Qdrant...")
        raw = discover_paper_ids()
        # Map by title keyword matching against labels
        paper_id_map: dict[str, str] = {}
        for pid, info in raw.items():
            title_lower = info["title"].lower()
            if "aion" in title_lower:
                paper_id_map["AION-1"] = pid
            elif "astrom3" in title_lower or "astro-m3" in title_lower:
                paper_id_map["AstroM3"] = pid
            elif "knowledge" in title_lower or "graph" in title_lower:
                paper_id_map["KnowledgeGraph"] = pid
            else:
                # Use first word of title as label
                label = info["title"].split()[0] if info["title"] else pid[:8]
                paper_id_map[label] = pid

    print(f"\nPaper ID map: {json.dumps(paper_id_map, indent=2)}")

    if not paper_id_map:
        print("ERROR: No papers found in Qdrant. Ingest papers first.")
        print("       POST /ingest via the research page or curl.")
        sys.exit(1)

    # Load questions
    questions = load_dataset(
        DATASET,
        paper_filter=args.papers,
        limit=args.limit,
    )
    print(f"\nLoaded {len(questions)} questions")

    # Run evaluation
    print(f"\nRunning evaluation: retrieval_only={args.retrieval_only} "
          f"use_judge={not args.no_judge}")
    all_results = evaluate(
        paper_id_map=   paper_id_map,
        questions=       questions,
        retrieval_only=  args.retrieval_only,
        use_judge=       not args.no_judge,
    )

    # Build and save report
    report = build_report(all_results, paper_id_map)

    ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = OUT_DIR

    results_path = run_dir / f"run_{ts}.json"
    with open(results_path, "w") as f:
        json.dump({
            "report": report,
            "results": [asdict(r) for r in all_results],
        }, f, indent=2, default=list)
    print(f"\nFull results saved to: {results_path}")

    write_summary(report, run_dir / f"run_{ts}_summary.txt")
    write_latex(report, run_dir / "latex_table.tex")


if __name__ == "__main__":
    main()
