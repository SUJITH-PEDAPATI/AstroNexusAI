"""
AstroNexus AI — BM25 Baseline Evaluation

Runs the full evaluation pipeline using BM25 retrieval
and produces a comparison table against dense retrieval.

Run from project root:
    python -m backend.evaluation.run_bm25_evaluation

Requires:
    pip install rank-bm25 numpy
    Qdrant running on localhost:6333 with QASPER papers already indexed.
    (Run run_evaluation.py first to index the papers.)

Outputs:
    output/evaluation/bm25_evaluation_report.csv
    output/evaluation/bm25_evaluation_summary.csv
    output/evaluation/bm25_evaluation_summary.txt
    output/evaluation/retrieval_comparison.csv
"""
from __future__ import annotations

import csv
import logging
import os
from collections import Counter
from pathlib import Path
from statistics import mean

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logging.basicConfig(
    level=  logging.INFO,
    format= "%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

OUT_DIR = Path("output/evaluation")


def retrieve_bm25_for_question(
    qa:    object,   # QAPair
    index_cache: dict,
    top_k: int = 5,
) -> object:
    """
    BM25 retrieval for a single QA pair.

    Builds and caches per-paper BM25 indexes so we don't rebuild
    the same index for every question from the same paper.
    """
    from backend.evaluation.models       import RetrievalResult
    from backend.evaluation.bm25_retriever import build_bm25_index, bm25_retrieve

    # Build index once per paper, cache for subsequent questions
    if qa.paper_id not in index_cache:
        logger.info(f"[BM25] Building index for paper '{qa.paper_id}'")
        index_cache[qa.paper_id] = build_bm25_index(paper_id=qa.paper_id)

    index   = index_cache[qa.paper_id]
    results = bm25_retrieve(index, qa.question, top_k=top_k)

    return RetrievalResult(
        question_id=      qa.question_id,
        question=         qa.question,
        retrieved_chunks= [r.text  for r in results],
        retrieved_scores= [r.score for r in results],
        evidence_chunks=  qa.evidence,
    )


def save_comparison_table(
    dense_report: object,
    bm25_report:  object,
    out_dir:      Path,
) -> None:
    """
    Write side-by-side comparison of dense vs BM25 metrics to CSV.
    This is the table that goes into your research paper.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "retrieval_comparison.csv"

    rows = [
        # Retrieval metrics
        ("Hit Rate",    f"{dense_report.avg_hit_rate:.4f}",        f"{bm25_report.avg_hit_rate:.4f}"),
        ("MRR",         f"{dense_report.avg_mrr:.4f}",             f"{bm25_report.avg_mrr:.4f}"),
        ("MAP",         f"{dense_report.avg_map:.4f}",             f"{bm25_report.avg_map:.4f}"),
        ("nDCG",        f"{dense_report.avg_ndcg:.4f}",            f"{bm25_report.avg_ndcg:.4f}"),
        ("Precision@1", f"{dense_report.avg_precision_at_1:.4f}",  f"{bm25_report.avg_precision_at_1:.4f}"),
        ("Precision@3", f"{dense_report.avg_precision_at_3:.4f}",  f"{bm25_report.avg_precision_at_3:.4f}"),
        ("Precision@5", f"{dense_report.avg_precision_at_5:.4f}",  f"{bm25_report.avg_precision_at_5:.4f}"),
        ("Recall@1",    f"{dense_report.avg_recall_at_1:.4f}",     f"{bm25_report.avg_recall_at_1:.4f}"),
        ("Recall@3",    f"{dense_report.avg_recall_at_3:.4f}",     f"{bm25_report.avg_recall_at_3:.4f}"),
        ("Recall@5",    f"{dense_report.avg_recall_at_5:.4f}",     f"{bm25_report.avg_recall_at_5:.4f}"),
        # Generation metrics
        ("Exact Match", f"{dense_report.avg_exact_match:.4f}",     f"{bm25_report.avg_exact_match:.4f}"),
        ("F1",          f"{dense_report.avg_f1:.4f}",              f"{bm25_report.avg_f1:.4f}"),
        ("ROUGE-1",     f"{dense_report.avg_rouge_1:.4f}",         f"{bm25_report.avg_rouge_1:.4f}"),
        ("ROUGE-2",     f"{dense_report.avg_rouge_2:.4f}",         f"{bm25_report.avg_rouge_2:.4f}"),
        ("ROUGE-L",     f"{dense_report.avg_rouge_l:.4f}",         f"{bm25_report.avg_rouge_l:.4f}"),
        ("BLEU",        f"{dense_report.avg_bleu:.4f}",            f"{bm25_report.avg_bleu:.4f}"),
    ]

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "Dense (BGE-M3)", "BM25 (Okapi)"])
        writer.writerows(rows)

    logger.info(f"[Comparison] Saved → {path}")

    # Print to terminal as well
    print(f"\n{'='*60}")
    print("DENSE vs BM25 COMPARISON")
    print(f"{'='*60}")
    print(f"  {'Metric':<15}  {'Dense':>12}  {'BM25':>12}")
    print(f"  {'-'*15}  {'-'*12}  {'-'*12}")
    for metric, dense_val, bm25_val in rows:
        print(f"  {metric:<15}  {dense_val:>12}  {bm25_val:>12}")
    print(f"{'='*60}")


def load_dense_report() -> object | None:
    """
    Load previously saved dense evaluation summary if it exists.
    Avoids re-running dense evaluation when only BM25 is needed.
    """
    summary_path = OUT_DIR / "evaluation_summary.csv"
    if not summary_path.exists():
        return None

    try:
        import csv
        from backend.evaluation.models import EvaluationReport

        data = {}
        with open(summary_path, encoding="utf-8") as f:
            for row in csv.reader(f):
                if len(row) == 2:
                    data[row[0]] = row[1]

        # Build a minimal EvaluationReport from the saved CSV
        def g(key, default=0.0):
            try:
                return float(data.get(key, default))
            except (ValueError, TypeError):
                return default

        return EvaluationReport(
            total_questions=         int(g("total_questions", 0)),
            answer_types=            {},
            avg_precision_at_1=      g("precision@1"),
            avg_precision_at_3=      g("precision@3"),
            avg_precision_at_5=      g("precision@5"),
            avg_recall_at_1=         g("recall@1"),
            avg_recall_at_3=         g("recall@3"),
            avg_recall_at_5=         g("recall@5"),
            avg_hit_rate=            g("hit_rate"),
            avg_mrr=                 g("mrr"),
            avg_map=                 g("map"),
            avg_ndcg=                g("ndcg"),
            avg_exact_match=         g("exact_match"),
            avg_f1=                  g("f1"),
            avg_rouge_1=             g("rouge_1"),
            avg_rouge_2=             g("rouge_2"),
            avg_rouge_l=             g("rouge_l"),
            avg_bleu=                g("bleu"),
            avg_answer_relevancy=    g("answer_relevancy"),
            avg_faithfulness=        g("faithfulness"),
            avg_context_precision=   g("context_precision"),
            avg_context_recall=      g("context_recall"),
        )
    except Exception as e:
        logger.warning(f"[Dense] Could not load saved report: {e}")
        return None


def run(
    max_papers:    int  = 10,
    max_questions: int  = 30,
    top_k:         int  = 5,
    use_ragas:     bool = False,
) -> None:

    from backend.evaluation.dataset_loader      import load_qasper
    from backend.evaluation.retrieval_evaluator import evaluate_retrieval
    from backend.evaluation.generation_evaluator import evaluate_generation
    from backend.evaluation.answer_generator    import generate_answer
    from backend.evaluation.report_generator    import (
        aggregate_metrics, save_csv, save_summary_csv, save_txt_report,
    )
    from backend.evaluation.models import GenerationResult

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 1: Load QASPER (same subset as dense evaluation)
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"[Step 1] Loading QASPER dataset")
    print(f"{'='*60}")

    qa_pairs, paper_texts = load_qasper(
        split="validation", max_papers=max_papers
    )
    qa_pairs = qa_pairs[:max_questions]
    print(f"  Papers   : {len(paper_texts)}")
    print(f"  QA pairs : {len(qa_pairs)}")
    print(f"  Types    : {dict(Counter(qa.answer_type for qa in qa_pairs))}")

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 2: BM25 retrieval evaluation
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"[Step 2] BM25 retrieval evaluation — {len(qa_pairs)} questions")
    print(f"{'='*60}")

    index_cache            = {}   # paper_id → BM25Index
    retrieval_metrics_list = []

    for i, qa in enumerate(qa_pairs, start=1):
        try:
            result  = retrieve_bm25_for_question(qa, index_cache, top_k=top_k)
            metrics = evaluate_retrieval(result)
            retrieval_metrics_list.append(metrics)

            if i % 10 == 0 or i == len(qa_pairs):
                hit = mean(float(m.hit_rate) for m in retrieval_metrics_list)
                mrr = mean(m.mrr            for m in retrieval_metrics_list)
                print(f"  [{i:>3}/{len(qa_pairs)}]  Hit Rate={hit:.3f}  MRR={mrr:.3f}")

        except Exception as e:
            logger.warning(f"  BM25 retrieval failed for {qa.question_id}: {e}")

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 3: Generation evaluation (same generator, different context)
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"[Step 3] Generation evaluation — {len(qa_pairs)} questions")
    print(f"{'='*60}")

    generation_metrics_list = []

    for i, qa in enumerate(qa_pairs, start=1):
        try:
            # Use BM25 retrieved chunks as context for generation
            bm25_result = retrieve_bm25_for_question(qa, index_cache, top_k=top_k)
            contexts    = bm25_result.retrieved_chunks

            if not contexts:
                gen_result = GenerationResult(
                    question_id=     qa.question_id,
                    question=        qa.question,
                    generated_answer="[NO_CONTEXT_RETRIEVED]",
                    ground_truth=    qa.ground_truth,
                    context_used=    [],
                )
            else:
                # Build context and generate via the same generator
                from backend.evaluation.answer_generator import _generate, _QA_PROMPT
                context_str = "\n\n---\n\n".join(
                    f"[Chunk {j+1}]\n{c}" for j, c in enumerate(contexts)
                )
                prompt    = _QA_PROMPT.format(
                    context=context_str, question=qa.question
                )
                generated = _generate(prompt)

                gen_result = GenerationResult(
                    question_id=     qa.question_id,
                    question=        qa.question,
                    generated_answer=generated,
                    ground_truth=    qa.ground_truth,
                    context_used=    contexts,
                )

            gen_metrics = evaluate_generation(gen_result, use_ragas=use_ragas)
            generation_metrics_list.append(gen_metrics)

            if i % 10 == 0 or i == len(qa_pairs):
                f1    = mean(m.f1_score for m in generation_metrics_list)
                rouge = mean(m.rouge_l  for m in generation_metrics_list)
                print(f"  [{i:>3}/{len(qa_pairs)}]  F1={f1:.3f}  ROUGE-L={rouge:.3f}")

        except Exception as e:
            logger.warning(f"  BM25 generation failed for {qa.question_id}: {e}")

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 4: Aggregate BM25 metrics
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"[Step 4] Aggregating BM25 metrics")
    print(f"{'='*60}")

    qa_type_map  = {qa.question_id: qa.answer_type for qa in qa_pairs}
    bm25_report  = aggregate_metrics(
        retrieval_metrics_list,
        generation_metrics_list,
        qa_type_map,
    )

    # Save BM25-specific reports
    # Save BM25-specific reports
    bm25_out = OUT_DIR

    save_csv(
        retrieval_metrics_list,
        generation_metrics_list,
        bm25_out,
        filename="bm25_evaluation_report.csv",
    )

    save_summary_csv(
        bm25_report,
        bm25_out,
        filename="bm25_evaluation_summary.csv",
    )

    save_txt_report(
        bm25_report,
        bm25_out,
        filename="bm25_evaluation_summary.txt",
    )

    logger.info(f"  Saved → {bm25_out / 'bm25_evaluation_report.csv'}")
    logger.info(f"  Saved → {bm25_out / 'bm25_evaluation_summary.csv'}")
    logger.info(f"  Saved → {bm25_out / 'bm25_evaluation_summary.txt'}")
    # ══════════════════════════════════════════════════════════════════════════
    # STEP 5: Load dense results and generate comparison table
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"[Step 5] Generating comparison table")
    print(f"{'='*60}")

    dense_report = load_dense_report()

    if dense_report is None:
        print(
            "\n  ⚠ Dense evaluation summary not found at "
            "output/evaluation/evaluation_summary.csv\n"
            "  Run run_evaluation.py first, then re-run this script.\n"
            "  BM25 results saved; comparison table skipped."
        )
    else:
        save_comparison_table(dense_report, bm25_report, OUT_DIR)
        print(f"\n  ✓ output/evaluation/retrieval_comparison.csv")

    # ══════════════════════════════════════════════════════════════════════════
    # FINAL SUMMARY
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("BM25 EVALUATION COMPLETE")
    print(f"{'='*60}")
    print(f"  Questions  : {bm25_report.total_questions}")
    print()
    print("  RETRIEVAL")
    print(f"    Hit Rate    : {bm25_report.avg_hit_rate:.4f}")
    print(f"    MRR         : {bm25_report.avg_mrr:.4f}")
    print(f"    MAP         : {bm25_report.avg_map:.4f}")
    print(f"    nDCG        : {bm25_report.avg_ndcg:.4f}")
    print(f"    Precision@3 : {bm25_report.avg_precision_at_3:.4f}")
    print(f"    Recall@5    : {bm25_report.avg_recall_at_5:.4f}")
    print()
    print("  GENERATION")
    print(f"    Exact Match : {bm25_report.avg_exact_match:.4f}")
    print(f"    F1          : {bm25_report.avg_f1:.4f}")
    print(f"    ROUGE-L     : {bm25_report.avg_rouge_l:.4f}")
    print(f"    BLEU        : {bm25_report.avg_bleu:.4f}")
    print()
    print("  OUTPUT FILES")
    print(f"    ✓ output/evaluation/bm25_evaluation_report.csv")
    print(f"    ✓ output/evaluation/bm25_evaluation_summary.csv")
    print(f"    ✓ output/evaluation/bm25_evaluation_summary.txt")
    if dense_report:
        print(f"    ✓ output/evaluation/retrieval_comparison.csv")
    print()


if __name__ == "__main__":
    run(
        max_papers=    10,
        max_questions= 30,
        top_k=         5,
        use_ragas=     False,
    )