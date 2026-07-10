"""
Report Generator for AstroNexus RAG Evaluation.

Aggregates metrics and saves:
    - evaluation_report.csv    (per-question details)
    - evaluation_summary.csv   (aggregated averages)
    - evaluation_summary.txt   (human-readable report)
"""
from __future__ import annotations

import csv
import logging
from pathlib import Path
from statistics import mean

from backend.evaluation.models import (
    EvaluationReport, RetrievalMetrics, GenerationMetrics
)

logger = logging.getLogger(__name__)


def aggregate_metrics(
    retrieval_metrics: list[RetrievalMetrics],
    generation_metrics: list[GenerationMetrics],
    qa_type_map: dict[str, str],   # question_id → answer_type
) -> EvaluationReport:
    """Aggregate per-question metrics into overall report."""

    def avg(values): return mean(values) if values else 0.0

    from collections import Counter
    type_dist = Counter(qa_type_map.values())

    return EvaluationReport(
        total_questions=  len(retrieval_metrics),
        answer_types=     dict(type_dist),

        # Retrieval
        avg_precision_at_1= avg([m.precision_at_k.get(1, 0) for m in retrieval_metrics]),
        avg_precision_at_3= avg([m.precision_at_k.get(3, 0) for m in retrieval_metrics]),
        avg_precision_at_5= avg([m.precision_at_k.get(5, 0) for m in retrieval_metrics]),
        avg_recall_at_1=    avg([m.recall_at_k.get(1, 0)    for m in retrieval_metrics]),
        avg_recall_at_3=    avg([m.recall_at_k.get(3, 0)    for m in retrieval_metrics]),
        avg_recall_at_5=    avg([m.recall_at_k.get(5, 0)    for m in retrieval_metrics]),
        avg_hit_rate=       avg([float(m.hit_rate)           for m in retrieval_metrics]),
        avg_mrr=            avg([m.mrr                       for m in retrieval_metrics]),
        avg_map=            avg([m.map_score                 for m in retrieval_metrics]),
        avg_ndcg=           avg([m.ndcg                      for m in retrieval_metrics]),

        # Generation
        avg_exact_match=       avg([m.exact_match       for m in generation_metrics]),
        avg_f1=                avg([m.f1_score          for m in generation_metrics]),
        avg_rouge_1=           avg([m.rouge_1           for m in generation_metrics]),
        avg_rouge_2=           avg([m.rouge_2           for m in generation_metrics]),
        avg_rouge_l=           avg([m.rouge_l           for m in generation_metrics]),
        avg_bleu=              avg([m.bleu              for m in generation_metrics]),
        avg_answer_relevancy=  avg([m.answer_relevancy  for m in generation_metrics]),
        avg_faithfulness=      avg([m.faithfulness      for m in generation_metrics]),
        avg_context_precision= avg([m.context_precision for m in generation_metrics]),
        avg_context_recall=    avg([m.context_recall    for m in generation_metrics]),

        retrieval_details=  retrieval_metrics,
        generation_details= generation_metrics,
    )


def save_csv(
    retrieval_metrics:  list[RetrievalMetrics],
    generation_metrics: list[GenerationMetrics],
    out_dir: Path,
) -> None:
    """Save per-question metrics to CSV."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # Per-question detail CSV
    detail_path = out_dir / "evaluation_report.csv"
    with open(detail_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "question_id",
            # Retrieval
            "precision@1", "precision@3", "precision@5",
            "recall@1",    "recall@3",    "recall@5",
            "hit_rate", "mrr", "map", "ndcg",
            # Generation
            "exact_match", "f1", "rouge_1", "rouge_2", "rouge_l", "bleu",
            "answer_relevancy", "faithfulness", "context_precision", "context_recall",
        ])

        # Build generation lookup
        gen_map = {m.question_id: m for m in generation_metrics}

        for rm in retrieval_metrics:
            gm = gen_map.get(rm.question_id)
            writer.writerow([
                rm.question_id,
                f"{rm.precision_at_k.get(1,0):.4f}",
                f"{rm.precision_at_k.get(3,0):.4f}",
                f"{rm.precision_at_k.get(5,0):.4f}",
                f"{rm.recall_at_k.get(1,0):.4f}",
                f"{rm.recall_at_k.get(3,0):.4f}",
                f"{rm.recall_at_k.get(5,0):.4f}",
                f"{float(rm.hit_rate):.4f}",
                f"{rm.mrr:.4f}",
                f"{rm.map_score:.4f}",
                f"{rm.ndcg:.4f}",
                f"{gm.exact_match:.4f}"      if gm else "n/a",
                f"{gm.f1_score:.4f}"         if gm else "n/a",
                f"{gm.rouge_1:.4f}"          if gm else "n/a",
                f"{gm.rouge_2:.4f}"          if gm else "n/a",
                f"{gm.rouge_l:.4f}"          if gm else "n/a",
                f"{gm.bleu:.4f}"             if gm else "n/a",
                f"{gm.answer_relevancy:.4f}" if gm else "n/a",
                f"{gm.faithfulness:.4f}"     if gm else "n/a",
                f"{gm.context_precision:.4f}" if gm else "n/a",
                f"{gm.context_recall:.4f}"   if gm else "n/a",
            ])

    logger.info(f"[Report] Per-question CSV → {detail_path}")


def save_summary_csv(report: EvaluationReport, out_dir: Path) -> None:
    """Save aggregated summary to CSV."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "evaluation_summary.csv"

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerows([
            ["total_questions",     report.total_questions],
            ["answer_types",        str(report.answer_types)],
            # Retrieval
            ["precision@1",         f"{report.avg_precision_at_1:.4f}"],
            ["precision@3",         f"{report.avg_precision_at_3:.4f}"],
            ["precision@5",         f"{report.avg_precision_at_5:.4f}"],
            ["recall@1",            f"{report.avg_recall_at_1:.4f}"],
            ["recall@3",            f"{report.avg_recall_at_3:.4f}"],
            ["recall@5",            f"{report.avg_recall_at_5:.4f}"],
            ["hit_rate",            f"{report.avg_hit_rate:.4f}"],
            ["mrr",                 f"{report.avg_mrr:.4f}"],
            ["map",                 f"{report.avg_map:.4f}"],
            ["ndcg",                f"{report.avg_ndcg:.4f}"],
            # Generation
            ["exact_match",         f"{report.avg_exact_match:.4f}"],
            ["f1",                  f"{report.avg_f1:.4f}"],
            ["rouge_1",             f"{report.avg_rouge_1:.4f}"],
            ["rouge_2",             f"{report.avg_rouge_2:.4f}"],
            ["rouge_l",             f"{report.avg_rouge_l:.4f}"],
            ["bleu",                f"{report.avg_bleu:.4f}"],
            ["answer_relevancy",    f"{report.avg_answer_relevancy:.4f}"],
            ["faithfulness",        f"{report.avg_faithfulness:.4f}"],
            ["context_precision",   f"{report.avg_context_precision:.4f}"],
            ["context_recall",      f"{report.avg_context_recall:.4f}"],
        ])

    logger.info(f"[Report] Summary CSV → {path}")


def save_txt_report(report: EvaluationReport, out_dir: Path) -> None:
    """Save human-readable report."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "evaluation_summary.txt"

    with open(path, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("ASTRONEXUS AI — RAG EVALUATION REPORT\n")
        f.write("=" * 60 + "\n\n")

        f.write(f"Total Questions : {report.total_questions}\n")
        f.write(f"Answer Types    : {report.answer_types}\n\n")

        f.write("─" * 60 + "\n")
        f.write("RETRIEVAL METRICS\n")
        f.write("─" * 60 + "\n")
        f.write(f"  Precision@1   : {report.avg_precision_at_1:.4f}\n")
        f.write(f"  Precision@3   : {report.avg_precision_at_3:.4f}\n")
        f.write(f"  Precision@5   : {report.avg_precision_at_5:.4f}\n")
        f.write(f"  Recall@1      : {report.avg_recall_at_1:.4f}\n")
        f.write(f"  Recall@3      : {report.avg_recall_at_3:.4f}\n")
        f.write(f"  Recall@5      : {report.avg_recall_at_5:.4f}\n")
        f.write(f"  Hit Rate      : {report.avg_hit_rate:.4f}\n")
        f.write(f"  MRR           : {report.avg_mrr:.4f}\n")
        f.write(f"  MAP           : {report.avg_map:.4f}\n")
        f.write(f"  nDCG          : {report.avg_ndcg:.4f}\n\n")

        f.write("─" * 60 + "\n")
        f.write("GENERATION METRICS\n")
        f.write("─" * 60 + "\n")
        f.write(f"  Exact Match       : {report.avg_exact_match:.4f}\n")
        f.write(f"  F1 Score          : {report.avg_f1:.4f}\n")
        f.write(f"  ROUGE-1           : {report.avg_rouge_1:.4f}\n")
        f.write(f"  ROUGE-2           : {report.avg_rouge_2:.4f}\n")
        f.write(f"  ROUGE-L           : {report.avg_rouge_l:.4f}\n")
        f.write(f"  BLEU              : {report.avg_bleu:.4f}\n")
        f.write(f"  Answer Relevancy  : {report.avg_answer_relevancy:.4f}\n")
        f.write(f"  Faithfulness      : {report.avg_faithfulness:.4f}\n")
        f.write(f"  Context Precision : {report.avg_context_precision:.4f}\n")
        f.write(f"  Context Recall    : {report.avg_context_recall:.4f}\n\n")

        f.write("─" * 60 + "\n")
        f.write("METRIC INTERPRETATION\n")
        f.write("─" * 60 + "\n")
        f.write("  Hit Rate > 0.7  : Retrieval is finding relevant chunks\n")
        f.write("  MRR    > 0.5  : First relevant chunk usually in top 2\n")
        f.write("  F1     > 0.4  : Reasonable answer overlap\n")
        f.write("  ROUGE-L > 0.3  : Good answer fluency\n")
        f.write("  Faithfulness > 0.7 : Answers grounded in retrieved context\n")
        f.write("=" * 60 + "\n")

    logger.info(f"[Report] Text report → {path}")