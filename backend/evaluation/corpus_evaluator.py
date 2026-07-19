"""
AstroNexus AI — Corpus-Level Evaluator

Computes paper-level retrieval metrics that measure whether the system
correctly identified the right paper from the full corpus — not just
whether it found the right chunk within a pre-selected paper.

Paper-level metrics:
    Top-1 Paper Accuracy  — was the first retrieved chunk from the correct paper?
    Paper Hit Rate        — did any top-k chunk come from the correct paper?
    Paper MRR             — reciprocal rank of first correct-paper chunk
    Paper Recall@k        — fraction of relevant papers found in top-k
                            (for QASPER each question has exactly one correct paper
                             so this equals Paper Hit Rate, but the function
                             supports multi-paper ground truth for future use)

These metrics do not exist in the single-paper evaluator and are only
meaningful in corpus-level retrieval.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from statistics import mean
from typing import Optional

logger = logging.getLogger(__name__)


# ── Paper-level result dataclass ───────────────────────────────────────────────

@dataclass
class PaperRetrievalMetrics:
    """Paper-level retrieval metrics for one QA pair."""
    question_id:         str
    correct_paper_id:    str
    retrieved_paper_ids: list[str]          # paper_id of each retrieved chunk, in rank order

    top1_paper_accuracy: float              # 1.0 if rank-1 chunk is from correct paper
    paper_hit_rate:      bool               # True if any chunk is from correct paper
    paper_mrr:           float              # 1/rank of first correct-paper chunk
    paper_recall_at_k:   dict[int, float]   # {1: x, 3: x, 5: x}


@dataclass
class CorpusPaperReport:
    """Aggregated paper-level metrics across all QA pairs."""
    total_questions:         int
    avg_top1_paper_accuracy: float
    avg_paper_hit_rate:      float
    avg_paper_mrr:           float
    avg_paper_recall_at_1:   float
    avg_paper_recall_at_3:   float
    avg_paper_recall_at_5:   float
    per_question:            list[PaperRetrievalMetrics] = field(default_factory=list)


# ── Metric functions ───────────────────────────────────────────────────────────

def compute_top1_paper_accuracy(
    retrieved_paper_ids: list[str],
    correct_paper_id:    str,
) -> float:
    """
    Top-1 Paper Accuracy = 1 if the first retrieved chunk is from the correct paper.

    This is the strictest paper-level metric. It requires not just finding
    the right paper but ranking a chunk from it first.
    Range: {0.0, 1.0}
    """
    if not retrieved_paper_ids:
        return 0.0
    return 1.0 if retrieved_paper_ids[0] == correct_paper_id else 0.0


def compute_paper_hit_rate(
    retrieved_paper_ids: list[str],
    correct_paper_id:    str,
) -> bool:
    """
    Paper Hit Rate = True if any retrieved chunk is from the correct paper.

    This is the most lenient paper-level metric. Even if the correct paper
    appears at rank 5, this scores True.
    Range: {True, False}
    """
    return correct_paper_id in retrieved_paper_ids


def compute_paper_mrr(
    retrieved_paper_ids: list[str],
    correct_paper_id:    str,
) -> float:
    """
    Paper MRR = 1 / rank of first chunk from the correct paper.

    MRR=1.0 → first chunk is from the correct paper.
    MRR=0.5 → second chunk is first from the correct paper.
    MRR=0.0 → no chunk from the correct paper retrieved.
    Range: [0, 1]
    """
    for rank, paper_id in enumerate(retrieved_paper_ids, start=1):
        if paper_id == correct_paper_id:
            return 1.0 / rank
    return 0.0


def compute_paper_recall_at_k(
    retrieved_paper_ids: list[str],
    correct_paper_ids:   list[str],
    k:                   int,
) -> float:
    """
    Paper Recall@K = (correct papers found in top-k) / (total correct papers)

    For QASPER each question has one correct paper, so this equals
    Paper Hit Rate when len(correct_paper_ids)==1. The function supports
    multi-paper ground truth for future extension.

    Range: [0, 1]
    """
    if not retrieved_paper_ids or not correct_paper_ids:
        return 0.0
    top_k   = set(retrieved_paper_ids[:k])
    correct = set(correct_paper_ids)
    found   = top_k & correct
    return min(len(found) / len(correct), 1.0)


def evaluate_paper_retrieval(
    question_id:         str,
    correct_paper_id:    str,
    retrieved_paper_ids: list[str],
    k_values:            list[int] = [1, 3, 5],
) -> PaperRetrievalMetrics:
    """
    Compute all paper-level metrics for a single QA pair.

    Args:
        question_id:         QA pair identifier
        correct_paper_id:    The paper_id that contains the answer
        retrieved_paper_ids: paper_id of each retrieved chunk, in rank order

    Returns:
        PaperRetrievalMetrics with all paper-level scores
    """
    return PaperRetrievalMetrics(
        question_id=         question_id,
        correct_paper_id=    correct_paper_id,
        retrieved_paper_ids= retrieved_paper_ids,
        top1_paper_accuracy= compute_top1_paper_accuracy(retrieved_paper_ids, correct_paper_id),
        paper_hit_rate=      compute_paper_hit_rate(retrieved_paper_ids, correct_paper_id),
        paper_mrr=           compute_paper_mrr(retrieved_paper_ids, correct_paper_id),
        paper_recall_at_k=   {
            k: compute_paper_recall_at_k(retrieved_paper_ids, [correct_paper_id], k)
            for k in k_values
        },
    )


def aggregate_paper_metrics(
    metrics: list[PaperRetrievalMetrics],
) -> CorpusPaperReport:
    """
    Aggregate per-question paper metrics into overall corpus report.

    Args:
        metrics: list of PaperRetrievalMetrics, one per QA pair

    Returns:
        CorpusPaperReport with averaged scores
    """
    if not metrics:
        return CorpusPaperReport(
            total_questions=         0,
            avg_top1_paper_accuracy= 0.0,
            avg_paper_hit_rate=      0.0,
            avg_paper_mrr=           0.0,
            avg_paper_recall_at_1=   0.0,
            avg_paper_recall_at_3=   0.0,
            avg_paper_recall_at_5=   0.0,
        )

    return CorpusPaperReport(
        total_questions=         len(metrics),
        avg_top1_paper_accuracy= mean(m.top1_paper_accuracy       for m in metrics),
        avg_paper_hit_rate=      mean(float(m.paper_hit_rate)      for m in metrics),
        avg_paper_mrr=           mean(m.paper_mrr                  for m in metrics),
        avg_paper_recall_at_1=   mean(m.paper_recall_at_k.get(1,0) for m in metrics),
        avg_paper_recall_at_3=   mean(m.paper_recall_at_k.get(3,0) for m in metrics),
        avg_paper_recall_at_5=   mean(m.paper_recall_at_k.get(5,0) for m in metrics),
        per_question=            metrics,
    )


def save_paper_metrics_csv(
    metrics:  list[PaperRetrievalMetrics],
    out_path: str = "output/evaluation/paper_metrics.csv",
) -> None:
    """Save per-question paper metrics to CSV."""
    import csv
    from pathlib import Path

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "question_id", "correct_paper_id",
            "top1_paper_accuracy", "paper_hit_rate", "paper_mrr",
            "paper_recall@1", "paper_recall@3", "paper_recall@5",
            "retrieved_paper_ids",
        ])
        for m in metrics:
            writer.writerow([
                m.question_id,
                m.correct_paper_id,
                f"{m.top1_paper_accuracy:.4f}",
                f"{float(m.paper_hit_rate):.4f}",
                f"{m.paper_mrr:.4f}",
                f"{m.paper_recall_at_k.get(1,0):.4f}",
                f"{m.paper_recall_at_k.get(3,0):.4f}",
                f"{m.paper_recall_at_k.get(5,0):.4f}",
                "|".join(m.retrieved_paper_ids),
            ])

    logger.info(f"[CorpusEval] Paper metrics → {out_path}")