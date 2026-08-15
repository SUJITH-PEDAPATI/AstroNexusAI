"""
AstroNexusAI Evaluation — Metric Computation
=============================================
All metrics computed here. No LLM calls — pure Python.
"""
from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Optional


# ── Retrieval metrics ─────────────────────────────────────────────────────────

def hit_at_k(retrieved_paper_ids: list[str], relevant_paper_ids: set[str], k: int) -> float:
    """1.0 if any of the top-k retrieved chunks come from a relevant paper."""
    return 1.0 if any(p in relevant_paper_ids for p in retrieved_paper_ids[:k]) else 0.0


def recall_at_k(retrieved_paper_ids: list[str], relevant_paper_ids: set[str], k: int) -> float:
    """Fraction of relevant papers found in top-k results."""
    if not relevant_paper_ids:
        return 0.0
    found = len(set(retrieved_paper_ids[:k]) & relevant_paper_ids)
    return found / len(relevant_paper_ids)


def precision_at_k(retrieved_paper_ids: list[str], relevant_paper_ids: set[str], k: int) -> float:
    """Fraction of top-k retrieved that are relevant."""
    if k == 0:
        return 0.0
    relevant_in_top_k = sum(1 for p in retrieved_paper_ids[:k] if p in relevant_paper_ids)
    return relevant_in_top_k / k


def mean_reciprocal_rank(retrieved_paper_ids: list[str], relevant_paper_ids: set[str]) -> float:
    """MRR: reciprocal of the rank of the first relevant result."""
    for rank, pid in enumerate(retrieved_paper_ids, start=1):
        if pid in relevant_paper_ids:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved_paper_ids: list[str], relevant_paper_ids: set[str], k: int) -> float:
    """
    nDCG@k: normalized discounted cumulative gain.

    Relevance is binary and defined at PAPER level (not chunk level).
    Each relevant paper contributes at most ONE hit — the first time it
    appears in the ranked list.  Duplicate chunks from the same paper do
    NOT increase DCG, because the information gain of seeing the same
    paper twice is zero.

    Bug fixed: the original version counted every chunk from a relevant
    paper separately, causing DCG > IDCG and nDCG > 1 whenever more than
    one chunk from the target paper was retrieved.

    Definition:
        DCG  = sum_{r=1}^{k}  rel(r) / log2(r + 1)   (first-hit per paper)
        IDCG = sum_{r=1}^{ideal_k}  1 / log2(r + 1)   (ideal ranking)
        nDCG = DCG / IDCG                              ∈ [0, 1]
    """
    # DCG: add gain only for the FIRST occurrence of each relevant paper
    seen_relevant: set[str] = set()
    dcg = 0.0
    for rank, pid in enumerate(retrieved_paper_ids[:k], start=1):
        if pid in relevant_paper_ids and pid not in seen_relevant:
            dcg += 1.0 / math.log2(rank + 1)
            seen_relevant.add(pid)

    # IDCG: assume all relevant papers appear at the top ranks
    ideal_k = min(k, len(relevant_paper_ids))
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_k + 1))
    return dcg / idcg if idcg > 0 else 0.0


# ── Token-level text metrics ──────────────────────────────────────────────────

def token_f1(generated: str, reference: str) -> float:
    """Token overlap F1 between generated answer and reference."""
    gen_toks = set(generated.lower().split())
    ref_toks = set(reference.lower().split())
    if not gen_toks or not ref_toks:
        return 0.0
    common = gen_toks & ref_toks
    if not common:
        return 0.0
    p = len(common) / len(gen_toks)
    r = len(common) / len(ref_toks)
    return 2 * p * r / (p + r)


def fact_coverage(answer: str, required_facts: list[str]) -> float:
    """Fraction of required key-facts present in the answer."""
    if not required_facts:
        return 1.0
    answer_lower = answer.lower()
    covered = sum(1 for fact in required_facts if fact.lower() in answer_lower)
    return covered / len(required_facts)


def grounding_score(answer: str, context_chunks: list[str]) -> float:
    """
    Fraction of answer sentences whose key terms appear in retrieved context.
    Mirrors live_evaluator._grounding_score() in the main codebase.
    """
    import re
    if not answer.strip() or not context_chunks:
        return 0.0
    context_text = " ".join(context_chunks).lower()
    sentences = [s.strip() for s in re.split(r'[.!?]+', answer) if len(s.strip()) > 10]
    if not sentences:
        return 0.0
    grounded = 0
    for sent in sentences:
        words = [w for w in sent.lower().split() if len(w) > 3]
        if not words:
            continue
        overlap = sum(1 for w in words if w in context_text) / len(words)
        if overlap >= 0.3:
            grounded += 1
    return grounded / len(sentences)


# ── Aggregate statistics ──────────────────────────────────────────────────────

@dataclass
class RetrievalMetrics:
    recall_at_1:   float = 0.0
    recall_at_3:   float = 0.0
    recall_at_5:   float = 0.0
    recall_at_10:  float = 0.0
    precision_at_1: float = 0.0
    precision_at_3: float = 0.0
    precision_at_5: float = 0.0
    mrr:           float = 0.0
    ndcg_at_10:    float = 0.0
    hit_at_1:      float = 0.0
    hit_at_5:      float = 0.0

    def to_dict(self) -> dict:
        return {
            "recall@1":   round(self.recall_at_1, 4),
            "recall@3":   round(self.recall_at_3, 4),
            "recall@5":   round(self.recall_at_5, 4),
            "recall@10":  round(self.recall_at_10, 4),
            "precision@1": round(self.precision_at_1, 4),
            "precision@3": round(self.precision_at_3, 4),
            "precision@5": round(self.precision_at_5, 4),
            "mrr":        round(self.mrr, 4),
            "ndcg@10":    round(self.ndcg_at_10, 4),
            "hit@1":      round(self.hit_at_1, 4),
            "hit@5":      round(self.hit_at_5, 4),
        }


@dataclass
class AnswerMetrics:
    token_f1:        float = 0.0
    fact_coverage:   float = 0.0
    grounding:       float = 0.0
    # From LLM judge (if enabled)
    judge_score:     Optional[float] = None   # 0-3
    judge_reason:    Optional[str]   = None
    missing_facts:   list[str]       = field(default_factory=list)

    @property
    def correctness(self) -> float:
        """Combined correctness: fact coverage + token F1, or judge score if available."""
        if self.judge_score is not None:
            return self.judge_score / 3.0
        return (self.fact_coverage + self.token_f1) / 2.0

    def to_dict(self) -> dict:
        return {
            "token_f1":       round(self.token_f1, 4),
            "fact_coverage":  round(self.fact_coverage, 4),
            "grounding":      round(self.grounding, 4),
            "correctness":    round(self.correctness, 4),
            "judge_score":    self.judge_score,
            "missing_facts":  self.missing_facts,
        }


@dataclass
class LatencyMetrics:
    retrieval_ms:   float = 0.0
    generation_ms:  float = 0.0
    total_ms:       float = 0.0


def aggregate_retrieval(metrics_list: list[RetrievalMetrics]) -> dict:
    if not metrics_list:
        return {}
    n = len(metrics_list)
    return {
        "n": n,
        "recall@1":    round(sum(m.recall_at_1  for m in metrics_list) / n, 4),
        "recall@3":    round(sum(m.recall_at_3  for m in metrics_list) / n, 4),
        "recall@5":    round(sum(m.recall_at_5  for m in metrics_list) / n, 4),
        "recall@10":   round(sum(m.recall_at_10 for m in metrics_list) / n, 4),
        "precision@5": round(sum(m.precision_at_5 for m in metrics_list) / n, 4),
        "mrr":         round(sum(m.mrr           for m in metrics_list) / n, 4),
        "ndcg@10":     round(sum(m.ndcg_at_10    for m in metrics_list) / n, 4),
        "hit@1":       round(sum(m.hit_at_1      for m in metrics_list) / n, 4),
        "hit@5":       round(sum(m.hit_at_5      for m in metrics_list) / n, 4),
    }


def aggregate_answers(metrics_list: list[AnswerMetrics]) -> dict:
    if not metrics_list:
        return {}
    n = len(metrics_list)
    grounding_vals = [m.grounding for m in metrics_list]
    halluc_rate = sum(1 for g in grounding_vals if g < 0.4) / n
    return {
        "n": n,
        "token_f1":       round(sum(m.token_f1       for m in metrics_list) / n, 4),
        "fact_coverage":  round(sum(m.fact_coverage  for m in metrics_list) / n, 4),
        "grounding":      round(sum(m.grounding      for m in metrics_list) / n, 4),
        "correctness":    round(sum(m.correctness    for m in metrics_list) / n, 4),
        "hallucination_rate": round(halluc_rate, 4),
        "grounded_pct":   round(1.0 - halluc_rate, 4),
    }


def aggregate_latency(latency_list: list[LatencyMetrics]) -> dict:
    if not latency_list:
        return {}
    import statistics
    totals = [l.total_ms for l in latency_list]
    ret    = [l.retrieval_ms for l in latency_list]
    gen    = [l.generation_ms for l in latency_list]
    return {
        "n": len(totals),
        "retrieval_mean_ms":   round(statistics.mean(ret), 1),
        "generation_mean_ms":  round(statistics.mean(gen), 1),
        "total_mean_ms":       round(statistics.mean(totals), 1),
        "total_median_ms":     round(statistics.median(totals), 1),
        "total_p95_ms":        round(sorted(totals)[int(0.95 * len(totals))], 1) if len(totals) >= 2 else totals[0],
    }
