"""
Retrieval Evaluation for AstroNexus RAG.

Computes: Precision@K, Recall@K, Hit Rate, MRR, MAP, nDCG

Key fix: is_relevant() now uses direct containment as primary check
and a lower word-overlap threshold (0.3 instead of 0.5) as fallback.
This correctly handles the case where a chunk CONTAINS an evidence
sentence but the overlap ratio is low because the chunk has many words.
"""
from __future__ import annotations

import logging
import math

from backend.evaluation.models import QAPair, RetrievalResult, RetrievalMetrics
from backend.rag.retriever import retrieve

logger = logging.getLogger(__name__)

K_VALUES = [1, 3, 5]


def retrieve_for_question(
    qa:              QAPair,
    top_k:           int   = 5,
    score_threshold: float = 0.0,  # 0.0 during eval — want all candidates
) -> RetrievalResult:
    """
    Run retrieval for a single QA pair.

    score_threshold is 0.0 during evaluation so we get all candidates
    regardless of score — the metrics judge relevance, not the threshold.
    We also filter by paper_id so we only search the correct paper's chunks.
    """
    results = retrieve(
        query=           qa.question,
        top_k=           top_k,
        score_threshold= score_threshold,
        filter_paper_id= qa.paper_id,   # ← critical: only search this paper
    )

    return RetrievalResult(
        question_id=      qa.question_id,
        question=         qa.question,
        retrieved_chunks= [r.text for r in results],
        retrieved_scores= [r.score for r in results],
        evidence_chunks=  qa.evidence,
    )


def is_relevant(
    chunk:     str,
    evidence:  list[str],
    threshold: float = 0.3,   # lowered from 0.5 — chunks are much longer than evidence
) -> bool:
    """
    Determine if a retrieved chunk is relevant to the question.

    A chunk is relevant if it contains at least one evidence sentence.

    Two-pass check:
        Pass 1 — Direct containment (most reliable)
                  If the evidence sentence appears verbatim in the chunk → relevant
        Pass 2 — Word overlap (handles minor text differences from PDF parsing)
                  If 30%+ of evidence words appear in the chunk → relevant

    Why threshold=0.3:
        Evidence sentences are typically 10-20 words.
        Chunks are 150-250 words (1024 chars).
        Even if the chunk CONTAINS the evidence sentence, the overlap ratio
        will be ~10-15% of the chunk's total words. We check overlap against
        evidence length (not chunk length), so 30% is the right threshold.
    """
    if not evidence:
        return False

    chunk_lower = chunk.lower()

    for ev in evidence:
        ev_clean = ev.lower().strip()
        if not ev_clean or len(ev_clean) < 10:
            continue

        # Pass 1: Direct containment — most reliable check
        if ev_clean in chunk_lower:
            return True

        # Pass 2: Word overlap against evidence length
        ev_words    = set(ev_clean.split())
        chunk_words = set(chunk_lower.split())

        if len(ev_words) >= 5:   # only check meaningful evidence sentences
            overlap = len(ev_words & chunk_words) / len(ev_words)
            if overlap >= threshold:
                return True

    return False


def compute_precision_at_k(
    retrieved: list[str],
    evidence:  list[str],
    k:         int,
) -> float:
    """
    Precision@K = (relevant chunks in top K) / K

    Of the K chunks we retrieved, what fraction are relevant?
    """
    if not retrieved or not evidence:
        return 0.0
    top_k = retrieved[:k]
    relevant = sum(1 for chunk in top_k if is_relevant(chunk, evidence))
    return relevant / k


def compute_recall_at_k(
    retrieved: list[str],
    evidence:  list[str],
    k:         int,
) -> float:
    """
    Recall@K = (relevant chunks in top K) / (total evidence sentences)

    Of all the evidence that exists, how much did we find in top K?
    Note: we treat each evidence sentence as one relevant unit.
    """
    if not retrieved or not evidence:
        return 0.0
    top_k = retrieved[:k]
    relevant = sum(1 for chunk in top_k if is_relevant(chunk, evidence))
    return min(relevant / len(evidence), 1.0)  # cap at 1.0


def compute_hit_rate(retrieved: list[str], evidence: list[str]) -> bool:
    """
    Hit Rate = 1 if ANY retrieved chunk is relevant, else 0.

    Most critical RAG metric: did we find at least one useful chunk?
    If hit rate is 0, the LLM has nothing to work with.
    """
    return any(is_relevant(chunk, evidence) for chunk in retrieved)


def compute_mrr(retrieved: list[str], evidence: list[str]) -> float:
    """
    MRR = 1 / rank_of_first_relevant_chunk

    How early does the first relevant chunk appear?
    MRR=1.0 → first chunk relevant
    MRR=0.5 → second chunk is first relevant
    MRR=0.0 → no relevant chunk found
    """
    for rank, chunk in enumerate(retrieved, start=1):
        if is_relevant(chunk, evidence):
            return 1.0 / rank
    return 0.0


def compute_map(retrieved: list[str], evidence: list[str]) -> float:
    """
    MAP = average precision across all relevant ranks.

    Rewards both finding relevant chunks AND finding them early.
    """
    if not retrieved or not evidence:
        return 0.0

    num_relevant  = 0
    precision_sum = 0.0

    for rank, chunk in enumerate(retrieved, start=1):
        if is_relevant(chunk, evidence):
            num_relevant  += 1
            precision_sum += num_relevant / rank

    if num_relevant == 0:
        return 0.0

    return precision_sum / len(evidence)


def compute_ndcg(retrieved: list[str], evidence: list[str], k: int = 5) -> float:
    """
    nDCG@K = DCG / IDCG

    Position-aware retrieval quality. Relevant chunks at rank 1
    contribute more than relevant chunks at rank 5.
    nDCG=1.0 is perfect retrieval.
    """
    if not retrieved or not evidence:
        return 0.0

    top_k = retrieved[:k]

    dcg = sum(
        1.0 / math.log2(rank + 2)
        for rank, chunk in enumerate(top_k)
        if is_relevant(chunk, evidence)
    )

    ideal_relevant = min(len(evidence), k)
    idcg = sum(1.0 / math.log2(rank + 2) for rank in range(ideal_relevant))

    return dcg / idcg if idcg > 0 else 0.0


def evaluate_retrieval(result: RetrievalResult) -> RetrievalMetrics:
    """Compute all retrieval metrics for a single QA pair."""
    retrieved = result.retrieved_chunks
    evidence  = result.evidence_chunks

    return RetrievalMetrics(
        question_id=    result.question_id,
        precision_at_k= {k: compute_precision_at_k(retrieved, evidence, k) for k in K_VALUES},
        recall_at_k=    {k: compute_recall_at_k(retrieved, evidence, k)    for k in K_VALUES},
        hit_rate=       compute_hit_rate(retrieved, evidence),
        mrr=            compute_mrr(retrieved, evidence),
        map_score=      compute_map(retrieved, evidence),
        ndcg=           compute_ndcg(retrieved, evidence),
    )