"""
Retrieval Evaluation for AstroNexus RAG.

Computes: Precision@K, Recall@K, Hit Rate, MRR, MAP, nDCG
"""
from __future__ import annotations

import logging
import math

from backend.evaluation.models import QAPair, RetrievalResult, RetrievalMetrics
from backend.rag.retriever import retrieve

logger = logging.getLogger(__name__)

K_VALUES = [1, 3, 5]   # evaluate at these K values


def retrieve_for_question(
    qa: QAPair,
    top_k: int = 5,
    score_threshold: float = 0.0,   # low threshold — we want all candidates for eval
) -> RetrievalResult:
    """
    Run retrieval for a single QA pair.
    Filters results to only chunks from the correct paper.
    """
    results = retrieve(
        query=qa.question,
        top_k=top_k,
        score_threshold=score_threshold,
        filter_paper_id=qa.paper_id,  # only search this paper's chunks
    )

    return RetrievalResult(
        question_id=      qa.question_id,
        question=         qa.question,
        retrieved_chunks= [r.text for r in results],
        retrieved_scores= [r.score for r in results],
        evidence_chunks=  qa.evidence,
    )


def is_relevant(chunk: str, evidence: list[str], threshold: float = 0.5) -> bool:
    """
    Check if a retrieved chunk is relevant (contains evidence).

    A chunk is relevant if it contains at least one evidence sentence
    (or a significant overlap — handles chunking boundary effects).
    """
    if not evidence:
        return False

    chunk_lower = chunk.lower()

    for ev in evidence:
        ev_lower = ev.lower().strip()
        if not ev_lower:
            continue

        # Direct containment check
        if ev_lower in chunk_lower:
            return True

        # Overlap check: evidence words in chunk
        ev_words   = set(ev_lower.split())
        chunk_words = set(chunk_lower.split())
        if len(ev_words) > 0:
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

    Measures: of the K chunks we retrieved, how many are relevant?
    """
    if not retrieved or not evidence:
        return 0.0

    top_k = retrieved[:k]
    relevant_count = sum(1 for chunk in top_k if is_relevant(chunk, evidence))
    return relevant_count / k


def compute_recall_at_k(
    retrieved: list[str],
    evidence:  list[str],
    k:         int,
) -> float:
    """
    Recall@K = (relevant chunks in top K) / (total relevant chunks)

    Measures: of all relevant chunks that exist, how many did we find in top K?
    Note: we approximate total relevant as number of evidence sentences.
    """
    if not retrieved or not evidence:
        return 0.0

    top_k = retrieved[:k]
    relevant_count = sum(1 for chunk in top_k if is_relevant(chunk, evidence))
    return relevant_count / len(evidence)


def compute_hit_rate(retrieved: list[str], evidence: list[str]) -> bool:
    """
    Hit Rate = 1 if any relevant chunk is in retrieved list, else 0.

    Measures: did we retrieve at least one useful chunk?
    Most important metric for RAG — if hit rate is low, LLM has nothing to work with.
    """
    return any(is_relevant(chunk, evidence) for chunk in retrieved)


def compute_mrr(retrieved: list[str], evidence: list[str]) -> float:
    """
    Mean Reciprocal Rank = 1 / rank_of_first_relevant_chunk

    Measures: how early does the first relevant chunk appear?
    MRR=1.0 means first chunk was relevant.
    MRR=0.5 means second chunk was first relevant.
    MRR=0.0 means no relevant chunk found.
    """
    for rank, chunk in enumerate(retrieved, start=1):
        if is_relevant(chunk, evidence):
            return 1.0 / rank
    return 0.0


def compute_map(retrieved: list[str], evidence: list[str]) -> float:
    """
    Mean Average Precision = average of Precision@K at each relevant rank.

    Measures: overall quality of the ranked retrieval list.
    Rewards finding relevant chunks early AND finding all of them.
    """
    if not retrieved or not evidence:
        return 0.0

    num_relevant = 0
    precision_sum = 0.0

    for rank, chunk in enumerate(retrieved, start=1):
        if is_relevant(chunk, evidence):
            num_relevant += 1
            precision_sum += num_relevant / rank

    if num_relevant == 0:
        return 0.0

    return precision_sum / len(evidence)


def compute_ndcg(retrieved: list[str], evidence: list[str], k: int = 5) -> float:
    """
    Normalized Discounted Cumulative Gain@K.

    Measures: retrieval quality with position-aware discounting.
    Relevant chunks at rank 1 contribute more than rank 5.
    nDCG=1.0 is perfect retrieval.

    Assumes binary relevance (relevant=1, not relevant=0).
    """
    if not retrieved or not evidence:
        return 0.0

    top_k = retrieved[:k]

    # DCG: sum of (relevance / log2(rank+1))
    dcg = sum(
        1.0 / math.log2(rank + 2)   # rank is 0-indexed, +2 for log2(2)=1 at rank 0
        for rank, chunk in enumerate(top_k)
        if is_relevant(chunk, evidence)
    )

    # Ideal DCG: assume all relevant chunks are at the top
    ideal_relevant = min(len(evidence), k)
    idcg = sum(1.0 / math.log2(rank + 2) for rank in range(ideal_relevant))

    return dcg / idcg if idcg > 0 else 0.0


def evaluate_retrieval(result: RetrievalResult) -> RetrievalMetrics:
    """
    Compute all retrieval metrics for a single QA pair.
    """
    retrieved = result.retrieved_chunks
    evidence  = result.evidence_chunks

    precision_at_k = {k: compute_precision_at_k(retrieved, evidence, k) for k in K_VALUES}
    recall_at_k    = {k: compute_recall_at_k(retrieved, evidence, k)    for k in K_VALUES}

    return RetrievalMetrics(
        question_id=    result.question_id,
        precision_at_k= precision_at_k,
        recall_at_k=    recall_at_k,
        hit_rate=       compute_hit_rate(retrieved, evidence),
        mrr=            compute_mrr(retrieved, evidence),
        map_score=      compute_map(retrieved, evidence),
        ndcg=           compute_ndcg(retrieved, evidence),
    )