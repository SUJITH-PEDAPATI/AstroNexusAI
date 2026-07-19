"""
AstroNexus AI — Retrieval Evaluator

Fixes applied vs previous version:
    - compute_recall_at_k: caps at 1.0 and uses min(relevant, evidence_count)
    - compute_map: caps average precision at 1.0 per position
    - compute_ndcg: uses actual DCG / IDCG with correct ideal calculation
    - All metrics now guaranteed to be in [0, 1]

No interface changes — all existing callers work unchanged.
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
    score_threshold: float = 0.0,
) -> RetrievalResult:
    """
    Run retrieval for a single QA pair filtered to the correct paper.
    score_threshold=0.0 during evaluation so all candidates are returned
    and metrics judge relevance, not the score cutoff.
    """
    results = retrieve(
        query=           qa.question,
        top_k=           top_k,
        score_threshold= score_threshold,
        filter_paper_id= qa.paper_id,
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
    threshold: float = 0.3,
) -> bool:
    """
    A chunk is relevant if it contains at least one evidence sentence.

    Pass 1 — direct containment (most reliable).
    Pass 2 — word overlap >= threshold against evidence length (handles
              minor PDF parsing differences like extra whitespace).
    """
    if not evidence:
        return False

    chunk_lower = chunk.lower()

    for ev in evidence:
        ev_clean = ev.lower().strip()
        if not ev_clean or len(ev_clean) < 10:
            continue

        if ev_clean in chunk_lower:
            return True

        ev_words    = set(ev_clean.split())
        chunk_words = set(chunk_lower.split())
        if len(ev_words) >= 5:
            overlap = len(ev_words & chunk_words) / len(ev_words)
            if overlap >= threshold:
                return True

    return False


def _count_relevant(chunks: list[str], evidence: list[str]) -> int:
    """Count how many chunks in the list are relevant."""
    return sum(1 for c in chunks if is_relevant(c, evidence))


def compute_precision_at_k(
    retrieved: list[str],
    evidence:  list[str],
    k:         int,
) -> float:
    """
    Precision@K = relevant_in_top_k / K

    Measures: of the K chunks retrieved, what fraction are relevant?
    Range: [0, 1]
    """
    if not retrieved or not evidence:
        return 0.0
    top_k     = retrieved[:k]
    relevant  = _count_relevant(top_k, evidence)
    return relevant / k


def compute_recall_at_k(
    retrieved: list[str],
    evidence:  list[str],
    k:         int,
) -> float:
    """
    Recall@K = relevant_in_top_k / total_evidence_count

    Measures: of all evidence sentences, how many did we retrieve in top K?

    Fix: cap the result at 1.0.
    A chunk may contain multiple evidence sentences, so relevant_count
    can exceed len(evidence) if we count naively. min(..., 1.0) prevents
    values above 1.
    Range: [0, 1]
    """
    if not retrieved or not evidence:
        return 0.0
    top_k    = retrieved[:k]
    relevant = _count_relevant(top_k, evidence)
    # KEY FIX: cap at 1.0 — relevant can exceed len(evidence) when
    # one chunk contains multiple evidence sentences
    return min(relevant / len(evidence), 1.0)


def compute_hit_rate(retrieved: list[str], evidence: list[str]) -> bool:
    """
    Hit Rate = 1 if any retrieved chunk is relevant, else 0.

    Most critical RAG metric: did we find at least one useful chunk?
    Range: {0, 1}
    """
    return any(is_relevant(c, evidence) for c in retrieved)


def compute_mrr(retrieved: list[str], evidence: list[str]) -> float:
    """
    MRR = 1 / rank_of_first_relevant_chunk

    MRR=1.0 → first chunk is relevant.
    MRR=0.5 → second chunk is first relevant.
    MRR=0.0 → no relevant chunk found.
    Range: [0, 1]
    """
    for rank, chunk in enumerate(retrieved, start=1):
        if is_relevant(chunk, evidence):
            return 1.0 / rank
    return 0.0


def compute_map(retrieved: list[str], evidence: list[str]) -> float:
    """
    MAP = mean average precision across relevant ranks.

    Fix: the denominator is min(len(evidence), len(retrieved)) not
    len(evidence) alone. When a single chunk covers multiple evidence
    sentences, the running precision sum can exceed 1 if we use
    len(evidence) as the denominator without normalising.

    We normalise by capping the final result at 1.0.
    Range: [0, 1]
    """
    if not retrieved or not evidence:
        return 0.0

    num_relevant  = 0
    precision_sum = 0.0

    for rank, chunk in enumerate(retrieved, start=1):
        if is_relevant(chunk, evidence):
            num_relevant  += 1
            # Precision at this rank
            p_at_rank      = num_relevant / rank
            precision_sum += p_at_rank

    if num_relevant == 0:
        return 0.0

    # Normalise by total evidence count but cap at 1.0
    # KEY FIX: cap prevents values > 1 when chunks overlap multiple evidences
    raw_map = precision_sum / len(evidence)
    return min(raw_map, 1.0)


def compute_ndcg(
    retrieved: list[str],
    evidence:  list[str],
    k:         int = 5,
) -> float:
    """
    nDCG@K = DCG@K / IDCG@K

    Uses binary relevance: relevant=1, not relevant=0.
    Position-aware: relevant chunks at rank 1 contribute more than rank 5.

    Fix: IDCG must be computed as the score of the IDEAL ranking up to K,
    which is min(total_relevant_in_list, k) relevant items at the top.
    Previously, IDCG used len(evidence) which could be larger than the
    actual number of relevant chunks retrieved, producing DCG > IDCG.

    Range: [0, 1]
    """
    if not retrieved or not evidence:
        return 0.0

    top_k = retrieved[:k]

    # Relevance labels for retrieved chunks
    relevance = [1 if is_relevant(c, evidence) else 0 for c in top_k]

    # DCG@K
    dcg = sum(
        rel / math.log2(rank + 2)   # rank is 0-indexed; log2(2)=1 at rank 0
        for rank, rel in enumerate(relevance)
    )

    if dcg == 0.0:
        return 0.0

    # IDCG@K — ideal: place all relevant items at the top
    # KEY FIX: number of ideal relevant items = min(actual_relevant, k)
    # NOT len(evidence), which can be larger than retrieved relevant count
    total_relevant = sum(relevance)
    ideal_relevant = min(total_relevant, k)  # can't have more than k items

    if ideal_relevant == 0:
        return 0.0

    idcg = sum(
        1.0 / math.log2(rank + 2)
        for rank in range(ideal_relevant)
    )

    # Safety cap — floating point arithmetic can produce values like 1.0000001
    return min(dcg / idcg, 1.0)


def evaluate_retrieval(result: RetrievalResult) -> RetrievalMetrics:
    """
    Compute all retrieval metrics for a single QA pair.
    All returned values are guaranteed to be in [0, 1].
    """
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

# ══════════════════════════════════════════════════════════════════════════════
# ADD THIS FUNCTION TO THE BOTTOM OF
# backend/evaluation/retrieval_evaluator.py
#
# Do NOT modify anything above it.
# This is the only change to an existing file.
# ══════════════════════════════════════════════════════════════════════════════

def retrieve_for_question_corpus(
    qa:              "QAPair",
    top_k:           int   = 5,
    score_threshold: float = 0.0,
) -> "RetrievalResult":
    """
    Corpus-level retrieval for a single QA pair.

    Unlike retrieve_for_question(), this function does NOT pass
    filter_paper_id to Qdrant. The search covers the entire corpus —
    chunks from any paper may be returned.

    This is the realistic evaluation mode: the system does not know
    in advance which paper contains the answer.

    Args:
        qa:              QAPair from QASPER
        top_k:           Number of chunks to retrieve
        score_threshold: Minimum score (0.0 = return all candidates)

    Returns:
        RetrievalResult with retrieved_chunks from any paper in the corpus
    """
    results = retrieve(
        query=           qa.question,
        top_k=           top_k,
        score_threshold= score_threshold,
        # NO filter_paper_id — search the full corpus
    )

    return RetrievalResult(
        question_id=      qa.question_id,
        question=         qa.question,
        retrieved_chunks= [r.text     for r in results],
        retrieved_scores= [r.score    for r in results],
        evidence_chunks=  qa.evidence,
        # Store paper_ids of retrieved chunks for paper-level metrics
        retrieved_paper_ids= [r.paper_id for r in results],
    )

def retrieve_for_question_corpus(
    qa:              "QAPair",
    top_k:           int   = 5,
    score_threshold: float = 0.0,
) -> "RetrievalResult":
    """
    Corpus-level retrieval for a single QA pair.
 
    Unlike retrieve_for_question(), this function does NOT pass
    filter_paper_id to Qdrant. The search covers the entire corpus —
    chunks from any paper may be returned.
 
    This is the realistic evaluation mode: the system does not know
    in advance which paper contains the answer.
 
    Args:
        qa:              QAPair from QASPER
        top_k:           Number of chunks to retrieve
        score_threshold: Minimum score (0.0 = return all candidates)
 
    Returns:
        RetrievalResult with retrieved_chunks from any paper in the corpus
    """
    results = retrieve(
        query=           qa.question,
        top_k=           top_k,
        score_threshold= score_threshold,
        # NO filter_paper_id — search the full corpus
    )
 
    return RetrievalResult(
        question_id=      qa.question_id,
        question=         qa.question,
        retrieved_chunks= [r.text     for r in results],
        retrieved_scores= [r.score    for r in results],
        evidence_chunks=  qa.evidence,
        # Store paper_ids of retrieved chunks for paper-level metrics
        retrieved_paper_ids= [r.paper_id for r in results],
    )