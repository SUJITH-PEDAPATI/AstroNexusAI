"""
Generation Evaluation for AstroNexus RAG.

Computes: Exact Match, F1, ROUGE, BLEU, and Ragas metrics.

Install: pip install rouge-score nltk ragas
"""
from __future__ import annotations

import logging
import os
import re
import string
from collections import Counter

from backend.evaluation.models import GenerationResult, GenerationMetrics

logger = logging.getLogger(__name__)


# ── Text normalization ─────────────────────────────────────────────────────────

def _normalize(text: str) -> str:
    """
    Standard QA normalization: lowercase, strip punctuation and articles.
    Same normalization used in SQuAD evaluation.
    """
    text = text.lower().strip()
    text = re.sub(r'\b(a|an|the)\b', ' ', text)   # remove articles
    text = ''.join(ch for ch in text if ch not in string.punctuation)
    text = ' '.join(text.split())   # collapse whitespace
    return text


# ── Exact Match ───────────────────────────────────────────────────────────────

def compute_exact_match(generated: str, ground_truths: list[str]) -> float:
    """
    Exact Match = 1 if normalized generated answer matches any ground truth.

    Best for: yes/no questions and short extractive answers.
    Worst for: long abstractive answers (very strict).
    """
    gen_norm = _normalize(generated)
    return float(any(_normalize(gt) == gen_norm for gt in ground_truths))


# ── Token-level F1 ────────────────────────────────────────────────────────────

def compute_token_f1(generated: str, ground_truths: list[str]) -> float:
    """
    Token F1 = harmonic mean of token precision and recall.

    Precision = (shared tokens) / (generated tokens)
    Recall    = (shared tokens) / (ground truth tokens)

    Takes the max F1 across all ground truths (multiple valid answers).
    Best for: extractive and short abstractive answers.
    """
    gen_tokens = _normalize(generated).split()

    best_f1 = 0.0
    for gt in ground_truths:
        gt_tokens = _normalize(gt).split()

        if not gen_tokens or not gt_tokens:
            continue

        common = Counter(gen_tokens) & Counter(gt_tokens)
        num_common = sum(common.values())

        if num_common == 0:
            continue

        precision = num_common / len(gen_tokens)
        recall    = num_common / len(gt_tokens)
        f1        = 2 * precision * recall / (precision + recall)
        best_f1   = max(best_f1, f1)

    return best_f1


# ── ROUGE ──────────────────────────────────────────────────────────────────────

def compute_rouge(generated: str, ground_truths: list[str]) -> dict[str, float]:
    """
    ROUGE-1, ROUGE-2, ROUGE-L scores.

    ROUGE-1: unigram overlap
    ROUGE-2: bigram overlap
    ROUGE-L: longest common subsequence

    Best for: abstractive answer evaluation.
    Takes max score across all ground truths.
    """
    try:
        from rouge_score import rouge_scorer
    except ImportError as e:
        raise ImportError("Install rouge-score: pip install rouge-score") from e

    scorer = rouge_scorer.RougeScorer(
        ['rouge1', 'rouge2', 'rougeL'],
        use_stemmer=True
    )

    best = {"rouge_1": 0.0, "rouge_2": 0.0, "rouge_l": 0.0}

    for gt in ground_truths:
        scores = scorer.score(gt, generated)
        best["rouge_1"] = max(best["rouge_1"], scores["rouge1"].fmeasure)
        best["rouge_2"] = max(best["rouge_2"], scores["rouge2"].fmeasure)
        best["rouge_l"] = max(best["rouge_l"], scores["rougeL"].fmeasure)

    return best


# ── BLEU ───────────────────────────────────────────────────────────────────────

def compute_bleu(generated: str, ground_truths: list[str]) -> float:
    """
    BLEU score (sentence-level).

    Measures n-gram precision of generated text vs references.
    Best for: short to medium length answers.
    Note: BLEU penalizes short answers (brevity penalty).
    """
    try:
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        import nltk
        nltk.download('punkt', quiet=True)
    except ImportError as e:
        raise ImportError("Install nltk: pip install nltk") from e

    gen_tokens = _normalize(generated).split()
    ref_tokens = [_normalize(gt).split() for gt in ground_truths]

    if not gen_tokens or not any(ref_tokens):
        return 0.0

    smoothing = SmoothingFunction().method1
    return sentence_bleu(ref_tokens, gen_tokens, smoothing_function=smoothing)


# ── Ragas Metrics ──────────────────────────────────────────────────────────────

def compute_ragas_metrics(
    question:     str,
    generated:    str,
    contexts:     list[str],
    ground_truth: str,
) -> dict[str, float]:
    """
    Compute Ragas metrics: Answer Relevancy, Faithfulness,
    Context Precision, Context Recall.

    Ragas uses LLMs internally to evaluate — requires an LLM API.
    These metrics capture what token overlap metrics miss:
        - Does the answer actually answer the question? (relevancy)
        - Is the answer supported by the retrieved context? (faithfulness)
        - Is the context relevant to the question? (context precision)
        - Does the context cover the ground truth? (context recall)

    Install: pip install ragas langchain-openai
    """
    try:
        from ragas import evaluate
        from ragas.metrics import (
            answer_relevancy,
            faithfulness,
            context_precision,
            context_recall,
        )
        from datasets import Dataset
    except ImportError:
        logger.warning(
            "[Ragas] ragas not installed — returning zeros. "
            "Install: pip install ragas"
        )
        return {
            "answer_relevancy":  0.0,
            "faithfulness":      0.0,
            "context_precision": 0.0,
            "context_recall":    0.0,
        }

    try:
        data = {
            "question":   [question],
            "answer":     [generated],
            "contexts":   [contexts],
            "ground_truth": [ground_truth],
        }
        dataset = Dataset.from_dict(data)

        result = evaluate(
            dataset,
            metrics=[
                answer_relevancy,
                faithfulness,
                context_precision,
                context_recall,
            ],
        )

        return {
            "answer_relevancy":  float(result["answer_relevancy"]),
            "faithfulness":      float(result["faithfulness"]),
            "context_precision": float(result["context_precision"]),
            "context_recall":    float(result["context_recall"]),
        }

    except Exception as e:
        logger.warning(f"[Ragas] Evaluation failed: {e}")
        return {
            "answer_relevancy":  0.0,
            "faithfulness":      0.0,
            "context_precision": 0.0,
            "context_recall":    0.0,
        }


# ── Main evaluation function ───────────────────────────────────────────────────

def evaluate_generation(
    result:       GenerationResult,
    use_ragas:    bool = True,
) -> GenerationMetrics:
    """
    Compute all generation metrics for a single QA pair.

    Args:
        result:    GenerationResult with question, generated answer, ground truths
        use_ragas: Whether to run Ragas metrics (requires LLM API call)
    """
    generated    = result.generated_answer
    ground_truth = result.ground_truth
    contexts     = result.context_used

    # Token-overlap metrics (fast, no API)
    rouge_scores = compute_rouge(generated, ground_truth)

    # Ragas metrics (LLM-based, slower)
    ragas_scores = (
        compute_ragas_metrics(
            question=     result.question,
            generated=    generated,
            contexts=     contexts,
            ground_truth= ground_truth[0] if ground_truth else "",
        )
        if use_ragas else
        {"answer_relevancy": 0.0, "faithfulness": 0.0,
         "context_precision": 0.0, "context_recall": 0.0}
    )

    return GenerationMetrics(
        question_id=      result.question_id,
        exact_match=      compute_exact_match(generated, ground_truth),
        f1_score=         compute_token_f1(generated, ground_truth),
        rouge_1=          rouge_scores["rouge_1"],
        rouge_2=          rouge_scores["rouge_2"],
        rouge_l=          rouge_scores["rouge_l"],
        bleu=             compute_bleu(generated, ground_truth),
        answer_relevancy= ragas_scores["answer_relevancy"],
        faithfulness=     ragas_scores["faithfulness"],
        context_precision=ragas_scores["context_precision"],
        context_recall=   ragas_scores["context_recall"],
    )