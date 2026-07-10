from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class QAPair:
    """A single question-answer pair from QASPER."""
    question_id:      str
    paper_id:         str
    question:         str
    ground_truth:     list[str]       # multiple valid answers possible
    evidence:         list[str]       # ground-truth evidence sentences
    answer_type:      str             # "extractive" | "abstractive" | "yes_no" | "unanswerable"


@dataclass
class RetrievalResult:
    """Result of retrieving chunks for a question."""
    question_id:      str
    question:         str
    retrieved_chunks: list[str]       # text of retrieved chunks
    retrieved_scores: list[float]     # cosine similarity scores
    evidence_chunks:  list[str]       # ground-truth evidence sentences


@dataclass
class GenerationResult:
    """Result of generating an answer."""
    question_id:      str
    question:         str
    generated_answer: str
    ground_truth:     list[str]
    context_used:     list[str]       # chunks passed to LLM


@dataclass
class RetrievalMetrics:
    """All retrieval metrics for one question."""
    question_id:  str
    precision_at_k: dict[int, float]  # {1: 0.5, 3: 0.33, 5: 0.2}
    recall_at_k:    dict[int, float]
    hit_rate:       bool              # True if any relevant chunk retrieved
    mrr:            float             # Mean Reciprocal Rank
    map_score:      float             # Average Precision
    ndcg:           float             # Normalized Discounted Cumulative Gain


@dataclass
class GenerationMetrics:
    """All generation metrics for one question."""
    question_id:      str
    exact_match:      float
    f1_score:         float
    rouge_1:          float
    rouge_2:          float
    rouge_l:          float
    bleu:             float
    answer_relevancy: float           # from Ragas
    faithfulness:     float           # from Ragas
    context_precision: float          # from Ragas
    context_recall:   float           # from Ragas


@dataclass
class EvaluationReport:
    """Aggregated evaluation report across all QA pairs."""
    total_questions:  int
    answer_types:     dict[str, int]  # distribution of answer types

    # Retrieval averages
    avg_precision_at_1:  float
    avg_precision_at_3:  float
    avg_precision_at_5:  float
    avg_recall_at_1:     float
    avg_recall_at_3:     float
    avg_recall_at_5:     float
    avg_hit_rate:        float
    avg_mrr:             float
    avg_map:             float
    avg_ndcg:            float

    # Generation averages
    avg_exact_match:      float
    avg_f1:               float
    avg_rouge_1:          float
    avg_rouge_2:          float
    avg_rouge_l:          float
    avg_bleu:             float
    avg_answer_relevancy: float
    avg_faithfulness:     float
    avg_context_precision: float
    avg_context_recall:   float

    # Per-question details
    retrieval_details:   list[RetrievalMetrics] = field(default_factory=list)
    generation_details:  list[GenerationMetrics] = field(default_factory=list)