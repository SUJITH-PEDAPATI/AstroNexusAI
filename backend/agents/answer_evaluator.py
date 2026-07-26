"""
AstroNexus AI — Answer Evaluator

Evaluates research agent answers for grounding, faithfulness, and precision.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional, Any

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    query:            str
    answer:           str
    confidence:       str = "high"             # "high" | "medium" | "low"
    grounding_score:  float = 1.0              # 0.0 to 1.0
    is_reliable:      bool = True
    warnings:         list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "query":           self.query,
            "confidence":      self.confidence,
            "grounding_score": self.grounding_score,
            "is_reliable":     self.is_reliable,
            "warnings":        self.warnings,
        }


class AnswerEvaluator:
    """
    Evaluates generated RAG answers against retrieved context chunks.
    """

    def evaluate(
        self,
        query:            str,
        answer:           str,
        retrieved_chunks: list[dict[str, Any]],
    ) -> EvaluationResult:
        warnings: list[str] = []

        if not retrieved_chunks:
            warnings.append("No context chunks provided for grounding evaluation.")
            return EvaluationResult(
                query=query,
                answer=answer,
                confidence="low",
                grounding_score=0.0,
                is_reliable=False,
                warnings=warnings,
            )

        if "insufficient context" in answer.lower() or "no document excerpts" in answer.lower():
            warnings.append("Answer indicates insufficient context available.")
            return EvaluationResult(
                query=query,
                answer=answer,
                confidence="low",
                grounding_score=0.2,
                is_reliable=True,
                warnings=warnings,
            )

        # Simple overlap check across retrieved chunks
        context_combined = " ".join(c.get("text", "") for c in retrieved_chunks).lower()
        words = [w.strip("?.,!\"'").lower() for w in answer.split() if len(w) > 4]

        if not words:
            score = 1.0
        else:
            matches = sum(1 for w in words if w in context_combined)
            score = round(matches / len(words), 2)

        is_reliable = score >= 0.3
        confidence  = "high" if score >= 0.7 else ("medium" if score >= 0.4 else "low")

        if not is_reliable:
            warnings.append("Answer contains low token overlap with retrieved context.")

        return EvaluationResult(
            query=query,
            answer=answer,
            confidence=confidence,
            grounding_score=score,
            is_reliable=is_reliable,
            warnings=warnings,
        )


_evaluator_instance: Optional[AnswerEvaluator] = None


def get_evaluator() -> AnswerEvaluator:
    global _evaluator_instance
    if _evaluator_instance is None:
        _evaluator_instance = AnswerEvaluator()
    return _evaluator_instance
