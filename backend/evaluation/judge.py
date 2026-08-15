"""
AstroNexusAI Evaluation — LLM Judge
====================================
Uses Ollama qwen3:4b (already running locally) to judge answer correctness.
Returns a score 0-3 with reasoning.

No external API calls. Uses the same Ollama instance as the main pipeline.
"""
from __future__ import annotations

import json
import logging
import os
import urllib.request
from typing import Optional

logger = logging.getLogger(__name__)

OLLAMA_BASE  = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
JUDGE_MODEL  = os.environ.get("JUDGE_MODEL", "qwen3:4b")

_JUDGE_SYSTEM = """\
You are a strict scientific answer evaluator for an astronomical AI system.
Evaluate how well the given answer addresses the question based on the
evidence provided. Be objective and concise. Output ONLY valid JSON.\
"""

_JUDGE_PROMPT = """\
QUESTION: {question}

EXPECTED KEY FACTS: {facts}

REFERENCE ANSWER: {reference}

SYSTEM ANSWER: {answer}

Rate the SYSTEM ANSWER on correctness and completeness (0-3):
  0 = Wrong or completely misses the question
  1 = Partially correct but missing key facts
  2 = Mostly correct with minor gaps
  3 = Fully correct and addresses all key facts

Output ONLY this JSON (no other text):
{{
  "score": <0|1|2|3>,
  "reasoning": "<one sentence>"
}}
"""


def judge(
    question:         str,
    answer:           str,
    reference_answer: str,
    required_facts:   list[str],
) -> tuple[Optional[float], Optional[str]]:
    """
    Ask Ollama to judge the answer.

    Returns:
        (score 0-3 normalised to 0.0-1.0, reasoning string)
        (None, None) on any failure — caller falls back to automated metrics.
    """
    prompt = _JUDGE_PROMPT.format(
        question=question,
        facts=", ".join(required_facts) if required_facts else "N/A",
        reference=reference_answer[:600],
        answer=answer[:800],
    )

    payload = json.dumps({
        "model":  JUDGE_MODEL,
        "prompt": f"{_JUDGE_SYSTEM}\n\n{prompt}",
        "stream": False,
        "options": {"temperature": 0, "num_predict": 120},
    }).encode()

    try:
        req = urllib.request.Request(
            f"{OLLAMA_BASE}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            raw = json.loads(resp.read()).get("response", "")

        # Parse JSON from model output — strip any markdown fences
        raw = raw.strip().strip("```json").strip("```").strip()
        parsed = json.loads(raw)
        score  = int(parsed.get("score", 0))
        reason = str(parsed.get("reasoning", ""))
        logger.debug(f"[Judge] score={score} reason={reason[:80]}")
        return float(score), reason

    except Exception as e:
        logger.debug(f"[Judge] failed: {e}")
        return None, None
