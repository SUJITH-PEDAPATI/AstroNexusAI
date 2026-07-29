"""
AstroNexus AI — Evaluation Logger

Persists per-turn evaluation scores and produces a session summary.

Files written:
    output/evaluation_log.json     — one entry per QA turn
    output/evaluation_summary.json — aggregated session stats (written on /quit)
"""
from __future__ import annotations

import json
import os
import time
import threading
from pathlib import Path
from typing import Any

_LOCK = threading.Lock()

# Resolve output directory relative to this file's project root
_OUTPUT_DIR   = Path(__file__).resolve().parents[2] / "output"
_LOG_FILE     = _OUTPUT_DIR / "evaluation_log.json"
_SUMMARY_FILE = _OUTPUT_DIR / "evaluation_summary.json"

# In-memory store of all turns this session
_turns: list[dict] = []


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────

def _ensure_output_dir() -> None:
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _load_log() -> list[dict]:
    """Load existing log file (if any) so we append rather than overwrite."""
    if _LOG_FILE.exists():
        try:
            with open(_LOG_FILE, encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    return data
        except Exception:
            pass
    return []


def _write_json(path: Path, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=str)


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def save_eval(
    *,
    query:       str,
    answer:      str,
    mode:        str,
    eval_dict:   dict | None,
    paper_id:    str | None = None,
    paper_title: str | None = None,
    top_score:   float = 0.0,
    latency_s:   float = 0.0,
) -> None:
    """
    Append one QA turn to the in-memory log and flush to disk.

    Parameters
    ----------
    query       : The user's question.
    answer      : The AI's final answer.
    mode        : Query mode ('paper_qa', 'general', etc.).
    eval_dict   : Full evaluation dict from LiveEvalResult.to_dict(), or None.
    paper_id    : Active paper ID (may be None for general queries).
    paper_title : Active paper title (may be None).
    top_score   : Top retrieval score from metadata.
    latency_s   : Wall-clock latency of the pipeline run.
    """
    _ensure_output_dir()

    # Guard against None eval_dict
    ev = eval_dict or {}
    ov = ev.get("overall", {}) if ev else {}

    entry = {
        "timestamp":   time.strftime("%Y-%m-%dT%H:%M:%S"),
        "turn":        len(_turns) + 1,
        "mode":        mode,
        "paper_id":    paper_id,
        "paper_title": paper_title,
        "query":       query,
        "answer":      answer[:500],          # cap for readability
        "top_score":   round(top_score, 4),
        "latency_s":   round(latency_s, 3),
        "grade":       ov.get("grade", "?"),
        "reliability": round(float(ov.get("reliability_score", 0.0)), 4),
        "evaluation":  ev,
    }

    # Load existing log BEFORE acquiring the lock to minimise lock-hold time
    existing_log = _load_log()
    existing_log.append(entry)

    with _LOCK:
        _turns.append(entry)
        try:
            _write_json(_LOG_FILE, existing_log)
        except Exception:
            pass


def save_summary() -> dict | None:
    """
    Compute aggregated statistics over all turns in this session and write
    them to *output/evaluation_summary.json*.

    Returns the summary dict, or None if there were no turns.
    """
    _ensure_output_dir()

    with _LOCK:
        turns = list(_turns)

    if not turns:
        return None

    paper_qa_turns = [t for t in turns if t.get("mode") == "paper_qa"]

    # ── Helpers ───────────────────────────────────────────────────────────────
    def _avg(key_path: list[str], source: list[dict]) -> float:
        """Return the mean of a nested numeric field across *source* turns."""
        vals = []
        for t in source:
            obj = t.get("evaluation", {})
            for k in key_path[:-1]:
                obj = obj.get(k, {}) if isinstance(obj, dict) else {}
            v = obj.get(key_path[-1], None) if isinstance(obj, dict) else None
            if v is not None:
                try:
                    vals.append(float(v))
                except (TypeError, ValueError):
                    pass
        return round(sum(vals) / len(vals), 4) if vals else 0.0

    def _sum(key_path: list[str], source: list[dict]) -> int:
        """Return the total (sum) of a nested integer field across *source* turns."""
        total = 0
        for t in source:
            obj = t.get("evaluation", {})
            for k in key_path[:-1]:
                obj = obj.get(k, {}) if isinstance(obj, dict) else {}
            v = obj.get(key_path[-1], None) if isinstance(obj, dict) else None
            if v is not None:
                try:
                    total += int(v)
                except (TypeError, ValueError):
                    pass
        return total

    # ── Answer quality averages ───────────────────────────────────────────────
    aq_turns = paper_qa_turns or turns
    aq = {
        "avg_bleu":      _avg(["answer_quality", "bleu"],      aq_turns),
        "avg_rouge1":    _avg(["answer_quality", "rouge1"],    aq_turns),
        "avg_rouge2":    _avg(["answer_quality", "rouge2"],    aq_turns),
        "avg_rougeL":    _avg(["answer_quality", "rougeL"],    aq_turns),
        "avg_precision": _avg(["answer_quality", "precision"], aq_turns),
        "avg_recall":    _avg(["answer_quality", "recall"],    aq_turns),
        "avg_f1":        _avg(["answer_quality", "f1"],        aq_turns),
    }

    fa = {
        "avg_grounding":             _avg(["faithfulness", "grounding"],         aq_turns),
        "avg_citation_coverage":     _avg(["faithfulness", "citation_coverage"], aq_turns),
        # hallucination_flags is an integer count — summing is more meaningful than averaging
        "total_hallucination_flags": _sum(["faithfulness", "hallucination_flags"], aq_turns),
    }

    ret = {
        "avg_top_score":  _avg(["retrieval", "top_score"],  aq_turns),
        "avg_mean_score": _avg(["retrieval", "mean_score"], aq_turns),
        "avg_coverage":   _avg(["retrieval", "coverage"],   aq_turns),
    }

    # ── Grade distribution ────────────────────────────────────────────────────
    grade_dist: dict[str, int] = {}
    for t in turns:
        g = t.get("grade", "?")
        grade_dist[g] = grade_dist.get(g, 0) + 1

    avg_latency = round(
        sum(t.get("latency_s", 0.0) for t in turns) / len(turns), 3
    )

    summary = {
        "session_end":    time.strftime("%Y-%m-%dT%H:%M:%S"),
        "total_turns":    len(turns),
        "paper_qa_turns": len(paper_qa_turns),
        "avg_latency_s":  avg_latency,
        "answer_quality": aq,
        "faithfulness":   fa,
        "retrieval":      ret,
        "overall": {
            "avg_reliability":    _avg(["overall", "reliability_score"], turns),
            "grade_distribution": grade_dist,
        },
    }

    try:
        _write_json(_SUMMARY_FILE, summary)
    except Exception:
        pass

    return summary
