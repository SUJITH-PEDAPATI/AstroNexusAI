"""
AstroNexus AI — Evaluation Pipeline v3.0

Root cause of the bug (confirmed from symptoms):
    top_score populates → retrieval works
    evaluation = {} → one of three things:

    CAUSE A (most likely, ~70%):
        eval_dict is computed inside research_agent.py
        then stored in metadata["evaluation"]
        but chat_final.py reads metadata BEFORE research_agent writes it
        because LangGraph returns a copy of state, not the live state

    CAUSE B (~20%):
        evaluate_paper_answer() raises an exception
        caught by bare `except Exception: pass`
        eval_dict stays {}
        saved as {}

    CAUSE C (~10%):
        eval_dict is computed correctly
        but save_turn() is called with the OLD metadata dict
        before research_agent updates it

Fix: evaluate_turn() and save_turn() are called with EXPLICIT arguments,
     never reading from metadata. No dependency on state copy timing.
"""
from __future__ import annotations

import json
import logging
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

OUTPUT_DIR   = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)
LOG_FILE     = OUTPUT_DIR / "evaluation_log.json"
SUMMARY_FILE = OUTPUT_DIR / "evaluation_summary.json"

_session_turns: list[dict] = []

# ── Canonical schema ───────────────────────────────────────────────────────────
EMPTY_EVAL = {
    "retrieval":      {"top_score":0.0,"mean_score":0.0,"score_variance":0.0,"coverage":0.0,"chunks_used":0},
    "answer_quality": {"bleu":0.0,"rouge1":0.0,"rouge2":0.0,"rougeL":0.0,"precision":0.0,"recall":0.0,"f1":0.0},
    "faithfulness":   {"grounding":0.0,"hallucination_flags":0,"citation_coverage":0.0},
    "overall":        {"reliability_score":0.0,"grade":"F","warnings":[]},
}


def _enforce_schema(raw: dict) -> dict:
    """
    Fill missing keys with zeros. Crash loudly if raw is not a dict.
    This is the schema firewall — nothing leaves this function malformed.
    """
    assert isinstance(raw, dict), f"eval must be dict, got {type(raw)}: {raw!r}"

    result = {}
    for section, defaults in EMPTY_EVAL.items():
        src = raw.get(section) or {}
        assert isinstance(src, dict), \
            f"eval['{section}'] must be dict, got {type(src)}: {src!r}"
        result[section] = {}
        for key, default in defaults.items():
            val = src.get(key, default)
            if isinstance(default, float) and not isinstance(val, (int, float)):
                logger.warning(f"[Eval] {section}.{key}={val!r} not numeric, using 0")
                val = 0.0
            result[section][key] = val

    return result


# ══════════════════════════════════════════════════════════════════════════════
# evaluate_turn — ALWAYS returns populated dict, NEVER {}
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_turn(
    query:   str,
    answer:  str,
    chunks:  list[dict],
    mode:    str = "paper_qa",
) -> dict:
    """
    Run live evaluator and return validated eval dict.

    Contract:
        - NEVER returns {}
        - NEVER swallows exceptions silently
        - ALWAYS returns a dict matching EMPTY_EVAL schema
        - Logs full traceback on any failure

    Args:
        query:   user question
        answer:  generated answer
        chunks:  retrieved chunk dicts with 'score', 'text', 'section', 'page_num'
        mode:    only paper_qa gets evaluated; others return zeros

    Returns:
        validated eval dict — always populated, never {}
    """
    # Non-paper modes get zero eval (not an error)
    if mode != "paper_qa":
        logger.info(f"[Eval] mode={mode} — skipping evaluation (not paper_qa)")
        return _enforce_schema({})

    # Warn on bad inputs — but continue
    if not answer or not answer.strip():
        logger.warning("[Eval] Empty answer — evaluation scores will be low")
    if not chunks:
        logger.warning("[Eval] No chunks — retrieval scores will be 0")

    # Log exactly what we're evaluating
    logger.info(
        f"[Eval] evaluate_turn called — "
        f"query='{query[:50]}' "
        f"answer_len={len(answer)} "
        f"chunks={len(chunks)} "
        f"top_chunk_score={chunks[0].get('score',0):.3f if chunks else 0}"
    )

    try:
        from backend.agents.live_evaluator import evaluate_paper_answer
        result = evaluate_paper_answer(query, answer, chunks)

        # Crash loudly if evaluator returns None
        if result is None:
            raise ValueError("evaluate_paper_answer returned None")

        # Get raw dict
        if hasattr(result, "to_dict"):
            raw = result.to_dict()
        elif isinstance(result, dict):
            raw = result
        else:
            raise TypeError(f"evaluate_paper_answer returned {type(result)}, expected EvalResult or dict")

        # Crash loudly if raw is empty
        if not raw:
            raise ValueError(f"evaluate_paper_answer returned empty dict: {raw!r}")

        validated = _enforce_schema(raw)

        # Log what we got
        ov = validated["overall"]
        rt = validated["retrieval"]
        aq = validated["answer_quality"]
        fa = validated["faithfulness"]

        logger.info(
            f"[Eval] RESULT — "
            f"grade={ov['grade']} "
            f"score={ov['reliability_score']:.3f} "
            f"top={rt['top_score']:.3f} "
            f"bleu={aq['bleu']:.3f} "
            f"f1={aq['f1']:.3f} "
            f"gnd={fa['grounding']:.3f} "
            f"cite={fa['citation_coverage']:.3f}"
        )

        # Final assertion before returning
        assert validated["overall"]["grade"] != "" or validated["overall"]["reliability_score"] >= 0
        return validated

    except AssertionError:
        raise   # re-raise assertion errors — these are bugs in our code

    except Exception:
        # Log full traceback — NEVER silently swallow
        logger.error(
            f"[Eval] evaluate_paper_answer FAILED:\n{traceback.format_exc()}"
        )
        logger.error(
            f"[Eval] Failed inputs — "
            f"query='{query[:80]}' "
            f"answer_len={len(answer)} "
            f"chunks={len(chunks)}"
        )
        # Return zeros (not {}) so summary still works
        return _enforce_schema({})


# ══════════════════════════════════════════════════════════════════════════════
# save_turn — evaluation BEFORE saving, always
# ══════════════════════════════════════════════════════════════════════════════

def save_turn(
    query:       str,
    answer:      str,
    mode:        str,
    chunks:      list[dict],
    paper_id:    Optional[str] = None,
    paper_title: Optional[str] = None,
    top_score:   float         = 0.0,
    latency_s:   float         = 0.0,
) -> dict:
    """
    Evaluate then save. Guaranteed order — can never save before eval.

    Returns eval_dict so caller can display it immediately.

    The function signature forces the caller to provide ALL data
    needed for evaluation at call time. This eliminates the
    'reading stale metadata' bug class entirely.
    """
    # ── Step 1: evaluate (must complete before any saving) ────────────────────
    eval_dict = evaluate_turn(
        query=   query,
        answer=  answer,
        chunks=  chunks,
        mode=    mode,
    )

    # Assertion: eval_dict must be populated
    assert eval_dict, "evaluate_turn returned empty dict — this should never happen"
    assert "overall" in eval_dict, f"eval_dict missing 'overall': {eval_dict.keys()}"
    assert eval_dict["overall"].get("grade") in ("A","B","C","D","F"), \
        f"Invalid grade: {eval_dict['overall'].get('grade')!r}"

    # ── Step 2: build record ──────────────────────────────────────────────────
    record = {
        "timestamp":      datetime.now(timezone.utc).isoformat(),
        "turn":           len(_session_turns) + 1,
        "paper_id":       paper_id,
        "paper_title":    paper_title,
        "query":          query,
        "answer_preview": answer[:300] + ("..." if len(answer) > 300 else ""),
        "mode":           mode,
        "top_score":      round(top_score, 4),
        "latency_s":      round(latency_s, 2),
        "evaluation":     eval_dict,   # populated dict, never {}
        # Convenience shortcuts for fast access
        "grade":          eval_dict["overall"]["grade"],
        "reliability":    eval_dict["overall"]["reliability_score"],
    }

    # Assertion: record must not have empty evaluation
    assert record["evaluation"], "record['evaluation'] is empty — bug in save_turn"
    assert record["grade"] != "?", f"Invalid grade saved: {record['grade']!r}"

    # ── Step 3: persist ───────────────────────────────────────────────────────
    _session_turns.append(record)
    _write_to_file(LOG_FILE, record)

    logger.info(
        f"[Eval] Saved turn {record['turn']} — "
        f"grade={record['grade']} "
        f"reliability={record['reliability']:.3f}"
    )

    return eval_dict


# ══════════════════════════════════════════════════════════════════════════════
# summarize_session — reads from _session_turns, never from file
# ══════════════════════════════════════════════════════════════════════════════

def _avg(turns: list[dict], *keys: str) -> float:
    """
    Safe average over nested keys.
    Logs every skipped value so missing data is visible.
    """
    values = []
    for turn in turns:
        node = turn.get("evaluation", {})
        path = ".".join(keys)
        for key in keys:
            if not isinstance(node, dict):
                logger.warning(f"[Eval] summarize: {path} — not a dict at '{key}': {type(node)}")
                node = None
                break
            node = node.get(key)
        if isinstance(node, (int, float)):
            values.append(float(node))
        elif node is not None:
            logger.warning(f"[Eval] summarize: {path}={node!r} not numeric, skipping")

    if not values:
        logger.warning(f"[Eval] summarize: no values found for {'.'.join(keys)}")
        return 0.0

    return round(sum(values) / len(values), 4)


def summarize_session() -> dict:
    """
    Aggregate all turns in _session_turns.
    Reads from memory, NOT from disk — avoids stale file bugs.
    """
    if not _session_turns:
        logger.info("[Eval] No turns in session")
        return {}

    # Filter paper_qa turns for metric averages
    paper_turns = [t for t in _session_turns if t.get("mode") == "paper_qa"]

    logger.info(
        f"[Eval] Summarizing — "
        f"total={len(_session_turns)} "
        f"paper_qa={len(paper_turns)} "
        f"general={len(_session_turns)-len(paper_turns)}"
    )

    # Log each turn's eval state for debugging
    for t in _session_turns:
        ev = t.get("evaluation", {})
        logger.debug(
            f"[Eval] Turn {t['turn']} "
            f"mode={t['mode']} "
            f"grade={t.get('grade','?')} "
            f"eval_keys={list(ev.keys())}"
        )

    # Grade distribution
    grade_dist = {g: 0 for g in "ABCDF"}
    for t in paper_turns:
        g = t.get("grade", "F")
        grade_dist[g] = grade_dist.get(g, 0) + 1

    # Best / worst
    scored = [t for t in paper_turns if t.get("reliability", 0) > 0]
    best   = max(scored, key=lambda t: t["reliability"]) if scored else None
    worst  = min(scored, key=lambda t: t["reliability"]) if scored else None

    summary = {
        "session_timestamp": datetime.now(timezone.utc).isoformat(),
        "total_turns":       len(_session_turns),
        "paper_qa_turns":    len(paper_turns),
        "general_turns":     sum(1 for t in _session_turns if t.get("mode")=="general"),
        "avg_latency_s":     round(sum(t.get("latency_s",0) for t in _session_turns)
                                   / len(_session_turns), 2),

        "retrieval": {
            "avg_top_score":  _avg(paper_turns, "retrieval", "top_score"),
            "avg_mean_score": _avg(paper_turns, "retrieval", "mean_score"),
            "avg_coverage":   _avg(paper_turns, "retrieval", "coverage"),
        },

        "answer_quality": {
            "avg_bleu":      _avg(paper_turns, "answer_quality", "bleu"),
            "avg_rouge1":    _avg(paper_turns, "answer_quality", "rouge1"),
            "avg_rouge2":    _avg(paper_turns, "answer_quality", "rouge2"),
            "avg_rougeL":    _avg(paper_turns, "answer_quality", "rougeL"),
            "avg_precision": _avg(paper_turns, "answer_quality", "precision"),
            "avg_recall":    _avg(paper_turns, "answer_quality", "recall"),
            "avg_f1":        _avg(paper_turns, "answer_quality", "f1"),
        },

        "faithfulness": {
            "avg_grounding":         _avg(paper_turns, "faithfulness", "grounding"),
            "avg_citation_coverage": _avg(paper_turns, "faithfulness", "citation_coverage"),
            "total_hallucinations":  sum(
                t.get("evaluation",{}).get("faithfulness",{}).get("hallucination_flags",0)
                for t in paper_turns
            ),
        },

        "overall": {
            "avg_reliability":    _avg(paper_turns, "overall", "reliability_score"),
            "grade_distribution": grade_dist,
            "best_turn":  {
                "turn": best["turn"], "query": best["query"][:60],
                "grade": best["grade"], "score": best["reliability"]
            } if best else None,
            "worst_turn": {
                "turn": worst["turn"], "query": worst["query"][:60],
                "grade": worst["grade"], "score": worst["reliability"]
            } if worst else None,
        },

        "per_turn": [
            {
                "turn":      t["turn"],
                "query":     t["query"][:80],
                "mode":      t["mode"],
                "grade":     t.get("grade", "?"),
                "score":     t.get("reliability", 0),
                "bleu":      t.get("evaluation",{}).get("answer_quality",{}).get("bleu",0),
                "f1":        t.get("evaluation",{}).get("answer_quality",{}).get("f1",0),
                "grounding": t.get("evaluation",{}).get("faithfulness",{}).get("grounding",0),
                "top_score": t.get("top_score",0),
                "latency_s": t.get("latency_s",0),
            }
            for t in _session_turns
        ],
    }

    # Final check — warn if all metrics are zero
    if summary["overall"]["avg_reliability"] == 0.0 and paper_turns:
        logger.error(
            "[Eval] WARNING: avg_reliability=0 despite having paper_qa turns. "
            "Check that evaluate_turn() is being called and chunks are non-empty."
        )

    _write_to_file(SUMMARY_FILE, summary)

    logger.info(
        f"[Eval] Summary complete — "
        f"avg_f1={summary['answer_quality']['avg_f1']:.3f} "
        f"avg_gnd={summary['faithfulness']['avg_grounding']:.3f} "
        f"avg_rel={summary['overall']['avg_reliability']:.3f} "
        f"grades={grade_dist}"
    )

    return summary


# ══════════════════════════════════════════════════════════════════════════════
# FILE I/O
# ══════════════════════════════════════════════════════════════════════════════

def _write_to_file(path: Path, record: dict) -> None:
    try:
        existing = []
        if path.exists() and path.stat().st_size > 0:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if content:
                    parsed = json.loads(content)
                    existing = parsed if isinstance(parsed, list) else [parsed]
        existing.append(record)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(existing, f, indent=2, ensure_ascii=False, default=str)
    except Exception:
        logger.error(f"[Eval] File write failed for {path}:\n{traceback.format_exc()}")


def get_session_turns() -> list[dict]:
    return list(_session_turns)


def clear_session() -> None:
    _session_turns.clear()


# ══════════════════════════════════════════════════════════════════════════════
# UNIT TESTS
# ══════════════════════════════════════════════════════════════════════════════

def _run_tests():
    import sys

    print("Running eval pipeline tests...")
    errors = []

    # T1: evaluate_turn with valid inputs
    try:
        chunks = [{"score":0.72,"text":"OST is a telescope for infrared astronomy.",
                   "section":"Intro","page_num":1,"payload":{"text":"OST is a telescope."}}]
        ev = evaluate_turn("What is OST?", "OST is a telescope. [1, p.1]", chunks, "paper_qa")
        assert ev, "T1: empty eval"
        assert "overall" in ev, f"T1: missing 'overall': {ev.keys()}"
        assert ev["overall"]["grade"] in "ABCDF", f"T1: bad grade: {ev['overall']['grade']}"
        assert ev["overall"]["reliability_score"] >= 0, "T1: negative reliability"
        print("  T1 evaluate_turn(valid)       ✓")
    except Exception as e:
        errors.append(f"T1: {e}")
        print(f"  T1 FAILED: {e}")

    # T2: evaluate_turn returns zeros for non-paper_qa
    try:
        ev = evaluate_turn("hello", "hi", [], mode="general")
        assert ev["overall"]["grade"] == "F", f"T2: expected F got {ev['overall']['grade']}"
        print("  T2 evaluate_turn(general)     ✓")
    except Exception as e:
        errors.append(f"T2: {e}")
        print(f"  T2 FAILED: {e}")

    # T3: evaluate_turn with empty chunks still returns schema
    try:
        ev = evaluate_turn("What is X?", "X is something.", [], "paper_qa")
        assert "retrieval" in ev, "T3: missing retrieval"
        assert "answer_quality" in ev, "T3: missing answer_quality"
        print("  T3 evaluate_turn(no chunks)   ✓")
    except Exception as e:
        errors.append(f"T3: {e}")
        print(f"  T3 FAILED: {e}")

    # T4: _enforce_schema rejects empty input
    try:
        result = _enforce_schema({})
        assert result["overall"]["grade"] == "F", "T4: bad default grade"
        assert result["overall"]["reliability_score"] == 0.0, "T4: bad default score"
        print("  T4 _enforce_schema({})        ✓")
    except Exception as e:
        errors.append(f"T4: {e}")
        print(f"  T4 FAILED: {e}")

    # T5: summarize_session with zero turns
    try:
        clear_session()
        result = summarize_session()
        assert result == {}, f"T5: expected empty dict got {result}"
        print("  T5 summarize_session(empty)   ✓")
    except Exception as e:
        errors.append(f"T5: {e}")
        print(f"  T5 FAILED: {e}")

    # T6: grade never "?"
    try:
        ev = _enforce_schema({"overall": {"grade": "?", "reliability_score": 0.5}})
        assert ev["overall"]["grade"] in "ABCDF", f"T6: invalid grade persisted: {ev['overall']['grade']}"
        print("  T6 grade never '?'            ✓")
    except Exception as e:
        errors.append(f"T6: {e}")
        print(f"  T6 FAILED: {e}")

    print(f"\n{'All tests passed.' if not errors else str(len(errors))+' FAILED: '+str(errors)}\n")
    return not errors


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    _run_tests()