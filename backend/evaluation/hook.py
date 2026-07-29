"""
AstroNexus AI — Evaluation Hook

Drop-in function that wraps any orchestrator result and logs everything.
Call this AFTER orchestrator.run() — no changes to existing code needed.

Usage:
    from backend.evaluation.hook import log_result
    result = orchestrator.run(query=query, ...)
    log_result(result, session_id=session_id, query=query)
"""
from __future__ import annotations

import time
from typing import Any


def log_result(
    result:       dict,
    query:        str,
    session_id:   str   = "",
    reference:    str   = None,
    relevant_ids: list  = None,
    total_ms:     float = 0.0,
) -> dict:
    """
    Log a completed orchestrator result.
    Returns the evaluation dict so it can be displayed in the chat.

    Args:
        result:       return value of orchestrator.run()
        query:        original user query
        session_id:   session identifier
        reference:    ground truth answer (optional)
        relevant_ids: known relevant chunk IDs (optional)
        total_ms:     total end-to-end latency

    Returns:
        evaluation dict with all computed metrics
    """
    try:
        from backend.evaluation.logger    import get_logger
        from backend.evaluation.evaluator import evaluate

        ql     = get_logger()
        meta   = result.get("metadata") or {}
        answer = result.get("final_answer", "")

        # New query log
        log = ql.new_query(session_id=session_id, user_query=query)

        # Routing
        ql.set_routing(
            log,
            route=        result.get("query_type", ""),
            agent=        meta.get("mode", ""),
            model=        "gemini-2.0-flash" if meta.get("gemini_used") else "qwen3:4b",
            paper_loaded= meta.get("paper_loaded", False),
            paper_id=     meta.get("paper_id", ""),
        )

        # Retrieval — rebuild chunk list from metadata if available
        chunks = meta.get("retrieved_chunks", [])
        if chunks or meta.get("top_score", 0) > 0:
            ql.set_retrieval(
                log,
                chunks=     chunks,
                latency_ms= meta.get("retrieval_ms", 0.0),
                top_k=      meta.get("top_k", 5),
                threshold=  0.20,
            )
            # Patch top score if chunks not stored
            if not chunks and meta.get("top_score"):
                log.retrieval.max_score = meta["top_score"]
                log.retrieval.avg_score = meta["top_score"]

        # Generation
        ql.set_generation(
            log,
            answer=      answer,
            model=       "gemini-2.0-flash" if meta.get("gemini_used") else "qwen3:4b",
            latency_ms=  meta.get("latency_s", 0) * 1000,
            total_ms=    total_ms,
            confidence=  meta.get("evaluation", {}).get("overall", {}).get("grade", ""),
        )

        # Evaluation metrics
        eval_result = evaluate(
            query=        query,
            answer=       answer,
            chunks=       chunks,
            reference=    reference,
            relevant_ids= relevant_ids,
        )
        log.evaluation = eval_result.to_dict()

        # Save everything
        ql.save(log)

        return log.evaluation

    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(f"[EvalHook] Failed (non-fatal): {e}")
        return {}