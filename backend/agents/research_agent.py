"""
AstroNexus AI — Research Agent v4.4
Uses identity module so every response sounds like AstroNexus AI.
"""
from __future__ import annotations

import json, logging, os, re, time, urllib.request
from backend.agents.state    import AgentState
from backend.agents.identity import (
    PAPER_SYSTEM, REFINE_SYSTEM, GENERAL_SYSTEM, ASTRONOMY_SYSTEM
)

logger = logging.getLogger(__name__)

OLLAMA_BASE   = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL  = os.environ.get("OLLAMA_MODEL",    "qwen3:4b")
GEMINI_KEY    = os.environ.get("GEMINI_API_KEY",  "")
ABSTAIN_SCORE = 0.20

_PAPER_PROMPT = """\
{fused_context}

ANSWER THE QUESTION using ONLY the document excerpts and graph context above.
Cite every fact: [chunk_number, p.page]
If not found: "AstroNexus AI could not find this in the uploaded paper."

ANSWER:"""

_REFINE_PROMPT = """\
CONTEXT (document + graph):
{fused_context}

DRAFT:
{draft}

QUESTION: {question}

IMPROVED ANSWER WITH FULL CITATIONS:"""

_LIVE_RE = re.compile(
    r'\b(weather|forecast|right now|today|air quality|real.?time)\b',
    re.IGNORECASE,
)


def _ollama(system: str, prompt: str) -> str:
    payload = json.dumps({
        "model":  OLLAMA_MODEL,
        "prompt": prompt,
        "system": system,
        "stream": False,
        "options": {"temperature": 0.1, "num_predict": 1200},
    }).encode()
    try:
        req = urllib.request.Request(
            f"{OLLAMA_BASE}/api/generate",
            data=payload, headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=90) as resp:
            return json.loads(resp.read()).get("response", "").strip()
    except Exception as e:
        logger.error(f"[ResearchAgent] Ollama: {e}")
        return ""


def _gemini(system: str, prompt: str) -> str:
    if not GEMINI_KEY: return ""
    try:
        from google import genai
        from google.genai import types
        client   = genai.Client(api_key=GEMINI_KEY)
        response = client.models.generate_content(
            model=    "gemini-2.0-flash",
            contents= f"{system}\n\n{prompt}",
            config=   types.GenerateContentConfig(
                temperature=0.1, max_output_tokens=1200
            ),
        )
        return response.text.strip()
    except Exception as e:
        logger.warning(f"[ResearchAgent] Gemini: {e}")
        return ""


def research_agent_node(state: AgentState) -> AgentState:
    t0         = time.perf_counter()
    query      = state.get("query", "")
    metadata   = state.get("metadata") or {}
    paper_id   = metadata.get("paper_id")
    history    = state.get("conversation_history") or []
    turn_count = state.get("turn_count", 1)
    paper_loaded = (
        metadata.get("paper_loaded", False)
        or state.get("paper_loaded", False)
    )

    logger.info(f"[ResearchAgent] T{turn_count} paper={paper_loaded} '{query[:50]}'")

    # ── Retrieve ──────────────────────────────────────────────────────────────
    chunk_dicts = []
    top_score   = 0.0
    try:
        from backend.rag.retriever import retrieve
        chunks = retrieve(query, top_k=5, paper_id=paper_id)
        chunk_dicts = [
            {
                "score":   c.score, "text": c.text,
                "section": c.section, "page_num": c.page_num,
                "title":   c.title,
                "payload": {"text":c.text,"section":c.section,"page_num":c.page_num},
            }
            for c in chunks
        ]
        top_score = chunks[0].score if chunks else 0.0
        logger.info(f"[ResearchAgent] {len(chunk_dicts)} chunks top={top_score:.4f}")
    except Exception as e:
        logger.error(f"[ResearchAgent] Retrieval: {e}")

    # ── Mode ──────────────────────────────────────────────────────────────────
    if _LIVE_RE.search(query):
        mode = "live_data"
    elif paper_loaded:
        mode = "paper_qa"
    else:
        mode = "general"

    logger.info(f"[ResearchAgent] mode={mode}")
    final_answer = ""
    eval_dict    = {}

    # ══════════════════════════════════════════════════════════════════════════
    # PAPER QA
    # ══════════════════════════════════════════════════════════════════════════
    if mode == "paper_qa":
        if not chunk_dicts or top_score < ABSTAIN_SCORE:
            final_answer = (
                f"**AstroNexus AI** could not find sufficient evidence in the "
                f"uploaded paper to answer this question "
                f"(retrieval score: {top_score:.3f}). "
                f"Please try rephrasing your question."
            )
        else:
            fused_prompt = ""
            try:
                from backend.agents.knowledge_fusion import fuse
                fused        = fuse(
                    query=                query,
                    qdrant_chunks=        chunk_dicts,
                    paper_node_id=        paper_id,
                    conversation_history= history,
                )
                fused_prompt = fused.prompt_text
                g            = fused.graph_entities
                logger.info(
                    f"[ResearchAgent] Graph authors={len(g.get('authors',[]))} "
                    f"kws={len(g.get('keywords',[]))} domain='{g.get('domain','')}'"
                )
            except Exception as e:
                logger.warning(f"[ResearchAgent] Fusion: {e}")
                fused_prompt = "\n\n".join(
                    f"[Chunk {i+1} | {c['section']} p.{c['page_num']}]\n{c['text']}"
                    for i, c in enumerate(chunk_dicts[:5])
                )

            draft        = _ollama(PAPER_SYSTEM, _PAPER_PROMPT.format(fused_context=fused_prompt))
            final_answer = _gemini(
                REFINE_SYSTEM,
                _REFINE_PROMPT.format(fused_context=fused_prompt, draft=draft, question=query)
            ) or draft or "AstroNexus AI was unable to generate an answer."

            # Evaluate
            try:
                from backend.agents.live_evaluator import evaluate_paper_answer
                ev        = evaluate_paper_answer(query, final_answer, chunk_dicts)
                eval_dict = ev.to_dict()
                logger.info(
                    f"[ResearchAgent] grade={ev.grade} BLEU={ev.bleu:.3f} "
                    f"F1={ev.f1:.3f} gnd={ev.grounding:.3f} cite={ev.citation_coverage:.3f}"
                )
                try:
                    from backend.agents.eval_logger import save_eval
                    save_eval(
                        query=query, answer=final_answer, mode=mode,
                        eval_dict=eval_dict, paper_id=paper_id,
                        top_score=top_score, latency_s=time.perf_counter()-t0,
                    )
                except Exception: pass
            except Exception as e:
                logger.warning(f"[ResearchAgent] Eval: {e}")

    # ══════════════════════════════════════════════════════════════════════════
    # LIVE DATA
    # ══════════════════════════════════════════════════════════════════════════
    elif mode == "live_data":
        try:
            from backend.agents.api_router import select_apis
            from backend.agents.api_fusion import call_apis, format_for_prompt
            api_data = format_for_prompt(call_apis(select_apis(query)))
            if api_data:
                final_answer = _gemini(
                    ASTRONOMY_SYSTEM,
                    f"Real-time data:\n{api_data}\n\nQuestion: {query}\n\nCite each source."
                )
        except Exception as e:
            logger.warning(f"[ResearchAgent] Live: {e}")

    # ══════════════════════════════════════════════════════════════════════════
    # GENERAL
    # ══════════════════════════════════════════════════════════════════════════
    else:
        draft        = _ollama(GENERAL_SYSTEM, f"Question: {query}")
        final_answer = _gemini(GENERAL_SYSTEM, f"Improve:\n{draft}\n\nQ: {query}") or draft

    if not final_answer:
        final_answer = "AstroNexus AI was unable to generate an answer. Please check Ollama is running."

    updated_history = list(history) + [{
        "turn": turn_count, "query": query,
        "answer": final_answer[:500], "mode": mode,
    }]

    elapsed = time.perf_counter() - t0
    logger.info(f"[ResearchAgent] done {elapsed:.1f}s")

    return {
        **state,
        "final_answer":         final_answer,
        "conversation_history": updated_history,
        "metadata": {
            **metadata,
            "mode":       mode,
            "top_score":  round(top_score, 4),
            "evaluation": eval_dict,
            "latency_s":  round(elapsed, 2),
        },
    }