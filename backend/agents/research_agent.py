"""
AstroNexus AI — Research Agent (complete pipeline)

Full flow on every query:
    1. Retrieve chunks from Qdrant (semantic search)
    2. Pull graph context from Neo4j (entities, keywords, authors, satellites)
    3. Pull image keywords from Neo4j (if image was uploaded this session)
    4. Knowledge Fusion — merge all three into structured prompt
    5. Ollama — fast local draft answer
    6. Gemini — refine + improve the Ollama draft
    7. Evaluator — confidence, grounding, hallucination, citations
    8. Return final answer + full evaluation metadata
"""
from __future__ import annotations

import json
import logging
import os
import urllib.request

from backend.agents.state import AgentState

logger = logging.getLogger(__name__)

OLLAMA_BASE  = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL",    "qwen3:4b")
GEMINI_KEY   = os.environ.get("GEMINI_API_KEY",  "")

_SYSTEM = """\
You are a scientific research assistant for AstroNexus AI.
You have access to:
  - Document excerpts retrieved from uploaded papers
  - Knowledge graph context (authors, models, datasets, keywords, satellites)
  - Image analysis keywords (if a satellite image was uploaded)

Rules:
  - Answer using ONLY the provided context
  - Cite section names when referencing specific content
  - If context is insufficient, say so — do not invent details
  - Do not fabricate section numbers, page numbers, equations, or scores
  - Use graph context to enrich answers with entity relationships"""

_REFINE_PROMPT = """\
You are refining a scientific answer.

Original answer from local model:
{ollama_answer}

Full context used:
{context}

Question: {question}

Improve the answer by:
1. Adding specific citations from the document excerpts (Section name, page)
2. Incorporating relevant graph facts (authors, models, datasets, satellites)
3. Correcting any inaccuracies
4. Making it more precise and well-structured

Refined answer:"""


# ══════════════════════════════════════════════════════════════════════════════
# LLM CALLS
# ══════════════════════════════════════════════════════════════════════════════

def _call_ollama(prompt: str) -> str:
    """Fast local draft via Ollama."""
    payload = json.dumps({
        "model":  OLLAMA_MODEL,
        "prompt": prompt,
        "system": _SYSTEM,
        "stream": False,
        "options": {"temperature": 0.2, "num_predict": 512},
    }).encode()

    try:
        req = urllib.request.Request(
            f"{OLLAMA_BASE}/api/generate",
            data=payload, headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.loads(resp.read()).get("response", "").strip()
    except Exception as e:
        logger.warning(f"[ResearchAgent] Ollama failed: {e}")
        return ""


def _call_gemini(question: str, context: str, ollama_draft: str) -> str:
    """Refine Ollama draft using Gemini for higher quality."""
    if not GEMINI_KEY:
        return ""
    try:
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_KEY)
        model    = genai.GenerativeModel("gemini-2.0-flash")
        prompt   = _REFINE_PROMPT.format(
            ollama_answer= ollama_draft,
            context=       context[:4000],
            question=      question,
        )
        response = model.generate_content(
            f"{_SYSTEM}\n\n{prompt}",
            generation_config={"temperature": 0.2, "max_output_tokens": 800},
        )
        return response.text.strip()
    except Exception as e:
        logger.warning(f"[ResearchAgent] Gemini refine failed: {e}")
        return ""


# ══════════════════════════════════════════════════════════════════════════════
# IMAGE KEYWORDS FROM NEO4J (for current session image)
# ══════════════════════════════════════════════════════════════════════════════

def _get_image_keywords_from_graph(image_path: str | None) -> list[str]:
    """Retrieve keywords that were written to Neo4j when the image was uploaded."""
    if not image_path:
        return []
    try:
        from backend.graph.neo4j_client import _get_driver
        driver = _get_driver()
        with driver.session() as s:
            rows = s.run(
                """
                MATCH (img:ImageSession {path: $path})-[:HAS_KEYWORD]->(k:Keyword)
                RETURN k.name AS kw
                """,
                path=str(image_path),
            ).data()
        return [r["kw"] for r in rows]
    except Exception as e:
        logger.warning(f"[ResearchAgent] Image keyword fetch failed: {e}")
        return []


# ══════════════════════════════════════════════════════════════════════════════
# VOICE INPUT KEYWORD EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════

def _extract_voice_keywords(query: str) -> list[str]:
    """
    Extract domain keywords from a voice-transcribed query
    and write them to Neo4j for session context enrichment.
    """
    from backend.graph.domain_classifier import DOMAIN_KEYWORDS
    import re

    found = []
    query_lower = query.lower()

    for domain, kw_groups in DOMAIN_KEYWORDS.items():
        for kw in kw_groups.get("primary", []) + kw_groups.get("secondary", []):
            if re.search(r'\b' + re.escape(kw.lower()) + r'\b', query_lower):
                found.append(kw)

    if found:
        try:
            from backend.graph.neo4j_client import _get_driver
            driver = _get_driver()
            with driver.session() as s:
                for kw in found:
                    s.run(
                        "MERGE (k:Keyword {name: $kw}) SET k.domain='voice_query'",
                        kw=kw,
                    )
        except Exception:
            pass

    return found


# ══════════════════════════════════════════════════════════════════════════════
# MAIN AGENT NODE
# ══════════════════════════════════════════════════════════════════════════════

def research_agent_node(state: AgentState) -> AgentState:
    """
    LangGraph node: full Qdrant + Neo4j + Ollama + Gemini + Evaluator pipeline.
    """
    query      = state.get("query", "")
    metadata   = state.get("metadata") or {}
    image_path = metadata.get("image_path")
    paper_id   = metadata.get("paper_id")
    audio_path = state.get("audio_path")

    logger.info(f"[ResearchAgent] ── Query: {query[:60]} ──")

    # ── Step 1: Retrieve from Qdrant ──────────────────────────────────────────
    retrieved_chunks = []
    chunk_dicts      = []

    try:
        from backend.rag.retriever import retrieve

        retrieved_chunks = retrieve(query, top_k=5)
        chunk_dicts = [
            {
                "score":   c.score,
                "text":    c.text,
                "section": c.section,
                "page_num":c.page_num,
                "title":   c.title,
                "payload": {
                    "text":    c.text,
                    "section": c.section,
                    "page_num":c.page_num,
                    "title":   c.title,
                },
            }
            for c in retrieved_chunks
        ]
        logger.info(
            f"[ResearchAgent] Qdrant: {len(retrieved_chunks)} chunks, "
            f"top score: {retrieved_chunks[0].score:.3f if retrieved_chunks else 0:.3f}"
        )
    except Exception as e:
        logger.warning(f"[ResearchAgent] Qdrant retrieval failed: {e}")

    # ── Step 2: Get image keywords from Neo4j ─────────────────────────────────
    image_keywords = _get_image_keywords_from_graph(image_path)
    if image_keywords:
        logger.info(f"[ResearchAgent] Image keywords from graph: {image_keywords}")

    # ── Step 3: Extract voice keywords (if voice query) ───────────────────────
    if audio_path and query:
        voice_kws = _extract_voice_keywords(query)
        if voice_kws:
            logger.info(f"[ResearchAgent] Voice keywords extracted: {voice_kws}")

    # ── Step 4: Knowledge Fusion (Qdrant + Neo4j + image keywords) ────────────
    try:
        from backend.agents.knowledge_fusion import fuse

        fused = fuse(
            query=          query,
            qdrant_chunks=  chunk_dicts,
            paper_node_id=  paper_id,
            image_keywords= image_keywords,
        )
        fused_prompt = fused.prompt_text
        logger.info("[ResearchAgent] Knowledge fusion complete")
    except Exception as e:
        logger.warning(f"[ResearchAgent] Knowledge fusion failed, using raw chunks: {e}")
        fused_prompt = "\n\n".join(c.get("text","") for c in chunk_dicts[:5])

    # ── Step 5: Ollama — fast local draft ─────────────────────────────────────
    logger.info("[ResearchAgent] Calling Ollama...")
    ollama_answer = _call_ollama(
        f"{fused_prompt}\n\nQuestion: {query}\n\nAnswer:"
    )
    if ollama_answer:
        logger.info(f"[ResearchAgent] Ollama draft: {ollama_answer[:80]}...")
    else:
        logger.warning("[ResearchAgent] Ollama returned empty response")
        ollama_answer = "I was unable to generate an initial draft."

    # ── Step 6: Gemini — refine the Ollama draft ──────────────────────────────
    logger.info("[ResearchAgent] Calling Gemini to refine...")
    gemini_answer = _call_gemini(query, fused_prompt, ollama_answer)

    # Use Gemini if it returned something, otherwise keep Ollama
    final_raw = gemini_answer if gemini_answer else ollama_answer
    logger.info(f"[ResearchAgent] Final answer source: {'Gemini' if gemini_answer else 'Ollama'}")

    # ── Step 7: Evaluate ──────────────────────────────────────────────────────
    try:
        from backend.agents.answer_evaluator import get_evaluator

        eval_result = get_evaluator().evaluate(
            query=            query,
            answer=           final_raw,
            retrieved_chunks= chunk_dicts,
        )
        final_answer = eval_result.answer

        logger.info(
            f"[ResearchAgent] Evaluation — "
            f"confidence={eval_result.confidence} "
            f"grounding={eval_result.grounding_score:.2f} "
            f"reliable={eval_result.is_reliable} "
            f"warnings={len(eval_result.warnings)}"
        )

        eval_dict = eval_result.to_dict()

    except Exception as e:
        logger.warning(f"[ResearchAgent] Evaluator failed: {e}")
        final_answer = final_raw
        eval_dict    = {"error": str(e)}

    # ── Step 8: Return ────────────────────────────────────────────────────────
    return {
        **state,
        "rag_context":  fused_prompt[:2000],
        "final_answer": final_answer,
        "metadata": {
            **metadata,
            "evaluation":      eval_dict,
            "image_keywords":  image_keywords,
            "ollama_used":     bool(ollama_answer),
            "gemini_used":     bool(gemini_answer),
        },
    }