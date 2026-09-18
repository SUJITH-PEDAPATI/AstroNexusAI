"""
AstroNexus AI — Web Search Agent
===================================
LangGraph node that handles web_search and hybrid query routes.

Pipeline:
    query → QueryRouter classifies as web_search or hybrid
         → WebSearchAgent runs
         → web_search_service.search()   [parallel with RAG if hybrid]
         → hybrid_retriever.merge()
         → Ollama/Gemini generates answer with source attribution
         → final_answer + web_sources in metadata

This node is purely additive — it does not replace ResearchAgent.
For hybrid queries, it calls the retriever AND the web search service,
then merges both contexts before generation.

Source attribution in the LLM answer:
    📄 [Section, p.N]    → from uploaded papers
    🌐 [source_name]     → from web search
"""
from __future__ import annotations

import json
import logging
import os
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError

from backend.agents.state import AgentState

logger = logging.getLogger(__name__)


def _get_provider() -> str:
    """Safely get the current web search provider name."""
    try:
        try:
            from backend.services.web_search import web_search_service
        except ImportError:
            from web_search import web_search_service
        return getattr(web_search_service, "provider", "ddg")
    except Exception:
        return "ddg"

OLLAMA_BASE  = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL",    "qwen3:4b")
GEMINI_KEY   = os.environ.get("GEMINI_API_KEY",  "")
WEB_RESULTS  = int(os.environ.get("WEB_SEARCH_RESULTS", "5"))
PARALLEL_TIMEOUT = 12   # seconds to wait for parallel retrieval

# ── System prompts ────────────────────────────────────────────────────────────

_WEB_SYSTEM = """\
You are AstroNexus AI, a scientific research assistant specialising in astronomy,
astrophysics, remote sensing, and AI research.

When answering using WEB SEARCH RESULTS:
  - Cite each fact with [🌐 source_name], e.g. [🌐 nasa.gov]
  - Only state facts present in the provided context
  - Clearly indicate if information is time-sensitive or subject to change

When answering using RESEARCH PAPER CONTEXT:
  - Cite with [Section, p.N] format
  - Do not hallucinate document contents

When answering using BOTH sources:
  - Clearly distinguish 📄 paper information from 🌐 web information
  - Label each claim with its source type

Never invent facts. If uncertain, say so."""

_WEB_PROMPT = """\
Context:
{context}

Question: {question}

Provide a comprehensive answer with source citations:"""


# ── LLM callers (reused from existing agents) ─────────────────────────────────

def _ollama(system: str, prompt: str) -> str:
    payload = json.dumps({
        "model":  OLLAMA_MODEL,
        "prompt": prompt,
        "system": system,
        "stream": False,
        "options": {"temperature": 0.2, "num_predict": 700},
    }).encode()
    try:
        req = urllib.request.Request(
            f"{OLLAMA_BASE}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.loads(resp.read()).get("response", "").strip()
    except Exception as e:
        logger.warning(f"[WebSearchAgent/Ollama] {e}")
        return ""


def _gemini(system: str, prompt: str) -> str:
    if not GEMINI_KEY:
        return ""
    try:
        from google import genai
        from google.genai import types
        client   = genai.Client(api_key=GEMINI_KEY)
        response = client.models.generate_content(
            model=    "gemini-2.5-flash",
            contents= f"{system}\n\n{prompt}",
            config=   types.GenerateContentConfig(
                temperature=0.2, max_output_tokens=800,
            ),
        )
        return response.text.strip()
    except Exception as e:
        logger.warning(f"[WebSearchAgent/Gemini] {e}")
        return ""


# ── Parallel retrieval helper ─────────────────────────────────────────────────

def _fetch_rag(query: str, paper_id: str | None) -> tuple[str, list]:
    """Retrieve RAG chunks. Returns (context_str, chunks_list)."""
    try:
        from backend.rag.retriever import retrieve, retrieve_for_rag
        chunks = retrieve(query, top_k=5, filter_paper_id=paper_id)
        text   = retrieve_for_rag(query, top_k=5)
        return text, chunks
    except Exception as e:
        logger.warning(f"[WebSearchAgent] RAG retrieval failed: {e}")
        return "", []


def _fetch_web(query: str, n: int) -> list:
    """Web search. Returns list[WebResult]."""
    try:
        try:
            from backend.services.web_search import web_search_service
        except ImportError:
            from web_search import web_search_service
        return web_search_service.search(query, n=n)
    except Exception as e:
        logger.warning(f"[WebSearchAgent] Web search failed: {e}")
        return []


# ── Main agent node ───────────────────────────────────────────────────────────

def web_search_agent_node(state: AgentState) -> AgentState:
    """
    LangGraph node: hybrid web + RAG research.

    Handles both:
        query_type == "web_search"  → web only
        query_type == "hybrid"      → web + local RAG in parallel
    """
    query      = state.get("query", "")
    query_type = state.get("query_type", "web_search")
    metadata   = dict(state.get("metadata") or {})
    paper_id   = metadata.get("paper_id")
    paper_loaded = state.get("paper_loaded") or bool(paper_id)

    logger.info(
        f"[WebSearchAgent] query_type={query_type} "
        f"paper_id={paper_id} query='{query[:55]}'"
    )

    rag_context = state.get("rag_context") or ""
    rag_chunks: list = []
    web_results: list = []

    # ── Parallel fetch for hybrid queries ──────────────────────────────────────
    if query_type == "hybrid" and paper_loaded:
        with ThreadPoolExecutor(max_workers=2) as pool:
            rag_future = pool.submit(_fetch_rag, query, paper_id)
            web_future = pool.submit(_fetch_web, query, WEB_RESULTS)
            try:
                rag_context, rag_chunks = rag_future.result(timeout=PARALLEL_TIMEOUT)
            except (TimeoutError, Exception) as e:
                logger.warning(f"[WebSearchAgent] RAG future failed: {e}")
            try:
                web_results = web_future.result(timeout=PARALLEL_TIMEOUT)
            except (TimeoutError, Exception) as e:
                logger.warning(f"[WebSearchAgent] Web future failed: {e}")
    else:
        # Web-only: just search the web
        web_results = _fetch_web(query, WEB_RESULTS)

    # ── Automatic RAG fallback if web came back empty ─────────────────────────
    if not web_results and not rag_context and paper_loaded:
        logger.info("[WebSearchAgent] Web empty → falling back to RAG")
        rag_context, rag_chunks = _fetch_rag(query, paper_id)

    # ── Merge contexts ────────────────────────────────────────────────────────
    try:
        from backend.services.hybrid_retriever import merge
    except ImportError:
        try:
            from hybrid_retriever import merge
        except ImportError as _imp_err:
            logger.error("[WebSearchAgent] hybrid_retriever missing — "
                         "copy hybrid_retriever.py to backend/services/. Error: %s", _imp_err)
            answer = ("I was routed to perform a web search, but the hybrid_retriever "
                      "module is missing. Please copy web_search.py and hybrid_retriever.py "
                      "to backend/services/ then restart the server.")
            return {**state, "final_answer": answer, "metadata": metadata}
    graph_context = state.get("graph_context") or ""
    hybrid        = merge(
        query=         query,
        rag_context=   rag_context or None,
        rag_chunks=    rag_chunks or None,
        web_results=   web_results or None,
        graph_context= graph_context or None,
    )

    if not hybrid.text:
        answer = (
            "Unable to retrieve information for your query. "
            "The web search returned no results and no local documents are loaded. "
            "Please try a more specific question or upload a research paper."
        )
        metadata["search_type"] = "failed"
        metadata["web_sources"] = []
        return {**state, "final_answer": answer, "metadata": metadata}

    # ── Generate answer ────────────────────────────────────────────────────────
    prompt = _WEB_PROMPT.format(context=hybrid.text[:4000], question=query)

    # Prefer Gemini for web + hybrid (better at multi-source attribution)
    answer = _gemini(_WEB_SYSTEM, prompt)
    if not answer:
        answer = _ollama(_WEB_SYSTEM, prompt)
    if not answer:
        answer = (
            "Unable to retrieve live web information. "
            "Answer generated from local knowledge only.\n\n"
            + (rag_context or "No local context available.")
        )

    # ── Build source indicator for the frontend ────────────────────────────────
    search_label_map = {
        "hybrid":      "📄 + 🌐 Hybrid Search",
        "web_search":  "🌐 Web Search",
        "local_rag":   "📄 Research Papers",
        "general_llm": "🧠 AI Knowledge",
    }

    metadata.update({
        "search_type":     hybrid.search_type,
        "search_label":    search_label_map.get(hybrid.search_type, "🌐 Web"),
        "web_sources":     hybrid.web_sources,
        "rag_top_score":   hybrid.rag_top_score,
        "rag_chunks_used": hybrid.rag_chunks_used,
        "web_results_n":   len(web_results),
        "provider":        _get_provider(),
    })

    logger.info(
        f"[WebSearchAgent] Done: {hybrid.search_type} "
        f"answer_len={len(answer)} chars"
    )

    return {
        **state,
        "rag_context":  rag_context or hybrid.text,
        "final_answer": answer,
        "metadata":     metadata,
    }