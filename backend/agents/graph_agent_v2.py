"""
AstroNexus AI — General Agent v2

Now uses External API Intelligence Layer for real-time data.

Flow:
    query
      ↓
    APIRouter.select_apis()     ← Neo4j keyword → API lookup
      ↓
    APIFusionLayer.call()       ← parallel API calls
      ↓
    Build enriched prompt       ← real data + general knowledge
      ↓
    Ollama draft
      ↓
    Gemini refine (if API data present)
      ↓
    Final answer with real-time data cited
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
You are a helpful assistant for AstroNexus AI.
When real-time data is provided in the context, use it to give accurate,
specific answers. Always mention the data source.
If no real-time data is available, answer from general knowledge and say so."""

_SCOPE_NOTE = """\

---
💡 AstroNexus AI is primarily designed for:
  • Scientific paper Q&A — upload a PDF and ask questions
  • Satellite image analysis — upload an image for AI analysis
  • Knowledge graph queries — ask about authors, models, datasets"""


def _call_ollama(prompt: str) -> str:
    payload = json.dumps({
        "model":  OLLAMA_MODEL,
        "prompt": prompt,
        "system": _SYSTEM,
        "stream": False,
        "options": {"temperature": 0.3, "num_predict": 600},
    }).encode()
    try:
        req = urllib.request.Request(
            f"{OLLAMA_BASE}/api/generate",
            data=payload, headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read()).get("response", "").strip()
    except Exception as e:
        logger.warning(f"[GeneralAgent] Ollama failed: {e}")
        return ""


def _call_gemini(prompt: str) -> str:
    if not GEMINI_KEY:
        return ""
    try:
        import google.genai as genai
        client   = genai.Client(api_key=GEMINI_KEY)
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=f"{_SYSTEM}\n\n{prompt}",
            config={"temperature": 0.3, "max_output_tokens": 600},
        )
        return response.text.strip()
    except Exception as e:
        logger.warning(f"[GeneralAgent] Gemini failed: {e}")
        return ""


def general_agent_node(state: AgentState) -> AgentState:
    """
    LangGraph node: handle general and out-of-scope queries.
    Uses External API Intelligence Layer for real-time data.
    """
    query = state.get("query", "")
    logger.info(f"[GeneralAgent] Query: {query[:60]}")

    # ── Step 1: API Intelligence Layer ────────────────────────────────────────
    api_context = ""
    api_names   = []

    try:
        from backend.agents.api_router  import select_apis
        from backend.agents.api_fusion  import call_apis, format_for_prompt

        candidates = select_apis(query)

        if candidates:
            logger.info(
                f"[GeneralAgent] APIs selected: "
                f"{[c.name for c in candidates]}"
            )
            results    = call_apis(candidates)
            api_context = format_for_prompt(results)
            api_names   = [r.api_name for r in results if r.success]
        else:
            logger.info("[GeneralAgent] No APIs triggered for this query")

    except Exception as e:
        logger.warning(f"[GeneralAgent] API layer failed (non-fatal): {e}")

    # ── Step 2: Build prompt ───────────────────────────────────────────────────
    prompt_parts = []
    if api_context:
        prompt_parts.append(api_context)
        prompt_parts.append("")
    prompt_parts.append(f"Question: {query}")
    if api_context:
        prompt_parts.append(
            "\nUse the real-time data above to answer specifically and accurately. "
            "Cite the data source."
        )
    else:
        prompt_parts.append(
            "\nAnswer from general knowledge. Note that this is general knowledge, "
            "not real-time data."
        )

    full_prompt = "\n".join(prompt_parts)

    # ── Step 3: Generate ───────────────────────────────────────────────────────
    answer = ""

    if api_context and GEMINI_KEY:
        # Real data present → Gemini for better synthesis
        logger.info("[GeneralAgent] Real-time data available → Gemini")
        answer = _call_gemini(full_prompt)

    if not answer:
        logger.info("[GeneralAgent] Calling Ollama...")
        answer = _call_ollama(full_prompt)

    if not answer:
        answer = (
            "I was unable to retrieve data for this query. "
            "Please check that Ollama is running."
        )

    # ── Step 4: Add scope note only if no real API data ────────────────────────
    if not api_names:
        answer += _SCOPE_NOTE
    else:
        answer += f"\n\n*Data sourced from: {', '.join(api_names)}*"

    logger.info(
        f"[GeneralAgent] Done — "
        f"apis_used={api_names} "
        f"chars={len(answer)}"
    )

    return {
        **state,
        "final_answer": answer,
        "metadata": {
            **(state.get("metadata") or {}),
            "apis_called":  api_names,
            "had_real_data": bool(api_names),
        },
    }