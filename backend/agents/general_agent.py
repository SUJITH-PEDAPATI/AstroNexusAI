"""
AstroNexus AI — General Agent v5.2 with identity
"""
from __future__ import annotations

import json, logging, os, urllib.request
from backend.agents.state    import AgentState
from backend.agents.identity import (
    ASTRONOMY_SYSTEM, SCIENCE_SYSTEM, GENERAL_SYSTEM, SCOPE_NOTE
)

logger = logging.getLogger(__name__)

OLLAMA_BASE  = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL",    "qwen3:4b")
GEMINI_KEY   = os.environ.get("GEMINI_API_KEY",  "")

ASTRONOMY_TERMS = {
    "star","galaxy","black hole","nebula","quasar","pulsar","exoplanet",
    "dark matter","dark energy","cosmology","universe","telescope","orbit",
    "nasa","esa","isro","jaxa","satellite","spacecraft","jwst","hubble",
    "astrophysics","spectroscopy","photometry","redshift","supernova",
    "remote sensing","earth observation","sentinel","landsat","sar","ndvi",
    "multispectral","hyperspectral","flood","wildfire","climate","atmosphere",
    "infrared","cosmic","asteroid","comet","planet",
}
SCIENCE_TERMS = {
    "physics","chemistry","biology","mathematics","quantum","relativity",
    "molecule","atom","dna","protein","evolution","calculus","neural",
    "machine learning","deep learning","algorithm","transformer",
    "computer science","programming","software","hardware",
}


def _classify_domain(query: str) -> str:
    q = query.lower()
    for t in ASTRONOMY_TERMS:
        if t in q: return "astronomy"
    for t in SCIENCE_TERMS:
        if t in q: return "science"
    return "general"


def _gemini(system: str, prompt: str) -> str:
    if not GEMINI_KEY: return ""
    try:
        from google import genai
        from google.genai import types
        client   = genai.Client(api_key=GEMINI_KEY)
        response = client.models.generate_content(
            model=    "gemini-2.5-flash",
            contents= f"{system}\n\n{prompt}",
            config=   types.GenerateContentConfig(temperature=0.2, max_output_tokens=800),
        )
        return response.text.strip()
    except Exception as e:
        logger.warning(f"[GeneralAgent] Gemini: {e}")
        return ""


def _ollama(system: str, prompt: str) -> str:
    payload = json.dumps({
        "model": OLLAMA_MODEL, "prompt": prompt, "system": system,
        "stream": False, "options": {"temperature": 0.3, "num_predict": 500},
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
        logger.warning(f"[GeneralAgent] Ollama: {e}")
        return ""


def _get_api_data(query: str) -> str:
    try:
        from backend.agents.api_router import select_apis
        from backend.agents.api_fusion import call_apis, format_for_prompt
        candidates = select_apis(query)
        if not candidates: return ""
        return format_for_prompt(call_apis(candidates)) or ""
    except Exception as e:
        logger.debug(f"[GeneralAgent] API: {e}")
        return ""


def general_agent_node(state: AgentState) -> AgentState:
    query    = state.get("query", "")
    metadata = state.get("metadata") or {}
    domain   = _classify_domain(query)

    logger.info(f"[GeneralAgent] domain={domain} '{query[:50]}'")

    answer = ""

    if domain == "astronomy":
        api_data = _get_api_data(query)
        if api_data:
            answer = _gemini(ASTRONOMY_SYSTEM,
                f"Real-time data:\n{api_data}\n\nQuestion: {query}\n\nAnswer:")
        if not answer:
            answer = _gemini(ASTRONOMY_SYSTEM, f"Question: {query}\n\nAnswer:")

    elif domain == "science":
        answer = _gemini(SCIENCE_SYSTEM, f"Question: {query}\n\nAnswer:")

    else:
        draft  = _ollama(GENERAL_SYSTEM, f"Question: {query}")
        answer = _gemini(GENERAL_SYSTEM,
            f"Improve this answer:\n{draft}\n\nQuestion: {query}") or draft

    if not answer:
        answer = _ollama(GENERAL_SYSTEM, f"Question: {query}") or \
                 "AstroNexus AI was unable to generate an answer."

    if domain != "astronomy":
        answer += SCOPE_NOTE

    return {
        **state,
        "final_answer": answer,
        "metadata": {**metadata, "domain": domain},
    }