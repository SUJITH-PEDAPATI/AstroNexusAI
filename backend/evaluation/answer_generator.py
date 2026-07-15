"""
AstroNexus AI — Answer Generator (Step 2 Fixed)

Backend priority:
    1. Ollama  (local — always available, no internet required)
    2. HuggingFace Inference API (fallback — requires internet)

If both fail, returns a clearly labeled error string so evaluation
metrics reflect the actual failure rather than silent empty responses.

Only this file changes — all other modules are untouched.
"""
from __future__ import annotations

import logging
import os
import time
from typing import Optional

from backend.evaluation.models import QAPair, GenerationResult
from backend.evaluation.retrieval_evaluator import retrieve_for_question

logger = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────────────────────────────────────

# Ollama settings (primary backend)
OLLAMA_BASE_URL  = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL     = os.environ.get("OLLAMA_MODEL",    "qwen2.5:7b")  # or mistral, llama3.2, etc.

# HuggingFace settings (fallback backend)
HF_MODEL_ID      = os.environ.get("HF_GENERATOR_MODEL", "mistralai/Mistral-7B-Instruct-v0.3")
HF_API_URL       = f"https://api-inference.huggingface.co/models/{HF_MODEL_ID}"

# Retry settings
MAX_RETRIES      = 3
RETRY_DELAY      = 5     # seconds between retries
REQUEST_TIMEOUT  = 60    # seconds per request

# Generation settings
MAX_NEW_TOKENS   = 256
TEMPERATURE      = 0.1   # low — factual QA needs determinism

# Scientific QA prompt — citation-aware, hallucination-resistant
_QA_PROMPT = """You are a scientific research assistant. Answer the question using ONLY the provided context.

Rules:
- If the answer is in the context, give a concise direct answer.
- Cite your source as [Section, p.N] after each factual claim.
- If the context does not contain the answer, say: "Not found in the provided context."
- Never invent facts, numbers, or citations.

Context:
{context}

Question: {question}

Answer:"""


# ══════════════════════════════════════════════════════════════════════════════
# BACKEND 1 — OLLAMA (primary)
# ══════════════════════════════════════════════════════════════════════════════

def _is_ollama_available() -> bool:
    """Check if Ollama server is reachable before attempting generation."""
    import urllib.request
    try:
        req = urllib.request.Request(
            f"{OLLAMA_BASE_URL}/api/tags",
            method="GET",
        )
        with urllib.request.urlopen(req, timeout=3) as resp:
            return resp.status == 200
    except Exception:
        return False


def _generate_ollama(prompt: str) -> Optional[str]:
    """
    Generate answer using local Ollama server.

    Uses /api/generate endpoint with stream=False for simplicity.
    Retries on connection errors but not on model errors.
    """
    import json
    import urllib.request
    import urllib.error

    url     = f"{OLLAMA_BASE_URL}/api/generate"
    payload = json.dumps({
        "model":  OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature":  TEMPERATURE,
            "num_predict":  MAX_NEW_TOKENS,
        },
    }).encode("utf-8")

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            req = urllib.request.Request(
                url,
                data=    payload,
                headers= {"Content-Type": "application/json"},
                method=  "POST",
            )
            with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
                if resp.status == 200:
                    data     = json.loads(resp.read().decode("utf-8"))
                    response = data.get("response", "").strip()
                    if response:
                        logger.info(f"[Generator] Ollama succeeded (attempt {attempt})")
                        return response
                    else:
                        logger.warning("[Generator] Ollama returned empty response")
                        return None
                else:
                    logger.warning(
                        f"[Generator] Ollama HTTP {resp.status} "
                        f"(attempt {attempt}/{MAX_RETRIES})"
                    )

        except urllib.error.URLError as e:
            logger.warning(
                f"[Generator] Ollama connection failed "
                f"(attempt {attempt}/{MAX_RETRIES}): {e.reason}"
            )
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY)

        except TimeoutError:
            logger.warning(
                f"[Generator] Ollama timeout after {REQUEST_TIMEOUT}s "
                f"(attempt {attempt}/{MAX_RETRIES})"
            )
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY)

        except Exception as e:
            logger.error(f"[Generator] Ollama unexpected error: {type(e).__name__}: {e}")
            break   # non-retryable error

    logger.error(f"[Generator] Ollama failed after {MAX_RETRIES} attempts")
    return None


# ══════════════════════════════════════════════════════════════════════════════
# BACKEND 2 — HUGGINGFACE INFERENCE API (fallback)
# ══════════════════════════════════════════════════════════════════════════════

def _generate_hf_api(prompt: str) -> Optional[str]:
    """
    Generate answer using HuggingFace Inference API.

    Fallback when Ollama is unavailable (e.g. not installed).
    Handles: 503 model loading, 429 rate limiting, DNS failures, timeouts.
    """
    token = os.environ.get("HF_API_TOKEN") or os.environ.get("HUGGINGFACE_API_KEY")
    if not token:
        logger.warning("[Generator] HF_API_TOKEN not set — cannot use HF fallback")
        return None

    try:
        import requests
    except ImportError:
        logger.warning("[Generator] requests not installed — cannot use HF fallback")
        return None

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type":  "application/json",
    }
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens":  MAX_NEW_TOKENS,
            "temperature":     TEMPERATURE,
            "return_full_text": False,
        },
        "options": {"wait_for_model": True},
    }

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.post(
                HF_API_URL,
                headers= headers,
                json=    payload,
                timeout= REQUEST_TIMEOUT,
            )

            if resp.status_code == 200:
                data      = resp.json()
                generated = data[0].get("generated_text", "").strip() if data else ""
                if generated:
                    logger.info(f"[Generator] HF API succeeded (attempt {attempt})")
                    return generated
                else:
                    logger.warning("[Generator] HF API returned empty text")
                    return None

            elif resp.status_code == 503:
                logger.warning(
                    f"[Generator] HF model loading (503) — "
                    f"attempt {attempt}/{MAX_RETRIES}, "
                    f"waiting {RETRY_DELAY * 4}s"
                )
                time.sleep(RETRY_DELAY * 4)   # model loading takes longer

            elif resp.status_code == 429:
                logger.warning(
                    f"[Generator] HF rate limited (429) — "
                    f"waiting {RETRY_DELAY * 12}s"
                )
                time.sleep(RETRY_DELAY * 12)

            elif resp.status_code == 401:
                logger.error("[Generator] HF API: invalid token (401)")
                return None   # not retryable

            else:
                logger.warning(
                    f"[Generator] HF API HTTP {resp.status_code}: "
                    f"{resp.text[:200]} (attempt {attempt}/{MAX_RETRIES})"
                )
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_DELAY)

        except Exception as e:
            # Catches DNS failures, connection refused, timeouts, etc.
            error_type = type(e).__name__
            logger.warning(
                f"[Generator] HF API error ({error_type}) "
                f"attempt {attempt}/{MAX_RETRIES}: {e}"
            )
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY)

    logger.error(f"[Generator] HF API failed after {MAX_RETRIES} attempts")
    return None


# ══════════════════════════════════════════════════════════════════════════════
# ROUTER — try Ollama first, then HF API
# ══════════════════════════════════════════════════════════════════════════════

def _generate(prompt: str) -> str:
    """
    Route generation request to the best available backend.

    Order:
        1. Ollama (local — check availability first, then generate)
        2. HF Inference API (network fallback)
        3. Labeled failure string (never empty — metrics stay meaningful)
    """
    # Backend 1: Ollama
    if _is_ollama_available():
        logger.info(f"[Generator] Using Ollama ({OLLAMA_MODEL})")
        result = _generate_ollama(prompt)
        if result:
            return result
        logger.warning("[Generator] Ollama available but generation failed — trying HF API")
    else:
        logger.info(
            f"[Generator] Ollama not available at {OLLAMA_BASE_URL} "
            f"— trying HF API fallback"
        )

    # Backend 2: HF API
    result = _generate_hf_api(prompt)
    if result:
        return result

    # Both failed — return labeled failure (never empty string)
    # This ensures evaluation metrics show 0 rather than undefined behavior
    logger.error(
        "[Generator] All backends failed. "
        "Start Ollama with: ollama serve && ollama pull qwen2.5:7b"
    )
    return "[GENERATION_FAILED: No backend available. Start Ollama or set HF_API_TOKEN.]"


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC API — unchanged interface
# ══════════════════════════════════════════════════════════════════════════════

def generate_answer(
    qa:    QAPair,
    top_k: int = 5,
) -> GenerationResult:
    """
    Full RAG generation for a single QA pair.

    Interface is identical to the previous version —
    no other module needs to change.

    Steps:
        1. Retrieve top-k chunks for the question
        2. Build context string from retrieved chunks
        3. Call _generate() which tries Ollama → HF API → labeled failure
        4. Return GenerationResult
    """
    # Step 1: Retrieve
    retrieval = retrieve_for_question(qa, top_k=top_k)
    contexts  = retrieval.retrieved_chunks

    if not contexts:
        logger.warning(
            f"[Generator] No context retrieved for question: "
            f"'{qa.question[:60]}'"
        )
        return GenerationResult(
            question_id=     qa.question_id,
            question=        qa.question,
            generated_answer="[NO_CONTEXT_RETRIEVED]",
            ground_truth=    qa.ground_truth,
            context_used=    [],
        )

    # Step 2: Build context string with section + page metadata
    context_parts = []
    for i, chunk in enumerate(contexts):
        context_parts.append(f"[Chunk {i+1}]\n{chunk}")
    context_str = "\n\n---\n\n".join(context_parts)

    # Step 3: Build prompt and generate
    prompt    = _QA_PROMPT.format(context=context_str, question=qa.question)
    generated = _generate(prompt)

    logger.info(
        f"[Generator] Q: '{qa.question[:50]}...' "
        f"→ A: '{generated[:80]}...'"
    )

    return GenerationResult(
        question_id=     qa.question_id,
        question=        qa.question,
        generated_answer=generated,
        ground_truth=    qa.ground_truth,
        context_used=    contexts,
    )