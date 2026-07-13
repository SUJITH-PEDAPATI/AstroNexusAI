"""
Answer Generator for AstroNexus RAG Evaluation.

Takes retrieved chunks + question → calls LLM → returns generated answer.
Uses HuggingFace Inference API (same as rest of pipeline).
"""
from __future__ import annotations

import logging
import os
import time

from backend.evaluation.models import QAPair, GenerationResult
from backend.evaluation.retrieval_evaluator import retrieve_for_question

logger = logging.getLogger(__name__)

# Prompt template for scientific QA
_QA_PROMPT_TEMPLATE = """You are a scientific assistant answering questions about research papers.
Answer the question based ONLY on the provided context.
Be concise and precise. If the answer is not in the context, say "Not found in context."

Context:
{context}

Question: {question}

Answer:"""


def _call_llm(prompt: str, token: str) -> str:
    """Call HF Inference API for answer generation."""
    import requests

    # model_id = "mistralai/Mistral-7B-Instruct-v0.3"
    model_id = "Qwen/Qwen2.5-7B-Instruct"
    url      = f"https://api-inference.huggingface.co/models/{model_id}"

    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 256,
            "temperature":    0.1,       # low temp for factual QA
            "return_full_text": False,
        },
        "options": {"wait_for_model": True},
    }

    for attempt in range(3):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=60)
            if resp.status_code == 200:
                raw = resp.json()
                return raw[0].get("generated_text", "").strip() if raw else ""
            elif resp.status_code == 503:
                logger.warning(f"[Generator] Model loading — retry {attempt+1}/3")
                time.sleep(20)
        except Exception as e:
            logger.warning(f"[Generator] API call failed: {e}")
            break

    return "Generation failed"


def generate_answer(
    qa:     QAPair,
    top_k:  int = 5,
) -> GenerationResult:
    """
    Full RAG generation for a single QA pair.

    Steps:
        1. Retrieve top-k chunks for the question
        2. Build context from retrieved chunks
        3. Call LLM with context + question
        4. Return GenerationResult
    """
    token = os.environ.get("HF_API_TOKEN") or os.environ.get("HUGGINGFACE_API_KEY")
    if not token:
        raise EnvironmentError("HF_API_TOKEN not found in environment")

    # Step 1: Retrieve
    retrieval = retrieve_for_question(qa, top_k=top_k)
    contexts  = retrieval.retrieved_chunks

    if not contexts:
        return GenerationResult(
            question_id=     qa.question_id,
            question=        qa.question,
            generated_answer="No context retrieved",
            ground_truth=    qa.ground_truth,
            context_used=    [],
        )

    # Step 2: Build context string
    context_str = "\n\n---\n\n".join(
        f"[Chunk {i+1}]\n{chunk}"
        for i, chunk in enumerate(contexts)
    )

    # Step 3: Build prompt and call LLM
    prompt = _QA_PROMPT_TEMPLATE.format(
        context=context_str,
        question=qa.question,
    )

    generated = _call_llm(prompt, token)

    return GenerationResult(
        question_id=     qa.question_id,
        question=        qa.question,
        generated_answer=generated,
        ground_truth=    qa.ground_truth,
        context_used=    contexts,
    )