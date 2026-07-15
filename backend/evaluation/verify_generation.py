"""
Generation pipeline verification script.

Run from project root BEFORE running the full evaluation:
    python -m backend.evaluation.verify_generation

Checks:
    1. Ollama is reachable
    2. Ollama model is available
    3. Generation produces a non-empty response
    4. HF API fallback (if token is set)
    5. End-to-end generate_answer() with a real QA pair
"""
from __future__ import annotations

import logging
import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logging.basicConfig(level=logging.INFO, format="%(message)s")


def check(label: str, passed: bool, detail: str = "") -> bool:
    icon = "✓" if passed else "✗"
    print(f"  {icon}  {label}")
    if detail:
        print(f"       {detail}")
    return passed


def run() -> None:
    print("\n" + "=" * 60)
    print("ASTRONEXUS — GENERATION PIPELINE VERIFICATION")
    print("=" * 60)

    from backend.evaluation.answer_generator import (
        OLLAMA_BASE_URL, OLLAMA_MODEL,
        _is_ollama_available, _generate_ollama,
        _generate_hf_api, _generate,
    )

    # ── Check 1: Ollama availability ──────────────────────────────────────────
    print("\n[Check 1] Ollama server")
    ollama_up = _is_ollama_available()
    check("Ollama reachable", ollama_up, f"URL: {OLLAMA_BASE_URL}")

    # ── Check 2: Ollama model list ────────────────────────────────────────────
    print("\n[Check 2] Ollama model availability")
    if ollama_up:
        try:
            import urllib.request, json
            with urllib.request.urlopen(
                f"{OLLAMA_BASE_URL}/api/tags", timeout=5
            ) as resp:
                data   = json.loads(resp.read())
                models = [m["name"] for m in data.get("models", [])]

            model_found = any(OLLAMA_MODEL in m for m in models)
            check(
                f"Model '{OLLAMA_MODEL}' available",
                model_found,
                f"Available models: {models}" if models else "No models found",
            )
            if not model_found:
                print(f"\n       → Pull the model first:")
                print(f"         ollama pull {OLLAMA_MODEL}")
        except Exception as e:
            check("Model list retrieval", False, str(e))
    else:
        check(f"Model '{OLLAMA_MODEL}' available", False,
              "Skipped — Ollama not reachable. Start with: ollama serve")

    # ── Check 3: Ollama generation ────────────────────────────────────────────
    print("\n[Check 3] Ollama generation")
    if ollama_up:
        test_prompt = "What is the capital of France? Answer in one word."
        result = _generate_ollama(test_prompt)
        passed = bool(result) and "[GENERATION_FAILED" not in result
        check("Ollama generates non-empty response", passed,
              f"Response: '{result[:100] if result else 'EMPTY'}'")
    else:
        check("Ollama generation", False, "Skipped — Ollama not reachable")

    # ── Check 4: HF API fallback ──────────────────────────────────────────────
    print("\n[Check 4] HuggingFace API fallback")
    token = os.environ.get("HF_API_TOKEN") or os.environ.get("HUGGINGFACE_API_KEY")
    if token:
        try:
            import socket
            socket.getaddrinfo("api-inference.huggingface.co", 443, timeout=3)
            dns_ok = True
        except Exception:
            dns_ok = False

        check("HF API DNS resolvable", dns_ok,
              "api-inference.huggingface.co" if dns_ok else
              "DNS failure — HF API not available in this environment")

        if dns_ok:
            result = _generate_hf_api("What is 2+2? Answer in one word.")
            passed = bool(result) and "[GENERATION_FAILED" not in result
            check("HF API generates response", passed,
                  f"Response: '{result[:80] if result else 'EMPTY'}'")
    else:
        check("HF_API_TOKEN set", False,
              "Set HF_API_TOKEN in .env to enable HF fallback")

    # ── Check 5: Router (_generate) ───────────────────────────────────────────
    print("\n[Check 5] Backend router")
    result = _generate("What is the main purpose of the Transformer architecture?")
    passed = bool(result) and result != "[NO_CONTEXT_RETRIEVED]"
    failed = "[GENERATION_FAILED" in result
    check("Router returns non-empty response", passed and not failed,
          f"Response: '{result[:100]}'")
    if failed:
        print("\n       → Both backends failed. Fix:")
        print("         Option A: ollama serve && ollama pull qwen2.5:7b")
        print("         Option B: set HF_API_TOKEN in .env (requires internet)")

    # ── Check 6: End-to-end generate_answer() ─────────────────────────────────
    print("\n[Check 6] End-to-end generate_answer() with real QA pair")
    try:
        from backend.evaluation.dataset_loader import load_qasper
        from backend.evaluation.answer_generator import generate_answer

        qa_pairs, _ = load_qasper(split="validation", max_papers=1)
        if qa_pairs:
            qa     = qa_pairs[0]
            result = generate_answer(qa, top_k=3)

            check("generate_answer() returns GenerationResult",
                  result is not None)
            check("Generated answer is not empty",
                  bool(result.generated_answer) and
                  result.generated_answer != "[NO_CONTEXT_RETRIEVED]",
                  f"Answer: '{result.generated_answer[:100]}'")
            check("Ground truth is present",
                  bool(result.ground_truth),
                  f"Ground truth: {result.ground_truth[:2]}")
            check("Context was used",
                  len(result.context_used) > 0,
                  f"{len(result.context_used)} chunks used")
        else:
            check("QASPER loads QA pairs", False, "No QA pairs found")

    except Exception as e:
        check("End-to-end generate_answer()", False, str(e))

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    if ollama_up:
        print("✓ Generation pipeline ready")
        print("  Run: python -m backend.evaluation.run_evaluation")
    else:
        print("✗ Generation backend not available")
        print()
        print("  Fix Option A — Start Ollama (recommended):")
        print("    1. Install Ollama: https://ollama.com/download")
        print("    2. ollama serve")
        print(f"    3. ollama pull {OLLAMA_MODEL}")
        print()
        print("  Fix Option B — Use HF API (requires internet):")
        print("    1. Add HF_API_TOKEN=hf_xxx to your .env file")
        print("    2. Ensure your machine can reach api-inference.huggingface.co")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    run()