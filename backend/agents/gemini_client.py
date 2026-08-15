"""
AstroNexus AI — Gemini Client (new SDK)

Drop-in replacement for all google.generativeai calls.
Uses google-genai (new SDK).

Install: pip install google-genai
"""
from __future__ import annotations
import os, logging
logger = logging.getLogger(__name__)

def call_gemini(system: str, prompt: str, max_tokens: int = 1200) -> str:
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        return ""
    try:
        from google import genai
        from google.genai import types
        client   = genai.Client(api_key=api_key)
        full     = f"{system}\n\n{prompt}" if system else prompt
        response = client.models.generate_content(
            model=    "gemini-3.1-flash-lite",
            contents= full,
            config=   types.GenerateContentConfig(
                temperature=       0.1,
                max_output_tokens= max_tokens,
            ),
        )
        return response.text.strip()
    except Exception as e:
        logger.warning(f"[Gemini] Failed: {e}")
        return ""