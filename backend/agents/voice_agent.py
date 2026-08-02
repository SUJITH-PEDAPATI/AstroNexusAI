"""
AstroNexus AI — Voice Agent

Handles voice-specific queries:
    STT → route to appropriate agent → TTS response
"""
from __future__ import annotations

import logging

from backend.agents.state import AgentState

logger = logging.getLogger(__name__)


def voice_agent_node(state: AgentState) -> AgentState:
    """
    LangGraph node: voice I/O coordination.

    If audio_path is set: transcribe it (already done by router).
    Takes final_answer from state and converts to speech.
    """
    query      = state.get("query", "")
    answer     = state.get("final_answer", "")
    audio_path = state.get("audio_path")

    logger.info(f"[VoiceAgent] Synthesising response for: {query[:40]}")

    # If no answer yet — generate a simple one
    if not answer:
        answer = f"Processing your query: {query}"

    audio_out = None
    try:
        from backend.voice.tts_service import TTSService
        out_path  = TTSService().speak(answer)
        audio_out = str(out_path)
        logger.info(f"[VoiceAgent] Audio response → {audio_out}")
    except Exception as e:
        logger.error(f"[VoiceAgent] TTS failed: {e}")

    return {**state, "final_answer": answer, "audio_out": audio_out}