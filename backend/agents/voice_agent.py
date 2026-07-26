"""
AstroNexus AI — Voice Agent

Handles voice synthesis (Text-to-Speech) for pipeline responses.
"""
from __future__ import annotations

import logging
from pathlib import Path
from backend.agents.state import AgentState

logger = logging.getLogger(__name__)


def voice_agent_node(state: AgentState) -> AgentState:
    """
    LangGraph node: Voice agent to synthesise answer text to speech.
    """
    text_to_speak = state.get("final_answer") or state.get("query", "")
    logger.info(f"[VoiceAgent] Synthesising voice output for: {text_to_speak[:60]}")

    try:
        from backend.voice.tts_service import TTSService
        out_path = TTSService().speak(text_to_speak)
        logger.info(f"[VoiceAgent] Voice audio created → {out_path}")
        return {**state, "voice_audio_path": str(out_path)}
    except Exception as e:
        logger.error(f"[VoiceAgent] TTS failed: {e}")
        return {**state, "error": f"Voice synthesis error: {e}"}
