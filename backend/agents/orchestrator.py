"""
AstroNexus AI — Agent Orchestrator

LangGraph orchestrator that routes user queries to specialized sub-agents:
    - Router Node (classifies query)
    - Research Agent Node (RAG over papers)
    - Graph Agent Node (Neo4j citations/entities)
    - Satellite Agent Node (DINOv2 + Gemini + SAM2 vision)
    - Voice Agent Node (TTS response)
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from backend.agents.state import AgentState
from backend.agents.router import router_node, route_decision
from backend.agents.research_agent import research_agent_node
from backend.agents.graph_agent_v2 import general_agent_node
from backend.agents.satellite_agent import satellite_agent_node
from backend.agents.voice_agent import voice_agent_node

logger = logging.getLogger(__name__)


def get_graph() -> None:
    """
    Warm-up / sanity check called at chat startup.

    Verifies that all pipeline node callables are importable and ready.
    Raises an exception if any required component is missing or broken.
    """
    nodes = [
        ("router_node",         router_node),
        ("route_decision",      route_decision),
        ("research_agent_node", research_agent_node),
        ("general_agent_node",  general_agent_node),
        ("satellite_agent_node",satellite_agent_node),
        ("voice_agent_node",    voice_agent_node),
    ]
    for name, fn in nodes:
        if not callable(fn):
            raise RuntimeError(f"Pipeline node '{name}' is not callable.")
    logger.debug("get_graph(): all pipeline nodes verified OK.")


def run(
    query:                str = "",
    image_path:           Optional[str] = None,
    audio_path:           Optional[str] = None,
    paper_loaded:         bool = False,
    paper_id:             Optional[str] = None,
    conversation_history: Optional[list] = None,
) -> dict[str, Any]:
    """
    Run the AstroNexus AI agent pipeline on a query.

    Args:
        query:                Text query
        image_path:           Optional satellite image path
        audio_path:           Optional input audio query path
        paper_loaded:         True if paper context is loaded
        paper_id:             Optional paper ID
        conversation_history: List of previous turn dicts for multi-turn context

    Returns:
        Resulting AgentState dict containing final_answer, query_type,
        conversation_history, etc.
    """
    history: list = conversation_history or []

    metadata: dict[str, Any] = {
        "paper_loaded": paper_loaded,
        "paper_id":     paper_id,
    }
    if image_path:
        metadata["image_path"] = image_path

    state: AgentState = {
        "query":                query,
        "image_path":           image_path,
        "audio_path":           audio_path,
        "metadata":             metadata,
        "conversation_history": history,
        "turn_count":           len(history) + 1,
    }

    # 1. Route query
    state = router_node(state)
    query_type = route_decision(state)

    # 2. Execute target agent node
    if query_type == "satellite":
        state = satellite_agent_node(state)
    elif query_type == "graph":
        state = general_agent_node(state)
    elif query_type == "voice":
        state = research_agent_node(state)
        state = voice_agent_node(state)
    else:  # "research" or default
        state = research_agent_node(state)

    return state
