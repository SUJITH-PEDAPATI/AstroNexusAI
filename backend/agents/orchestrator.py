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
from backend.agents.graph_agent_v2 import graph_agent_node
from backend.agents.satellite_agent import satellite_agent_node
from backend.agents.voice_agent import voice_agent_node

logger = logging.getLogger(__name__)


def run(
    query:        str = "",
    image_path:   Optional[str] = None,
    audio_path:   Optional[str] = None,
    paper_loaded: bool = False,
    paper_id:     Optional[str] = None,
) -> dict[str, Any]:
    """
    Run the AstroNexus AI agent pipeline on a query.

    Args:
        query:        Text query
        image_path:   Optional satellite image path
        audio_path:   Optional input audio query path
        paper_loaded: True if paper context is loaded
        paper_id:     Optional paper ID

    Returns:
        Resulting AgentState dict containing final_answer, query_type, etc.
    """
    metadata: dict[str, Any] = {
        "paper_loaded": paper_loaded,
        "paper_id":     paper_id,
    }
    if image_path:
        metadata["image_path"] = image_path

    state: AgentState = {
        "query":      query,
        "image_path": image_path,
        "audio_path": audio_path,
        "metadata":   metadata,
    }

    # 1. Route query
    state = router_node(state)
    query_type = route_decision(state)

    # 2. Execute target agent node
    if query_type == "satellite":
        state = satellite_agent_node(state)
    elif query_type == "graph":
        state = graph_agent_node(state)
    elif query_type == "voice":
        state = research_agent_node(state)
        state = voice_agent_node(state)
    else:  # "research" or default
        state = research_agent_node(state)

    return state
