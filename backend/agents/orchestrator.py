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


def legacy_get_graph() -> None:
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


def legacy_run(
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


# ==============================================================================
# NEW ORCHESTRATOR CODE (Merged from Downloads)
# ==============================================================================

"""
AstroNexus AI — LangGraph Orchestrator

Builds and runs the multi-agent pipeline:

    ┌─────────────┐
    │   Router    │   classifies query type
    └──────┬──────┘
           │
    ┌──────▼──────────────────────────────────┐
    │  research | satellite | graph | voice   │  conditional branch
    └──────┬──────────────────────────────────┘
           │
    ┌──────▼──────┐
    │ VoiceAgent  │   TTS if voice mode (optional)
    └──────┬──────┘
           │
         END

Install: pip install langgraph langchain-core
"""


import logging
from typing import Any

logger = logging.getLogger(__name__)


def build_graph():
    """
    Build and compile the LangGraph StateGraph.
    Returns a compiled graph ready for .invoke() calls.
    """
    from langgraph.graph import StateGraph, END

    from backend.agents.state           import AgentState
    from backend.agents.router          import router_node, route_decision
    from backend.agents.research_agent  import research_agent_node
    from backend.agents.satellite_agent import satellite_agent_node
    from backend.agents.graph_agent_v2  import general_agent_node as graph_agent_node
    from backend.agents.voice_agent     import voice_agent_node
    from backend.agents.web_search_agent import web_search_agent_node

    graph = StateGraph(AgentState)

    # ── Add nodes ──────────────────────────────────────────────────────────────
    graph.add_node("router",    router_node)
    graph.add_node("research",  research_agent_node)
    graph.add_node("satellite", satellite_agent_node)
    graph.add_node("graph",     graph_agent_node)
    graph.add_node("voice",      voice_agent_node)
    graph.add_node("web_search", web_search_agent_node)
    graph.add_node("hybrid",     web_search_agent_node)

    # ── Entry point ────────────────────────────────────────────────────────────
    graph.set_entry_point("router")

    # ── Conditional routing ────────────────────────────────────────────────────
    graph.add_conditional_edges(
        "router",
        route_decision,
        {
            "research":   "research",
            "satellite":  "satellite",
            "graph":      "graph",
            "voice":      "research",    # voice queries → research then TTS
            "web_search": "web_search",  # live/current queries → web
            "hybrid":     "hybrid",      # needs both RAG + web
        },
    )

    # ── All agents → voice (TTS) if needed, then END ──────────────────────────
    # For simplicity in V1: agents → END directly
    # Voice output is handled inside voice_agent when query_type == "voice"
    graph.add_edge("research",  END)
    graph.add_edge("satellite", END)
    graph.add_edge("graph",      END)
    graph.add_edge("web_search", END)
    graph.add_edge("hybrid",     END)
    graph.add_edge("voice",     END)

    return graph.compile()


# ── Singleton compiled graph ───────────────────────────────────────────────────
_graph = None


def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


def run(
    query:                str = "",
    image_path:           Optional[str] = None,
    audio_path:           Optional[str] = None,
    paper_loaded:         bool = False,
    paper_id:             Optional[str] = None,
    conversation_history: Optional[list] = None,
) -> dict:
    """
    Run the full AstroNexus multi-agent pipeline.

    Args:
        query:                User question (text)
        image_path:           Optional satellite image path
        audio_path:           Optional audio file for voice input
        paper_loaded:         True if paper context is loaded
        paper_id:             Optional paper ID
        conversation_history: List of previous turn dicts for multi-turn context

    Returns:
        {
            "query":            str,
            "query_type":       str,
            "final_answer":     str,
            "rag_context":      str | None,
            "graph_context":    str | None,
            "satellite_result": dict | None,
            "audio_out":        str | None,
            "error":            str | None,
        }
    """
    from langchain_core.messages import HumanMessage

    metadata = {
        "paper_loaded": paper_loaded,
        "paper_id":     paper_id,
    }
    if image_path:
        metadata["image_path"] = image_path

    history = conversation_history or []

    initial_state = {
        "messages":             [HumanMessage(content=query)],
        "query":                query,
        "audio_path":           audio_path,
        "image_path":           image_path,
        "paper_loaded":         paper_loaded,
        "conversation_history": history,
        "turn_count":           len(history) + 1,
        "query_type":           None,
        "rag_context":          None,
        "graph_context":        None,
        "satellite_result":     None,
        "final_answer":         None,
        "audio_out":            None,
        "error":                None,
        "metadata":             metadata,
    }

    logger.info(f"[Orchestrator] Running pipeline for: {query[:60]}")

    graph        = get_graph()
    final_state  = graph.invoke(initial_state)

    return {
        "query":            final_state.get("query"),
        "query_type":       final_state.get("query_type"),
        "final_answer":     final_state.get("final_answer"),
        "rag_context":      final_state.get("rag_context"),
        "graph_context":    final_state.get("graph_context"),
        "satellite_result": final_state.get("satellite_result"),
        "audio_out":        final_state.get("audio_out"),
        "error":            final_state.get("error"),
    }

