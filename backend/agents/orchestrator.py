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
from __future__ import annotations

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
    from backend.agents.graph_agent     import graph_agent_node
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
    query:                str,
    audio_path:           str | None  = None,
    image_path:           str | None  = None,
    paper_loaded:         bool        = False,
    paper_id:             str | None  = None,
    conversation_history: list | None = None,
) -> dict:
    """
    Run the full AstroNexus multi-agent pipeline.

    Args:
        query:                User question (text)
        audio_path:           Optional audio file for voice input
        image_path:           Optional satellite image path
        paper_loaded:         True when a specific paper is in context
        paper_id:             ID of the paper currently in context
        conversation_history: Previous turns [{turn, query, answer, query_type}]

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
            "metadata":         dict,
        }
    """
    from langchain_core.messages import HumanMessage

    meta: dict = {}
    if image_path:
        meta["image_path"] = image_path
    if paper_id:
        meta["paper_id"] = paper_id

    initial_state = {
        "messages":              [HumanMessage(content=query)],
        "query":                 query,
        "audio_path":            audio_path,
        "query_type":            None,
        "rag_context":           None,
        "graph_context":         None,
        "satellite_result":      None,
        "final_answer":          None,
        "audio_out":             None,
        "error":                 None,
        "paper_loaded":          paper_loaded,
        "conversation_history":  conversation_history or [],
        "turn_count":            len(conversation_history) if conversation_history else 0,
        "metadata":              meta,
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
        "metadata":         final_state.get("metadata") or {},
    }