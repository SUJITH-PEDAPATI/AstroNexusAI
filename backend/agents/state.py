"""
AstroNexus AI — Agent State

Defines the AgentState TypedDict used across LangGraph nodes and agents.
"""
from __future__ import annotations

from typing import Optional, TypedDict, Any


class AgentState(TypedDict, total=False):
    """
    Shared state passed between LangGraph agent nodes.
    """
    query:            str
    query_type:       str                       # "research" | "satellite" | "graph" | "voice"
    audio_path:       Optional[str]             # input speech file if any
    image_path:       Optional[str]             # input satellite image if any

    # RAG / Research context
    retrieved_docs:   list[dict[str, Any]]
    rag_context:      str

    # Sub-agent outputs
    research_answer:  Optional[str]
    graph_answer:     Optional[str]
    satellite_result: Optional[dict[str, Any]]
    voice_audio_path: Optional[str]

    # Final combined response
    final_answer:     Optional[str]
    error:            Optional[str]
