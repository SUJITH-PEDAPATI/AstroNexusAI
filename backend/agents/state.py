"""
AstroNexus AI — Agent State v3.0

Added in v3:
    conversation_history  list of previous turns for multi-turn context
"""
from __future__ import annotations

from typing import Annotated, Optional
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    # ── Core ──────────────────────────────────────────────────────────────────
    messages:     Annotated[list, add_messages]
    query:        str
    audio_path:   Optional[str]

    # ── Routing ───────────────────────────────────────────────────────────────
    query_type:   Optional[str]
    metadata:     Optional[dict]

    # ── Agent outputs ─────────────────────────────────────────────────────────
    rag_context:       Optional[str]
    graph_context:     Optional[str]
    satellite_result:  Optional[dict]
    final_answer:      Optional[str]
    audio_out:         Optional[str]

    # ── Session state ─────────────────────────────────────────────────────────
    paper_loaded:  Optional[bool]
    paper_id:      Optional[str]   # ID of the paper currently in context
    turn_count:    Optional[int]
    error:         Optional[str]

    # ── Conversation history (NEW in v3) ──────────────────────────────────────
    conversation_history: Optional[list]
    # Each entry: {"turn": int, "query": str, "answer": str, "query_type": str}