"""
AstroNexus AI — Orchestrator v3.0

Change from v2:
    - Accepts conversation_history in run()
    - Returns updated conversation_history in result
    - Caller passes history back on next turn for multi-turn sessions
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_graph = None


def build_graph():
    from langgraph.graph import StateGraph, END
    from backend.agents.state           import AgentState
    from backend.agents.router          import router_node, route_decision
    from backend.agents.research_agent  import research_agent_node
    from backend.agents.satellite_agent import satellite_agent_node
    from backend.agents.graph_agent_v2  import general_agent_node as graph_agent_node
    from backend.agents.voice_agent     import voice_agent_node
    from backend.agents.general_agent   import general_agent_node

    graph = StateGraph(AgentState)
    graph.add_node("router",    router_node)
    graph.add_node("research",  research_agent_node)
    graph.add_node("satellite", satellite_agent_node)
    graph.add_node("graph",     graph_agent_node)
    graph.add_node("voice",     voice_agent_node)
    graph.add_node("general",   general_agent_node)
    graph.set_entry_point("router")
    graph.add_conditional_edges(
        "router", route_decision,
        {
            "research":   "research",
            "satellite":  "satellite",
            "graph":      "graph",
            "voice":      "research",
            "general":    "general",
            # web_search / hybrid → general until full web agent is ready
            "web_search": "general",
            "hybrid":     "general",
        },
    )
    for node in ["research","satellite","graph","voice","general"]:
        graph.add_edge(node, END)
    return graph.compile()


def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


def run(
    query:                str,
    audio_path:           str | None   = None,
    image_path:           str | None   = None,
    paper_loaded:         bool         = False,
    paper_id:             str | None   = None,
    conversation_history: list | None  = None,   # ← pass previous turns
    prev_state:           dict | None  = None,
) -> dict:
    """
    Run the AstroNexus pipeline.

    For multi-turn conversations:
        result1 = run("What is the main contribution?", paper_id=pid)
        history = result1["conversation_history"]

        result2 = run("What are its limitations?",
                      paper_id=pid,
                      conversation_history=history)   # ← pass back
        history = result2["conversation_history"]

        result3 = run("Can you compare it with BERT?",
                      paper_id=pid,
                      conversation_history=history)
    """
    from langchain_core.messages import HumanMessage

    history     = conversation_history or []
    prev_route  = (prev_state or {}).get("query_type")
    prev_turn   = (prev_state or {}).get("turn_count", 0)

    initial_state = {
        "messages":            [HumanMessage(content=query)],
        "query":               query,
        "audio_path":          audio_path,
        "query_type":          prev_route,
        "metadata":            {
            "image_path":   image_path,
            "paper_id":     paper_id,
            "paper_loaded": paper_loaded,
        },
        "rag_context":         None,
        "graph_context":       None,
        "satellite_result":    None,
        "final_answer":        None,
        "audio_out":           None,
        "paper_loaded":        paper_loaded,
        "turn_count":          prev_turn + 1,
        "error":               None,
        "conversation_history":history,         # ← injected into state
    }

    logger.info(
        f"[Orchestrator] Turn {prev_turn+1} | "
        f"'{query[:50]}' | "
        f"history={len(history)} turns"
    )

    final_state = get_graph().invoke(initial_state)

    return {
        "query":                final_state.get("query"),
        "query_type":           final_state.get("query_type"),
        "final_answer":         final_state.get("final_answer"),
        "rag_context":          final_state.get("rag_context"),
        "graph_context":        final_state.get("graph_context"),
        "satellite_result":     final_state.get("satellite_result"),
        "audio_out":            final_state.get("audio_out"),
        "error":                final_state.get("error"),
        "conversation_history": final_state.get("conversation_history", history),  # ← returned
        "_state":               final_state,
    }