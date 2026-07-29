"""
Debug script: trace paper_loaded/paper_id through the full LangGraph pipeline.
Run: python -m backend.tests.debug_langgraph
"""
from dotenv import load_dotenv
load_dotenv()

import backend.agents.research_agent as ra

orig_node = ra.research_agent_node

def patched_node(state):
    meta = state.get("metadata") or {}
    print("  [patched_node] State snapshot:")
    print(f"    paper_loaded (top-level) : {state.get('paper_loaded')}")
    print(f"    paper_loaded (metadata)  : {meta.get('paper_loaded')}")
    print(f"    paper_id     (metadata)  : {meta.get('paper_id')}")
    return orig_node(state)

ra.research_agent_node = patched_node

# ── Build a fresh LangGraph (skip cached _graph) ─────────────────────────────
try:
    from langgraph.graph import StateGraph, END
    from backend.agents.state import AgentState
    from backend.agents.router import router_node, route_decision
    from backend.agents.general_agent import general_agent_node
    from backend.agents.satellite_agent import satellite_agent_node
    from backend.agents.voice_agent import voice_agent_node

    graph = StateGraph(AgentState)
    graph.add_node("router",    router_node)
    graph.add_node("research",  patched_node)
    graph.add_node("satellite", satellite_agent_node)
    graph.add_node("general",   general_agent_node)
    graph.add_node("voice",     patched_node)   # voice → research path
    graph.set_entry_point("router")
    graph.add_conditional_edges(
        "router", route_decision,
        {
            "research":  "research",
            "satellite": "satellite",
            "graph":     "research",   # graph falls back to research
            "voice":     "voice",
            "general":   "general",
        },
    )
    for node in ["research", "satellite", "general", "voice"]:
        graph.add_edge(node, END)

    g = graph.compile()
    USE_LANGGRAPH = True
    print("[debug] LangGraph compiled OK")

except ImportError as e:
    USE_LANGGRAPH = False
    print(f"[debug] LangGraph not available ({e}), falling back to orchestrator.run()")

# ── Initial state ─────────────────────────────────────────────────────────────
QUERY    = "What are the scientific goals of this paper?"
PAPER_ID = "d542e31bb239eca29da245eea786d9e5"

init_state = {
    "query":                QUERY,
    "image_path":           None,
    "audio_path":           None,
    "query_type":           None,
    "rag_context":          None,
    "graph_context":        None,
    "satellite_result":     None,
    "final_answer":         None,
    "audio_out":            None,
    "paper_loaded":         True,
    "turn_count":           1,
    "error":                None,
    "conversation_history": [],
    "metadata": {
        "image_path":   None,
        "paper_id":     PAPER_ID,
        "paper_loaded": True,
    },
}

# ── Run ───────────────────────────────────────────────────────────────────────
if USE_LANGGRAPH:
    print(f"\n[debug] Invoking graph with query: {QUERY!r}\n")
    result = g.invoke(init_state)
else:
    print(f"\n[debug] Invoking orchestrator.run() with query: {QUERY!r}\n")
    from backend.agents.orchestrator import run
    result = run(
        query=        QUERY,
        paper_loaded= True,
        paper_id=     PAPER_ID,
    )

# ── Output ────────────────────────────────────────────────────────────────────
meta = result.get("metadata") or {}
print()
print(f"mode            : {meta.get('mode')}")
print(f"top_score       : {meta.get('top_score')}")
print(f"final_answer    : {str(result.get('final_answer', ''))[:150]}")
