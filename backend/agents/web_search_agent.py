from backend.agents.state import AgentState
import logging

logger = logging.getLogger(__name__)

def web_search_agent_node(state: AgentState) -> AgentState:
    """
    Dummy/stub node for the web search agent.
    Replace this with the actual web search or hybrid agent logic.
    """
    logger.info("[WebSearchAgent] Stub node invoked (real logic missing)")
    
    answer = (
        "I'm sorry, I was routed to perform a web search, but the web search "
        "capabilities are currently unavailable because the agent logic is missing."
    )
    
    return {
        **state,
        "final_answer": answer,
        "metadata": {
            **(state.get("metadata") or {}),
            "agent_used": "web_search_stub"
        }
    }
