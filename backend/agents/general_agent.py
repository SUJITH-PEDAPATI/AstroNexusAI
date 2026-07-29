"""
AstroNexus AI — General Agent (compatibility shim)

The actual implementation lives in graph_agent_v2.py.
This module re-exports general_agent_node so that both import paths work:

    from backend.agents.general_agent import general_agent_node   # ← this file
    from backend.agents.graph_agent_v2 import general_agent_node  # ← original
"""
from backend.agents.graph_agent_v2 import general_agent_node  # noqa: F401

__all__ = ["general_agent_node"]
