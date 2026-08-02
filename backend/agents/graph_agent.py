# This file exists only as a compatibility shim.
# The actual implementation is in graph_agent_v2.py
from backend.agents.graph_agent_v2 import general_agent_node as graph_agent_node  # noqa: F401

__all__ = ["graph_agent_node"]
