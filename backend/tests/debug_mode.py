"""
Debug script: monkey-patch detect_query_mode to trace what research_agent receives.
Run: python -m backend.tests.debug_mode
"""
from dotenv import load_dotenv
load_dotenv()

import backend.agents.research_agent as ra

orig = ra.detect_query_mode

def patched(query, top_score, has_chunks, paper_loaded=False):
    print("  detect_query_mode called:")
    print(f"    query        = {query[:60]!r}")
    print(f"    top_score    = {top_score}")
    print(f"    has_chunks   = {has_chunks}")
    print(f"    paper_loaded = {paper_loaded}")
    result = orig(query, top_score, has_chunks, paper_loaded)
    print(f"    -> mode       = {result}")
    return result

ra.detect_query_mode = patched

from backend.agents.orchestrator import run

result = run(
    query="What are the scientific goals of this paper?",
    paper_loaded=True,
    paper_id="d542e31bb239eca29da245eea786d9e5",
)

meta = result.get("metadata") or {}
print()
print(f"Final mode        : {meta.get('mode')}")
print(f"Final answer[:100]: {str(result.get('final_answer',''))[:100]}")
