from dataclasses import dataclass

@dataclass
class RouteDecision:
    route: str
    confidence: float
    reason: str

def classify(query: str, has_paper: bool = False) -> RouteDecision:
    """
    Dummy/stub classifier since the original logic was missing.
    Update this with the actual classification logic (LLM or rule-based) 
    for 'web_search' / 'hybrid' routes.
    """
    return RouteDecision(
        route="research", 
        confidence=1.0, 
        reason="Default fallback (classification logic missing)"
    )
