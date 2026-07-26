"""
AstroNexus AI — Satellite Agent

Handles satellite image analysis queries:
    Image → DINOv2 + Gemini + SAM2 → structured result + description
"""
from __future__ import annotations

import logging

from backend.agents.state import AgentState

logger = logging.getLogger(__name__)


def satellite_agent_node(state: AgentState) -> AgentState:
    """
    LangGraph node: satellite image analysis.

    Expects state["metadata"]["image_path"] to contain the image path.
    Falls back to a helpful message if no image is provided.
    """
    query      = state.get("query", "")
    metadata   = state.get("metadata") or {}
    image_path = metadata.get("image_path")

    logger.info(f"[SatelliteAgent] Query: {query[:60]}")

    if not image_path:
        answer = (
            "No satellite image path provided. "
            "Please supply an image_path in the request metadata "
            "to run satellite analysis."
        )
        return {**state, "final_answer": answer}

    try:
        from backend.vision.satellite_pipeline import analyze_satellite_image

        logger.info(f"[SatelliteAgent] Analysing: {image_path}")
        result = analyze_satellite_image(
            image_path,
            run_florence= True,
            run_sam2=     True,
        )

        # Build a natural language summary from the result
        parts = []

        if result.florence2 and result.florence2.caption:
            parts.append(f"Caption: {result.florence2.caption}")

        if result.florence2 and result.florence2.detailed_caption:
            parts.append(result.florence2.detailed_caption)

        if result.sam2 and result.sam2.get("mask_count", 0) > 0:
            n = result.sam2["mask_count"]
            parts.append(
                f"SAM2 identified {n} distinct regions in the image. "
                f"Largest region covers "
                f"{result.sam2['segments'][0]['area']} pixels."
            )

        if result.dinov2:
            parts.append(
                f"DINOv2 extracted a {result.dinov2.vector_dim}-dimensional "
                f"feature embedding for similarity search."
            )

        answer = "\n\n".join(parts) if parts else "Analysis complete."

        sat_result = {
            "pipeline_stages": result.pipeline_stages,
            "caption":  result.florence2.caption if result.florence2 else None,
            "sam2":     result.sam2,
            "errors":   result.errors,
        }

    except Exception as e:
        logger.error(f"[SatelliteAgent] Analysis failed: {e}")
        answer     = f"Satellite analysis error: {e}"
        sat_result = {"error": str(e)}

    logger.info(f"[SatelliteAgent] Answer: {answer[:100]}...")
    return {**state, "satellite_result": sat_result, "final_answer": answer}