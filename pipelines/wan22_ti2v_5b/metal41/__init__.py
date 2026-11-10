"""Wan2.2 TI2V-5B text-to-video and image-to-video."""

from __future__ import annotations

from typing import Any

from wan22_ti2v_5b.metal41.runtime import handle_chat, pipeline
from wan22_ti2v_5b.metal41.request import list_resolutions, video_modes

PIPELINE = "wan22_ti2v_5b"

handle_video = pipeline
main = pipeline.main
open_runtime = pipeline.open_runtime


def _video(name: str, description: str, *, still: bool) -> dict[str, Any]:
    properties: dict[str, Any] = {
        "prompt": {"type": "string", "description": "Motion prompt."},
        "size": {"type": "string", "description": "Required WIDTHxHEIGHT."},
        "duration": {"type": "number", "description": "Length in seconds."},
        "seed": {"type": "integer", "description": "Omit for random."},
        "quality": {"type": "string", "description": "auto, low, medium, high, xhigh, max."},
        "negative_prompt": {"type": "string", "description": "Empty uses the sample negative."},
        "guidance_scale": {"description": "Number or [low, high]; default 1."},
        "steps": {"type": "integer", "description": "Euler 1-40; 0 uses 4."},
    }
    required = ["prompt", "size"]
    if still:
        properties["image"] = {"description": "image_ id, PNG, or file://."}
        properties["prompt"] = {"type": "string", "description": "Motion prompt; still is frame 0."}
        required.append("image")
    return {
        "name": name,
        "description": description,
        "parameters": {"type": "object", "properties": properties, "required": required},
        "task": True,
        "handler": "handle_video",
        "result": "video",
    }


def mcp_tools():
    return pipeline.mcp_tools()


__all__ = [
    "pipeline",
    "handle_chat",
    "handle_video",
    "list_resolutions",
    "main",
    "mcp_tools",
    "open_runtime",
    "video_modes",
]
