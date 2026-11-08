"""Wan2.2 TI2V-5B pipeline.

``favlo --wan22-ti2v-5b`` loads this file by path. The model lives in the
``wan22_ti2v_5b`` package beside it. This file does not import another pipeline.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

PIPELINE = "wan22-ti2v-5b"

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import wan22_ti2v_5b as _pipeline  # noqa: E402


def __getattr__(name: str) -> Any:
    return getattr(_pipeline, name)


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


def mcp_tools() -> list[dict[str, Any]]:
    """Local tools. The host publishes the pipeline prefix. This loads no weights."""
    return [
        _video(
            "generate_video_t2v",
            "MP4 from text when the user asked for video or motion, not a still.",
            still=False,
        ),
        _video(
            "generate_video_i2v",
            "MP4 from a still when the user asked for video or motion, not a still.",
            still=True,
        ),
    ]
