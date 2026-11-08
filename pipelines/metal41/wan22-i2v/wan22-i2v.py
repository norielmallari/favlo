"""Wan 2.2 I2V pipeline.

``favlo --wan22-i2v`` loads this file by path. The model lives in the
``wan22_i2v`` package beside it. This file does not import another pipeline.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

PIPELINE = "wan22-i2v"

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import wan22_i2v as _pipeline  # noqa: E402


def __getattr__(name: str) -> Any:
    return getattr(_pipeline, name)


def mcp_tools() -> list[dict[str, Any]]:
    """Local tools. The host publishes the pipeline prefix. This loads no weights."""
    return [
        {
            "name": "generate_video",
            "description": "MP4 from a still when the user asked for video or motion, not a still.",
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {"type": "string", "description": "Motion prompt; still is frame 0."},
                    "size": {"type": "string", "description": "Required WIDTHxHEIGHT."},
                    "image": {"description": "image_ id, PNG, or file://."},
                    "duration": {"type": "number", "description": "Length in seconds."},
                    "seed": {"type": "integer", "description": "Omit for random."},
                    "quality": {"type": "string", "description": "auto, low, medium, high, xhigh, max."},
                    "negative_prompt": {"type": "string", "description": "Empty uses the sample negative."},
                    "guidance_scale": {"description": "Number or [low, high]; default 1."},
                    "steps": {"type": "integer", "description": "Euler 1-40; 0 uses 4."},
                },
                "required": ["prompt", "size", "image"],
            },
            "task": True,
            "handler": "handle_video",
            "result": "video",
        }
    ]
