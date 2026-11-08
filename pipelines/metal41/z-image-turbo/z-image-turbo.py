"""Z-Image-Turbo pipeline.

``favlo --z-image-turbo`` loads this file by path. The model lives in the
``z_image_turbo`` package beside it. This file does not import another pipeline.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

PIPELINE = "z-image-turbo"

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import z_image_turbo as _pipeline  # noqa: E402


def __getattr__(name: str) -> Any:
    return getattr(_pipeline, name)


def mcp_tools() -> list[dict[str, Any]]:
    """Local tools. The host publishes the pipeline prefix. This loads no weights."""
    return [
        {
            "name": "generate_image",
            "description": "Still PNG for an image or picture request only, not video.",
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {"type": "string", "description": "Image prompt."},
                    "size": {"type": "string", "description": "WxH; auto is 1024x1024."},
                    "quality": {"type": "string", "description": "Quality preset."},
                    "seed": {"type": "integer", "description": "Omit for random."},
                    "transparent": {"type": "boolean", "description": "RGBA PNG after the draw."},
                    "include_image": {"type": "boolean", "description": "Inline PNG if at most 1 MiB."},
                },
                "required": ["prompt"],
            },
            "task": True,
            "handler": "handle_image",
            "result": "image",
        }
    ]
