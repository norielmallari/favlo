"""Z-Image pipeline.

``favlo --z-image-base`` loads this file by path. The sampler lives in the
``z_image_base`` package beside it. The forward lives in ``z_image``.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

PIPELINE = "z-image-base"

_ROOT = Path(__file__).resolve().parent
_SHARED = _ROOT.parent / "z-image"
if str(_SHARED) not in sys.path:
    sys.path.insert(0, str(_SHARED))
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import z_image_base as _pipeline  # noqa: E402


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
                    "negative_prompt": {
                        "type": "string",
                        "description": "Unconditional caption when empty.",
                    },
                    "size": {"type": "string", "description": "WxH; auto is 1024x1024."},
                    "quality": {"type": "string", "description": "Quality preset."},
                    "seed": {"type": "integer", "description": "Omit for random."},
                    "steps": {"type": "integer", "description": "Velocities from 1 to 100."},
                    "guidance": {
                        "type": "number",
                        "description": "CFG scale. 0 skips the negative.",
                    },
                    "shift": {"type": "number", "description": "Flow-matching shift. Default 6."},
                    "cfg_truncation": {
                        "type": "number",
                        "description": "Drop the negative after this 1-sigma.",
                    },
                    "cfg_normalization": {
                        "type": "number",
                        "description": "L2 cap times the positive norm.",
                    },
                },
                "required": ["prompt"],
            },
            "task": True,
            "handler": "handle_image",
            "result": "image",
        }
    ]
