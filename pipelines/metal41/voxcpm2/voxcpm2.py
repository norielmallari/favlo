"""VoxCPM2 pipeline.

``favlo --voxcpm2`` loads this file by path. The model lives in the
``voxcpm2`` package beside it. This file does not import another pipeline.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

PIPELINE = "voxcpm2"

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import voxcpm2 as _pipeline  # noqa: E402


def __getattr__(name: str) -> Any:
    return getattr(_pipeline, name)


def mcp_tools() -> list[dict[str, Any]]:
    """Local tools. The host publishes the pipeline prefix. This loads no weights."""
    return [
        {
            "name": "generate_voice",
            "description": "Speak one line as mp3, wav, or flac.",
            "parameters": {
                "type": "object",
                "properties": {
                    "input": {"type": "string", "description": "Line to speak."},
                    "voice": {"type": "string", "description": "Empty speaks with no reference."},
                    "instructions": {"type": "string", "description": "Style prefix."},
                    "response_format": {"type": "string", "description": "mp3, wav, or flac."},
                    "include_audio": {"type": "boolean", "description": "Inline audio if at most 1 MiB."},
                },
                "required": ["input"],
            },
            "task": True,
            "handler": "handle_voice",
            "result": "audio",
        }
    ]
