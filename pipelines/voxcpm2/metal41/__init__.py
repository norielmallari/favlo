"""VoxCPM2 voice pipeline."""


PIPELINE = "voxcpm2"


from typing import Any
from voxcpm2.metal41.commands import main
from voxcpm2.metal41.runtime import handle_chat, handle_voice, open_runtime

__all__ = ["handle_chat", "handle_voice", "main", "open_runtime"]



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
                },
                "required": ["input"],
            },
            "task": True,
            "handler": "handle_voice",
            "result": "audio",
        }
    ]
