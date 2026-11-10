"""ACE-Step 1.5 XL Turbo. Neural work stays in the Metal shaders."""

from __future__ import annotations


PIPELINE = "ace_step_15_turbo"


from typing import Any

__all__ = ["main", "open_runtime", "handle_chat", "handle_music"]


def __getattr__(name: str) -> Any:
    if name == "main":
        from ace_step_15_turbo.metal41.commands import main

        return main
    if name in {"open_runtime", "handle_chat", "handle_music"}:
        from ace_step_15_turbo.metal41 import runtime

        return getattr(runtime, name)
    message = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(message)



def mcp_tools() -> list[dict[str, Any]]:
    """Local tools. The host publishes the pipeline prefix. This loads no weights."""
    return [
        {
            "name": "generate_music",
            "description": "One music clip as mp3, wav, or flac.",
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {"type": "string", "description": "Caption or style."},
                    "response_format": {"type": "string", "description": "mp3, wav, or flac."},
                    "lyrics": {"type": "string", "description": "Lyrics, when used."},
                    "seconds": {"type": "number", "description": "Duration in seconds."},
                    "seed": {"type": "integer", "description": "Omit for random."},
                    "vocal_language": {"type": "string", "description": "Vocal language."},
                    "bpm": {"type": "integer", "description": "Tempo."},
                    "keyscale": {"type": "string", "description": "Key."},
                    "timesignature": {"type": "string", "description": "Meter."},
                },
                "required": ["prompt"],
            },
            "task": True,
            "handler": "handle_music",
            "result": "audio",
        }
    ]
