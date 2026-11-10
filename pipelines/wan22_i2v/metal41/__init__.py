"""Wan 2.2 image-to-video pipeline."""


PIPELINE = "wan22_i2v"


from typing import Any
from wan22_i2v.metal41.chat import handle_chat, handle_video
from wan22_i2v.metal41.commands import main
from wan22_i2v.metal41.runtime import open_runtime
from wan22_i2v.metal41.schedule import list_resolutions, video_modes

__all__ = [
    "handle_chat",
    "handle_video",
    "list_resolutions",
    "main",
    "open_runtime",
    "video_modes",
]



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
