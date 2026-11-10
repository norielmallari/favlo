"""Z-Image-Turbo pipeline."""


PIPELINE = "z_image_turbo"


from typing import Any
from z_image_turbo.metal41.chat import handle_chat, handle_image
from z_image_turbo.metal41.commands import main
from z_image_turbo.metal41.runtime import open_runtime

__all__ = ["handle_chat", "handle_image", "main", "open_runtime"]



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
                },
                "required": ["prompt"],
            },
            "task": True,
            "handler": "handle_image",
            "result": "image",
        }
    ]
