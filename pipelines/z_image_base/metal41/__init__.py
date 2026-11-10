"""Z-Image pipeline."""


PIPELINE = "z_image_base"


from typing import Any
from z_image_base.metal41.chat import handle_chat, handle_image
from z_image_base.metal41.commands import main
from z_image_base.metal41.runtime import open_runtime

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
