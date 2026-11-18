"""Image rows on the language prompt.

The CUDA language opening does not run the vision tower. A text prompt is unchanged.
"""

from __future__ import annotations

from typing import Any

_IMAGE = frozenset({"image", "input_image", "image_url"})


def prepare_prompt(runtime: Any, messages: list[dict[str, Any]], ids: list[int]) -> list[int]:
    """Keep a text prompt. An image part has no tower on this opening."""
    for message in messages:
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, list):
            continue
        for part in content:
            if isinstance(part, dict) and part.get("type") in _IMAGE:
                message_text = "qwen35_4b cuda does not run the vision tower"
                raise RuntimeError(message_text)
    runtime._vision_rows = None
    runtime._vision_index = None
    runtime._rope_device = None
    runtime._rope_delta = 0
    return ids
