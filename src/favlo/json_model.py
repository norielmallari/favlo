"""Compact JSON for model-facing tool schemas and results."""

from __future__ import annotations

import json
from typing import Any


def omit_empty(payload: Any) -> Any:
    """Drop ``None``, empty strings, empty lists, and empty dicts recursively."""
    if isinstance(payload, dict):
        out: dict[str, Any] = {}
        for key, value in payload.items():
            cleaned = omit_empty(value)
            if cleaned is None or cleaned == "" or cleaned == [] or cleaned == {}:
                continue
            out[key] = cleaned
        return out
    if isinstance(payload, list):
        items = [omit_empty(item) for item in payload]
        return [item for item in items if item is not None and item != "" and item != [] and item != {}]
    return payload


def dumps_for_model(payload: Any, *, drop_empty: bool = True) -> str:
    """Serialize ``payload`` without whitespace. Use for tool results the model reads."""
    body = omit_empty(payload) if drop_empty else payload
    return json.dumps(body, separators=(",", ":"), ensure_ascii=False)
