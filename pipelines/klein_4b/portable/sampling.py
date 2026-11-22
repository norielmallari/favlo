"""Sampling card. Diffusion does not read it. The registry requires one variant."""

from __future__ import annotations

from typing import Any

DEFAULT_VARIANT = "generate"
CONTEXT_LENGTH = 512

VARIANTS = {
    "generate": {
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": 0,
        "presence_penalty": 0.0,
    },
}


def requested_thinking(body: dict[str, Any]) -> bool | None:
    """Diffusion has no thinking flag."""
    del body
    return None
