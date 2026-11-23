"""Sampling card for Qwen3.6-35B-A3B.

Registration copies the thinking and instruct rows. A request that names no
variant uses ``DEFAULT_VARIANT``. The official card exposes thinking only as
on or off. ``resolve_effort`` maps every other OpenAI name onto ``none`` or
``high``.
"""

from __future__ import annotations

from typing import Any

CONTEXT_LENGTH = 131_072
DEFAULT_VARIANT = "thinking"
DEFAULT_THINKING = True

EFFORT_ORDER = ("none", "minimal", "low", "medium", "high", "xhigh", "max")
SUPPORTED_EFFORTS = frozenset({"none", "high"})

VARIANTS = {
    "thinking": {
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 20,
        "presence_penalty": 1.5,
        "reasoning_effort": "high",
    },
    "instruct": {
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
        "presence_penalty": 1.5,
        "reasoning_effort": "none",
    },
}

CODING = {
    "temperature": 0.6,
    "top_p": 0.95,
    "top_k": 20,
    "presence_penalty": 0.0,
}


def resolve_effort(requested: str | None) -> str | None:
    """Map a request effort onto ``none`` or ``high``."""
    if requested is None:
        return None
    name = requested.strip().lower()
    if name in SUPPORTED_EFFORTS:
        return name
    if name not in EFFORT_ORDER:
        return None
    index = EFFORT_ORDER.index(name)
    for cursor in range(index + 1, len(EFFORT_ORDER)):
        if EFFORT_ORDER[cursor] in SUPPORTED_EFFORTS:
            return EFFORT_ORDER[cursor]
    for cursor in range(index - 1, -1, -1):
        if EFFORT_ORDER[cursor] in SUPPORTED_EFFORTS:
            return EFFORT_ORDER[cursor]
    return None


def thinking_on(effort: str | None) -> bool:
    resolved = resolve_effort(effort)
    if resolved is None:
        return DEFAULT_THINKING
    return resolved != "none"


def mode_card(name: str | None, cards: dict[str, Any] | None = None) -> dict[str, Any]:
    """The named variant, or the default when the name is empty."""
    source = cards if isinstance(cards, dict) else VARIANTS
    chosen = name if isinstance(name, str) and name in source else DEFAULT_VARIANT
    card = source.get(chosen)
    if not isinstance(card, dict):
        card = VARIANTS[DEFAULT_VARIANT]
    return dict(card)
