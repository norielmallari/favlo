"""Sampling card for Qwen3.6-35B-A3B.

The two registry rows are the model card's general-thinking and instruct sets.
Registration copies this module. A request that names no variant uses
``DEFAULT_VARIANT``. Precise coding stays here because the registry has no
third variant name.
"""

from typing import Any

DEFAULT_VARIANT = "thinking"
# Completion tokens drawn inside an open think block before the close is forced.
THINKING_BUDGET = 512

VARIANTS = {
    "thinking": {
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 20,
        "presence_penalty": 1.5,
    },
    "instruct": {
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
        "presence_penalty": 1.5,
    },
}

CODING = {
    "temperature": 0.6,
    "top_p": 0.95,
    "top_k": 20,
    "presence_penalty": 0.0,
}


def requested_thinking(body: dict[str, Any]) -> bool | None:
    """The request's thinking flag, or None when the body omitted it."""
    if "enable_thinking" in body and body["enable_thinking"] is not None:
        return bool(body["enable_thinking"])
    kwargs = body.get("chat_template_kwargs")
    if isinstance(kwargs, dict) and kwargs.get("enable_thinking") is not None:
        return bool(kwargs["enable_thinking"])
    return None


def mode_card(
    selection: Any, requested: bool | None
) -> tuple[bool, dict[str, float | int] | None]:
    """The row this pipeline applies.

    A ``:thinking`` or ``:instruct`` id keeps that row. A bare id uses
    ``requested`` when the body set it, and ``DEFAULT_VARIANT`` when the body
    omitted it.
    """
    variant = getattr(selection, "variant", None) if selection is not None else None
    if variant in VARIANTS:
        stored = getattr(selection, "sampling", None)
        card = stored if isinstance(stored, dict) and stored else VARIANTS[variant]
        return variant == "thinking", dict(card)
    if requested is None:
        name = getattr(selection, "default_variant", None) if selection is not None else None
        name = name if isinstance(name, str) and name in VARIANTS else DEFAULT_VARIANT
    else:
        name = "thinking" if requested else "instruct"
    cards = getattr(selection, "variants", None) if selection is not None else None
    source = cards if isinstance(cards, dict) else VARIANTS
    card = source.get(name) if isinstance(source, dict) else None
    if not isinstance(card, dict):
        card = VARIANTS[name]
    return name == "thinking", dict(card)
