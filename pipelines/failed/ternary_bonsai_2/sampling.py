"""Sampling card for Qwen3.8-27B.

Registration copies this module onto the registry row. A request that names
no variant uses ``DEFAULT_VARIANT``. The language forward applies the row
the request selects. A request field overrides that row.
"""

from typing import Any

DEFAULT_VARIANT = "thinking"

VARIANTS = {
    "thinking": {
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 20,
        "presence_penalty": 0.0,
    },
    "instruct": {
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
        "presence_penalty": 1.5,
    },
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
    """The row this pipeline applies to the chat template.

    A ``:thinking`` or ``:instruct`` id keeps that row. A bare id uses
    ``requested`` when the body set it, and ``DEFAULT_VARIANT`` when the body
    omitted it. The forward samples from the returned card.
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
