"""Sampling card for Gemma 4 26B-A4B.

Both rows use top-p 0.95 and top-k 64 from the model card, with temperature
0.95. The template difference is ``enable_thinking``. The card has no
reasoning-effort ladder.
"""

from typing import Any

DEFAULT_VARIANT = "thinking"
CONTEXT_LENGTH = 131072

_CARD = {
    "temperature": 0.95,
    "top_p": 0.95,
    "top_k": 64,
    "presence_penalty": 0.0,
}

VARIANTS = {
    "thinking": dict(_CARD),
    "instruct": dict(_CARD),
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
) -> tuple[bool, dict[str, float | int]]:
    """The row this pipeline applies."""
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
