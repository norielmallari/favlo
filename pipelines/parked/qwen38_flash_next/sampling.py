"""Sampling card for Qwen3.8-Flash-Next.

Registration copies this module onto the registry row. A request that names
no variant uses ``DEFAULT_VARIANT``. A variant sets sampling defaults only.
Thinking comes from the request or the registry ``enable_thinking`` field.
"""

from typing import Any

DEFAULT_VARIANT = "thinking"
DEFAULT_THINKING = True

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


def sampling_card(selection: Any) -> dict[str, float | int]:
    """The sampling row for the selected id. The suffix does not change thinking."""
    variant = getattr(selection, "variant", None) if selection is not None else None
    stored = getattr(selection, "sampling", None) if selection is not None else None
    if isinstance(variant, str) and variant in VARIANTS:
        name = variant
        if isinstance(stored, dict) and stored:
            return dict(stored)
    else:
        name = getattr(selection, "default_variant", None) if selection is not None else None
        name = name if isinstance(name, str) and name in VARIANTS else DEFAULT_VARIANT
        if isinstance(stored, dict) and stored and variant is None:
            return dict(stored)
    cards = getattr(selection, "variants", None) if selection is not None else None
    source = cards if isinstance(cards, dict) else VARIANTS
    card = source.get(name) if isinstance(source, dict) else None
    if not isinstance(card, dict):
        card = VARIANTS[name]
    return dict(card)


def thinking_for(selection: Any, requested: bool | None) -> bool:
    """Open thinking from the request, else the registry row, else the default."""
    if requested is not None:
        return requested
    flag = getattr(selection, "enable_thinking", None) if selection is not None else None
    if flag is not None:
        return bool(flag)
    return DEFAULT_THINKING


def mode_card(
    selection: Any, requested: bool | None
) -> tuple[bool, dict[str, float | int]]:
    """Thinking flag and the sampling card for this selection.

    The variant id only supplies the sampling card. ``requested`` and the
    registry ``enable_thinking`` field control the think block.
    """
    return thinking_for(selection, requested), sampling_card(selection)
