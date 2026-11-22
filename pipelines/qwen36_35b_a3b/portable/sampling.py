"""Sampling card for Qwen3.6-35B-A3B.

The two registry rows are the model card's general-thinking and instruct sets.
Registration copies this module. A request that names no variant uses
``DEFAULT_VARIANT``. Precise coding stays here because the registry has no
third variant name.

Each variant stores sampling defaults and ``reasoning_effort``. The official
card exposes thinking only as ``enable_thinking`` on or off. Every OpenAI
effort is still accepted. ``resolve_effort`` maps an unsupported name onto
``none`` or ``high``. ``high`` is thinking on; ``none`` is thinking off.
"""

from typing import Any

DEFAULT_VARIANT = "thinking"
DEFAULT_THINKING = True

# Official card: on/off only. ``high`` stands for thinking on.
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
    """Map a request effort onto a value this pipeline supports.

    An omitted effort stays omitted. A supported name is unchanged. An
    unsupported name uses the next higher supported value. When no higher
    supported value exists, it falls back to the next lower supported value
    (for example ``xhigh`` → ``high``).
    """
    if requested is None:
        return None
    name = requested.strip().lower()
    if not name:
        return None
    if name in SUPPORTED_EFFORTS:
        return name
    if name not in EFFORT_ORDER:
        return name
    index = EFFORT_ORDER.index(name)
    for candidate in EFFORT_ORDER[index + 1 :]:
        if candidate in SUPPORTED_EFFORTS:
            return candidate
    for candidate in reversed(EFFORT_ORDER[:index]):
        if candidate in SUPPORTED_EFFORTS:
            return candidate
    return name


def requested_thinking(body: dict[str, Any]) -> bool | None:
    """The request's thinking flag, or None when the body omitted it."""
    if "enable_thinking" in body and body["enable_thinking"] is not None:
        return bool(body["enable_thinking"])
    kwargs = body.get("chat_template_kwargs")
    if isinstance(kwargs, dict) and kwargs.get("enable_thinking") is not None:
        return bool(kwargs["enable_thinking"])
    return None


_SAMPLING_KEYS = ("temperature", "top_p", "top_k", "presence_penalty")


def _sampling_fields(card: dict[str, Any]) -> dict[str, float | int]:
    """Numeric sampler fields only. ``reasoning_effort`` stays off this dict."""
    out: dict[str, float | int] = {}
    for key in _SAMPLING_KEYS:
        if key not in card:
            continue
        value = card[key]
        if key == "top_k":
            out[key] = int(value)
        else:
            out[key] = float(value)
    return out


def sampling_card(selection: Any) -> dict[str, float | int]:
    """The sampling row for the selected id."""
    variant = getattr(selection, "variant", None) if selection is not None else None
    stored = getattr(selection, "sampling", None) if selection is not None else None
    if isinstance(variant, str) and variant in VARIANTS:
        name = variant
        if isinstance(stored, dict) and stored:
            return _sampling_fields(stored)
    else:
        name = getattr(selection, "default_variant", None) if selection is not None else None
        name = name if isinstance(name, str) and name in VARIANTS else DEFAULT_VARIANT
        if isinstance(stored, dict) and stored and variant is None:
            return _sampling_fields(stored)
    cards = getattr(selection, "variants", None) if selection is not None else None
    source = cards if isinstance(cards, dict) else VARIANTS
    card = source.get(name) if isinstance(source, dict) else None
    if not isinstance(card, dict):
        card = VARIANTS[name]
    return _sampling_fields(card)


def thinking_for(selection: Any, requested: str | bool | None) -> bool:
    """Open thinking from a resolved effort, else the selection, else the default.

    A string is an OpenAI effort. ``resolve_effort`` maps it onto ``none`` or
    ``high``, and ``none`` closes the think block. A bool is that decision
    already made. None means the request omitted effort: the selection's
    ``reasoning_effort`` (from the variant card or the row) wins, then
    ``enable_thinking``, then ``DEFAULT_THINKING``.
    """
    if isinstance(requested, str):
        resolved = resolve_effort(requested)
        if resolved is not None:
            return resolved != "none"
    elif requested is not None:
        return bool(requested)
    stored = getattr(selection, "reasoning_effort", None) if selection is not None else None
    if isinstance(stored, str):
        resolved = resolve_effort(stored)
        if resolved is not None:
            return resolved != "none"
    flag = getattr(selection, "enable_thinking", None) if selection is not None else None
    if flag is not None:
        return bool(flag)
    return DEFAULT_THINKING


def mode_card(
    selection: Any, requested: str | bool | None
) -> tuple[bool, dict[str, float | int]]:
    """Thinking flag and the sampling card for this selection.

    ``requested`` is the request effort, or a bool when the caller already
    resolved it. An omitted effort uses the variant's ``reasoning_effort``.
    Changing the effort does not swap the sampling card.
    """
    return thinking_for(selection, requested), sampling_card(selection)
