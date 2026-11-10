"""Sampling card for Qwen3.8-27B.

Registration copies this module onto the registry row. A request that names
no variant uses ``DEFAULT_VARIANT``. The language forward applies the row
the request selects. A request field overrides that row. A variant sets
sampling defaults only. Thinking comes from the request or the registry
``enable_thinking`` / ``reasoning_effort`` fields.

The official template accepts ``low``, ``medium``, and ``xhigh`` (default)
when thinking is on. Every OpenAI effort is still accepted.
``resolve_effort`` maps an unsupported name onto that set.
"""

from typing import Any

DEFAULT_VARIANT = "thinking"
DEFAULT_THINKING = True
DEFAULT_REASONING_EFFORT = "xhigh"

# Official template: low / medium / xhigh. ``none`` closes thinking.
EFFORT_ORDER = ("none", "minimal", "low", "medium", "high", "xhigh", "max")
SUPPORTED_EFFORTS = frozenset({"none", "low", "medium", "xhigh"})

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


def resolve_effort(requested: str | None) -> str | None:
    """Map a request effort onto a value this pipeline supports.

    An omitted effort stays omitted. A supported name is unchanged. An
    unsupported name uses the next higher supported value. When no higher
    supported value exists, it falls back to the next lower supported value
    (for example ``high`` → ``xhigh``, ``minimal`` → ``low``).
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


def thinking_for(selection: Any, requested: str | bool | None) -> bool:
    """Open thinking from a resolved effort, else the registry, else the default.

    A string is an OpenAI effort. ``resolve_effort`` maps it onto ``none``,
    ``low``, ``medium``, or ``xhigh``, and ``none`` closes the think block. A
    bool is that decision already made. None means the request omitted effort:
    the row's ``reasoning_effort`` wins, then ``enable_thinking``, then
    ``DEFAULT_THINKING``.
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


def template_effort(selection: Any, requested: str | bool | None) -> str | None:
    """Effort string for the checkpoint template, or None for its xhigh default.

    ``none`` and a closed think block omit the argument. A bool leaves the
    template default. A stored ``reasoning_effort`` applies when the request
    omitted the field.
    """
    if isinstance(requested, str):
        resolved = resolve_effort(requested)
        if resolved is None or resolved == "none":
            return None
        return resolved
    if requested is not None:
        return None
    stored = getattr(selection, "reasoning_effort", None) if selection is not None else None
    if isinstance(stored, str):
        resolved = resolve_effort(stored)
        if resolved is not None and resolved != "none":
            return resolved
    return None


def mode_card(
    selection: Any, requested: str | bool | None
) -> tuple[bool, dict[str, float | int]]:
    """Thinking flag and the sampling card for this selection.

    The variant id only supplies the sampling card. ``requested`` is the
    request effort, or a bool when the caller already resolved it. Changing
    the effort does not swap the sampling card.
    """
    return thinking_for(selection, requested), sampling_card(selection)
