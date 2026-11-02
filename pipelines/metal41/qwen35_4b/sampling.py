"""Sampling card from the Qwen3.5-4B model card.

Registration copies the thinking and instruct rows as sampling presets.
Precise-coding stays here for harness use and is not a registered suffix.
A variant sets sampling defaults only. Thinking comes from the request or
the registry ``enable_thinking`` field.

The official card exposes thinking only as ``enable_thinking`` on/off. It has
no graduated ``reasoning_effort`` ladder. Favlo still accepts every OpenAI
effort and maps an unsupported name up to the next supported value.
"""

from __future__ import annotations

CONTEXT_LENGTH = 262_144

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
    },
    "instruct": {
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
        "presence_penalty": 1.5,
    },
}

PRECISE_CODING = {
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
