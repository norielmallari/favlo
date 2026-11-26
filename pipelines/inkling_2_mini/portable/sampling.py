"""Sampling card for Inkling 2 Mini.

``generation_config.json`` publishes temperature 1.0 and ``do_sample``. The
published run settings are temperature 1.0, top-p 1.0, and top-k 0, with no
presence penalty. The checkpoint has no thinking or instruct sampler. A
request sets the effort with ``reasoning_effort``. ``high`` is the template
default, 0.9. The template has no ``preserve_thinking`` switch. It renders
an assistant ``reasoning_content`` when that field is present, so the
registry row keeps the field. A request that sets it false drops the field
before render.
"""

DEFAULT_VARIANT = None
DEFAULT_THINKING = True
DEFAULT_REASONING_EFFORT = "high"
PRESERVE_THINKING = True
CONTEXT_LENGTH = 131072

EFFORT_ORDER = ("none", "minimal", "low", "medium", "high", "xhigh", "max")
SUPPORTED_EFFORTS = frozenset(EFFORT_ORDER)

# The template's own map. A string effort is rendered by the checkpoint Jinja.
EFFORT_LEVEL = {
    "none": 0.0,
    "minimal": 0.1,
    "low": 0.2,
    "medium": 0.7,
    "high": 0.9,
    "xhigh": 0.99,
    "max": 0.99,
}

CARD = {
    "temperature": 1.0,
    "top_p": 1.0,
    "top_k": 0,
    "presence_penalty": 0.0,
}

VARIANTS: dict[str, dict[str, float | int]] = {}


def template_effort(value: str | float | int | None) -> float:
    """The number the checkpoint template accepts, in ``[0.0, 0.99]``.

    A listed name uses the template map. A number, or a numeric string, is
    clamped into that range. The template writes this number into the prompt.
    """
    if value is None:
        return EFFORT_LEVEL["high"]
    if isinstance(value, bool):
        return _bad_effort(value)
    if isinstance(value, int | float):
        return _clamp_effort(float(value))
    name = value.strip().lower()
    if not name:
        return EFFORT_LEVEL["high"]
    if name in EFFORT_LEVEL:
        return EFFORT_LEVEL[name]
    try:
        return _clamp_effort(float(name))
    except ValueError:
        mapped = resolve_effort(name)
    if isinstance(mapped, str) and mapped in EFFORT_LEVEL:
        return EFFORT_LEVEL[mapped]
    return _bad_effort(value)


def _clamp_effort(number: float) -> float:
    if number < 0.0:
        return 0.0
    if number > 0.99:
        return 0.99
    return number


def _bad_effort(value: object) -> float:
    message = f"reasoning_effort {value} is not a template level"
    raise ValueError(message)


def resolve_effort(requested: str | None) -> str | None:
    """Map a request effort onto a value this pipeline supports.

    An omitted effort stays omitted. A supported name is unchanged. An
    unsupported name uses the next higher supported value, then the next lower.
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
