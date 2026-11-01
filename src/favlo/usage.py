"""OpenAI token fields on a chat completion result."""

from __future__ import annotations

from typing import Any

_PROMPT_FIELDS = (
    "audio_tokens",
    "cache_write_tokens",
    "cached_tokens",
    "image_tokens",
    "text_tokens",
)
_COMPLETION_FIELDS = (
    "accepted_prediction_tokens",
    "audio_tokens",
    "reasoning_tokens",
    "rejected_prediction_tokens",
    "text_tokens",
)


def complete_chat_usage(payload: dict[str, Any]) -> None:
    """Fill the standard usage fields and ``max_prompt_tokens`` on one result."""
    kind = payload.get("object")
    if kind not in {"chat.completion", "chat.completion.chunk"}:
        return
    usage = payload.get("usage")
    if not isinstance(usage, dict):
        return
    prompt = _count(usage.get("prompt_tokens"))
    completion = _count(usage.get("completion_tokens"))
    if (
        prompt is not None
        and completion is not None
        and _count(usage.get("total_tokens")) is None
    ):
        usage["total_tokens"] = prompt + completion
    usage["prompt_tokens_details"] = _prompt_details(usage.get("prompt_tokens_details"), prompt)
    usage["completion_tokens_details"] = _completion_details(
        usage.get("completion_tokens_details"),
        completion,
    )
    cap = _prompt_limit(usage, payload)
    if cap is not None:
        usage["max_prompt_tokens"] = cap


def _count(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _fields(source: object, names: tuple[str, ...]) -> tuple[dict[str, Any], dict[str, int]]:
    raw = source if isinstance(source, dict) else {}
    return raw, {name: _count(raw.get(name)) or 0 for name in names}


def _prompt_details(source: object, prompt: int | None) -> dict[str, int]:
    raw, filled = _fields(source, _PROMPT_FIELDS)
    if "text_tokens" not in raw and prompt is not None:
        filled["text_tokens"] = max(0, prompt - filled["audio_tokens"] - filled["image_tokens"])
    return filled


def _completion_details(source: object, completion: int | None) -> dict[str, int]:
    raw, filled = _fields(source, _COMPLETION_FIELDS)
    if "text_tokens" not in raw and completion is not None:
        spent = filled["audio_tokens"] + filled["reasoning_tokens"]
        filled["text_tokens"] = max(0, completion - spent)
    return filled


def _prompt_limit(usage: dict[str, Any], payload: dict[str, Any]) -> int | None:
    for candidate in (
        usage.get("max_prompt_tokens"),
        payload.get("max_prompt_tokens"),
        usage.get("context_length"),
    ):
        cap = _count(candidate)
        if cap is not None and cap > 0:
            return cap
    return _registered_cap(payload.get("model"))


def _registered_cap(model: object) -> int | None:
    if not isinstance(model, str) or not model.strip():
        return None
    try:
        from favlo.registry import resolve

        row = resolve(model.strip())
    except LookupError:
        return None
    cap = _count(row.max_prompt_tokens)
    if cap is not None and cap > 0:
        return cap
    window = _count(row.context_length)
    if window is not None and window > 0:
        return window
    return None
