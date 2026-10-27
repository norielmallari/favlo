"""Agent Framework client for the listener's chat completions.

Every chat caller uses this client. Reasoning from a completion is sent back
on ``reasoning_content``. A reply whose text already contains the tool-call
markup is sent as that text.
"""

from __future__ import annotations

from typing import Any


def reasoning_text(message: Any) -> str:
    """``reasoning_content`` on a completion message or a stream delta."""
    if message is None:
        return ""
    value = getattr(message, "reasoning_content", None)
    if isinstance(value, str) and value:
        return value
    extra = getattr(message, "model_extra", None)
    if isinstance(extra, dict):
        raw = extra.get("reasoning_content")
        if isinstance(raw, str) and raw:
            return raw
    return ""


def read_reasoning(message: Any, contents: list[Any]) -> list[Any]:
    """Keep ``reasoning_content`` as reasoning, beside the reply text."""
    text = reasoning_text(message)
    if not text:
        return contents
    from agent_framework import Content

    return [Content.from_text_reasoning(text=text), *contents]


def prepare_messages(message: object, built: list[dict[str, object]]) -> list[dict[str, object]]:
    """Put reasoning on ``reasoning_content`` for the chat template.

    The listener reads that field when it renders the next prompt. A reply
    whose text already contains the tool-call markup is sent as that text.
    """
    if getattr(message, "role", None) != "assistant":
        return built
    reasoning = "".join(
        part.text or ""
        for part in getattr(message, "contents", ())
        if getattr(part, "type", None) == "text_reasoning"
        and not getattr(part, "protected_data", None)
    )
    if not reasoning:
        return built
    kept: list[dict[str, object]] = []
    placed = False
    for item in built:
        if item.get("role") != "assistant":
            kept.append(item)
            continue
        text = _content_text(item.get("content"))
        if text == reasoning and "tool_calls" not in item:
            continue
        copied = dict(item)
        if not placed:
            copied["reasoning_content"] = reasoning
            placed = True
        if "<tool_call>" in text:
            copied.pop("tool_calls", None)
        kept.append(copied)
    if not placed:
        kept.append({"role": "assistant", "reasoning_content": reasoning, "content": ""})
    return kept


def history_message(role: str, text: str, reasoning: str = "") -> Any:
    """One stored turn. Reasoning stays a reasoning part, not reply text."""
    from agent_framework import Content, Message

    parts: list[Any] = []
    if reasoning:
        parts.append(Content.from_text_reasoning(text=reasoning))
    parts.append(text)
    return Message(role, parts)


def open_chat_client(
    model: str,
    base_url: str,
    async_client: Any | None = None,
) -> Any:
    """Chat client that sends reasoning back on the next request."""
    from agent_framework.openai import OpenAIChatCompletionClient

    kwargs: dict[str, Any] = {
        "model": model,
        "api_key": "favlo",
        "base_url": base_url,
        "response_parser": read_reasoning,
        "message_preparer": prepare_messages,
    }
    if async_client is not None:
        kwargs["async_client"] = async_client
    return OpenAIChatCompletionClient(**kwargs)


def _content_text(content: object) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for item in content:
        if isinstance(item, str):
            parts.append(item)
        elif isinstance(item, dict) and item.get("type") == "text":
            parts.append(str(item.get("text") or ""))
    return "".join(parts)
