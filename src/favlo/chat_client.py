"""Agent Framework client for the listener's chat completions.

Every chat caller uses this client. Reasoning from a completion is sent back
on ``reasoning_content``. Tool calls stay on ``tool_calls``.
"""

from __future__ import annotations

from typing import Any


def reasoning_text(message: Any) -> str:
    """The trace on a completion message or a stream delta.

    ``reasoning`` is the OpenAI field. ``reasoning_content`` is the same trace
    on clients that still read that name.
    """
    if message is None:
        return ""
    for name in ("reasoning", "reasoning_content"):
        value = getattr(message, name, None)
        if isinstance(value, str) and value.strip():
            return value
    extra = getattr(message, "model_extra", None)
    if isinstance(extra, dict):
        for name in ("reasoning", "reasoning_content"):
            raw = extra.get(name)
            if isinstance(raw, str) and raw.strip():
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
    """Put reasoning on ``reasoning_content`` for the next request.

    The listener reads that field when it renders the next prompt. ``tool_calls``
    stay on the message the framework built.
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
    middleware: Any | None = None,
) -> Any:
    """Chat client that sends reasoning back on the next request."""
    kwargs: dict[str, Any] = {
        "model": model,
        "api_key": "favlo",
        "base_url": base_url,
    }
    if async_client is not None:
        kwargs["async_client"] = async_client
    if middleware is not None:
        kwargs["middleware"] = middleware
    client = _chat_client_type()(**kwargs)
    if async_client is None:
        http = getattr(client, "client", None)
        if http is not None:
            http.timeout = 3600.0
    return client


async def close_chat_client(client: Any) -> None:
    """Close the listener client the framework opened for this chat."""
    http = getattr(client, "client", None)
    closer = getattr(http, "close", None)
    if closer is not None:
        closed = closer()
        if hasattr(closed, "__await__"):
            await closed
    own = getattr(client, "close", None)
    if own is None or own is closer:
        return
    closed = own()
    if hasattr(closed, "__await__"):
        await closed


def prompt_window() -> Any:
    """Context provider that compacts once a prompt reaches half its cap."""
    return _window_type()()


def _chat_client_type() -> type:
    global _CLIENT
    if _CLIENT is None:
        from agent_framework.openai import OpenAIChatClient

        class FavloChatClient(OpenAIChatClient):
            """Responses client. Keeps the prompt cap from usage when it is present."""

            max_prompt_tokens: int | None = None
            prompt_tokens: int | None = None

            def _parse_usage_from_openai(self, usage: Any) -> Any:
                details = super()._parse_usage_from_openai(usage)
                cap = _usage_cap(usage)
                if cap is not None:
                    self.max_prompt_tokens = cap
                prompt = getattr(usage, "input_tokens", None)
                if not isinstance(prompt, int) or isinstance(prompt, bool):
                    prompt = getattr(usage, "prompt_tokens", None)
                if isinstance(prompt, int) and not isinstance(prompt, bool) and prompt > 0:
                    self.prompt_tokens = prompt
                return _with_prompt_cap(details, usage)

        _CLIENT = FavloChatClient
    return _CLIENT


def _window_type() -> type:
    global _WINDOW
    if _WINDOW is None:
        from agent_framework import ContextProvider

        class PromptWindow(ContextProvider):
            """Leaves older messages out once a prompt reaches half the cap."""

            after_run_once_per_turn = True

            def __init__(self) -> None:
                super().__init__("prompt_window")

            async def after_run(
                self,
                *,
                agent: Any,
                session: Any,
                context: Any,
                state: dict[str, Any],
            ) -> None:
                await _compact_half(agent, session, context, state)

        _WINDOW = PromptWindow
    return _WINDOW


def _with_prompt_cap(details: Any, usage: Any) -> Any:
    cap = _usage_cap(usage)
    if cap is None:
        return details
    if details is None:
        from agent_framework import UsageDetails

        details = UsageDetails()
    details["max_prompt_tokens"] = cap
    return details


def _usage_cap(usage: Any) -> int | None:
    for name in ("max_prompt_tokens", "context_length"):
        value = getattr(usage, name, None)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value
        extra = getattr(usage, "model_extra", None)
        if isinstance(extra, dict):
            raw = extra.get(name)
            if isinstance(raw, int) and not isinstance(raw, bool) and raw > 0:
                return raw
    return None


async def _compact_half(agent: Any, session: Any, context: Any, state: dict[str, Any]) -> None:
    client = getattr(agent, "client", None)
    usage = getattr(getattr(context, "response", None), "usage_details", None)
    if not isinstance(usage, dict):
        usage = {}
    cap = _stored_count(getattr(client, "max_prompt_tokens", None))
    used = _stored_count(getattr(client, "prompt_tokens", None))
    if cap is None:
        cap = _stored_count(usage.get("max_prompt_tokens"))
    if used is None:
        used = _stored_count(usage.get("input_token_count"))
    if cap is not None:
        state["max_prompt_tokens"] = cap
    else:
        cap = _stored_count(state.get("max_prompt_tokens"))
    if used is not None:
        state["prompt_tokens"] = used
    else:
        used = _stored_count(state.get("prompt_tokens"))
    if cap is None or used is None:
        return
    half = max(1, cap // 2)
    if used < half:
        return
    messages = _stored_messages(session)
    if messages is None:
        return
    await _compress(messages, used, half)


def _stored_count(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        return None
    return value


def _stored_messages(session: Any) -> list[Any] | None:
    memory = getattr(session, "state", None)
    if not isinstance(memory, dict):
        return None
    history = memory.get("in_memory")
    if not isinstance(history, dict):
        return None
    messages = history.get("messages")
    if not isinstance(messages, list) or len(messages) < 2:
        return None
    from agent_framework import Message

    if any(not isinstance(item, Message) for item in messages):
        return None
    return messages


async def _compress(messages: list[Any], used: int, half: int) -> None:
    from agent_framework import TruncationStrategy
    from agent_framework._compaction import (
        CharacterEstimatorTokenizer,
        _run_compaction_strategy,
        _serialize_message,
    )

    base = CharacterEstimatorTokenizer()
    included = [
        message
        for message in messages
        if not message.additional_properties.get("_excluded")
    ]
    estimate = sum(base.count_tokens(_serialize_message(message)) for message in included)
    factor = used / estimate if estimate else 1.0
    tokenizer = _ScaledTokenizer(factor)
    _clear_token_counts(messages)
    strategy = TruncationStrategy(
        max_n=half,
        compact_to=half,
        tokenizer=tokenizer,
        preserve_system=True,
    )
    await _run_compaction_strategy(
        messages,
        strategy=strategy,
        tokenizer=tokenizer,
        phase="after_run",
    )


def _clear_token_counts(messages: list[Any]) -> None:
    for message in messages:
        group = message.additional_properties.get("_group")
        if isinstance(group, dict):
            group.pop("token_count", None)
            group.pop("token_count_basis", None)


class _ScaledTokenizer:
    """Character estimate scaled so the included text matches the reported prompt."""

    def __init__(self, factor: float) -> None:
        from agent_framework._compaction import CharacterEstimatorTokenizer

        self.factor = factor
        self._base = CharacterEstimatorTokenizer()

    def count_tokens(self, text: str) -> int:
        return max(1, int(round(self._base.count_tokens(text) * self.factor)))


_CLIENT: type | None = None
_WINDOW: type | None = None


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
