"""Prompt compaction for the web runner."""

from __future__ import annotations

from typing import Any


def percent_window(percent: int) -> Any:
    """Context provider that compacts at ``percent`` of max prompt tokens."""
    capped = max(50, min(95, int(percent)))
    return _window_type(capped)()


def _window_type(percent: int) -> type:
    from agent_framework import ContextProvider
    from favlo.chat_client import _clear_token_counts, _stored_count, _stored_messages

    class PercentWindow(ContextProvider):
        """Leaves older messages out once a prompt reaches the configured percent."""

        after_run_once_per_turn = True

        def __init__(self) -> None:
            super().__init__(f"prompt_window_{percent}")
            self.percent = percent

        async def after_run(
            self,
            *,
            agent: Any,
            session: Any,
            context: Any,
            state: dict[str, Any],
        ) -> None:
            await _compact_percent(agent, session, context, state, self.percent)

    return PercentWindow


async def _compact_percent(
    agent: Any,
    session: Any,
    context: Any,
    state: dict[str, Any],
    percent: int,
) -> None:
    from favlo.chat_client import _stored_count, _stored_messages

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
    line = max(1, (cap * percent) // 100)
    if used < line:
        return
    messages = _stored_messages(session)
    if messages is None:
        return
    await _compress(messages, used, line)


async def _compress(messages: list[Any], used: int, line: int) -> None:
    from agent_framework import TruncationStrategy
    from agent_framework._compaction import (
        CharacterEstimatorTokenizer,
        _run_compaction_strategy,
        _serialize_message,
    )
    from favlo.chat_client import _ScaledTokenizer, _clear_token_counts

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
        max_n=line,
        compact_to=line,
        tokenizer=tokenizer,
        preserve_system=True,
    )
    await _run_compaction_strategy(
        messages,
        strategy=strategy,
        tokenizer=tokenizer,
        phase="after_run",
    )
