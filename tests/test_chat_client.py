"""The shared chat client sends reasoning back on the next request."""

import asyncio
from types import SimpleNamespace

from favlo.chat_client import history_message, prepare_messages, prompt_window, read_reasoning


class _Part:
    def __init__(self, kind: str, text: str) -> None:
        self.type = kind
        self.text = text
        self.protected_data = None


class _Message:
    def __init__(self, contents: list[_Part]) -> None:
        self.role = "assistant"
        self.contents = contents


def test_reasoning_returns_on_its_own_field() -> None:
    message = _Message(
        [
            _Part("text_reasoning", "look\n"),
            _Part("text", "Hi"),
        ]
    )
    built = [
        {"role": "assistant", "content": [{"type": "text", "text": "look\n"}]},
        {
            "role": "assistant",
            "content": "Hi",
            "tool_calls": [{"id": "call_0"}],
        },
    ]
    prepared = prepare_messages(message, built)
    assert prepared == [
        {
            "role": "assistant",
            "content": "Hi",
            "tool_calls": [{"id": "call_0"}],
            "reasoning_content": "look\n",
        }
    ]


def test_the_reasoning_field_is_the_trace() -> None:
    class _Delta:
        model_extra = {"reasoning": "Look at the notes."}

    parsed = read_reasoning(_Delta(), [])
    assert parsed[0].text == "Look at the notes."


def test_reasoning_delta_is_not_reply_text() -> None:
    class _Delta:
        model_extra = {"reasoning_content": "look"}

    parsed = read_reasoning(_Delta(), [])
    assert parsed[0].type == "text_reasoning"
    assert parsed[0].text == "look"


def test_stored_reasoning_returns_on_the_next_request() -> None:
    message = history_message("assistant", "Hi", "look\n")
    prepared = prepare_messages(message, [{"role": "assistant", "content": "Hi"}])
    assert prepared == [
        {"role": "assistant", "content": "Hi", "reasoning_content": "look\n"}
    ]


def test_the_client_keeps_the_prompt_cap() -> None:
    import pytest

    pytest.importorskip("openai")
    from openai.types.completion_usage import CompletionUsage

    from favlo.chat_client import close_chat_client, open_chat_client

    client = open_chat_client("example-model", "http://127.0.0.1:8129/v1")
    usage = CompletionUsage.model_validate(
        {
            "prompt_tokens": 10,
            "completion_tokens": 2,
            "total_tokens": 12,
            "max_prompt_tokens": 4096,
            "prompt_tokens_details": {"cached_tokens": 4},
        }
    )
    try:
        details = client._parse_usage_from_openai(usage)
    finally:
        asyncio.run(close_chat_client(client))
    assert details["max_prompt_tokens"] == 4096
    assert details["prompt/cached_tokens"] == 4
    assert client.max_prompt_tokens == 4096
    assert client.prompt_tokens == 10


def test_half_the_prompt_cap_leaves_older_messages_out() -> None:
    from agent_framework import AgentSession, Message

    session = AgentSession()
    messages = [
        Message("user", ["alpha " * 80]),
        Message("assistant", ["beta " * 80]),
        Message("user", ["gamma " * 80]),
        Message("assistant", ["delta " * 80]),
    ]
    session.state["in_memory"] = {"messages": messages}
    window = prompt_window()
    context = SimpleNamespace(
        response=SimpleNamespace(
            usage_details={"max_prompt_tokens": 300, "input_token_count": 80}
        )
    )
    agent = SimpleNamespace(client=SimpleNamespace(max_prompt_tokens=100, prompt_tokens=80))

    async def once() -> None:
        await window.after_run(agent=agent, session=session, context=context, state={})

    asyncio.run(once())
    excluded = [item.additional_properties.get("_excluded") is True for item in messages]
    assert any(excluded[:-1])
    assert excluded[-1] is False


def test_a_short_prompt_keeps_every_message() -> None:
    from agent_framework import AgentSession, Message

    session = AgentSession()
    messages = [Message("user", ["alpha"]), Message("assistant", ["beta"])]
    session.state["in_memory"] = {"messages": messages}
    window = prompt_window()
    context = SimpleNamespace(
        response=SimpleNamespace(
            usage_details={"max_prompt_tokens": 100, "input_token_count": 10}
        )
    )

    async def once() -> None:
        await window.after_run(agent=None, session=session, context=context, state={})

    asyncio.run(once())
    assert all(item.additional_properties.get("_excluded") is not True for item in messages)
