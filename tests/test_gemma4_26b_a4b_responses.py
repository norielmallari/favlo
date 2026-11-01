"""Responses generation for Gemma 4. The sampler is not loaded."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from favlo.reply import MemoryReply
from gemma4_26b_a4b.chat import handle_responses
from gemma4_26b_a4b.geometry import CONTEXT

_TRACE = (
    "<|channel>thought\nLook at the notes.\n<channel|>"
    "The notes are short.\n"
)
_CALL = '<|channel>thought\n<channel|><|tool_call>call:glob{pattern:"**/*.txt"}<tool_call|>'


class _Runtime:
    def __init__(self, text: str, *, hit_limit: bool = False) -> None:
        self.root = Path(".")
        self.text = text
        self.prompt_tokens = 4
        self._cached_tokens = 2
        self._hit_limit = hit_limit
        self._stop = False
        self.limit = 0
        self.top_k = 0
        self.presence = -1.0
        self.temperature = -1.0
        self.stops: Any = None

    def generate(
        self,
        prompt: list[int],
        limit: int,
        temperature: float,
        top_p: float,
        top_k: int,
        presence: float,
        **kwargs: Any,
    ) -> tuple[list[int], str, str]:
        del prompt, top_p
        self.limit = limit
        self.top_k = top_k
        self.presence = presence
        self.temperature = temperature
        self.stops = kwargs.get("stops")
        on_token = kwargs.get("on_token")
        if callable(on_token):
            on_token(9)
        return [9], "", ""


def _install(monkeypatch: Any, runtime: _Runtime) -> list[tuple[Any, bool, bool]]:
    seen: list[tuple[Any, bool, bool]] = []

    def encode(
        root: Path,
        messages: list[dict[str, Any]],
        thinking: bool,
        tools: list[dict[str, Any]],
        saw_reasoning: bool,
    ) -> tuple[str, list[int]]:
        del root, tools
        seen.append((messages, thinking, saw_reasoning))
        return "<|turn>model\n", [1, 2, 3]

    monkeypatch.setattr("gemma4_26b_a4b.chat._encode", encode)
    monkeypatch.setattr(
        "gemma4_26b_a4b.chat.decode_text", lambda root, tokens: runtime.text if tokens else ""
    )
    monkeypatch.setattr("gemma4_26b_a4b.chat.collect_images", lambda messages: [])
    monkeypatch.setattr("gemma4_26b_a4b.stream.launch", _launch_local)
    return seen


def _launch_local(root: Path, lead: str, channel_open: bool) -> Any:
    from gemma4_26b_a4b.chat import decode_text
    from gemma4_26b_a4b.stream import launch_local

    return launch_local(root, lead, channel_open, decode_text)


def test_a_response_carries_the_trace_and_the_reply(monkeypatch: Any) -> None:
    runtime = _Runtime(_TRACE)
    _install(monkeypatch, runtime)
    reply = MemoryReply("gemma4-26b-a4b")
    handle_responses.complete(
        reply,
        runtime,
        {"model": "gemma4-26b-a4b", "input": "Hi", "temperature": 0.2, "top_k": 3},
        None,
        None,
        {},
    )
    body = reply.body
    assert body["object"] == "response"
    assert body["status"] == "completed"
    kinds = [item["type"] for item in body["output"]]
    assert kinds == ["reasoning", "message"]
    assert body["output"][0]["content"][0]["text"] == "Look at the notes."
    assert "encrypted_content" not in body["output"][0]
    assert body["output"][1]["content"][0]["text"] == "The notes are short.\n"
    assert body["usage"]["input_tokens"] == 4
    assert body["usage"]["output_tokens"] == 1
    assert body["usage"]["input_tokens_details"]["cached_tokens"] == 2
    assert runtime.temperature == 0.2
    assert runtime.top_k == 64
    assert runtime.presence == 0.0
    assert runtime.stops is None
    assert runtime.limit == 4096


def test_store_false_replays_the_trace(monkeypatch: Any) -> None:
    runtime = _Runtime(_TRACE)
    _install(monkeypatch, runtime)
    reply = MemoryReply()
    handle_responses.complete(
        reply,
        runtime,
        {
            "input": "Hi",
            "store": False,
            "reasoning": {"effort": "medium", "summary": "auto"},
        },
        None,
        None,
        {},
    )
    item = reply.body["output"][0]
    assert item["encrypted_content"] == "Look at the notes."
    assert item["summary"][0]["text"] == "Look at the notes."


def test_an_empty_thought_is_not_a_reasoning_item(monkeypatch: Any) -> None:
    runtime = _Runtime(_CALL)
    _install(monkeypatch, runtime)
    reply = MemoryReply()
    handle_responses.complete(reply, runtime, {"input": "Hi"}, None, None, {})
    kinds = [item["type"] for item in reply.body["output"]]
    assert kinds == ["function_call"]
    assert reply.body["output"][0]["name"] == "glob"
    assert reply.body["status"] == "completed"


def test_tool_choice_none_drops_the_call(monkeypatch: Any) -> None:
    runtime = _Runtime(_CALL + "done")
    _install(monkeypatch, runtime)
    reply = MemoryReply()
    handle_responses.complete(
        reply, runtime, {"input": "Hi", "tool_choice": "none"}, None, None, {}
    )
    kinds = [item["type"] for item in reply.body["output"]]
    assert kinds == ["message"]
    assert "glob" not in reply.body["output"][0]["content"][0]["text"]
    assert "done" in reply.body["output"][0]["content"][0]["text"]


def test_a_filled_cap_is_incomplete(monkeypatch: Any) -> None:
    runtime = _Runtime("The", hit_limit=True)
    _install(monkeypatch, runtime)
    reply = MemoryReply()
    handle_responses.complete(
        reply, runtime, {"input": "Hi", "max_output_tokens": 16}, None, None, {}
    )
    assert reply.body["status"] == "incomplete"
    assert reply.body["incomplete_details"]["reason"] == "max_output_tokens"
    assert runtime.limit == 16


def test_a_short_cap_is_rejected() -> None:
    reply = MemoryReply()
    handle_responses.complete(
        reply, None, {"input": "Hi", "max_output_tokens": 4}, None, None, {}
    )
    assert reply.status == 400


def test_audio_and_a_hosted_tool_are_rejected() -> None:
    audio = MemoryReply()
    handle_responses.complete(
        audio,
        None,
        {"input": [{"type": "input_audio", "data": "x"}]},
        None,
        None,
        {},
    )
    hosted = MemoryReply()
    handle_responses.complete(
        hosted,
        None,
        {"input": "Hi", "tools": [{"type": "file_search"}]},
        None,
        None,
        {},
    )
    assert audio.status == 400
    assert hosted.status == 400


def test_effort_none_is_thinking_off(monkeypatch: Any) -> None:
    runtime = _Runtime("Hello")
    seen = _install(monkeypatch, runtime)
    reply = MemoryReply()
    handle_responses.complete(
        reply,
        runtime,
        {"input": "Hi", "reasoning": {"effort": "none"}},
        None,
        None,
        {},
    )
    assert seen[0][1] is False


def test_a_stored_response_is_prepended(monkeypatch: Any) -> None:
    runtime = _Runtime("Hello")
    seen = _install(monkeypatch, runtime)
    state: dict[str, Any] = {}
    first = MemoryReply()
    handle_responses.complete(first, runtime, {"input": "Hi", "store": True}, None, None, state)
    second = MemoryReply()
    handle_responses.complete(
        second,
        runtime,
        {"input": "Next", "previous_response_id": first.body["id"], "store": True},
        None,
        None,
        state,
    )
    flat = str(seen[-1][0])
    assert "Hi" in flat
    assert "Hello" in flat
    assert "Next" in flat
    assert second.body["previous_response_id"] == first.body["id"]


def test_launch_reports_a_missing_tokenizer(tmp_path: Path) -> None:
    from gemma4_26b_a4b.stream import launch

    failed = False
    try:
        launch(tmp_path, "", False)
    except RuntimeError:
        failed = True
    assert failed


def test_a_stream_sends_one_delta_per_token(monkeypatch: Any) -> None:
    parts = ["Look", " at", " the", " notes."]
    runtime = _Runtime("")

    def generate(
        prompt: list[int],
        limit: int,
        temperature: float,
        top_p: float,
        top_k: int,
        presence: float,
        **kwargs: Any,
    ) -> tuple[list[int], str, str]:
        del prompt, limit, temperature, top_p, top_k, presence
        on_token = kwargs.get("on_token")
        assert callable(on_token)
        for index in range(len(parts)):
            on_token(index + 1)
        return list(range(1, len(parts) + 1)), "", ""

    runtime.generate = generate  # type: ignore[method-assign]

    def decode(root: Path, tokens: list[int]) -> str:
        del root
        return "".join(parts[: len(tokens)])

    def encode(
        root: Path,
        messages: list[dict[str, Any]],
        thinking: bool,
        tools: list[dict[str, Any]],
        saw_reasoning: bool,
    ) -> tuple[str, list[int]]:
        del root, messages, thinking, tools, saw_reasoning
        return "<|turn>model\n", [1]

    monkeypatch.setattr("gemma4_26b_a4b.chat._encode", encode)
    monkeypatch.setattr("gemma4_26b_a4b.chat.decode_text", decode)
    monkeypatch.setattr("gemma4_26b_a4b.chat.collect_images", lambda messages: [])
    monkeypatch.setattr("gemma4_26b_a4b.stream.launch", _launch_local)
    reply = MemoryReply()
    handle_responses.complete(
        reply, runtime, {"input": "Hi", "stream": True}, None, None, {}
    )
    deltas = [
        event["delta"]
        for event in reply.events
        if event["type"] == "response.output_text.delta"
    ]
    assert deltas == parts


def test_a_stream_shares_one_response_id(monkeypatch: Any) -> None:
    runtime = _Runtime(_TRACE)
    _install(monkeypatch, runtime)
    reply = MemoryReply()
    handle_responses.complete(
        reply, runtime, {"input": "Hi", "stream": True, "store": False}, None, None, {}
    )
    kinds = [event["type"] for event in reply.events]
    assert kinds[0] == "response.created"
    assert "response.in_progress" in kinds
    assert "response.output_text.delta" in kinds
    assert "response.reasoning_text.delta" in kinds
    assert "response.reasoning_text.done" not in kinds
    assert kinds[-1] == "response.completed"
    opened = reply.events[0]["response"]["id"]
    assert reply.events[-1]["response"]["id"] == opened
    assert reply.sent == []
    reasoning = next(item for item in reply.events[-1]["response"]["output"] if item["type"] == "reasoning")
    assert reasoning["encrypted_content"] == "Look at the notes."


def test_a_streamed_call_sends_the_arguments_once(monkeypatch: Any) -> None:
    runtime = _Runtime(_CALL)
    _install(monkeypatch, runtime)
    reply = MemoryReply()
    handle_responses.complete(reply, runtime, {"input": "Hi", "stream": True}, None, None, {})
    kinds = [event["type"] for event in reply.events]
    assert "response.function_call_arguments.delta" not in kinds
    added = next(
        event
        for event in reply.events
        if event["type"] == "response.output_item.added" and event["item"]["type"] == "function_call"
    )
    done = next(event for event in reply.events if event["type"] == "response.function_call_arguments.done")
    assert added["item"]["arguments"] == ""
    assert added["item"]["name"] == "glob"
    assert "pattern" in done["arguments"]


def test_truncation_auto_drops_the_oldest_item(monkeypatch: Any) -> None:
    runtime = _Runtime("ok")
    seen: list[int] = []

    def encode(
        root: Path,
        messages: list[dict[str, Any]],
        thinking: bool,
        tools: list[dict[str, Any]],
        saw_reasoning: bool,
    ) -> tuple[str, list[int]]:
        del root, thinking, tools, saw_reasoning
        seen.append(len(messages))
        if len(messages) > 2:
            return "x", [0] * (CONTEXT + 1)
        return "x", [1]

    monkeypatch.setattr("gemma4_26b_a4b.chat._encode", encode)
    monkeypatch.setattr("gemma4_26b_a4b.chat.decode_text", lambda root, tokens: "ok")
    monkeypatch.setattr("gemma4_26b_a4b.chat.collect_images", lambda messages: [])
    reply = MemoryReply()
    handle_responses.complete(
        reply,
        runtime,
        {
            "input": [
                {"type": "message", "role": "user", "content": "one"},
                {"type": "message", "role": "assistant", "content": "two"},
                {"type": "message", "role": "user", "content": "three"},
            ],
            "truncation": "auto",
        },
        None,
        None,
        {},
    )
    assert reply.status == 200
    assert seen[-1] <= 2


def test_truncation_disabled_rejects_an_overlong_prompt(monkeypatch: Any) -> None:
    runtime = _Runtime("ok")

    def encode(*args: Any, **kwargs: Any) -> tuple[str, list[int]]:
        del args, kwargs
        return "x", [0] * (CONTEXT + 1)

    monkeypatch.setattr("gemma4_26b_a4b.chat._encode", encode)
    reply = MemoryReply()
    handle_responses.complete(reply, runtime, {"input": "Hi"}, None, None, {})
    assert reply.status == 400
