"""Responses generation for Gemma 4. The sampler is not loaded."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from favlo.reply import MemoryReply
from gemma4_26b_a4b.metal41.geometry import CONTEXT
from gemma4_26b_a4b.portable.chat import handle_responses

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


def _install(monkeypatch: Any, runtime: _Runtime) -> list[tuple[Any, bool]]:
    seen: list[tuple[Any, bool]] = []

    def encode(
        root: Path,
        messages: list[dict[str, Any]],
        thinking: bool,
        tools: list[dict[str, Any]],
    ) -> tuple[str, list[int]]:
        del root, tools
        seen.append((messages, thinking))
        return "<|turn>model\n", [1, 2, 3]

    monkeypatch.setattr("gemma4_26b_a4b.portable.chat._encode", encode)
    monkeypatch.setattr(
        "gemma4_26b_a4b.portable.chat.decode_text", lambda root, tokens: runtime.text if tokens else ""
    )
    monkeypatch.setattr("gemma4_26b_a4b.metal41.vision.collect_images", lambda messages: [])
    monkeypatch.setattr("gemma4_26b_a4b.metal41.stream.launch", _launch_local)
    return seen


def _launch_local(root: Path, lead: str, channel_open: bool) -> Any:
    from gemma4_26b_a4b.metal41.stream import launch_local
    from gemma4_26b_a4b.portable.chat import decode_text

    return launch_local(root, lead, channel_open, decode_text)


def test_a_response_carries_the_trace_and_the_reply(monkeypatch: Any) -> None:
    runtime = _Runtime(_TRACE)
    _install(monkeypatch, runtime)
    reply = MemoryReply("gemma4_26b_a4b")
    handle_responses.handle_responses(
        reply,
        runtime,
        {"model": "gemma4_26b_a4b", "input": "Hi", "temperature": 0.2, "top_k": 3},
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
    assert body["output"][0]["encrypted_content"] == "Look at the notes."
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
    handle_responses.handle_responses(
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
    handle_responses.handle_responses(reply, runtime, {"input": "Hi"}, None, None, {})
    kinds = [item["type"] for item in reply.body["output"]]
    assert kinds == ["function_call"]
    assert reply.body["output"][0]["name"] == "glob"
    assert reply.body["status"] == "completed"


def test_tool_choice_none_drops_the_call(monkeypatch: Any) -> None:
    runtime = _Runtime(_CALL + "done")
    _install(monkeypatch, runtime)
    reply = MemoryReply()
    handle_responses.handle_responses(
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
    handle_responses.handle_responses(
        reply, runtime, {"input": "Hi", "max_output_tokens": 16}, None, None, {}
    )
    assert reply.body["status"] == "incomplete"
    assert reply.body["incomplete_details"]["reason"] == "max_output_tokens"
    assert runtime.limit == 16


def test_a_short_cap_is_rejected() -> None:
    reply = MemoryReply()
    handle_responses.handle_responses(
        reply, None, {"input": "Hi", "max_output_tokens": 4}, None, None, {}
    )
    assert reply.status == 400


def test_audio_and_a_hosted_tool_are_rejected() -> None:
    audio = MemoryReply()
    handle_responses.handle_responses(
        audio,
        None,
        {"input": [{"type": "input_audio", "data": "x"}]},
        None,
        None,
        {},
    )
    hosted = MemoryReply()
    handle_responses.handle_responses(
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
    handle_responses.handle_responses(
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
    handle_responses(first, runtime, {"input": "Hi", "store": True}, None, None, state)
    second = MemoryReply()
    handle_responses(
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
    from gemma4_26b_a4b.metal41.stream import launch

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
    ) -> tuple[str, list[int]]:
        del root, messages, thinking, tools
        return "<|turn>model\n", [1]

    monkeypatch.setattr("gemma4_26b_a4b.portable.chat._encode", encode)
    monkeypatch.setattr("gemma4_26b_a4b.portable.chat.decode_text", decode)
    monkeypatch.setattr("gemma4_26b_a4b.metal41.vision.collect_images", lambda messages: [])
    monkeypatch.setattr("gemma4_26b_a4b.metal41.stream.launch", _launch_local)
    reply = MemoryReply()
    handle_responses.handle_responses(
        reply, runtime, {"input": "Hi", "stream": True}, None, None, {}
    )
    deltas = [
        event["delta"]
        for event in reply.events
        if event["type"] == "response.output_text.delta"
    ]
    assert deltas == parts


def test_a_stream_shares_one_response_id(monkeypatch: Any) -> None:
    from favlo.responses import ResponseWire

    runtime = _Runtime(_TRACE)
    _install(monkeypatch, runtime)
    inner = MemoryReply()
    reply = ResponseWire(inner)
    handle_responses.handle_responses(
        reply, runtime, {"input": "Hi", "stream": True, "store": False}, None, None, {}
    )
    kinds = [event["type"] for event in inner.events]
    assert kinds[0] == "response.created"
    assert "response.in_progress" in kinds
    assert "response.output_text.delta" in kinds
    assert "response.reasoning_text.delta" in kinds
    assert "response.reasoning_text.done" not in kinds
    assert kinds[-1] == "response.completed"
    opened = inner.events[0]["response"]["id"]
    assert inner.events[-1]["response"]["id"] == opened
    assert inner.sent == []
    reasoning = next(item for item in inner.events[-1]["response"]["output"] if item["type"] == "reasoning")
    assert reasoning["encrypted_content"] == "Look at the notes."


def test_a_streamed_call_sends_the_arguments_once(monkeypatch: Any) -> None:
    runtime = _Runtime(_CALL)
    _install(monkeypatch, runtime)
    reply = MemoryReply()
    handle_responses.handle_responses(reply, runtime, {"input": "Hi", "stream": True}, None, None, {})
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
    ) -> tuple[str, list[int]]:
        del root, thinking, tools
        seen.append(len(messages))
        if len(messages) > 2:
            return "x", [0] * (CONTEXT + 1)
        return "x", [1]

    monkeypatch.setattr("gemma4_26b_a4b.portable.chat._encode", encode)
    monkeypatch.setattr("gemma4_26b_a4b.portable.chat.decode_text", lambda root, tokens: "ok")
    monkeypatch.setattr("gemma4_26b_a4b.metal41.vision.collect_images", lambda messages: [])
    reply = MemoryReply()
    handle_responses.handle_responses(
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

    monkeypatch.setattr("gemma4_26b_a4b.portable.chat._encode", encode)
    reply = MemoryReply()
    handle_responses.handle_responses(reply, runtime, {"input": "Hi"}, None, None, {})
    assert reply.status == 400


def test_a_dropped_thought_shares_the_prompt_before_the_generation_suffix() -> None:
    root = Path("models/gemma-4-26B-A4B-it-fp8-e20")
    if not (root / "tokenizer.json").is_file():
        pytest.skip("tokenizer is not installed")
    from gemma4_26b_a4b.portable.chat import decode_text, encode_chat, generation_cut

    picture = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "Describe the picture."},
            ],
        }
    ]
    follow = [
        *picture,
        {"role": "assistant", "content": "A horse is running."},
        {"role": "user", "content": "Build ledger.py."},
    ]
    prompt = encode_chat(root, picture, False)
    cut = generation_cut(root, prompt)
    assert cut is not None
    assert "<|image|>" in decode_text(root, prompt[:cut])
    nxt = encode_chat(root, follow, False)
    assert nxt[:cut] == prompt[:cut]
    assert nxt[: len(prompt)] == prompt
    plain = encode_chat(root, picture, True)
    plain_cut = generation_cut(root, plain)
    assert plain_cut is not None
    assert "<|image|>" in decode_text(root, plain[:plain_cut])
    kept = encode_chat(root, follow, True)
    assert kept[:plain_cut] == plain[:plain_cut]
    assert kept[: len(plain)] == plain

    tools = [
        {
            "type": "function",
            "function": {
                "name": "read",
                "description": "read",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]
    opened_messages = [
        {"role": "user", "content": "Read the file."},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "1",
                    "type": "function",
                    "function": {"name": "read", "arguments": {}},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "1", "name": "read", "content": "ok"},
    ]
    opened = encode_chat(root, opened_messages, True, tools)
    opened_cut = generation_cut(root, opened)
    assert opened_cut is not None
    dropped = encode_chat(
        root,
        [
            *opened_messages,
            {
                "role": "assistant",
                "reasoning_content": "The file is short.",
                "content": "Done.",
            },
            {"role": "user", "content": "Now write ledger.py."},
        ],
        True,
        tools,
    )
    assert dropped[:opened_cut] == opened[:opened_cut]
    assert dropped[: len(opened)] != opened


def test_a_covered_image_span_is_not_encoded(monkeypatch: pytest.MonkeyPatch) -> None:
    from gemma4_26b_a4b.metal41.forward import Gemma4Runtime
    from gemma4_26b_a4b.metal41.geometry import BOI_TOKEN, EOI_TOKEN, IMAGE_TOKEN

    monkeypatch.setattr("gemma4_26b_a4b.metal41.vision.image_soft_tokens", lambda _blob: 2)
    block = [BOI_TOKEN, IMAGE_TOKEN, IMAGE_TOKEN, EOI_TOKEN]

    class _Walk:
        def __init__(self) -> None:
            self._stop = False
            self._vision = self
            self.notes: list[str] = []
            self.encoded: list[bytes] = []
            self.prefilled: list[list[int]] = []

        def _status(self, text: str) -> None:
            self.notes.append(text)

        def encode(self, blob: bytes, note: Any = None) -> tuple[object, int]:
            del note
            self.encoded.append(blob)
            return object(), 2

        def _prefill(self, tokens: list[int], *, finish: bool = True) -> None:
            del finish
            self.prefilled.append(list(tokens))

        def _accept_rows(self, source: object, count: int, *, finish: bool) -> None:
            del source, count, finish

    covered = _Walk()
    Gemma4Runtime._prefill_visual(covered, block, len(block), [b"cached"])
    assert covered.notes == ["vision cached 2"]
    assert covered.encoded == []
    assert covered.prefilled == []

    later = _Walk()
    Gemma4Runtime._prefill_visual(later, block + block, len(block), [b"cached", b"fresh"])
    assert later.notes == ["vision cached 2"]
    assert later.encoded == [b"fresh"]
    assert later.prefilled == [[BOI_TOKEN], [EOI_TOKEN]]


def test_a_preserve_thinking_value_is_ignored(monkeypatch: Any) -> None:
    runtime = _Runtime("ok")
    _install(monkeypatch, runtime)
    for value in ("true", False, True):
        reply = MemoryReply("gemma4_26b_a4b")
        handle_responses.handle_responses(
            reply,
            runtime,
            {"input": "Hi", "preserve_thinking": value},
            None,
            None,
            {},
        )
        assert reply.status == 200


def test_a_later_user_keeps_the_tool_trace() -> None:
    root = Path("models/gemma-4-26B-A4B-it-fp8-e20")
    if not (root / "tokenizer.json").is_file():
        pytest.skip("tokenizer is not installed")
    from gemma4_26b_a4b.portable.chat import decode_text, encode_chat

    tools = [
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "read",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
            },
        }
    ]
    call = {
        "id": "c1",
        "type": "function",
        "function": {"name": "read_file", "arguments": {"path": "a.py"}},
    }
    user = {"role": "user", "content": "Read a.py"}
    assistant = {
        "role": "assistant",
        "reasoning_content": "I should read the file first.",
        "content": "",
        "tool_calls": [call],
    }
    tool = {"role": "tool", "tool_call_id": "c1", "name": "read_file", "content": "print(1)\n"}
    mid = [user, assistant, tool]
    follow = [
        *mid,
        {"role": "assistant", "reasoning_content": "The file prints one.", "content": "It prints 1."},
        {"role": "user", "content": "Now change it."},
    ]
    for thinking in (False, True):
        text = decode_text(root, encode_chat(root, mid, thinking, tools))
        nxt = decode_text(root, encode_chat(root, follow, thinking, tools))
        assert "I should read the file first." in text
        assert "I should read the file first." in nxt
        assert "The file prints one." not in nxt


def test_a_tool_trace_stays_and_the_previous_prompt_is_shared() -> None:
    root = Path("models/gemma-4-26B-A4B-it-fp8-e20")
    if not (root / "tokenizer.json").is_file():
        pytest.skip("tokenizer is not installed")
    from gemma4_26b_a4b.portable.chat import decode_text, encode_chat

    tools = [
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "read",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
            },
        }
    ]
    call = {
        "id": "c1",
        "type": "function",
        "function": {"name": "read_file", "arguments": {"path": "a.py"}},
    }
    asked = [
        {"role": "user", "content": "Call read_file with path a.py."},
        {
            "role": "assistant",
            "reasoning_content": "older-trace",
            "content": "",
            "tool_calls": [call],
        },
        {"role": "tool", "tool_call_id": "c1", "name": "read_file", "content": "print(1)\n"},
        {"role": "user", "content": "Call read_file with path b.py."},
    ]
    plain = [
        {"role": "user", "content": "Hello."},
        {"role": "assistant", "reasoning_content": "older-trace", "content": "Hi."},
        {"role": "user", "content": "Again."},
    ]
    for thinking in (False, True):
        kept_ids = encode_chat(root, asked, thinking, tools)
        kept = decode_text(root, kept_ids)
        assert "older-trace" in kept
        assert "older-trace" not in decode_text(root, encode_chat(root, plain, thinking, tools))
        follow = encode_chat(
            root,
            [*asked, {"role": "user", "content": "Call read_file with path c.py."}],
            thinking,
            tools,
        )
        assert follow[: len(kept_ids)] == kept_ids


def test_an_image_pad_shifts_the_cut(monkeypatch: pytest.MonkeyPatch) -> None:
    from gemma4_26b_a4b.metal41.geometry import IMAGE_TOKEN
    from gemma4_26b_a4b.portable.chat import expanded_cut

    monkeypatch.setattr("gemma4_26b_a4b.metal41.vision.image_soft_tokens", lambda _blob: 4)
    text = [1, IMAGE_TOKEN, 2, 3]
    assert expanded_cut(text, 2, [b"png"]) == 7
    assert expanded_cut(text, 1, [b"png"]) == 1
    assert expanded_cut(text, None, [b"png"]) is None


def test_prefix_copy_keeps_the_scratch_name() -> None:
    from types import SimpleNamespace

    from favlo.memory import memory
    from gemma4_26b_a4b.metal41.prefix import RuntimeAdapter

    runtime = SimpleNamespace(
        _owner="gemma4-prefix-copy",
        _prefix_scratch="file-agent-evict",
        _prefix_scratch_n=10**9,
    )
    copied = RuntimeAdapter(runtime)._scratch(16)
    try:
        assert copied != "file-agent-evict"
        assert runtime._prefix_scratch == "file-agent-evict"
    finally:
        memory.release(copied)
