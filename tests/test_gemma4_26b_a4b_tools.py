"""Gemma 4 tool-call text and the canonical chat template. No weights."""

from __future__ import annotations

import json

import pytest

from gemma4_26b_a4b.chat import (
    _delta,
    _finish_reason,
    _stop_strings,
    completion_fields,
    cut_stop,
    openai_message,
    render_chat,
    reply_text,
)
from gemma4_26b_a4b.tools import StreamText, parse_tool_calls


def test_gemma_tool_call_quotes() -> None:
    text = (
        "<|channel>thought\nlook\n<channel|>"
        '<|tool_call>call:read_file{file_path:<|"|>ledger.py<|"|>,offset:0}<tool_call|>'
    )
    content, calls = parse_tool_calls(text.split("<channel|>", 1)[1])
    assert content is None
    assert len(calls) == 1
    assert calls[0]["id"] == "call_0"
    assert calls[0]["function"]["name"] == "read_file"
    assert json.loads(calls[0]["function"]["arguments"]) == {
        "file_path": "ledger.py",
        "offset": 0,
    }


def test_gemma_tool_call_json_arguments() -> None:
    text = '<|tool_call>call:glob{"pattern": "**/*.py"}<tool_call|>'
    content, calls = parse_tool_calls(text)
    assert content is None
    assert json.loads(calls[0]["function"]["arguments"]) == {"pattern": "**/*.py"}


def test_gemma_tool_call_keeps_prose() -> None:
    content, calls = parse_tool_calls("done")
    assert content == "done"
    assert calls == []


def test_reply_keeps_the_thought_body() -> None:
    reasoning, answer = reply_text("<|channel>thought\nlook\n<channel|>done")
    assert reasoning == "look"
    assert answer == "done"


def test_reply_keeps_text_before_the_thought() -> None:
    reasoning, answer = reply_text("hello\n<|channel>thought\nlook\n<channel|>done")
    assert reasoning == "look"
    assert answer == "hello\ndone"


def test_open_thought_is_not_the_reply() -> None:
    reasoning, answer = reply_text("<|channel>thought\nlook")
    assert reasoning == "look"
    assert answer == ""
    trace, content, calls = completion_fields("<|channel>thought\nlook")
    assert trace == "look"
    assert content is None
    assert calls == []


def test_text_after_a_tool_call_stays_in_the_reply() -> None:
    text = (
        '<|tool_call>call:glob{pattern:<|"|>**/txt<|"|>}<tool_call|>'
        "<|channel>thought\nlook\n<channel|>\nhere"
    )
    trace, content, calls = completion_fields(text)
    assert trace == "look"
    assert content == "here"
    assert calls[0]["function"]["name"] == "glob"
    assert "<|channel>thought" not in (content or "")
    assert "<tool_call" not in (content or "")


def test_reply_after_a_call_that_ends_the_thought() -> None:
    text = (
        "<|channel>thought\nlook\n"
        "<|tool_call>call:read_file{}<tool_call|>\n"
        "<channel|>\ndone"
    )
    trace, content, calls = completion_fields(text)
    assert trace == "look"
    assert content == "done"
    assert calls[0]["function"]["name"] == "read_file"


def _joined(parts: list[tuple[str, str]], channel: str) -> str:
    return "".join(text for kind, text in parts if kind == channel).strip("\n")


def test_open_channel_streams_reasoning_before_the_reply() -> None:
    gate = StreamText(channel_open=True)
    assert _joined(gate.push("look"), "reasoning") == "look"
    assert _joined(gate.push("\n<channel|>\n"), "reasoning") == ""
    assert _joined(gate.push("done"), "content") == "done"
    assert gate.finish() == []


def test_a_later_thought_streams_after_the_call() -> None:
    gate = StreamText(channel_open=False)
    assert gate.push('<|tool_call>call:glob{pattern:<|"|>**/txt<|"|>}<tool_call|>') == []
    streamed = gate.push("<|channel>thought\nlook\n<channel|>\nhere")
    streamed.extend(gate.finish())
    assert _joined(streamed, "reasoning") == "look"
    assert _joined(streamed, "content") == "here"
    assert all("<tool_call" not in text for _channel, text in streamed)


def test_tool_call_before_the_close_ends_the_thought() -> None:
    text = "<|channel>thought\nlook\n<|tool_call>call:read_file{}<tool_call|>"
    reasoning, answer = reply_text(text)
    assert reasoning == "look"
    assert answer.startswith("<|tool_call>call:read_file")
    trace, content, calls = completion_fields(text)
    assert trace == "look"
    assert content is None
    assert calls[0]["function"]["name"] == "read_file"
    assert "<|channel>thought" not in (content or "")


def test_prompt_opened_thought_stays_reasoning_until_close() -> None:
    reasoning, answer = reply_text("look\n<channel|>done", channel_open=True)
    assert reasoning == "look"
    assert answer == "done"
    reasoning, answer = reply_text("still looking", channel_open=True)
    assert reasoning == "still looking"
    assert answer == ""


def test_thinking_off_closes_the_ghost_channel() -> None:
    text = render_chat([{"role": "user", "content": "Hi"}], False)
    assert text.endswith("<|turn>model\n<|channel>thought\n<channel|>")
    assert "<|think|>" not in text


def test_thinking_on_starts_a_system_turn() -> None:
    text = render_chat([{"role": "user", "content": "Hi"}], True)
    assert "<|turn>system\n<|think|>\n<turn|>\n" in text
    assert text.endswith("<|turn>model\n")


def test_json_arguments_render_as_gemma_calls() -> None:
    messages = [
        {"role": "user", "content": "read"},
        {
            "role": "assistant",
            "reasoning_content": "old thought",
            "tool_calls": [
                {
                    "id": "call_0",
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "arguments": json.dumps({"file_path": "ledger.py", "offset": 0}),
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_0", "content": "ok"},
        {"role": "user", "content": "next"},
    ]
    text = render_chat(messages, True, tools=[_read_tool()])
    assert 'call:read_file{file_path:<|"|>ledger.py<|"|>,offset:0}' in text
    assert '{"file_path"' not in text
    assert "old thought" not in text
    assert text.endswith("<|turn>model\n")


def test_tool_result_reopens_the_thought_channel() -> None:
    messages = [
        {"role": "user", "content": "read"},
        {
            "role": "assistant",
            "reasoning_content": "look",
            "tool_calls": [
                {
                    "id": "call_0",
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "arguments": {"file_path": "ledger.py"},
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_0", "content": "ok"},
    ]
    text = render_chat(messages, True, tools=[_read_tool()])
    assert "<|channel>thought\nlook\n<channel|>" in text
    assert text.endswith("<tool_response|><|channel>thought\n")


def test_plain_answer_thought_is_not_replayed() -> None:
    text = render_chat(
        [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "reasoning_content": "secret", "content": "Hello"},
        ],
        True,
    )
    assert "secret" not in text
    assert "Hello" in text


def test_string_arguments_must_be_json() -> None:
    messages = [
        {"role": "user", "content": "read"},
        {
            "role": "assistant",
            "tool_calls": [
                {"function": {"name": "read_file", "arguments": "ledger.py"}}
            ],
        },
    ]
    with pytest.raises(ValueError):
        render_chat(messages, False)


def test_an_empty_thought_is_not_a_trace() -> None:
    text = '<|channel>thought\n<channel|><|tool_call>call:glob{pattern:"**/*.txt"}<tool_call|>'
    reasoning, content, calls = completion_fields(text)
    assert reasoning == ""
    message = openai_message(reasoning, content, calls)
    assert message["content"] is None
    assert "reasoning" not in message
    assert "reasoning_content" not in message
    assert message["tool_calls"][0]["function"]["name"] == "glob"
    assert _delta("reasoning", "\n") is None


def test_a_thought_and_a_reply_are_openai_fields() -> None:
    text = (
        "<|channel>thought\nLook at the notes.\n<channel|>"
        "The notes are short.\n"
    )
    reasoning, content, calls = completion_fields(text)
    message = openai_message(reasoning, content, calls)
    assert message["reasoning"] == "Look at the notes."
    assert message["reasoning_content"] == "Look at the notes."
    assert message["content"] == "The notes are short.\n"
    assert "tool_calls" not in message
    delta = _delta("reasoning", "\nLook at the notes.\n")
    assert delta is not None
    assert delta["reasoning"] == delta["reasoning_content"]


def test_a_stop_string_is_left_out_of_the_reply() -> None:
    assert cut_stop("alpha STOPWORD omega", ["STOPWORD"]) == "alpha "
    assert cut_stop("alpha", ["STOPWORD"]) == "alpha"
    assert cut_stop("STOPWORD omega", ["STOPWORD"]) == ""
    assert _stop_strings({"stop": "STOPWORD"}) == ["STOPWORD"]
    assert _stop_strings({"stop": ["STOPWORD", ""]}) == ["STOPWORD"]
    assert _stop_strings({}) == []


def test_filling_the_cap_is_length() -> None:
    class Runtime:
        _hit_limit = True

    assert _finish_reason(Runtime(), []) == "length"
    Runtime._hit_limit = False
    assert _finish_reason(Runtime(), []) == "stop"
    assert _finish_reason(Runtime(), [{"id": "call_0"}]) == "tool_calls"


def _read_tool() -> dict[str, object]:
    return {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read",
            "parameters": {
                "type": "object",
                "properties": {"file_path": {"type": "string"}},
                "required": ["file_path"],
            },
        },
    }
