"""Gemma 4 tool-call text and the canonical chat template. No weights."""

from __future__ import annotations

import json

import pytest

from gemma4_26b_a4b.chat import render_chat, reply_text
from gemma4_26b_a4b.tools import parse_tool_calls


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
