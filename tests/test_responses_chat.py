"""Chat Completions is a thin Responses translation."""

from __future__ import annotations

from favlo.reply import MemoryReply
from favlo.responses import ChatWire, ResponseWire, chat_to_responses, response_to_chat, template_messages


def test_wires_expose_started_and_payload() -> None:
    for wire_cls in (ChatWire, ResponseWire):
        inner = MemoryReply(model_id="example-model")
        wire = wire_cls(inner)
        assert wire.started is False
        assert wire.payload == {}
        inner.begin_events()
        assert wire.started is True
        inner.json(200, {"ok": True})
        assert wire.payload == {"ok": True}


def test_chat_reasoning_content_becomes_a_reasoning_item() -> None:
    translated = chat_to_responses(
        {
            "model": "example-model",
            "messages": [
                {"role": "user", "content": "hi"},
                {
                    "role": "assistant",
                    "content": "hello",
                    "reasoning_content": "plan the greeting",
                },
                {"role": "user", "content": "again"},
            ],
            "preserve_thinking": True,
            "stream": True,
        }
    )
    assert translated["preserve_thinking"] is True
    assert translated["stream"] is True
    assert [item.get("type") for item in translated["input"]] == [
        "message",
        "reasoning",
        "message",
        "message",
    ]
    assert translated["input"][1] == {
        "type": "reasoning",
        "content": [{"type": "reasoning_text", "text": "plan the greeting"}],
    }
    messages, saw = template_messages(translated)
    assert saw is True
    assert messages[1]["reasoning_content"] == "plan the greeting"
    assert messages[1]["content"] == "hello"


def test_chat_tool_call_keeps_reasoning_ahead_of_the_call() -> None:
    translated = chat_to_responses(
        {
            "messages": [
                {"role": "user", "content": "read"},
                {
                    "role": "assistant",
                    "content": None,
                    "reasoning_content": "look",
                    "tool_calls": [
                        {
                            "id": "call_0",
                            "type": "function",
                            "function": {
                                "name": "read_file",
                                "arguments": '{"path":"ledger.py"}',
                            },
                        }
                    ],
                },
            ]
        }
    )
    kinds = [item.get("type") for item in translated["input"]]
    assert kinds == ["message", "reasoning", "function_call"]
    messages, saw = template_messages(translated)
    assert saw is True
    assert messages[1]["reasoning_content"] == "look"
    assert messages[1]["tool_calls"][0]["function"]["name"] == "read_file"


def test_response_to_chat_round_trips_reasoning() -> None:
    chat = response_to_chat(
        {
            "id": "resp_1",
            "object": "response",
            "created_at": 1,
            "model": "example-model",
            "output": [
                {
                    "type": "reasoning",
                    "content": [{"type": "reasoning_text", "text": "plan"}],
                },
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "done"}],
                },
            ],
            "usage": {"input_tokens": 3, "output_tokens": 2},
        }
    )
    message = chat["choices"][0]["message"]
    assert message["reasoning_content"] == "plan"
    assert message["content"] == "done"
    again = chat_to_responses(
        {
            "messages": [
                {"role": "user", "content": "go"},
                message,
            ]
        }
    )
    assert again["input"][1]["type"] == "reasoning"
    assert again["input"][1]["content"][0]["text"] == "plan"


def test_chat_effort_fields_map_onto_reasoning() -> None:
    by_effort = chat_to_responses(
        {"messages": [{"role": "user", "content": "x"}], "reasoning_effort": "high"}
    )
    assert by_effort["reasoning"] == {"effort": "high"}
    by_flag = chat_to_responses(
        {"messages": [{"role": "user", "content": "x"}], "enable_thinking": False}
    )
    assert by_flag["reasoning"] == {"effort": "none"}
    by_body = chat_to_responses(
        {
            "messages": [{"role": "user", "content": "x"}],
            "reasoning": {"effort": "medium"},
        }
    )
    assert by_body["reasoning"] == {"effort": "medium"}
