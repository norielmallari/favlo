"""Tool-call parsing and the checkpoint template's tool round trip."""

import json

import pytest
from qwen38_flash_next.formats import DEFAULT_RECORD
from qwen38_flash_next.tools import (
    ContentGate,
    ThinkStream,
    normalize_messages,
    parse_tool_calls,
    repeated_cycle,
    split_thinking,
)


def test_parse_tool_call_keeps_multiline_arguments() -> None:
    text = (
        "Creating the note.\n\n"
        "<tool_call>\n"
        "<function=create_file>\n"
        "<parameter=path>\n"
        "note.txt\n"
        "</parameter>\n"
        "<parameter=content>\n"
        "The harbor log is open.\n"
        "Second line\n"
        "</parameter>\n"
        "</function>\n"
        "</tool_call>"
    )
    content, calls = parse_tool_calls(text)
    assert content == "Creating the note."
    assert calls[0]["id"] == "call_0"
    assert calls[0]["function"]["name"] == "create_file"
    arguments = json.loads(calls[0]["function"]["arguments"])
    assert arguments == {"path": "note.txt", "content": "The harbor log is open.\nSecond line"}


def test_parse_two_tool_calls() -> None:
    text = (
        "<tool_call>\n<function=create_file>\n<parameter=path>\na.txt\n</parameter>\n"
        "</function>\n</tool_call>\n"
        "<tool_call>\n<function=confirm_file>\n<parameter=path>\na.txt\n</parameter>\n"
        "</function>\n</tool_call>"
    )
    content, calls = parse_tool_calls(text)
    assert content is None
    assert [call["function"]["name"] for call in calls] == ["create_file", "confirm_file"]
    assert calls[1]["id"] == "call_1"


def test_plain_text_is_not_a_tool_call() -> None:
    content, calls = parse_tool_calls("The harbor log is closed.")
    assert content == "The harbor log is closed."
    assert calls == []


def test_content_gate_holds_a_partial_marker() -> None:
    gate = ContentGate()
    assert gate.push("Hello <tool") == "Hello "
    assert gate.push("_call>\n<function=create_file>") == ""
    assert gate.finish() == ""


def test_content_gate_emits_text_without_a_call() -> None:
    gate = ContentGate()
    assert gate.push("Hello") == "Hello"
    assert gate.finish() == ""


def test_normalize_parses_argument_json() -> None:
    messages = normalize_messages(
        [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_0",
                        "type": "function",
                        "function": {
                            "name": "create_file",
                            "arguments": '{"path": "note.txt", "content": "hi"}',
                        },
                    }
                ],
            }
        ]
    )
    function = messages[0]["tool_calls"][0]["function"]
    assert messages[0]["content"] == ""
    assert function["arguments"] == {"path": "note.txt", "content": "hi"}


def test_template_renders_a_tool_round_trip() -> None:
    config_path = DEFAULT_RECORD / "tokenizer_config.json"
    if not config_path.is_file():
        pytest.skip("quantized record is not installed")
    from jinja2 import Environment

    template = json.loads(config_path.read_text())["chat_template"]
    messages = normalize_messages(
        [
            {"role": "user", "content": "Create the note."},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_0",
                        "type": "function",
                        "function": {
                            "name": "create_file",
                            "arguments": json.dumps(
                                {"path": "note.txt", "content": "The harbor log is open."}
                            ),
                        },
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_0", "content": "created note.txt"},
        ]
    )
    rendered = Environment().from_string(template).render(
        messages=messages,
        tools=[{"type": "function", "function": {"name": "create_file"}}],
        add_generation_prompt=True,
        enable_thinking=False,
        raise_exception=RuntimeError,
    )
    assert "<function=create_file>" in rendered
    assert "<parameter=path>" in rendered
    assert "note.txt" in rendered
    assert "<tool_response>" in rendered
    assert "created note.txt" in rendered


def test_function_name_stays_inside_the_request() -> None:
    from qwen38_flash_next.tools import continuing_ids, function_name_options, tool_names

    names = tool_names(
        [
            {"type": "function", "function": {"name": "create_file"}},
            {"type": "function", "function": {"name": "edit_file"}},
            {"type": "function", "function": {"name": "confirm_file"}},
        ]
    )
    assert names == ["create_file", "edit_file", "confirm_file"]
    assert function_name_options("hello", names) is None
    assert function_name_options("<tool_call>\n<function=", names) == [
        "create_file>",
        "edit_file>",
        "confirm_file>",
    ]
    assert function_name_options("<tool_call>\n<function=cre", names) == ["ate_file>"]
    closed = "<tool_call>\n<function=create_file>\n</function>\n</tool_call>"
    assert function_name_options(closed, names) is None
    pieces = ["", "make", "create", "edit", "confirm", ">", "ate_file>"]
    assert continuing_ids(["create_file>", "edit_file>", "confirm_file>"], pieces) == [2, 3, 4]
    assert continuing_ids(["ate_file>"], pieces) == [6]


def test_required_function_forces_the_opening() -> None:
    from qwen38_flash_next.tools import forced_call_options, required_function

    assert required_function("auto") is None
    assert (
        required_function({"type": "function", "function": {"name": "edit_file"}}) == "edit_file"
    )
    assert forced_call_options("", "edit_file") == ["<tool_call>\n<function=edit_file>\n"]
    assert forced_call_options("", "read_file", close=True) == [
        "<tool_call>\n<function=read_file>\n</function>\n</tool_call>"
    ]
    from qwen38_flash_next.tools import call_closes

    tools = [
        {"type": "function", "function": {"name": "read_file", "parameters": {"properties": {}}}},
        {
            "type": "function",
            "function": {
                "name": "edit_file",
                "parameters": {"properties": {"path": {"type": "string"}}},
            },
        },
    ]
    assert call_closes(tools, "read_file")
    assert not call_closes(tools, "edit_file")
    assert forced_call_options("<tool_call>\n<function=edit_file>\n", "edit_file") is None
    assert forced_call_options("Done", "edit_file") is None


def test_canonical_continuation_follows_the_encoder() -> None:
    from qwen38_flash_next.tools import canonical_continuation

    def encode(text: str) -> list[int]:
        return [ord(char) for char in text]

    def decode(ids: list[int]) -> str:
        return "".join(chr(item) for item in ids)

    target = "<tool_call>\n<function=read_file>\n"
    produced: list[int] = []
    while True:
        rest = target[len(decode(produced)) :]
        if not rest:
            break
        nxt = canonical_continuation(produced, rest, encode, decode)
        assert nxt is not None
        produced.append(nxt)
    assert produced == encode(target)


def test_think_stream_holds_a_partial_close_tag() -> None:
    stream = ThinkStream(True)
    assert stream.push("The ninth buoy is bl") == [("reasoning", "The ninth buoy is bl")]
    assert stream.push("ue.</thi") == [("reasoning", "ue.")]
    assert stream.push("nk>\nMara Ellison Voss.") == [("content", "\nMara Ellison Voss.")]
    assert stream.finish() == []


def test_think_stream_closed_request_is_content() -> None:
    stream = ThinkStream(False)
    assert stream.push("Mara Ellison Voss") == [("content", "Mara Ellison Voss")]
    assert stream.finish() == []


def test_repeated_cycle_is_the_next_token_of_the_span() -> None:
    cycle = [4, 5, 6, 7, 8, 9]
    assert repeated_cycle(cycle * 3) == 4
    assert repeated_cycle(cycle * 2 + [4]) is None
    assert repeated_cycle([1, 1, 1]) == 1
    assert repeated_cycle([1, 1]) is None
    assert repeated_cycle([7, 8, 7, 8, 7, 8]) == 7


def test_stuck_ids_are_the_small_recent_set() -> None:
    from qwen38_flash_next.tools import stuck_ids

    assert set(stuck_ids([1, 2, 3] * 8)) == {1, 2, 3}
    assert stuck_ids(list(range(24))) == []


def test_tail_repeats_when_the_ending_already_appeared() -> None:
    from qwen38_flash_next.tools import tail_repeats

    phrase = "The question is from Mara Voss and the buoy stays blue. "
    assert tail_repeats(phrase * 4)
    assert not tail_repeats("The ninth buoy is blue and the name is Mara Ellison Voss.")


def test_split_thinking_keeps_reasoning_when_the_tag_is_missing() -> None:
    reasoning, content = split_thinking("still thinking", True)
    assert reasoning == "still thinking"
    assert content == ""


def test_completion_carries_created() -> None:
    from qwen38_flash_next.commands import _completion

    payload = _completion("hello", 3, 1, "stop")
    assert isinstance(payload["created"], int)
    assert payload["created"] > 0
