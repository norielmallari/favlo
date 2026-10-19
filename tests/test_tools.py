"""Tool-call parsing and the checkpoint template's tool round trip."""

import json

import pytest
from qwen38_flash_next.formats import DEFAULT_RECORD
from qwen38_flash_next.tools import (
    ContentGate,
    ThinkStream,
    normalize_messages,
    parse_tool_calls,
    penalty_ids,
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


def test_penalty_ids_use_the_recent_window() -> None:
    ids = list(range(30))
    assert penalty_ids(ids, 20, once=True) == list(range(10, 30))
    assert penalty_ids([1, 2, 1, 2], 20, once=True) == [1, 2]
    assert penalty_ids([1, 2, 1, 2], 20, once=False) == [1, 2, 1, 2]
    assert penalty_ids(ids, 0, once=True) == ids


def test_split_thinking_keeps_reasoning_when_the_tag_is_missing() -> None:
    reasoning, content = split_thinking("still thinking", True)
    assert reasoning == "still thinking"
    assert content == ""


def test_completion_carries_created() -> None:
    from qwen38_flash_next.commands import _completion

    payload = _completion("hello", 3, 1, "stop")
    assert isinstance(payload["created"], int)
    assert payload["created"] > 0


_AGENT = (
    "You are a repository agent. Call read_file with the path the user names. "
    "Think through the files already returned before that call. "
    "After the file comes back, think through what it does and how it relates "
    "to those files, then reply with that analysis."
)
_CLOSED_CALL = "<tool_call>\n<function=read_file>\n</function>\n</tool_call>"


def _read_prompt(path: str) -> str:
    return f"Call read_file with path {path}."


def _render(
    template: object,
    tokenizer: object,
    messages: list[dict[str, object]],
    tools: list[object],
) -> list[int]:
    text = template.render(  # type: ignore[attr-defined]
        messages=normalize_messages(messages),
        tools=tools,
        add_generation_prompt=True,
        enable_thinking=False,
        raise_exception=RuntimeError,
    )
    return list(tokenizer.encode(text).ids)  # type: ignore[attr-defined]


def test_read_file_turns_extend_the_rendered_prefix() -> None:
    config_path = DEFAULT_RECORD / "tokenizer_config.json"
    tokenizer_path = DEFAULT_RECORD / "tokenizer.json"
    if not config_path.is_file() or not tokenizer_path.is_file():
        pytest.skip("quantized record is not installed")
    from jinja2 import Environment
    from tokenizers import Tokenizer

    template = Environment().from_string(json.loads(config_path.read_text())["chat_template"])
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    tools = [{"type": "function", "function": {"name": "read_file"}}]
    first_path = "qwen38_flash_next/runtime.py"
    second_path = "qwen38_flash_next/session.py"
    first = [
        {"role": "system", "content": _AGENT},
        {"role": "user", "content": _read_prompt(first_path)},
    ]
    call = {
        "id": "call_0",
        "type": "function",
        "function": {"name": "read_file", "arguments": "{}"},
    }
    after_call = [
        *first,
        {"role": "assistant", "content": None, "tool_calls": [call]},
        {"role": "tool", "tool_call_id": "call_0", "content": "def forward():\n    return 1\n"},
    ]
    ids1 = _render(template, tokenizer, first, tools)
    completion = list(tokenizer.encode(_CLOSED_CALL).ids)
    assert len(completion) < 64
    ids2 = _render(template, tokenizer, after_call, tools)
    assert ids2[: len(ids1) + len(completion)] == ids1 + completion
    replied = [
        *after_call,
        {"role": "assistant", "content": "read"},
        {"role": "user", "content": _read_prompt(second_path)},
    ]
    ids_reply = _render(template, tokenizer, after_call, tools)
    reply_ids = list(tokenizer.encode("read").ids)
    ids3 = _render(template, tokenizer, replied, tools)
    assert ids3[: len(ids_reply) + len(reply_ids)] == ids_reply + reply_ids


def test_forced_opening_matches_the_tokenizer() -> None:
    tokenizer_path = DEFAULT_RECORD / "tokenizer.json"
    if not tokenizer_path.is_file():
        pytest.skip("quantized record is not installed")
    from qwen38_flash_next.tools import canonical_continuation, forced_call_options
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    name = "read_file"

    def walk(close: bool) -> list[int]:
        produced: list[int] = []
        while True:
            text = tokenizer.decode(produced)
            options = forced_call_options(text, name, close=close)
            if options is None:
                break
            nxt = canonical_continuation(
                produced,
                options[0],
                lambda piece: tokenizer.encode(piece).ids,
                tokenizer.decode,
            )
            assert nxt is not None
            produced.append(nxt)
            assert len(produced) < 64
        return produced

    opening = f"<tool_call>\n<function={name}>\n"
    opened = walk(False)
    assert tokenizer.decode(opened) == opening
    assert opened == list(tokenizer.encode(opening).ids)
    closed = walk(True)
    assert tokenizer.decode(closed) == _CLOSED_CALL
    assert closed == list(tokenizer.encode(_CLOSED_CALL).ids)
