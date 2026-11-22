"""Gemma 4 tool-call text and the canonical chat template. No weights."""

from __future__ import annotations

import json

import pytest

from favlo.responses import chat_to_responses, template_messages
from gemma4_26b_a4b.portable.chat import (
    _stop_strings,
    advance_thought,
    close_model_turn,
    completion_fields,
    cut_stop,
    encode_chat,
    encode_text,
    render_chat,
    reply_text,
)
from gemma4_26b_a4b.portable.tools import StreamText, parse_tool_calls


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
    assert gate.push("look") == [("reasoning", "look")]
    assert gate.push(" here") == [("reasoning", " here")]
    assert _joined(gate.push("\n<channel|>\n"), "reasoning") == ""
    assert _joined(gate.push("done"), "content") == "done"
    assert gate.finish() == []


def test_a_repeated_open_is_not_the_trace() -> None:
    text = (
        "t)<|channel>thought\nFix the return.\n<channel|>"
        "<|tool_call>call:edit{}<tool_call|>"
    )
    reasoning, answer = reply_text(text, channel_open=True)
    assert reasoning == "Fix the return."
    assert "<|channel>" not in reasoning
    assert answer.startswith("<|tool_call>call:edit")
    gate = StreamText(channel_open=True)
    parts = gate.push("t)<|chan")
    parts.extend(gate.push("nel>thought\nFix the return.\n<channel|>"))
    assert _joined(parts, "reasoning") == "Fix the return."
    assert "<|channel>" not in _joined(parts, "reasoning")


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


def test_a_spent_thought_budget_closes_the_channel() -> None:
    open_ids = [10, 11]
    close_ids = [20]
    drawn = [10, 11]
    inside, reasoned, inject = advance_thought(
        drawn, inside=False, reasoned=0, budget=3, open_ids=open_ids, close_ids=close_ids
    )
    assert (inside, reasoned, inject) == (True, 0, [])
    for token, count in ((1, 1), (2, 2)):
        drawn.append(token)
        inside, reasoned, inject = advance_thought(
            drawn,
            inside=inside,
            reasoned=reasoned,
            budget=3,
            open_ids=open_ids,
            close_ids=close_ids,
        )
        assert (inside, reasoned, inject) == (True, count, [])
    drawn.append(3)
    inside, reasoned, inject = advance_thought(
        drawn, inside=inside, reasoned=reasoned, budget=3, open_ids=open_ids, close_ids=close_ids
    )
    assert (inside, reasoned, inject) == (False, 0, [20])
    drawn.extend(inject)
    drawn.append(4)
    inside, reasoned, inject = advance_thought(
        drawn, inside=inside, reasoned=reasoned, budget=3, open_ids=open_ids, close_ids=close_ids
    )
    assert (inside, reasoned, inject) == (False, 0, [])
    drawn = [7, 20]
    inside, reasoned, inject = advance_thought(
        drawn, inside=True, reasoned=2, budget=3, open_ids=open_ids, close_ids=close_ids
    )
    assert (inside, reasoned, inject) == (False, 0, [])


def test_chat_template_compiles_once(monkeypatch: pytest.MonkeyPatch) -> None:
    from gemma4_26b_a4b.portable import chat

    calls = {"n": 0}
    original = chat.Environment.from_string

    def counting(self: object, source: str, *args: object, **kwargs: object) -> object:
        calls["n"] += 1
        return original(self, source, *args, **kwargs)

    monkeypatch.setattr(chat.Environment, "from_string", counting)
    chat._compiled = None
    render_chat([{"role": "user", "content": "Hi"}], False)
    render_chat([{"role": "user", "content": "Hi"}], True)
    assert calls["n"] == 1


def test_tokenizer_loads_once(monkeypatch: pytest.MonkeyPatch, tmp_path: object) -> None:
    from pathlib import Path
    from types import SimpleNamespace

    from gemma4_26b_a4b.portable import chat

    root = Path(str(tmp_path))
    loads = {"n": 0}

    class _Tokenizer:
        def encode(self, text: str, add_special_tokens: bool = True) -> object:
            del text, add_special_tokens
            return SimpleNamespace(ids=[1])

    def from_file(path: str) -> _Tokenizer:
        del path
        loads["n"] += 1
        return _Tokenizer()

    monkeypatch.setattr(chat.Tokenizer, "from_file", from_file)
    chat._tokenizers.pop(root, None)
    chat._GENERATION_SUFFIX.pop(root, None)
    encode_text(root, "a")
    encode_text(root, "b")
    chat._generation_suffix(root)
    chat._generation_suffix(root)
    assert loads["n"] == 1


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
    assert "old thought" in text
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


def test_empty_tool_thought_stays_in_the_next_prompt() -> None:
    """An omitted whitespace thought still matches the tokens already on device."""
    call = {
        "id": "call_0",
        "type": "function",
        "function": {
            "name": "read_file",
            "arguments": {"file_path": "ledger.py"},
        },
    }
    messages = [
        {"role": "user", "content": "read"},
        {"role": "assistant", "tool_calls": [call]},
        {"role": "tool", "tool_call_id": "call_0", "content": "ok"},
    ]
    text = render_chat(messages, True, tools=[_read_tool()])
    assert "<|turn>model\n<|channel>thought\n<channel|><|tool_call>" in text
    assert text.count("<|channel>thought\n<channel|>") == 1
    assert text.endswith("<tool_response|><|channel>thought\n")


def test_empty_tool_thought_replays_after_a_tool_result() -> None:
    first = {
        "id": "call_0",
        "type": "function",
        "function": {
            "name": "read_file",
            "arguments": {"file_path": "ledger.py"},
        },
    }
    second = {
        "id": "call_1",
        "type": "function",
        "function": {
            "name": "read_file",
            "arguments": {"file_path": "notes.txt"},
        },
    }
    messages = [
        {"role": "user", "content": "read"},
        {"role": "assistant", "tool_calls": [first]},
        {"role": "tool", "tool_call_id": "call_0", "content": "ok"},
        {"role": "assistant", "tool_calls": [second]},
        {"role": "tool", "tool_call_id": "call_1", "content": "ok"},
    ]
    text = render_chat(messages, True, tools=[_read_tool()])
    assert (
        "<tool_response|><|channel>thought\n<channel|><|tool_call>call:read_file"
        in text
    )
    assert text.count("<|channel>thought\n<channel|>") == 2


def test_a_tool_thought_stays_on_the_next_user() -> None:
    call = {
        "id": "call_0",
        "type": "function",
        "function": {"name": "read_file", "arguments": '{"file_path":"ledger.py"}'},
    }
    messages = [
        {"role": "user", "content": "read"},
        {
            "role": "assistant",
            "content": None,
            "reasoning_content": "look",
            "tool_calls": [call],
        },
        {"role": "tool", "tool_call_id": "call_0", "content": "ok"},
    ]
    text = _rendered_chat(messages)
    assert "<|channel>thought\nlook\n<channel|>" in text
    assert text.endswith("<tool_response|><|channel>thought\n")
    messages.append({"role": "user", "content": "next"})
    assert "<|channel>thought\nlook\n<channel|>" in _rendered_chat(messages)


def test_a_plain_chat_thought_stays_out_of_the_prompt() -> None:
    messages = [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello", "reasoning_content": "secret"},
        {"role": "user", "content": "next"},
    ]
    assert "secret" not in _rendered_chat(messages)


def _rendered_chat(messages: list[dict[str, object]]) -> str:
    body = {"messages": messages, "tools": [_read_tool()]}
    translated = chat_to_responses(body)
    prepared, _seen = template_messages(translated)
    return render_chat(prepared, True, tools=[_read_tool()])


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
    assert content is None or content == ""
    assert calls[0]["function"]["name"] == "glob"


def test_a_thought_and_a_reply_are_completion_fields() -> None:
    text = (
        "<|channel>thought\nLook at the notes.\n<channel|>"
        "The notes are short.\n"
    )
    reasoning, content, calls = completion_fields(text)
    assert reasoning == "Look at the notes."
    assert content == "The notes are short.\n"
    assert calls == []


def test_a_stop_string_is_left_out_of_the_reply() -> None:
    assert cut_stop("alpha STOPWORD omega", ["STOPWORD"]) == "alpha "
    assert cut_stop("alpha", ["STOPWORD"]) == "alpha"
    assert cut_stop("STOPWORD omega", ["STOPWORD"]) == ""
    assert _stop_strings({"stop": "STOPWORD"}) == ["STOPWORD"]
    assert _stop_strings({"stop": ["STOPWORD", ""]}) == ["STOPWORD"]
    assert _stop_strings({}) == []


def test_unsupported_effort_steps_up_or_falls_back() -> None:
    from gemma4_26b_a4b.portable.sampling import resolve_effort

    assert resolve_effort(None) is None
    assert resolve_effort("minimal") == "minimal"
    assert resolve_effort("high") == "high"
    assert resolve_effort("none") == "minimal"
    assert resolve_effort("low") == "high"
    assert resolve_effort("medium") == "high"
    assert resolve_effort("xhigh") == "high"
    assert resolve_effort("max") == "high"


def test_effort_opens_thinking_without_changing_the_card() -> None:
    from types import SimpleNamespace

    from gemma4_26b_a4b.portable.chat import _request_effort
    from gemma4_26b_a4b.portable.sampling import VARIANTS, mode_card

    instruct = SimpleNamespace(
        variant="instruct",
        sampling={"temperature": 0.95, "top_p": 0.95, "top_k": 64, "presence_penalty": 0.0},
        variants=VARIANTS,
        default_variant="thinking",
        enable_thinking=None,
        reasoning_effort="minimal",
    )
    thinking, card = mode_card(instruct, _request_effort({"reasoning": {"effort": "low"}}))
    assert thinking is True
    assert card["temperature"] == 0.95
    assert "reasoning_effort" not in card
    closed, still = mode_card(instruct, _request_effort({"reasoning": {"effort": "none"}}))
    assert closed is False
    assert still["temperature"] == 0.95
    minimal, same = mode_card(instruct, _request_effort({"reasoning": {"effort": "minimal"}}))
    assert minimal is False
    assert same["temperature"] == 0.95
    omitted, _card = mode_card(instruct, _request_effort({}))
    assert omitted is False
    bare = SimpleNamespace(
        variant=None,
        sampling=None,
        variants=VARIANTS,
        default_variant="thinking",
        enable_thinking=None,
        reasoning_effort="high",
    )
    open_by_default, thinking_card = mode_card(bare, _request_effort({}))
    assert open_by_default is True
    assert thinking_card["temperature"] == 0.95
    assert "reasoning_effort" not in thinking_card


def test_a_required_thinking_call_shares_the_empty_thought() -> None:
    from pathlib import Path

    root = Path("models/gemma-4-26B-A4B-it-fp8-e20")
    if not (root / "tokenizer.json").is_file():
        pytest.skip("tokenizer is not installed")
    tools = [_read_tool()]
    user = [{"role": "user", "content": "Read console.py"}]
    first = encode_chat(root, user, True, tools)
    model_turn = encode_text(root, "<|turn>model\n")
    empty = encode_text(root, "<|channel>thought\n<channel|>")
    drawn = close_model_turn(first, model_turn, empty)
    assert drawn[-len(empty) :] == empty
    lead = encode_text(root, "<|tool_call>call:read_file{")
    opening = [*drawn, *lead]
    call = {
        "id": "c1",
        "type": "function",
        "function": {"name": "read_file", "arguments": {"file_path": "console.py"}},
    }
    follow = encode_chat(
        root,
        [
            *user,
            {"role": "assistant", "content": "", "tool_calls": [call]},
            {"role": "tool", "tool_call_id": "c1", "name": "read_file", "content": "print(1)\n"},
        ],
        True,
        tools,
    )
    assert follow[: len(opening)] == opening


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
