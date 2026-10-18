"""Long-prompt harness: file selection, rate parsing, and prefix extension."""

import json
from pathlib import Path

import pytest
from harness.long_prompt import (
    INSTRUCTIONS,
    echo_reasoning,
    extensions_hold,
    parse_requests,
    read_prompt,
    read_reasoning,
    read_repository_file,
    source_files,
    tool_call_text,
)
from qwen38_flash_next.formats import DEFAULT_RECORD
from qwen38_flash_next.tools import canonical_continuation, forced_call_options, normalize_messages

_LOG = """\
listening on http://127.0.0.1:9
prompt tokens 100
cache miss
incoming tokens 100
prefill 50.00 tok/s (100 tok in 2.00s)
output tokens 20
decode 10.00 tok/s (20 tok in 2.00s)
prefix load 120 tokens
prompt tokens 180
cache hit 120
incoming tokens 60
prefill 100.00 tok/s (60 tok in 0.60s)
output tokens 4
decode 8.00 tok/s (4 tok in 0.50s)
prompt tokens 300
cache hit 184
incoming tokens 116
prefill 80.00 tok/s (116 tok in 1.45s)
output tokens 4
decode 8.00 tok/s (4 tok in 0.50s)
"""


def test_source_files_skip_models_and_read_largest_first(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("a" * 10)
    (tmp_path / "b.metal").write_text("b" * 30)
    (tmp_path / "empty.py").write_text("")
    (tmp_path / "note.bin").write_text("no")
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "c.py").write_text("c" * 100)
    (tmp_path / ".venv").mkdir()
    (tmp_path / ".venv" / "d.py").write_text("d" * 100)
    found = source_files(tmp_path)
    assert [path.name for path in found] == ["b.metal", "a.py"]
    allowed = {path.relative_to(tmp_path).as_posix() for path in found}
    assert read_repository_file(tmp_path, "b.metal", allowed) == "b" * 30
    assert read_repository_file(tmp_path, "B.METAL", allowed) == "b" * 30
    assert read_repository_file(tmp_path, "missing.py", allowed, "b.metal") == "b" * 30
    with pytest.raises(ValueError):
        read_repository_file(tmp_path, "../secret.txt", allowed)
    with pytest.raises(ValueError):
        read_repository_file(tmp_path, "models/c.py", allowed)


def test_parse_requests_and_prefix_extension() -> None:
    requests = parse_requests(_LOG)
    assert len(requests) == 3
    assert requests[0].missed
    assert requests[0].prefix_hit is None
    assert requests[0].prefix_load is None
    assert requests[0].decode_tokens == 20
    assert requests[0].prefill.startswith("prefill ")
    assert requests[0].steady == ()
    assert requests[1].prefix_load == 120
    assert requests[1].prefix_hit == 120
    assert requests[1].prompt_tokens == 180
    assert requests[2].prefix_load is None
    assert requests[2].prefix_hit == 184
    assert extensions_hold(requests)
    assert not extensions_hold(requests[:1])
    missed = parse_requests(_LOG.replace("cache hit 120", "cache miss"))
    assert not extensions_hold(missed)
    short = parse_requests(_LOG.replace("cache hit 120", "cache hit 100"))
    assert not extensions_hold(short)
    memory = parse_requests(_LOG.replace("prefix load 120 tokens\n", ""))
    assert not extensions_hold(memory)
    later = _LOG.replace(
        "prompt tokens 300\ncache hit 184",
        "prefix load 184 tokens\nprompt tokens 300\ncache hit 184",
    )
    assert not extensions_hold(parse_requests(later))
    after_prompt = """\
prompt tokens 10
cache miss
prefill 1.00 tok/s (10 tok in 10.00s)
decode 1.00 tok/s (2 tok in 2.00s)
prompt tokens 20
prefix load 12 tokens
cache hit 12
prefill 1.00 tok/s (8 tok in 8.00s)
decode 1.00 tok/s (1 tok in 1.00s)
"""
    loaded = parse_requests(after_prompt)
    assert loaded[1].prefix_load == 12
    assert extensions_hold(loaded)
    followed = """\
prompt tokens 10
cache miss
prefill 1.00 tok/s (10 tok in 10.00s)
decode 1.00 tok/s (2 tok in 2.00s)
prompt tokens 20
cache hit 12
prefill 1.00 tok/s (8 tok in 8.00s)
decode 1.00 tok/s (4 tok in 4.00s)
prefix load 24 tokens
prompt tokens 40
cache hit 24
prefill 1.00 tok/s (16 tok in 16.00s)
decode 1.00 tok/s (3 tok in 3.00s)
"""
    agent = parse_requests(followed)
    assert agent[1].prefix_load is None
    assert agent[1].prefix_hit == 12
    assert agent[2].prefix_load == 24
    assert extensions_hold(agent)
    warm = """\
prefix load 50 tokens
prompt tokens 50
cache hit 50
prefill 1.00 tok/s (0 tok in 0.00s)
decode 1.00 tok/s (2 tok in 2.00s)
prompt tokens 80
cache hit 52
prefill 1.00 tok/s (28 tok in 2.00s)
decode 1.00 tok/s (4 tok in 4.00s)
prefix load 84 tokens
prompt tokens 100
cache hit 84
prefill 1.00 tok/s (16 tok in 2.00s)
decode 1.00 tok/s (1 tok in 1.00s)
"""
    assert extensions_hold(parse_requests(warm))


def _encode(
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
        {"role": "system", "content": INSTRUCTIONS},
        {"role": "user", "content": read_prompt(first_path)},
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
    ids1 = _encode(template, tokenizer, first, tools)
    completion = list(tokenizer.encode(tool_call_text()).ids)
    assert len(completion) < 64
    ids2 = _encode(template, tokenizer, after_call, tools)
    assert ids2[: len(ids1) + len(completion)] == ids1 + completion
    replied = [
        *after_call,
        {"role": "assistant", "content": "read"},
        {"role": "user", "content": read_prompt(second_path)},
    ]
    ids_reply = _encode(template, tokenizer, after_call, tools)
    reply_ids = list(tokenizer.encode("read").ids)
    ids3 = _encode(template, tokenizer, replied, tools)
    assert ids3[: len(ids_reply) + len(reply_ids)] == ids_reply + reply_ids


def test_forced_opening_matches_the_tokenizer() -> None:
    tokenizer_path = DEFAULT_RECORD / "tokenizer.json"
    if not tokenizer_path.is_file():
        pytest.skip("quantized record is not installed")
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
    assert tokenizer.decode(closed) == tool_call_text()
    assert closed == list(tokenizer.encode(tool_call_text()).ids)


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
            _Part("text", "<tool_call>\n<function=read_file>\n"),
        ]
    )
    built = [
        {"role": "assistant", "content": [{"type": "text", "text": "look\n"}]},
        {
            "role": "assistant",
            "content": "<tool_call>\n<function=read_file>\n",
            "tool_calls": [{"id": "call_0"}],
        },
    ]
    echoed = echo_reasoning(message, built)
    assert echoed == [
        {
            "role": "assistant",
            "content": "<tool_call>\n<function=read_file>\n",
            "reasoning_content": "look\n",
        }
    ]


def test_reasoning_delta_is_not_reply_text() -> None:
    class _Delta:
        model_extra = {"reasoning_content": "look"}

    parsed = read_reasoning(_Delta(), [])
    assert parsed[0].type == "text_reasoning"
    assert parsed[0].text == "look"
