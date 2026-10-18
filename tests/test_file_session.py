"""File-agent ladder: file selection, rate parsing, and prefix extension."""

from pathlib import Path

import pytest
from harness.file_session import (
    echo_reasoning,
    extensions_hold,
    model_stem,
    parse_requests,
    read_reasoning,
    read_repository_file,
    source_files,
)

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


def test_model_stem_drops_a_variant_suffix() -> None:
    assert model_stem("qwen38-27b") == "qwen38-27b"
    assert model_stem("qwen38-27b:thinking") == "qwen38-27b"


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
