"""The coding harness grades a workspace. These tests do not start serve."""

from __future__ import annotations

import asyncio
import json
from io import StringIO
from pathlib import Path

import pytest
from agent_framework import ChatResponseUpdate, Content
from harness.benchmarks.serve import Request, apply_sampling, card_model_id
from harness.coding.__main__ import argument_parser
from harness.coding.agent_tools import coding_tools
from harness.coding.grade import grade
from harness.coding.loop import agent_options, prefix_problems
from harness.turn_printer import print_updates
from harness.file_session import parse_requests
from harness.coding.session import REQUIRED_TOOLS, session_for
from harness.coding.workspace import PROMPT, dispatch, locate, prepare_workspace

_MODULE = "def collect(before, after):\n    return after - before\n"


def test_thinking_card_id() -> None:
    assert card_model_id("qwen38-27b", thinking=True) == "qwen38-27b:thinking"


def test_prefix_problems_require_a_disk_load() -> None:
    memory = """\
prompt tokens 10
cache miss
prefill 1.00 tok/s (10 tok in 10.00s)
decode 1.00 tok/s (2 tok in 2.00s)
prompt tokens 20
cache hit 12
prefill 1.00 tok/s (8 tok in 8.00s)
decode 1.00 tok/s (1 tok in 1.00s)
"""
    assert prefix_problems(parse_requests(memory)) == ["prefix was not loaded from disk"]
    loaded = """\
prompt tokens 10
cache miss
prefill 1.00 tok/s (10 tok in 10.00s)
decode 1.00 tok/s (2 tok in 2.00s)
prefix load 12 tokens
prompt tokens 20
cache hit 12
prefill 1.00 tok/s (8 tok in 8.00s)
decode 1.00 tok/s (1 tok in 1.00s)
prompt tokens 40
cache hit 21
prefill 1.00 tok/s (19 tok in 19.00s)
decode 1.00 tok/s (1 tok in 1.00s)
"""
    assert prefix_problems(parse_requests(loaded)) == []
    assert prefix_problems([]) == ["the ledger made fewer than two requests"]
    boundary = """\
prompt tokens 10
cache miss
prefill 1.00 tok/s (10 tok in 10.00s)
decode 1.00 tok/s (2 tok in 2.00s)
prefix load 12 tokens
prompt tokens 20
cache hit 12
prefill 1.00 tok/s (8 tok in 8.00s)
decode 1.00 tok/s (4 tok in 4.00s)
prompt tokens 30
cache hit 24
prefill 1.00 tok/s (6 tok in 6.00s)
decode 1.00 tok/s (3 tok in 3.00s)
prefix load 30 tokens
prompt tokens 50
cache hit 30
prefill 1.00 tok/s (20 tok in 20.00s)
decode 1.00 tok/s (2 tok in 2.00s)
"""
    assert prefix_problems(parse_requests(boundary)) == []
    drifted = """\
prompt tokens 10
cache miss
prefill 1.00 tok/s (10 tok in 10.00s)
decode 1.00 tok/s (2 tok in 2.00s)
prefix load 12 tokens
prompt tokens 20
cache hit 12
prefill 1.00 tok/s (8 tok in 8.00s)
decode 1.00 tok/s (4 tok in 4.00s)
prefix load 99 tokens
prompt tokens 50
cache hit 99
prefill 1.00 tok/s (0 tok in 0.00s)
decode 1.00 tok/s (1 tok in 1.00s)
"""
    assert prefix_problems(parse_requests(drifted)) == ["prefix did not extend"]


def test_thinking_body_omits_sampling_fields() -> None:
    request = apply_sampling(
        Request(enable_thinking=True),
        {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "presence_penalty": 0.0},
    )
    options = agent_options(request, mtp=True, mtp_depth=3)
    extra = options["extra_body"]
    assert isinstance(extra, dict)
    assert options["reasoning"] == {"effort": "medium"}
    assert extra["favlo:mtp"] is True
    assert extra["favlo:mtp_depth"] == 3
    assert options["store"] is False
    assert "temperature" not in options
    assert "temperature" not in extra
    assert "top_p" not in extra
    assert "top_k" not in extra
    assert "presence_penalty" not in extra
    assert "max_tokens" not in extra
    off = agent_options(request, mtp=False, mtp_depth=3)
    assert off["extra_body"]["favlo:mtp"] is False
    four = agent_options(request, mtp=True, mtp_depth=4)
    assert four["extra_body"]["favlo:mtp_depth"] == 4


def test_mtp_defaults_off() -> None:
    args = argument_parser().parse_args(["--model", "qwen36-35b-a3b"])
    assert args.mtp == "off"
    assert args.mtp_depth == 3
    assert args.vision is False
    assert args.thinking == "on"
    instruct = argument_parser().parse_args(["--model", "qwen36-35b-a3b", "--thinking", "off"])
    assert instruct.thinking == "off"


def _update(*contents: Content, finish_reason: str | None = None) -> ChatResponseUpdate:
    return ChatResponseUpdate(
        contents=list(contents),
        role="assistant",
        response_id="request-1",
        finish_reason=finish_reason,
    )


def test_a_stream_prints_reasoning_then_the_reply() -> None:
    updates = [
        _update(Content.from_text_reasoning(text="Need ")),
        _update(Content.from_text_reasoning(text="the file")),
        _update(Content.from_text(text="ok")),
        _update(
            Content.from_function_call(call_id="call_0", name="read_file", arguments='{"')
        ),
        _update(Content.from_function_call(call_id="call_0", name="", arguments='a": 1}')),
        _update(finish_reason="tool_calls"),
    ]
    sink = StringIO()
    print_updates(updates, sink)
    text = sink.getvalue()
    assert text.index("--- request 1 reasoning ---") < text.index("Need the file")
    assert text.index("Need the file") < text.index("--- request 1 reply ---")
    assert text.index("--- request 1 reply ---") < text.index("ok")
    assert text.index("ok") < text.index("--- request 1 tool_calls ---")
    assert 'read_file {"a": 1}' in text.split("--- request 1 tool_calls ---", 1)[1]


def test_a_later_turn_prints_when_the_response_id_stays() -> None:
    updates = [
        _update(Content.from_function_call(call_id="c0", name="glob", arguments="{}")),
        _update(finish_reason="tool_calls"),
        _update(Content.from_text(text="done")),
        _update(finish_reason="stop"),
    ]
    sink = StringIO()
    print_updates(updates, sink)
    text = sink.getvalue()
    assert "--- request 2 reply ---" in text
    assert text.index("glob") < text.index("--- request 2 reply ---")
    assert text.index("--- request 2 reply ---") < text.index("done")


def test_reasoning_without_a_reply_keeps_an_empty_reply_section() -> None:
    updates = [
        _update(Content.from_text_reasoning(text="draft reply stays here")),
        _update(finish_reason="stop"),
    ]
    sink = StringIO()
    print_updates(updates, sink)
    text = sink.getvalue()
    assert "tool_calls" not in text
    assert text.index("draft reply stays here") < text.index("--- request 1 reply ---")


def test_tool_schemas_match_the_contract(tmp_path: Path) -> None:
    prepare_workspace(tmp_path)
    session = session_for(tmp_path)
    try:
        tools = coding_tools(session)
        assert [tool.name for tool in tools] == list(REQUIRED_TOOLS)
        glob = tools[1]
        result = asyncio.run(
            glob.invoke(arguments={"pattern": "**/*.txt", "path": "notes"}, skip_parsing=True)
        )
    finally:
        session.close()
    body = json.loads(result)
    assert "notes/brief.txt" in body["paths"]


def test_grep_accepts_a_text_boolean(tmp_path: Path) -> None:
    prepare_workspace(tmp_path)
    session = session_for(tmp_path)
    try:
        grep = next(tool for tool in coding_tools(session) if tool.name == "grep_search")
        result = asyncio.run(
            grep.invoke(
                arguments={
                    "pattern": "collect",
                    "path": "notes/brief.txt",
                    "literal": "true",
                },
                skip_parsing=True,
            )
        )
    finally:
        session.close()
    body = json.loads(result)
    assert body["match_count"] >= 1


def test_path_stays_inside_the_workspace(tmp_path: Path) -> None:
    locate(tmp_path, "ledger.py")
    with pytest.raises(ValueError):
        locate(tmp_path, "../ledger.py")
    outside = tmp_path.parent / "outside.txt"
    outside.write_text("nope", encoding="utf-8")
    (tmp_path / "escape").symlink_to(outside)
    with pytest.raises(ValueError):
        locate(tmp_path, "escape")


def test_prompt_does_not_name_a_tool() -> None:
    for name in REQUIRED_TOOLS:
        assert name not in PROMPT
    assert "notes/brief.txt" in PROMPT
    assert "def collect(before, after):" in PROMPT
    assert "return after - before" in PROMPT
    assert 'python3 -c "import ledger; print(ledger.collect(15, 21))"' in PROMPT


def test_an_omitted_idempotency_key_still_writes(tmp_path: Path) -> None:
    prepare_workspace(tmp_path)
    session = session_for(tmp_path)
    arguments = {"file_path": "ledger.py", "content": _MODULE}
    try:
        write = next(tool for tool in coding_tools(session) if tool.name == "write_file")
        first = json.loads(asyncio.run(write.invoke(arguments=arguments, skip_parsing=True)))
        second = json.loads(asyncio.run(write.invoke(arguments=dict(arguments), skip_parsing=True)))
        shell = next(tool for tool in coding_tools(session) if tool.name == "run_shell_command")
        started = json.loads(
            asyncio.run(
                shell.invoke(
                    arguments={"command": "true"},
                    skip_parsing=True,
                )
            )
        )
    finally:
        session.close()
    assert first["file_path"] == "ledger.py"
    assert second == first
    assert (tmp_path / "ledger.py").read_text(encoding="utf-8") == _MODULE
    assert started["status"] == "queued"
    assert started["job_id"]


def test_a_supplied_idempotency_key_is_kept(tmp_path: Path) -> None:
    prepare_workspace(tmp_path)
    session = session_for(tmp_path)
    try:
        write = next(tool for tool in coding_tools(session) if tool.name == "write_file")
        asyncio.run(
            write.invoke(
                arguments={
                    "file_path": "ledger.py",
                    "content": _MODULE,
                    "idempotency_key": "create-ledger",
                },
                skip_parsing=True,
            )
        )
        again = json.loads(
            asyncio.run(
                write.invoke(
                    arguments={
                        "file_path": "ledger.py",
                        "content": _MODULE,
                        "idempotency_key": "other-attempt",
                    },
                    skip_parsing=True,
                )
            )
        )
    finally:
        session.close()
    assert again["error"] == "conflict"


def test_a_missing_required_field_names_that_field(tmp_path: Path) -> None:
    prepare_workspace(tmp_path)
    session = session_for(tmp_path)
    try:
        write = next(tool for tool in coding_tools(session) if tool.name == "write_file")
        with pytest.raises(TypeError, match="file_path"):
            asyncio.run(write.invoke(arguments={"content": "x"}, skip_parsing=True))
    finally:
        session.close()


def test_hidden_suite_passes_a_correct_module(tmp_path: Path) -> None:
    body = json.loads(
        dispatch(
            tmp_path,
            "write_file",
            {
                "file_path": "ledger.py",
                "content": _MODULE,
                "idempotency_key": "create-ledger",
            },
        )
    )
    assert body["file_path"] == "ledger.py"
    assert body["content_hash"]
    result = grade(tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr


def test_hidden_suite_fails_an_empty_workspace(tmp_path: Path) -> None:
    result = grade(tmp_path)
    assert result.returncode != 0
