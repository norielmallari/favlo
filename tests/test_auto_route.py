"""Auto selects a model from the tool calls on the turn, then the pipeline runs."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from favlo.auto_route import (
    PUBLIC_ID,
    AutoRouteError,
    auto_map,
    auto_row,
    prepare,
    request_label,
    select_task,
    tool_tasks,
)
from favlo.listener import bind_log


def _mapping() -> dict[str, str]:
    return {
        "default": "example-chat",
        "coding": "example-coder",
    }


def _tools() -> dict[str, str]:
    return {"write_file": "coding", "read_file": "coding", "glob": "coding"}


def _settings() -> dict:
    return {
        "auto": {
            **_mapping(),
            "tools": {"coding": ["write_file", "read_file", "glob"]},
        }
    }


def _call(name: str) -> dict[str, str]:
    return {"type": "function_call", "name": name, "call_id": f"call_{name}", "arguments": "{}"}


def _user(text: str) -> dict[str, str]:
    return {"type": "message", "role": "user", "content": text}


@pytest.fixture
def log_lines() -> list[str]:
    lines: list[str] = []
    bind_log(lines.append)
    yield lines
    bind_log(None)


def test_the_request_log_names_auto_and_the_alias() -> None:
    assert request_label("example-coder") == "auto:example-coder"
    assert request_label("") == PUBLIC_ID


def test_map_requires_default() -> None:
    assert auto_map({"auto": {"coding": "example-coder"}}) is None
    assert auto_map({"auto": _mapping()}) == _mapping()


def test_tool_names_map_to_the_task() -> None:
    assert tool_tasks(_settings()) == _tools()
    assert tool_tasks({"auto": _mapping()}) == {}


def test_no_tool_call_uses_default(
    monkeypatch: pytest.MonkeyPatch, log_lines: list[str]
) -> None:
    monkeypatch.setattr("favlo.auto_route.auto_map", lambda settings=None: _mapping())
    monkeypatch.setattr("favlo.auto_route.tool_tasks", lambda settings=None: _tools())
    body = {"model": PUBLIC_ID, "input": "hello", "tools": [{"type": "function", "name": "write_file"}]}
    routed = prepare(body, None)
    assert routed["model"] == "example-chat"
    assert routed["input"] == "hello"
    assert routed["tools"] == body["tools"]
    assert "auto default example-chat" in log_lines


def test_a_listed_tool_selects_coding(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("favlo.auto_route.auto_map", lambda settings=None: _mapping())
    monkeypatch.setattr("favlo.auto_route.tool_tasks", lambda settings=None: _tools())
    body = {
        "model": PUBLIC_ID,
        "input": [_user("build the ledger"), _call("glob"), _call("write_file")],
    }
    assert select_task(body, None, _tools()) == "coding"
    assert prepare(body, None)["model"] == "example-coder"


def test_a_user_message_with_no_call_uses_default() -> None:
    body = {
        "model": PUBLIC_ID,
        "input": [_user("build it"), _call("write_file"), _user("thanks")],
    }
    assert select_task(body, None, _tools()) == "default"


def test_a_tool_result_keeps_the_call(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("favlo.auto_route.auto_map", lambda settings=None: _mapping())
    monkeypatch.setattr("favlo.auto_route.tool_tasks", lambda settings=None: _tools())
    state = {
        "responses": {"resp_1": {"id": "resp_1", "output": [_call("write_file")]}},
        "response_inputs": {"resp_1": [_user("build it")]},
    }
    body = {
        "model": PUBLIC_ID,
        "previous_response_id": "resp_1",
        "input": [{"type": "function_call_output", "call_id": "call_write_file", "output": "ok"}],
    }
    assert select_task(body, state, _tools()) == "coding"
    assert prepare(body, state)["model"] == "example-coder"


def test_an_unmapped_tool_uses_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("favlo.auto_route.auto_map", lambda settings=None: _mapping())
    monkeypatch.setattr("favlo.auto_route.tool_tasks", lambda settings=None: _tools())
    body = {"model": PUBLIC_ID, "input": [_user("hold this"), _call("locker")]}
    assert prepare(body, None)["model"] == "example-chat"
    assert select_task(body, None, _tools()) == "default"


def test_a_missing_map_fails_before_an_open(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("favlo.auto_route.auto_map", lambda settings=None: None)
    with pytest.raises(AutoRouteError):
        prepare({"model": PUBLIC_ID, "input": "hello"}, None)


def test_listing_row_uses_the_default_context(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "favlo.auto_route.resolve",
        lambda _alias: SimpleNamespace(context_length=1024),
    )
    monkeypatch.setattr("favlo.auto_route.auto_map", lambda settings=None: _mapping())
    assert auto_row(created=1, settings={"auto": _mapping()})["context_length"] == 1024

