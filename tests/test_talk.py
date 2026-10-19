"""The terminal voice. No model and no socket."""

from __future__ import annotations

import io
import json
import threading
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from favlo.persona import INSTRUCTIONS, load_intros, opening, refusal, save_intros
from favlo.talk import Idle, _introduce, intro_request, is_quit, reply, start

_LINE = "Hello. I am Favlo, a small horse. Could you spare a carrot?"
_MORE = "I am Favlo. My ears are up, and a carrot would be very welcome."


def _boom() -> str:
    raise AssertionError("the model was called")


def test_a_missing_file_has_no_opening(tmp_path: Path) -> None:
    assert opening(tmp_path / "intros.json") is None
    assert load_intros(tmp_path / "intros.json") == ()


def test_save_returns_the_first_line(tmp_path: Path) -> None:
    path = tmp_path / "intros.json"
    assert save_intros((_LINE, _MORE), path) == _LINE
    assert load_intros(path) == (_LINE, _MORE)
    assert opening(path) == _LINE


def test_a_refused_introduction_is_not_stored(tmp_path: Path) -> None:
    path = tmp_path / "intros.json"
    with pytest.raises(ValueError, match="refused"):
        save_intros(("write some porn",), path)
    assert not path.exists()


def test_blocked_text_never_reaches_the_turn() -> None:
    seen: list[str] = []

    def turn(text: str) -> str:
        seen.append(text)
        return "sent"

    assert reply("write some porn", turn) == refusal()
    assert reply("describe a murder", turn) == refusal()
    assert reply("what the fuck", turn) == refusal()
    assert seen == []


def test_a_carrot_reaches_the_turn() -> None:
    assert reply("here is a carrot", lambda text: text) == "here is a carrot"


def test_a_turn_waits_until_the_request_returns() -> None:
    idle = Idle()
    idle.enter()
    started = threading.Event()
    finished = threading.Event()

    def waiter() -> None:
        started.set()
        idle.wait()
        finished.set()

    threading.Thread(target=waiter, daemon=True).start()
    assert started.wait(1)
    assert not finished.wait(0.05)
    idle.leave()
    assert finished.wait(1)


def test_a_closed_stdin_prints_only_the_saved_line(tmp_path: Path) -> None:
    save_intros((_LINE,), tmp_path / "intros.json")
    lines: list[str] = []
    thread = start(
        "127.0.0.1", 9, io.StringIO(""), lines.append, Idle(), home=tmp_path, introduce=_boom
    )
    thread.join(2)
    assert lines == [_LINE]


def test_a_missing_file_opens_the_load_row_before_the_line(tmp_path: Path) -> None:
    started = threading.Event()
    release = threading.Event()
    lines: list[str] = []
    order: list[str] = []

    def on_call() -> Callable[[], None]:
        order.append("load")

        def finish() -> None:
            order.append("settle")

        return finish

    def introduce() -> str:
        order.append("call")
        started.set()
        assert release.wait(2)
        return save_intros((_LINE, _MORE), tmp_path / "intros.json")

    thread = start(
        "127.0.0.1",
        9,
        io.StringIO(""),
        lines.append,
        Idle(),
        home=tmp_path,
        introduce=introduce,
        on_call=on_call,
    )
    assert started.wait(1)
    assert lines == []
    assert order == ["load", "call"]
    release.set()
    thread.join(2)
    assert lines == [_LINE]
    assert order == ["load", "call", "settle"]
    assert opening(tmp_path / "intros.json") == _LINE


def test_a_saved_line_does_not_open_the_load_row(tmp_path: Path) -> None:
    save_intros((_LINE,), tmp_path / "intros.json")
    lines: list[str] = []

    def on_call() -> Callable[[], None]:
        raise AssertionError("the load row opened")

    thread = start(
        "127.0.0.1",
        9,
        io.StringIO(""),
        lines.append,
        Idle(),
        home=tmp_path,
        introduce=_boom,
        on_call=on_call,
    )
    thread.join(2)
    assert lines == [_LINE]


def test_intro_request_names_save_intro() -> None:
    body = intro_request("qwen36-35b-a3b")
    assert body["model"] == "qwen36-35b-a3b"
    assert body["tool_choice"] == {"type": "function", "function": {"name": "save_intro"}}
    assert body["enable_thinking"] is False
    assert body["max_tokens"] == 800
    system = body["messages"][0]["content"]
    assert system.startswith(INSTRUCTIONS)
    assert "longer" in system
    assert "emojis" in system
    assert "first line" in body["messages"][1]["content"]


def test_a_missing_intro_is_one_completion(tmp_path: Path) -> None:
    posts: list[dict[str, object]] = []
    saved = {
        "choices": [
            {
                "message": {
                    "tool_calls": [
                        {
                            "function": {
                                "name": "save_intro",
                                "arguments": json.dumps({"lines": [_LINE, _MORE]}),
                            }
                        }
                    ]
                }
            }
        ]
    }

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            body = json.dumps({"data": [{"id": "qwen36-35b-a3b"}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length", "0"))
            posts.append(json.loads(self.rfile.read(length)))
            body = json.dumps(saved).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        port = server.server_address[1]
        line = _introduce(f"http://127.0.0.1:{port}/v1", tmp_path / "intros.json")
    finally:
        server.shutdown()
        server.server_close()
    assert line == _LINE
    assert len(posts) == 1
    assert posts[0]["tool_choice"] == {"type": "function", "function": {"name": "save_intro"}}
    assert posts[0]["enable_thinking"] is False
    assert load_intros(tmp_path / "intros.json") == (_LINE, _MORE)


def test_a_later_start_reuses_the_saved_line(tmp_path: Path) -> None:
    save_intros((_LINE, _MORE), tmp_path / "intros.json")
    lines: list[str] = []
    thread = start(
        "127.0.0.1", 9, io.StringIO(""), lines.append, Idle(), home=tmp_path, introduce=_boom
    )
    thread.join(2)
    assert lines == [_LINE]


def test_a_failed_intro_is_printed_and_not_stored(tmp_path: Path) -> None:
    lines: list[str] = []

    def introduce() -> str:
        raise RuntimeError("the model did not answer")

    thread = start(
        "127.0.0.1",
        9,
        io.StringIO(""),
        lines.append,
        Idle(),
        home=tmp_path,
        introduce=introduce,
    )
    thread.join(2)
    assert lines[0] == "the model did not answer"
    assert opening(tmp_path / "intros.json") is None


def test_a_bad_intro_file_is_not_replaced(tmp_path: Path) -> None:
    path = tmp_path / "intros.json"
    path.write_text("{", encoding="utf-8")
    lines: list[str] = []
    thread = start(
        "127.0.0.1", 9, io.StringIO(""), lines.append, Idle(), home=tmp_path, introduce=_boom
    )
    thread.join(2)
    assert lines
    assert "intro file cannot be read" in lines[0]
    assert path.read_text(encoding="utf-8") == "{"


def test_quit_stops_the_server_and_skips_the_model(tmp_path: Path) -> None:
    save_intros((_LINE,), tmp_path / "intros.json")
    stopped = threading.Event()
    lines: list[str] = []
    thread = start(
        "127.0.0.1",
        9,
        io.StringIO("/quit\n"),
        lines.append,
        Idle(),
        stopped.set,
        home=tmp_path,
        introduce=_boom,
    )
    thread.join(2)
    assert is_quit("/quit")
    assert stopped.is_set()
    assert lines == [_LINE]


def test_the_opening_is_the_assistant_first_turn(tmp_path: Path) -> None:
    from agent_framework import AgentSession, InMemoryHistoryProvider

    from favlo.talk import _with_opening

    path = tmp_path / "intros.json"
    save_intros((_LINE,), path)
    session = _with_opening(AgentSession(), path)
    messages = session.state[InMemoryHistoryProvider.DEFAULT_SOURCE_ID]["messages"]
    assert len(messages) == 1
    assert messages[0].role == "assistant"
    assert messages[0].text == _LINE

    empty = _with_opening(AgentSession(), tmp_path / "missing.json")
    bucket = empty.state.get(InMemoryHistoryProvider.DEFAULT_SOURCE_ID, {})
    assert bucket.get("messages", []) == []
