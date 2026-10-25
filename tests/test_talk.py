"""The terminal voice. No model and no socket."""

from __future__ import annotations

import io
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from favlo.persona import INSTRUCTIONS, load_intros, opening, save_intros
from favlo.talk import Idle, _introduce, intro_request, is_quit, is_reload, reply, start

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


def test_each_start_shows_the_next_line(tmp_path: Path) -> None:
    path = tmp_path / "intros.json"
    save_intros((_LINE, _MORE), path)

    def once() -> list[str]:
        lines: list[str] = []
        thread = start(
            "127.0.0.1", 9, io.StringIO(""), lines.append, Idle(), home=tmp_path, introduce=_boom
        )
        thread.join(2)
        return lines

    assert once() == [f"favlo: {_LINE}"]
    assert opening(path) == _MORE
    assert once() == [f"favlo: {_MORE}"]
    assert once() == [f"favlo: {_LINE}"]
    assert load_intros(path) == (_MORE, _LINE)


def test_a_line_reaches_the_turn() -> None:
    seen: list[str] = []

    def turn(text: str) -> str:
        seen.append(text)
        return text

    assert reply("write some porn", turn) == "write some porn"
    assert reply("here is a carrot", turn) == "here is a carrot"
    assert seen == ["write some porn", "here is a carrot"]


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
    assert lines == [f"favlo: {_LINE}"]


def test_a_missing_file_asks_for_the_line(tmp_path: Path) -> None:
    started = threading.Event()
    release = threading.Event()
    lines: list[str] = []

    def introduce() -> str:
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
    )
    assert started.wait(1)
    assert lines == []
    release.set()
    thread.join(2)
    assert lines == [f"favlo: {_LINE}"]
    assert opening(tmp_path / "intros.json") == _MORE
    assert load_intros(tmp_path / "intros.json") == (_MORE, _LINE)


def test_a_saved_line_does_not_ask_again(tmp_path: Path) -> None:
    save_intros((_LINE,), tmp_path / "intros.json")
    lines: list[str] = []
    thread = start(
        "127.0.0.1",
        9,
        io.StringIO(""),
        lines.append,
        Idle(),
        home=tmp_path,
        introduce=_boom,
    )
    thread.join(2)
    assert lines == [f"favlo: {_LINE}"]


def test_registered_models_reads_the_registry(tmp_path: Path) -> None:
    from favlo.talk import chat_model_id, registered_models

    assert registered_models(tmp_path) == "no models are registered"
    with pytest.raises(RuntimeError, match="no model is registered"):
        chat_model_id(tmp_path)
    registry = tmp_path / "models"
    registry.mkdir()
    (registry / "registry.json").write_text(
        json.dumps(
            {
                "version": 1,
                "default": "qwen36-35b-a3b:instruct",
                "models": {
                    "Qwen3.6-35B-A3B-fp4": {
                        "alias": "qwen36-35b-a3b",
                        "pipeline": "qwen36-35b-a3b",
                        "runtime": "metal41",
                        "kv_cache": "e4m3",
                        "variants": {
                            "thinking": {
                                "temperature": 1.0,
                                "top_p": 0.95,
                                "top_k": 20,
                                "presence_penalty": 1.5,
                            },
                            "instruct": {
                                "temperature": 0.7,
                                "top_p": 0.8,
                                "top_k": 20,
                                "presence_penalty": 1.5,
                            },
                        },
                        "default_variant": "thinking",
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    assert registered_models(tmp_path) == (
        "Qwen3.6-35B-A3B-fp4 pipeline qwen36-35b-a3b alias qwen36-35b-a3b kv e4m3"
    )
    assert chat_model_id(tmp_path) == "qwen36-35b-a3b:instruct"


def test_a_tool_update_is_not_spoken() -> None:
    from types import SimpleNamespace

    from favlo.talk import assistant_piece, spoken_text

    assert assistant_piece(SimpleNamespace(role="assistant", text="Hello")) == "Hello"
    assert assistant_piece(SimpleNamespace(role="tool", text="guide text")) == ""
    assert assistant_piece(SimpleNamespace(role=None, text="")) == ""
    call = "<tool_call>\n<function=list_models>\n</function>\n</tool_call>"
    assert spoken_text(call) == ""
    assert spoken_text(f"Looking.\n{call}") == "Looking."
    assert spoken_text("Here <tool") == "Here"
    assert spoken_text(f"{call}\nQwen3.6-35B-A3B-fp4") == "Qwen3.6-35B-A3B-fp4"


def test_quit_closes_a_stream_the_turn_left_open() -> None:
    import asyncio
    import warnings

    from favlo.talk import _finish_loop

    closed: list[str] = []
    held: list[object] = []

    async def events():
        try:
            yield "token"
            await asyncio.sleep(0)
        finally:
            closed.append("stream")

    async def start() -> None:
        iterator = events()
        held.append(iterator)
        await iterator.__anext__()

    loop = asyncio.new_event_loop()
    loop.run_until_complete(start())
    assert closed == []
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        _finish_loop(loop, {})
    assert closed == ["stream"]


def test_one_opening_is_not_enough() -> None:
    from favlo.talk import accepted_openings

    assert accepted_openings(["Hi there"]) is None
    assert accepted_openings(["Hi there", "hi there"]) is None
    lines = [f"Hello {number}, could you spare a carrot?" for number in range(4)]
    assert accepted_openings(lines) == lines


def test_intro_request_names_save_intro() -> None:
    body = intro_request("qwen36-35b-a3b")
    assert body["model"] == "qwen36-35b-a3b"
    assert body["tool_choice"] == {"type": "function", "function": {"name": "save_intro"}}
    assert body["enable_thinking"] is False
    assert body["max_tokens"] == 1200
    system = body["messages"][0]["content"]
    assert system.startswith(INSTRUCTIONS)
    assert "You refuse violent, sexual, NSFW, erotic, and obscene requests." in INSTRUCTIONS
    assert "You refuse sexual or romantic content." in INSTRUCTIONS
    assert "one short sentence" in INSTRUCTIONS
    assert "read_setup" in INSTRUCTIONS
    assert "qwen36-35b-a3b" in INSTRUCTIONS
    assert "qwen38-27b" in INSTRUCTIONS
    assert "z-image-turbo" in INSTRUCTIONS
    assert "wan22-i2v" in INSTRUCTIONS
    assert "ace-step-15-turbo" in INSTRUCTIONS
    assert "run_setup" in INSTRUCTIONS
    assert "Do not use violent, sexual, NSFW, erotic, or obscene language" in system
    assert "longer" in system
    assert "emojis" in system
    assert "next line" in body["messages"][1]["content"]


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
                                "arguments": json.dumps(
                                    {
                                        "lines": [
                                            _LINE,
                                            _MORE,
                                            "Favlo here. The carrot can wait one sentence.",
                                            "A small horse says hello and hopes for a carrot.",
                                        ]
                                    }
                                ),
                            }
                        }
                    ]
                }
            }
        ]
    }

    class Handler(BaseHTTPRequestHandler):
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
        registry = tmp_path / "models"
        registry.mkdir()
        (registry / "registry.json").write_text(
            json.dumps(
                {
                    "version": 1,
                    "default": "qwen36-35b-a3b:instruct",
                    "models": {
                        "Qwen3.6-35B-A3B-fp4": {
                            "alias": "qwen36-35b-a3b",
                            "pipeline": "qwen36-35b-a3b",
                        }
                    },
                }
            ),
            encoding="utf-8",
        )
        line = _introduce(
            f"http://127.0.0.1:{port}/v1",
            tmp_path / "intros.json",
            root=tmp_path,
        )
    finally:
        server.shutdown()
        server.server_close()
    assert line == _LINE
    assert len(posts) == 1
    assert posts[0]["model"] == "qwen36-35b-a3b:instruct"
    assert posts[0]["tool_choice"] == {"type": "function", "function": {"name": "save_intro"}}
    assert posts[0]["enable_thinking"] is False
    assert load_intros(tmp_path / "intros.json")[:2] == (_LINE, _MORE)
    assert len(load_intros(tmp_path / "intros.json")) == 4


def test_a_later_start_reuses_the_saved_line(tmp_path: Path) -> None:
    save_intros((_LINE, _MORE), tmp_path / "intros.json")
    lines: list[str] = []
    thread = start(
        "127.0.0.1", 9, io.StringIO(""), lines.append, Idle(), home=tmp_path, introduce=_boom
    )
    thread.join(2)
    assert lines == [f"favlo: {_LINE}"]


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


def test_reload_stays_in_chat(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    save_intros((_LINE,), tmp_path / "intros.json")
    calls: list[str] = []
    monkeypatch.setattr(
        "favlo.talk.pipeline_command",
        lambda _host, _port, action: calls.append(action),
    )
    stopped = threading.Event()
    lines: list[str] = []
    thread = start(
        "127.0.0.1",
        9,
        io.StringIO("/reload\n/quit\n"),
        lines.append,
        Idle(),
        stopped.set,
        home=tmp_path,
        introduce=_boom,
    )
    thread.join(2)
    assert is_reload("/reload")
    assert calls == ["reload"]
    assert stopped.is_set()
    assert "favlo: reloaded" in lines


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
    assert lines == [f"favlo: {_LINE}"]


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
