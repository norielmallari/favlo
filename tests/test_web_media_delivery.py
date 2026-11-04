"""A claimed generate ends the turn, then the finished file is appended."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "web-api"))

from web_api.media_delivery import (  # noqa: E402
    _QUEUED,
    _TASKS,
    ReplyAfterGenerate,
    announce_cue,
    favlo_links,
    flush_queued,
    note_task,
    store_links,
    take_tasks,
)
from web_api.store import Rejected  # noqa: E402


def test_a_claimed_generate_makes_the_next_call_text_only() -> None:
    _TASKS.set(None)
    middleware = ReplyAfterGenerate()
    first: dict[str, object] = {}
    second: dict[str, object] = {}

    async def call_next() -> None:
        return None

    claimed: list[tuple[str, str, str]] = []
    again: list[tuple[str, str, str]] = []

    async def turn() -> None:
        await middleware.process(_Options(first), call_next)
        note_task("task-1", "http://127.0.0.1:8129/mcp", kind="image")
        await middleware.process(_Options(second), call_next)
        claimed.extend(take_tasks())
        again.extend(take_tasks())

    asyncio.run(turn())
    assert "tool_choice" not in first
    assert second["tool_choice"] == "none"
    assert claimed == [("task-1", "http://127.0.0.1:8129/mcp", "image")]
    assert again == []


def test_favlo_links_come_from_the_task_result() -> None:
    payload = {
        "content": [
            {"type": "text", "text": "image_abc"},
            {"type": "resource_link", "uri": "favlo://images/image_abc.png"},
            {"type": "text", "text": "also favlo://images/image_abc.png"},
        ]
    }
    assert favlo_links(payload) == ["favlo://images/image_abc.png"]


def test_finished_links_append_once() -> None:
    store = _Store()
    events = store_links(store, "thread-1", "favlo://images/image_abc.png")
    assert events[0]["payload"]["text"] == "favlo://images/image_abc.png"
    assert store.messages[0]["role"] == "assistant"
    assert store_links(store, "thread-1", "favlo://images/image_abc.png") == []
    assert len(store.messages) == 1


def test_delivery_waits_while_a_run_is_open() -> None:
    _QUEUED.clear()
    store = _Store(busy=True)
    assert store_links(store, "thread-1", "favlo://images/image_abc.png") == []
    assert store.messages == []
    assert _QUEUED["thread-1"] == ["favlo://images/image_abc.png"]
    store.busy = False
    # Announce cannot open a run in this stub; fall back to appending the links.
    store.reject_greeting = True

    async def flush() -> None:
        await flush_queued(_App(store), "thread-1")

    asyncio.run(flush())
    assert store.messages[0]["content"] == "favlo://images/image_abc.png"
    assert "thread-1" not in _QUEUED


def test_delivery_marks_task_ids_on_the_event() -> None:
    store = _Store()
    events = store_links(
        store,
        "thread-1",
        "favlo://images/image_abc.png",
        task_ids=["task-1"],
    )
    assert events[0]["payload"]["task_ids"] == ["task-1"]


def test_announce_cue_lists_every_link() -> None:
    cue = announce_cue(["favlo://images/a.png", "favlo://images/b.png"])
    assert "favlo://images/a.png" in cue
    assert "favlo://images/b.png" in cue
    assert "Do not call tools." in cue


def test_links_in_a_host_cue_do_not_count_as_delivered() -> None:
    from web_api.host_cues import host_cue_message
    from web_api.media_delivery import _links_present

    store = _Store()
    store.messages.append(
        host_cue_message("ready\nfavlo://images/image_abc.png", message_id="cue-1")
    )
    assert not _links_present(store, "thread-1", ["favlo://images/image_abc.png"])
    # Fallback append still runs when only the host cue named the link.
    events = store_links(store, "thread-1", "favlo://images/image_abc.png")
    assert events[0]["payload"]["text"] == "favlo://images/image_abc.png"
    assert _links_present(store, "thread-1", ["favlo://images/image_abc.png"])


class _Options:
    def __init__(self, options: dict[str, object]) -> None:
        self.options = options


class _Store:
    def __init__(self, *, busy: bool = False) -> None:
        self.messages: list[dict[str, object]] = []
        self.said: list[str] = []
        self.busy = busy
        self.reject_greeting = False

    def running_run_ids(self, session_id: str) -> list[str]:
        assert session_id == "thread-1"
        return ["run-1"] if self.busy else []

    def session_exists(self, session_id: str) -> bool:
        assert session_id == "thread-1"
        return True

    def gallery_for_session(self, session_id: str) -> str | None:
        return None

    def begin_greeting(self, session_id: str, run_id: str, cue: str) -> dict[str, object]:
        assert session_id == "thread-1"
        assert run_id
        assert cue
        if self.reject_greeting:
            raise Rejected("stub announce")
        return {"run_id": run_id, "session_id": session_id, "events": []}

    def thread_snapshot(self, scope: str, thread_id: str) -> dict[str, object]:
        assert scope == "favlo"
        assert thread_id == "thread-1"
        return {"messages": self.messages}

    def append_thread_snapshot_message(
        self, scope: str, thread_id: str, message: dict[str, object]
    ) -> None:
        assert scope == "favlo"
        assert thread_id == "thread-1"
        self.messages.append(message)

    def record_assistant(
        self,
        session_id: str,
        text: str,
        *,
        task_ids: list[str] | None = None,
    ) -> list[dict[str, object]]:
        assert session_id == "thread-1"
        self.said.append(text)
        payload: dict[str, object] = {"text": text}
        if task_ids:
            payload["task_ids"] = task_ids
        return [{"type": "message.created", "payload": payload}]


class _App:
    def __init__(self, store: _Store) -> None:
        self.state = _State(store)


class _State:
    def __init__(self, store: _Store) -> None:
        self.store = store
        self.bus = _Bus()
        self.cancels: dict[str, object] = {}


class _Bus:
    async def publish(self, event: dict[str, object]) -> None:
        return None
