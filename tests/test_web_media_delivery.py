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
    TrackGenerateTasks,
    bound_task,
    favlo_links,
    flush_queued,
    media_middleware,
    note_task,
    resolve_pending_task,
    store_links,
    take_tasks,
    task_cue,
)
from web_api.store import Rejected  # noqa: E402


def test_a_claimed_generate_leaves_later_tool_calls_available() -> None:
    _TASKS.set(None)
    middleware = TrackGenerateTasks()
    first: dict[str, object] = {}
    second: dict[str, object] = {}

    async def call_next() -> None:
        return None

    claimed: list[tuple[str, str, str, str]] = []
    again: list[tuple[str, str, str, str]] = []

    async def turn() -> None:
        await middleware.process(_Options(first), call_next)
        note_task("task-1", "http://127.0.0.1:8129/mcp", kind="image")
        await middleware.process(_Options(second), call_next)
        claimed.extend(take_tasks())
        again.extend(take_tasks())

    asyncio.run(turn())
    assert media_middleware()
    assert "tool_choice" not in first
    assert "tool_choice" not in second
    assert claimed == [("task-1", "http://127.0.0.1:8129/mcp", "image", "")]
    assert again == []


def test_a_show_hint_uses_the_open_task() -> None:
    _TASKS.set([])
    note_task(
        "task-real",
        "http://127.0.0.1:8129/mcp",
        kind="image",
        filename="star.png",
    )
    assert resolve_pending_task("image", "task-real") == "task-real"
    assert resolve_pending_task("image", "star.png") == "task-real"
    assert resolve_pending_task("image", "task-reaI") == "task-real"
    note_task(
        "task-other",
        "http://127.0.0.1:8129/mcp",
        kind="image",
        filename="moon.png",
    )
    assert resolve_pending_task("image", "moon.png") == "task-other"
    assert resolve_pending_task("image", "task-reaI") == "task-reaI"
    assert bound_task("image", "moon.png") == "task-other"
    _TASKS.set([])
    assert bound_task("image", "moon.png") is None
    _TASKS.set(None)


def test_favlo_links_come_from_the_task_result() -> None:
    payload = {
        "content": [
            {"type": "text", "text": "image_abc"},
            {"type": "resource_link", "uri": "favlo://images/image_abc.png"},
            {"type": "text", "text": "also favlo://images/image_abc.png"},
        ]
    }
    assert favlo_links(payload) == ["favlo://images/image_abc.png"]
    filed = {"content": [{"type": "text", "text": '{"mediaUri":"file:///images/image_abc.png"}'}]}
    assert favlo_links(filed) == ["file:///images/image_abc.png"]


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
    store.reject_greeting = True
    _QUEUED["thread-1"] = ["Task task-1 completed."]

    async def flush() -> None:
        await flush_queued(_App(store), "thread-1")

    asyncio.run(flush())
    assert store.messages == []
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


def test_a_finished_task_sends_only_the_fence() -> None:
    from web_api.host_cues import host_cue_message
    from web_api.media_delivery import media_turn_messages

    text = "<media_completed>\nstar.png\n</media_completed>"
    messages = media_turn_messages(text, message_id="cue-1")
    assert messages == [host_cue_message(text, message_id="cue-1")]
    assert messages[0]["content"] == text
    assert "system_reminder" not in messages[0]["content"]
    assert "favlo://" not in messages[0]["content"]
    assert media_turn_messages("The user just created this gallery.", message_id="cue-2") is None


def test_task_cue_fences_the_file_name() -> None:
    from web_api.host_cues import host_cue_message

    cue = task_cue(
        [
            ("task-a", "completed", ""),
            ("task-b", "failed", "moon.png"),
            ("task-c", "cancelled", "dot.png"),
            ("task-e", "completed", "star.png"),
        ]
    )
    assert cue == (
        "<media_error>\nmoon.png\n</media_error>\n"
        "<media_cancelled>\ndot.png\n</media_cancelled>\n"
        "<media_completed>\nstar.png\n</media_completed>"
    )
    assert "task-a" not in cue
    assert "favlo://" not in cue
    assert task_cue([(" ", "completed", ""), ("task-d", "working", "star.png")]) == ""
    row = host_cue_message(cue, message_id="cue-1")
    assert row["content"] == cue
    assert "system_reminder" not in row["content"]


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
