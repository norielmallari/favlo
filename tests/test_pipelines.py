"""Pipeline manager. No GPU and no daemon."""

from __future__ import annotations

import importlib
import shutil
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

from favlo.cli import drop_pipeline_modules, load_pipeline
from favlo.http_app import _run_once
from favlo.pipelines import pipelines


class _Runtime:
    def __init__(self) -> None:
        self.closes = 0
        self.halts = 0
        self._closed = False

    def halt(self) -> None:
        self.halts += 1

    def close(self) -> None:
        self._closed = True
        self.closes += 1

    def generate(self) -> str:
        if self._closed:
            self._closed = False
        return "open"


def _state() -> dict:
    return {
        "lock": threading.Lock(),
        "runtime": None,
        "record": None,
        "pipeline": None,
        "module": None,
        "memory_owners": None,
        "cancel": threading.Event(),
    }


def test_close_reaches_every_runtime_and_a_second_close_is_quiet() -> None:
    first = _Runtime()
    second = _Runtime()
    state = _state()
    state["runtime"] = first
    state["record"] = "one"
    pipelines.add(first, ())
    pipelines.add(second, ())
    try:
        assert pipelines.close(state, SimpleNamespace()) == "one"
        assert first.closes == 1
        assert second.closes == 1
        assert first.halts == 1
        assert second.halts == 1
        assert state["runtime"] is None
        assert pipelines.close(state, SimpleNamespace()) is None
        assert first.closes == 1
        assert second.closes == 1
    finally:
        pipelines.close(_state(), SimpleNamespace())


def test_close_lets_the_next_generate_open_again() -> None:
    runtime = _Runtime()
    runtime._closed = True
    state = _state()
    state["runtime"] = runtime
    pipelines.add(runtime, ())
    try:
        pipelines.close(state, SimpleNamespace())
        assert runtime._closed is True
        assert runtime.generate() == "open"
        assert runtime._closed is False
    finally:
        pipelines.close(_state(), SimpleNamespace())


def test_drop_pipeline_modules_reloads_the_next_load(
    tmp_path: Path,
) -> None:
    tree = tmp_path / "pipelines" / "metal41"
    package = tree / "pipeline_reload_probe"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("VALUE = 1\n", encoding="utf-8")
    (tree / "probe.py").write_text(
        "import sys\n"
        "from pathlib import Path\n"
        "_ROOT = Path(__file__).resolve().parent\n"
        "if str(_ROOT) not in sys.path:\n"
        "    sys.path.insert(0, str(_ROOT))\n"
        "import pipeline_reload_probe as probe\n"
        "VALUE = probe.VALUE\n",
        encoding="utf-8",
    )
    favlo_name = "favlo.cli"
    assert favlo_name in sys.modules
    first = load_pipeline("probe", tmp_path, runtime="metal41")
    assert first.VALUE == 1
    (package / "__init__.py").write_text("VALUE = 2\n", encoding="utf-8")
    stale = load_pipeline("probe", tmp_path, runtime="metal41")
    assert stale.VALUE == 1
    shutil.rmtree(package / "__pycache__", ignore_errors=True)
    importlib.invalidate_caches()
    dropped = drop_pipeline_modules(tmp_path)
    assert dropped >= 1
    assert favlo_name in sys.modules
    fresh = load_pipeline("probe", tmp_path, runtime="metal41")
    assert fresh.VALUE == 2
    drop_pipeline_modules(tmp_path)
    assert "pipeline_reload_probe" not in sys.modules


def test_a_request_after_unload_does_not_open(monkeypatch) -> None:
    called: list[int] = []

    def bind(*_args: object, **_kwargs: object) -> None:
        called.append(1)

    monkeypatch.setattr("favlo.pipelines.bind_request", bind)
    state = _state()
    state["unload"] = True
    reply = SimpleNamespace(started=False, status=0, payload=None)

    def json(status: int, payload: dict) -> None:
        reply.status = status
        reply.payload = payload

    reply.json = json
    _run_once(
        SimpleNamespace(),
        state,
        "chat",
        "application/json",
        b"{}",
        reply,
    )
    assert called == []
    assert reply.status == 409


def test_reload_drops_a_checkpoint_file(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("favlo.cli.drop_pipeline_modules", lambda root=None: 0)
    checkpoint = tmp_path / "video_hold.pkl"
    checkpoint.write_bytes(b"hold")
    state = _state()
    state["video_hold"] = checkpoint
    try:
        pipelines.reload(state, SimpleNamespace())
        assert "video_hold" not in state
        assert not checkpoint.exists()
    finally:
        pipelines.close(_state(), SimpleNamespace())


def test_reload_drops_waiting_queues(monkeypatch) -> None:
    monkeypatch.setattr("favlo.cli.drop_pipeline_modules", lambda root=None: 0)
    from favlo.http_app import _mcp_dispatch
    from favlo.schedule import ClientClosed, Scheduler

    schedule = Scheduler()
    state = _state()
    state["schedule"] = schedule
    state["media_tasks"] = {
        "running": {"started": True, "paused": False},
        "queued": {"started": False, "paused": False},
    }
    state["media_order"] = ["running", "queued"]
    held = schedule.acquire("media")
    closed = threading.Event()

    def wait_chat() -> None:
        try:
            schedule.acquire("chat")
        except ClientClosed:
            closed.set()

    chat = threading.Thread(target=wait_chat)
    chat.start()
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline and schedule.waiting("chat") != 1:
        time.sleep(0.01)
    try:
        pipelines.reload(state, SimpleNamespace())
        assert closed.wait(2)
        assert schedule.waiting("chat") == 0
        assert state["media_tasks"]["queued"]["dropped"] is True
        assert state["media_tasks"]["running"].get("dropped") is not True
        assert state["media_order"] == ["running"]
        assert _mcp_dispatch(SimpleNamespace(), state, "image", {"prompt": "a cat"}, "queued") == {
            "status": "cancelled"
        }
    finally:
        schedule.release(held)
        chat.join(2)
        pipelines.close(_state(), SimpleNamespace())


def test_reload_drops_a_paused_hold(monkeypatch) -> None:
    monkeypatch.setattr("favlo.cli.drop_pipeline_modules", lambda root=None: 0)
    state = _state()
    state["video_hold"] = {"hold": object()}
    try:
        _closed, modules = pipelines.reload(state, SimpleNamespace())
        assert modules == 0
        assert "video_hold" not in state
    finally:
        pipelines.close(_state(), SimpleNamespace())
