"""Child preempt delivery. No GPU."""

from __future__ import annotations

import json
import os
import threading
import time
from types import SimpleNamespace

from favlo.pipeline_worker import PipelineChild, _call


class _Saw:
    def __init__(self) -> None:
        self.flags: list[bool] = []

    def handle_image(self, reply, runtime, body, selection, args, state) -> None:
        del runtime, body, selection, args
        self.flags.append(bool(state["preempt"].is_set()))
        reply.json(200, {"status": "in_progress"})


class _Proto:
    def write(self, blob: bytes) -> None:
        del blob


def _control() -> dict[str, threading.Event]:
    return {
        "preempt": threading.Event(),
        "cancel": threading.Event(),
        "disconnect": threading.Event(),
        "halt": threading.Event(),
    }


class _ChatSaw:
    def __init__(self) -> None:
        self.stopped = False

    def handle_chat(self, reply, runtime, body, selection, args, state) -> None:
        del runtime, body, selection, args, state
        self.stopped = bool(reply.stopped())
        reply.json(200, {"id": "chat", "choices": []})


def test_a_clean_exit_waits_for_the_pages_taken_while_resident(monkeypatch) -> None:
    seen: dict[str, object] = {}

    class _Ledger:
        def confirm_released(self, before: int, nbytes: int, timeout: float | None = None) -> bool:
            seen["before"] = before
            seen["nbytes"] = nbytes
            seen["timeout"] = timeout
            return True

        def forget(self, owners: object) -> None:
            seen["owners"] = owners

    monkeypatch.setattr("favlo.memory.memory", _Ledger())
    child = PipelineChild(SimpleNamespace(), None)
    child._proc = SimpleNamespace(returncode=0)
    child._baseline = 40_000_000_000
    child._while_resident = 16_000_000_000
    child._release_from = 50_000
    child._held = 30_000_000_000
    child.counts = {"chat": 30_000_000_000}
    child._forget()
    assert seen["before"] == 16_000_000_000
    assert seen["nbytes"] == 24_000_000_000
    assert seen["timeout"] is None
    assert seen["owners"] == ("chat",)


def test_a_short_reclaim_still_drops_the_owners(monkeypatch) -> None:
    seen: dict[str, object] = {}
    lines: list[str] = []

    class _Ledger:
        def confirm_released(self, before: int, nbytes: int, timeout: float | None = None) -> bool:
            del before, nbytes, timeout
            return False

        def forget(self, owners: object) -> None:
            seen["owners"] = owners

    monkeypatch.setattr("favlo.memory.memory", _Ledger())
    monkeypatch.setattr("favlo.pipeline_worker._log_line", lines.append)
    child = PipelineChild(SimpleNamespace(), None)
    child._proc = SimpleNamespace(returncode=0)
    child._baseline = 40_000_000_000
    child._while_resident = 16_000_000_000
    child._release_from = 50_000
    child._held = 30_000_000_000
    child.counts = {"chat": 30_000_000_000}
    child._forget()
    assert seen["owners"] == ("chat",)
    assert lines == ["reclaimable pages stopped short"]


def test_a_disconnected_chat_is_stopped_before_it_streams() -> None:
    module = _ChatSaw()
    _call(
        _Proto(),
        object(),
        module,
        SimpleNamespace(),
        None,
        {"kind": "chat", "body": {}, "disconnect": True},
        _control(),
        False,
        None,
    )
    assert module.stopped is True


def test_a_call_keeps_preempt_set_after_the_child_clears_flags() -> None:
    module = _Saw()
    control = _control()
    control["preempt"].set()
    _call(
        _Proto(),
        object(),
        module,
        SimpleNamespace(),
        None,
        {"kind": "image", "body": {}, "preempt": True},
        control,
        False,
        None,
    )
    assert module.flags == [True]
    _call(
        _Proto(),
        object(),
        module,
        SimpleNamespace(),
        None,
        {"kind": "image", "body": {}, "preempt": False},
        control,
        False,
        None,
    )
    assert module.flags == [True, False]


def test_the_watcher_signals_preempt_while_it_stays_set() -> None:
    child = PipelineChild.__new__(PipelineChild)
    child._write_lock = threading.Lock()
    read_fd, write_fd = os.pipe()
    os.set_blocking(read_fd, False)
    child._control = write_fd
    state = {"preempt": threading.Event()}
    state["preempt"].set()
    stop = threading.Event()
    reply = SimpleNamespace(stopped=None)
    watcher = threading.Thread(target=child._watch, args=(reply, state, stop))
    watcher.start()
    time.sleep(0.2)
    stop.set()
    watcher.join(2)
    os.close(write_fd)
    chunks: list[bytes] = []
    while True:
        try:
            blob = os.read(read_fd, 4096)
        except BlockingIOError:
            break
        if not blob:
            break
        chunks.append(blob)
    os.close(read_fd)
    names = [
        json.loads(line)["name"]
        for line in b"".join(chunks).decode().splitlines()
        if line
    ]
    assert names.count("preempt") >= 2
