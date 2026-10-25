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
