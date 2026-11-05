"""Host progress consumer. No device and no checkpoint."""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace

from favlo.host_progress import HostProgress


class Clock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


def test_counts_coalesce_latest_wins() -> None:
    seen: list[tuple] = []
    wake = threading.Event()

    def enqueue(item: tuple) -> None:
        seen.append(item)
        wake.set()

    clock = Clock()
    host = HostProgress(enqueue=enqueue, wake=lambda: None, clock=clock)
    progress = SimpleNamespace(
        closed=False, fed=False, started=100.0, total=10, unit="tok", detail="", gauge=None
    )
    try:
        host.count(progress, 1)
        host.count(progress, 2)
        host.count(progress, 7)
        host.flush()
        counts = [item for item in seen if item[0] == "count"]
        assert counts
        assert counts[-1][2][0] == 7
        assert counts[-1][2][5] == 0.0
        assert len(counts) <= 3
    finally:
        host.close()


def test_end_flushes_a_pending_count_first() -> None:
    seen: list[tuple] = []
    host = HostProgress(enqueue=seen.append, wake=lambda: None, clock=Clock())
    progress = SimpleNamespace(
        closed=False, fed=False, started=100.0, total=0, unit="tok", detail="", gauge=None
    )
    try:
        host.begin(progress)
        host.count(progress, 3)
        progress.closed = True
        host.end(progress, "decode settled")
        host.flush()
        kinds = [item[0] for item in seen]
        assert kinds == ["begin", "count", "end"]
        assert seen[1][2][0] == 3
        assert seen[2][2] == "decode settled"
    finally:
        host.close()


def test_note_reaches_the_console_queue() -> None:
    seen: list[tuple] = []
    host = HostProgress(enqueue=seen.append, wake=lambda: None, clock=Clock())
    try:
        host.note("prompt", {"tokens": 4})
        host.flush()
        assert seen == [("note", "prompt", {"tokens": 4})]
    finally:
        host.close()


def test_flush_waits_for_the_host_thread() -> None:
    seen: list[tuple] = []
    host = HostProgress(enqueue=seen.append, wake=lambda: None, clock=Clock())
    progress = SimpleNamespace(
        closed=False, fed=False, started=100.0, total=0, unit="tok", detail="", gauge=None
    )
    try:
        for done in range(20):
            host.count(progress, done)
        host.flush()
        assert any(item[0] == "count" and item[2][0] == 19 for item in seen)
    finally:
        host.close()
