"""CPU-bound progress consumer between the GPU thread and the console queue.

The GPU thread posts begin, count, end, and note. This module's thread builds
elapsed facts, coalesces counts latest-wins per open handle, and enqueues onto
the console action queue the CLI already drains. It does not paint and it does
not sanitize.
"""

from __future__ import annotations

import math
import queue
import threading
from collections.abc import Callable
from typing import Any


class HostProgress:
    """One producer inbox per console. Posts return before the console queue runs."""

    def __init__(
        self,
        *,
        enqueue: Callable[[tuple[Any, ...]], None],
        wake: Callable[[], None],
        clock: Callable[[], float],
    ) -> None:
        self._enqueue = enqueue
        self._wake = wake
        self._clock = clock
        self._inbox: queue.SimpleQueue[tuple[Any, ...]] = queue.SimpleQueue()
        self._counts: dict[int, tuple[Any, int]] = {}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._idle = threading.Event()
        self._idle.set()

    def begin(self, progress: Any) -> None:
        self._post(("begin", progress))

    def count(self, progress: Any, done: int) -> None:
        if progress is None or getattr(progress, "closed", False):
            return
        if isinstance(done, bool) or not isinstance(done, int):
            return
        with self._lock:
            self._counts[id(progress)] = (progress, int(done))
        progress.fed = True
        self._post(("count",))

    def end(self, progress: Any, summary: str) -> None:
        self._post(("end", progress, summary))

    def note(self, topic: str, fields: dict[str, Any]) -> None:
        self._post(("note", topic, fields))

    def flush(self) -> None:
        """Block until every posted item has reached the console queue."""
        self._ensure()
        done = threading.Event()
        self._inbox.put(("flush", done))
        self._idle.clear()
        if not done.wait(2.0):
            return
        self._idle.wait(2.0)

    def close(self) -> None:
        self._stop.set()
        self._inbox.put(("stop",))
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=1)
        self._drain()

    def _post(self, item: tuple[Any, ...]) -> None:
        self._idle.clear()
        self._inbox.put(item)
        self._ensure()

    def _ensure(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name="favlo-host-progress", daemon=True
        )
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                item = self._inbox.get(timeout=0.05)
            except queue.Empty:
                self._flush_counts()
                self._mark_idle()
                continue
            if item[0] == "stop":
                self._flush_counts()
                self._mark_idle()
                return
            if item[0] == "flush":
                self._drain()
                item[1].set()
                self._mark_idle()
                continue
            self._handle(item)
            self._drain()
        self._drain()
        self._mark_idle()

    def _drain(self) -> None:
        while True:
            self._flush_counts()
            try:
                item = self._inbox.get_nowait()
            except queue.Empty:
                return
            if item[0] == "stop":
                self._flush_counts()
                return
            if item[0] == "flush":
                self._flush_counts()
                item[1].set()
                continue
            self._handle(item)

    def _handle(self, item: tuple[Any, ...]) -> None:
        kind = item[0]
        if kind == "begin":
            self._flush_counts()
            self._enqueue(("begin", item[1]))
            self._wake()
            return
        if kind == "count":
            return
        if kind == "end":
            progress, summary = item[1], item[2]
            self._flush_counts()
            self._enqueue(("end", progress, summary))
            self._wake()
            return
        if kind == "note":
            self._enqueue(("note", item[1], item[2]))
            self._wake()

    def _flush_counts(self) -> None:
        with self._lock:
            pending = self._counts
            self._counts = {}
        if not pending:
            return
        now = self._clock()
        for progress, done in pending.values():
            started = float(getattr(progress, "started", now) or now)
            elapsed = now - started
            if elapsed < 0 or not math.isfinite(elapsed):
                elapsed = 0.0
            fact = (
                int(done),
                int(getattr(progress, "total", 0) or 0),
                str(getattr(progress, "unit", "tok") or "tok"),
                str(getattr(progress, "detail", "") or ""),
                getattr(progress, "gauge", None),
                float(elapsed),
            )
            self._enqueue(("count", progress, fact))
        self._wake()

    def _mark_idle(self) -> None:
        if self._inbox.empty():
            with self._lock:
                empty = not self._counts
            if empty:
                self._idle.set()
