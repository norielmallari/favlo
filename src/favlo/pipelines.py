"""Process pipeline manager.

``pipelines`` is the only registry of runtimes this process has opened.
``close`` halts every one of them, then calls ``close`` so a media pipeline
can open its device again on the next request. ``reload`` does that and
then drops the imported pipeline modules.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

_HOLDS = ("image_hold", "video_hold", "music_hold", "voice_hold")


class Pipelines:
    """Runtimes opened in this process. One resident model stays the listener's rule."""

    def __init__(self) -> None:
        self._items: list[Any] = []
        self._owners: dict[int, tuple[str, ...]] = {}

    def add(self, runtime: Any, owners: Any) -> None:
        """Remember a runtime ``open`` returned. A repeat keeps the first owners."""
        if runtime is None:
            return
        if runtime not in self._items:
            self._items.append(runtime)
        self._owners.setdefault(id(runtime), _owners(owners))

    def retire(self, runtime: Any, owners: Any = None) -> None:
        """Close one runtime and forget it. A runtime that is already closed stays closed."""
        self._forget(runtime)
        if runtime is None and not _owners(owners):
            return
        from favlo.listener import _close_runtime

        _close_runtime(runtime, _owners(owners) or None)

    def close(self, state: dict[str, Any], args: Any, *, wait: float | None = None) -> str | None:
        """Halt, wait for the request lock, and close every registered runtime.

        ``wait`` is the process-exit budget. The HTTP close waits until the
        request releases the lock. Neither path sets the smoke hold.
        """
        self._halt(state, args)
        lock: threading.Lock = state.setdefault("lock", threading.Lock())
        if wait is None:
            lock.acquire()
            held = True
        else:
            held = lock.acquire(timeout=wait)
        try:
            return self._close_locked(state, args, drop_modules=False)
        finally:
            if held:
                lock.release()

    def reload(self, state: dict[str, Any], args: Any) -> tuple[str | None, int]:
        """Close every runtime, drop paused holds, and drop imported pipeline modules."""
        self._halt(state, args)
        lock: threading.Lock = state.setdefault("lock", threading.Lock())
        lock.acquire()
        try:
            closed = self._close_locked(state, args, drop_modules=True)
            from favlo.cli import drop_pipeline_modules

            return closed, drop_pipeline_modules()
        finally:
            lock.release()

    def _halt(self, state: dict[str, Any], args: Any) -> None:
        state["unload"] = True
        cancel = state.get("cancel")
        if not isinstance(cancel, threading.Event):
            cancel = threading.Event()
            state["cancel"] = cancel
        cancel.set()
        seen: set[int] = set()
        for runtime in list(self._items):
            _halt(runtime)
            seen.add(id(runtime))
        current = state.get("runtime")
        if current is not None and id(current) not in seen:
            _halt(current)
        shutdown = getattr(args, "shutdown", None)
        extra = shutdown.runtime if shutdown is not None else None
        if extra is not None and id(extra) not in seen:
            _halt(extra)

    def _close_locked(self, state: dict[str, Any], args: Any, *, drop_modules: bool) -> str | None:
        folder = state.get("record")
        closed = folder if isinstance(folder, str) and folder else None
        current = state.get("runtime")
        runtimes = list(self._items)
        if current is not None and current not in runtimes:
            runtimes.append(current)
        resident_owners = _owners(state.get("memory_owners"))
        for runtime in runtimes:
            owners = self._owners.get(id(runtime), ())
            if runtime is current and not owners:
                owners = resident_owners
            self.retire(runtime, owners)
        state["runtime"] = None
        state["record"] = None
        state["pipeline"] = None
        state["module"] = None
        state["memory_owners"] = None
        shutdown = getattr(args, "shutdown", None)
        if shutdown is not None:
            shutdown.runtime = None
        if drop_modules:
            for key in _HOLDS:
                value = state.pop(key, None)
                if isinstance(value, Path):
                    value.unlink(missing_ok=True)
        cancel = state.get("cancel")
        if isinstance(cancel, threading.Event):
            cancel.clear()
        state["unload"] = False
        if closed is not None:
            from favlo.listener import _log

            _log(f"released {closed}")
        return closed

    def _forget(self, runtime: Any) -> None:
        self._items = [item for item in self._items if item is not runtime]
        if runtime is not None:
            self._owners.pop(id(runtime), None)


def _owners(owners: Any) -> tuple[str, ...]:
    if not owners:
        return ()
    return tuple(owner for owner in owners if isinstance(owner, str) and owner)


def _halt(runtime: Any) -> None:
    halt = getattr(runtime, "halt", None)
    if callable(halt):
        halt()


pipelines = Pipelines()
