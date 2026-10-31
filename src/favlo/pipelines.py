"""Process pipeline manager.

This is the registry of runtimes this process has opened, and it chooses
which request is in service and which runtime serves it. A request waits
in the admission queues until that runtime can serve it. ``close`` halts
every runtime, then calls ``close`` so a media pipeline can open its device
again on the next request. ``reload`` does that, drops every waiter, and
then drops the imported pipeline modules. A crash sweeps that pipeline's
owners on the memory ledger and leaves the listener up.
"""

from __future__ import annotations

import asyncio
import inspect
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

_HOLDS = ("image_hold", "video_hold", "music_hold", "voice_hold")


class Pipelines:
    """Runtimes opened in this process.

    One chat or media model stays resident. An embeddings process can sit
    beside it until that request returns.
    """

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
        close_runtime(runtime, _owners(owners) or None)

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
        """Close every runtime, drop the queues, and drop imported pipeline modules."""
        self._halt(state, args)
        _flush_queues(state)
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


def _flush_queues(state: dict[str, Any]) -> None:
    from favlo.schedule import Scheduler

    schedule = state.get("schedule")
    if isinstance(schedule, Scheduler):
        schedule.flush()
    _drop_waiting_tasks(state)


def _drop_waiting_tasks(state: dict[str, Any]) -> None:
    tasks = state.get("media_tasks")
    if not isinstance(tasks, dict):
        return
    dropped: set[str] = set()
    for task_id, record in tasks.items():
        if isinstance(task_id, str) and isinstance(record, dict) and not record.get("started"):
            record["dropped"] = True
            dropped.add(task_id)
    order = state.get("media_order")
    if isinstance(order, list) and dropped:
        state["media_order"] = [item for item in order if item not in dropped]


def _halt(runtime: Any) -> None:
    halt = getattr(runtime, "halt", None)
    if callable(halt):
        halt()


pipelines = Pipelines()

_HANDLERS = {
    "chat": "handle_chat",
    "embed": "handle_embed",
    "image": "handle_image",
    "video": "handle_video",
    "music": "handle_music",
    "voice": "handle_voice",
}
_MEDIA = frozenset({"image", "video", "music", "voice"})
_running_job: str | None = None
_job_gate = threading.Lock()


class Opened:
    """The runtime the manager opened for one request."""

    def __init__(
        self,
        selection: Any,
        runtime: Any,
        module: Any,
        *,
        embed_child: Any = None,
        drop_embed: bool = False,
    ) -> None:
        self.selection = selection
        self.runtime = runtime
        self.module = module
        self.embed_child = embed_child
        self.drop_embed = drop_embed


def pipeline_module(selection: Any) -> Any:
    """The entry file the registry row names."""
    from favlo.cli import load_pipeline

    return load_pipeline(
        selection.pipeline,
        runtime=selection.runtime,
        custom=selection.custom,
    )


def _accepts(opener: Any, name: str) -> bool:
    try:
        parameters = inspect.signature(opener).parameters
    except (TypeError, ValueError):
        return False
    return name in parameters


def open_runtime(selection: Any, console: Any, max_memory: str | None) -> Any:
    """Open the pipeline named on the registry row. The previous runtime is already closed."""
    from favlo.memory import mint_owner

    module = pipeline_module(selection)
    opener = getattr(module, "open_runtime", None)
    if not callable(opener):
        message = f"{selection.pipeline} does not open a runtime"
        raise RuntimeError(message)
    kwargs: dict[str, Any] = {
        "kv_cache": selection.kv_cache,
        "console": console,
        "resident_percent": selection.resident_percent,
    }
    if _accepts(opener, "max_memory"):
        kwargs["max_memory"] = max_memory
    if _accepts(opener, "owner"):
        kwargs["owner"] = mint_owner(selection.pipeline)
    if _accepts(opener, "mtp"):
        kwargs["mtp"] = bool(getattr(selection, "mtp", False))
    if _accepts(opener, "parts"):
        kwargs["parts"] = selection.parts or {}
    return opener(selection.path, **kwargs)


def opening_owners(before: Any, runtime: Any) -> tuple[str, ...]:
    """Owners this open added, plus the runtime's own string when it has one."""
    from favlo.memory import memory

    found = set(memory.owners()) - set(before)
    owner = getattr(runtime, "_owner", None)
    if isinstance(owner, str) and owner:
        found.add(owner)
    return tuple(owner for owner in found if owner)


def _sweep_owners(owners: Any) -> None:
    if not owners:
        return
    from favlo.memory import memory

    for owner in owners:
        if isinstance(owner, str) and owner:
            memory.sweep(owner)


def _sweep_new(before: Any) -> None:
    from favlo.memory import memory

    memory.sweep_new(before)


def close_runtime(runtime: Any, owners: Any = None) -> None:
    """Close one runtime and sweep the owners recorded for that opening."""
    if runtime is None and not owners:
        return
    import gc

    import objc

    try:
        if runtime is not None:
            with objc.autorelease_pool():
                runtime.close()
                gc.collect()
    finally:
        _sweep_owners(owners)


def start_pipeline(selection: Any, args: Any, schedule: Any = None) -> Any:
    """Open the pipeline in its own process. The manager keeps the handle."""
    from favlo.pipeline_worker import PipelineChild

    child = PipelineChild(
        selection,
        getattr(args, "max_memory", None),
        console=getattr(args, "console_ui", None),
    )
    child.admission = schedule
    child.open()
    return child


def install_runtime(state: dict[str, Any], args: Any, selection: Any) -> None:
    """Switch the resident runtime to ``selection``. The same process stays."""
    if (
        state.get("runtime") is not None
        and state.get("record") == selection.folder
        and state.get("pipeline") == selection.pipeline
    ):
        return
    from favlo.listener import _log
    from favlo.memory import memory

    old = state.get("runtime")
    previous = state.get("memory_owners")
    state["runtime"] = None
    state["record"] = None
    state["pipeline"] = None
    state["module"] = None
    state["memory_owners"] = None
    pipelines.retire(old, previous)
    ui = getattr(args, "console_ui", None)
    if ui is not None:
        ui.pipeline = selection.pipeline
        ui.load_detail = selection.alias or selection.folder
    schedule = admission(state)
    before = memory.owners()
    try:
        child = start_pipeline(selection, args, schedule)
    except Exception:
        _sweep_new(before)
        raise

    def _crashed() -> None:
        if state.get("runtime") is child:
            drop_runtime(state, args)

    child.on_crash = _crashed
    memory.report(getattr(child, "counts", {}) or {})
    state["runtime"] = child
    state["memory_owners"] = tuple(child.counts)
    pipelines.add(child, state["memory_owners"])
    state["record"] = selection.folder
    state["pipeline"] = selection.pipeline
    state["module"] = child.proxy
    shutdown = getattr(args, "shutdown", None)
    if shutdown is not None:
        shutdown.runtime = child
    if ui is not None and ui.keeps_lines:
        label = selection.alias or selection.folder
        _log(f"loaded {label} {selection.folder}")


def drop_runtime(state: dict[str, Any], args: Any) -> str | None:
    """Close the resident pipeline, sweep its owners, and leave the listener up."""
    from favlo.listener import _log

    old = state.get("runtime")
    owners = state.get("memory_owners")
    folder = state.get("record")
    state["runtime"] = None
    state["record"] = None
    state["pipeline"] = None
    state["module"] = None
    state["memory_owners"] = None
    shutdown = getattr(args, "shutdown", None)
    if shutdown is not None:
        shutdown.runtime = None
    pipelines.retire(old, owners)
    if isinstance(folder, str) and folder:
        _log(f"released {folder}")
        return folder
    return None


def _require_kind(kind: str, selection: Any, module: Any) -> None:
    name = _HANDLERS.get(kind)
    if name is not None and callable(getattr(module, name, None)):
        return
    message = f"{selection.pipeline} does not serve {kind}"
    raise LookupError(message)


def bind_request(body: dict[str, Any], state: dict[str, Any], args: Any, kind: str = "chat") -> Any:
    """Resolve the model and open the runtime that serves ``kind``."""
    if not getattr(args, "lazy", False):
        return None
    model = body.get("model")
    if not isinstance(model, str) or not model:
        message = "model is required"
        raise LookupError(message)
    from favlo.registry import resolve

    selection = resolve(model, resident=state.get("record"))
    module = pipeline_module(selection)
    _require_kind(kind, selection, module)
    install_runtime(state, args, selection)
    runtime = state.get("runtime")
    state["requests"] = int(state.get("requests") or 0) + 1
    if runtime is not None:
        runtime._request_id = state["requests"]
    return selection


def _restore_console(ui: Any, saved: Any) -> None:
    if ui is None or saved is None:
        return
    ui.pipeline, ui.load_detail = saved


def open_embed(state: dict[str, Any], args: Any, selection: Any) -> Any:
    """Open an embeddings process beside the resident model. That model stays."""
    from favlo.listener import _log
    from favlo.memory import memory

    ui = getattr(args, "console_ui", None)
    saved = None
    if ui is not None:
        saved = (ui.pipeline, ui.load_detail)
        ui.pipeline = selection.pipeline
        ui.load_detail = selection.alias or selection.folder
    schedule = admission(state)
    before = memory.owners()
    try:
        child = start_pipeline(selection, args, schedule)
    except Exception:
        _sweep_new(before)
        _restore_console(ui, saved)
        raise
    child._embed_console = saved
    memory.report(getattr(child, "counts", {}) or {})
    pipelines.add(child, tuple(getattr(child, "counts", {}) or ()))
    if ui is not None and ui.keeps_lines:
        label = selection.alias or selection.folder
        _log(f"loaded {label} {selection.folder}")
    return child


def close_embed(child: Any, args: Any) -> None:
    """Close the embeddings process opened for one request and sweep its owners."""
    from favlo.listener import _log

    if child is None:
        return
    folder = getattr(getattr(child, "selection", None), "folder", None)
    ui = getattr(args, "console_ui", None)
    saved = getattr(child, "_embed_console", None)
    try:
        pipelines.retire(child, tuple(getattr(child, "counts", {}) or ()))
    finally:
        _restore_console(ui, saved)
    if isinstance(folder, str) and folder:
        _log(f"released {folder}")


def _open_embed_for(body: dict[str, Any], state: dict[str, Any], args: Any) -> Opened:
    model = body.get("model")
    if not isinstance(model, str) or not model.strip():
        from favlo.registry import embeddings_default

        model = embeddings_default()
        if not model:
            message = "no default embeddings model is registered"
            raise LookupError(message)
        body["model"] = model
    from favlo.registry import resolve

    selection = resolve(model, resident=state.get("record"))
    module = pipeline_module(selection)
    _require_kind("embed", selection, module)
    same = (
        state.get("runtime") is not None
        and state.get("record") == selection.folder
        and state.get("pipeline") == selection.pipeline
    )
    if same:
        opened = Opened(selection, state.get("runtime"), state.get("module"), drop_embed=True)
    else:
        child = open_embed(state, args, selection)
        opened = Opened(selection, child, child.proxy, embed_child=child)
    state["requests"] = int(state.get("requests") or 0) + 1
    if opened.runtime is not None:
        opened.runtime._request_id = state["requests"]
    return opened


def open_for(kind: str, body: dict[str, Any], state: dict[str, Any], args: Any) -> Opened:
    """Open the runtime that serves ``kind``. A different resident waits, then switches."""
    if kind == "embed" and getattr(args, "lazy", False):
        return _open_embed_for(body, state, args)
    selection = bind_request(body, state, args, kind=kind)
    return Opened(selection, state.get("runtime"), state.get("module"))


def handler_for(kind: str, module: Any) -> Any:
    """The handler on the runtime the manager opened. A missing kind stays missing."""
    name = _HANDLERS.get(kind)
    if name is None or module is None:
        return None
    try:
        call = getattr(module, name)
    except AttributeError:
        return None
    if not callable(call):
        return None
    return call


def admission(state: dict[str, Any]) -> Any:
    """The admission queues. The manager creates them on the first request."""
    from favlo.schedule import ensure

    return ensure(state)


def admit(state: dict[str, Any], kind: str, cancel: Any = None) -> Any:
    """Wait until ``kind`` is the request in service. A cleared waiter returns none."""
    from favlo.schedule import ClientClosed

    queue = "media" if kind in _MEDIA else kind
    try:
        return admission(state).acquire(queue, cancel=cancel)
    except ClientClosed:
        return None


def release_admission(state: dict[str, Any], ticket: Any) -> None:
    admission(state).release(ticket)


def admission_epoch(state: dict[str, Any]) -> int:
    return int(admission(state).epoch)


def foreground_busy(state: dict[str, Any]) -> bool:
    return bool(admission(state).foreground())


def run_job(task_id: str, work: Callable[[], Any]) -> Any:
    """Run one media task. The next task waits until this one returns."""
    global _running_job
    with _job_gate:
        _running_job = task_id
        try:
            return work()
        finally:
            if _running_job == task_id:
                _running_job = None


async def execute_job(task_id: str, work: Callable[[], Any]) -> Any:
    """The worker entry. The tools/call that queued the job has returned."""
    return await asyncio.to_thread(run_job, task_id, work)
