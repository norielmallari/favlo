"""One pipeline process.

The pipeline manager spawns this module and keeps the registry, the job queue,
and the memory ledger. This process loads the pipeline and creates the Metal device.
Its console events are forwarded to the daemon, which writes the telemetry
the CLI paints. Before this process exits it sweeps its owners and blocks
until the ledger says those heaps are gone. When the process exits, including
a clean close, the daemon blocks the next call until the pages that process
wired are back, then drops the ledger entry. A crash also fails the job.
"""

from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any

_HOLDS = ("image_hold", "video_hold", "music_hold", "voice_hold")
_KINDS = {
    "chat": "handle_chat",
    "embed": "handle_embed",
    "image": "handle_image",
    "video": "handle_video",
    "music": "handle_music",
    "voice": "handle_voice",
}
_RUNTIME: list[Any] = []


def _served_kinds(module: Any, stub: bool) -> list[str]:
    if stub:
        return list(_KINDS)
    return [kind for kind, name in _KINDS.items() if callable(getattr(module, name, None))]


class ChildExited(RuntimeError):
    """The pipeline process ended before the call returned."""


class _Proxy:
    """Handlers the listener calls. Each call is one message to the child."""

    _kinds = {name: kind for kind, name in _KINDS.items()}

    def __init__(self, child: PipelineChild) -> None:
        self._child = child

    def __getattr__(self, name: str) -> Any:
        kind = self._kinds.get(name)
        if kind is None or kind not in self._child.kinds:
            message = f"{type(self).__name__} has no {name}"
            raise AttributeError(message)

        def handler(
            reply: Any,
            runtime: Any,
            body: Any,
            selection: Any,
            args: Any,
            state: Any,
        ) -> None:
            del runtime, selection, args
            payload = body if isinstance(body, dict) else {}
            self._child.call(kind, reply, payload, state)

        return handler

    def idle_heartbeat(self, runtime: Any, selection: Any) -> None:
        del runtime, selection
        self._child.heartbeat()


class PipelineChild:
    """A spawned pipeline process. ``halt`` and ``close`` match an in-process runtime."""

    def __init__(
        self,
        selection: Any,
        max_memory: str | None,
        *,
        console: Any = None,
        stub: bool = False,
    ) -> None:
        self.selection = selection
        self.max_memory = max_memory
        self.stub = stub
        self._ui = console
        self.counts: dict[str, int] = {}
        self.kinds: set[str] = set()
        self.proxy = _Proxy(self)
        self.on_crash: Any = None
        self._proc: subprocess.Popen[bytes] | None = None
        self._control: int | None = None
        self._baseline: int | None = None
        self._while_resident: int | None = None
        self._release_from: int | None = None
        self._held = 0
        self._messages: queue.Queue[dict[str, Any]] = queue.Queue()
        self._write_lock = threading.Lock()
        self._shut = False
        self._forgotten = False
        self._released = threading.Event()
        self._busy = False
        self.admission: Any = None

    def open(self) -> None:
        """Spawn the process and wait until it has reported its owners."""
        self._hold_memory()
        try:
            self._start()
            self._send(
                {
                    "op": "open",
                    "selection": _selection_payload(self.selection),
                    "max_memory": self.max_memory,
                }
            )
            message = self._next({"opened", "error", "gone"})
            if message["op"] != "opened":
                self.close()
                detail = message.get("message") or "pipeline process failed"
                raise RuntimeError(str(detail))
            counts = message.get("owners")
            if isinstance(counts, dict):
                self.counts = {str(key): int(value) for key, value in counts.items()}
            else:
                self.counts = {}
            reported = message.get("kinds")
            if isinstance(reported, list):
                self.kinds = {str(item) for item in reported}
            self._note_resident()
        finally:
            self._release_memory()

    def call(self, kind: str, reply: Any, body: dict[str, Any], state: dict[str, Any]) -> None:
        """Run one handler in the child and copy its reply onto ``reply``."""
        proc = self._proc
        if proc is None or proc.poll() is not None:
            raise ChildExited("pipeline process is gone")
        stop = threading.Event()
        watcher = threading.Thread(
            target=self._watch,
            args=(reply, state, stop),
            name="favlo-pipeline-watch",
            daemon=True,
        )
        self._busy = True
        watcher.start()
        streamed = False
        try:
            self._send(
                {
                    "op": "call",
                    "kind": kind,
                    "body": body,
                    "homes": _homes(state),
                    "holds": _hold_paths(state),
                    "preempt": _flagged(state, "preempt"),
                    "disconnect": bool(callable(getattr(reply, "stopped", None)) and reply.stopped()),
                }
            )
            while True:
                message = self._next({"result", "event", "stream", "done", "error", "gone"})
                op = message["op"]
                if op == "stream":
                    streamed = True
                    try:
                        reply.begin_events()
                    except OSError:
                        pass
                elif op == "event":
                    payload = message.get("payload")
                    if isinstance(payload, dict):
                        try:
                            reply.event(payload)
                        except OSError:
                            pass
                elif op == "done":
                    reply.done()
                elif op == "gone":
                    raise ChildExited("pipeline process exited")
                elif op == "error":
                    detail = message.get("message") or "pipeline process failed"
                    raise RuntimeError(str(detail))
                else:
                    _apply_holds(state, message.get("holds"))
                    _apply_tables(state, message.get("tables"))
                    payload = message.get("payload")
                    if not streamed and isinstance(payload, dict):
                        status = message.get("status")
                        reply.json(int(status) if isinstance(status, int) else 200, payload)
                    return
        finally:
            self._busy = False
            stop.set()

    def heartbeat(self) -> None:
        if self._busy or self._shut or self._proc is None or self._proc.poll() is not None:
            return
        try:
            self._send({"op": "heartbeat"})
        except ChildExited:
            return

    def halt(self) -> None:
        self._signal("halt")

    def close(self) -> None:
        """Ask the process to exit, then drop its owners once the ledger is clear."""
        from favlo.memory import reclaimable_bytes

        self._hold_memory()
        try:
            with self._write_lock:
                already = self._shut
                self._shut = True
                if self._release_from is None:
                    self._release_from = reclaimable_bytes()
            if not already:
                self._signal("halt")
                try:
                    self._send({"op": "close"})
                except ChildExited:
                    pass
                proc = self._proc
                if proc is not None and proc.poll() is None:
                    proc.wait()
            self._forget()
            self._released.wait()
        finally:
            self._release_memory()

    def _start(self) -> None:
        from favlo.memory import reclaimable_bytes

        self._baseline = reclaimable_bytes()
        read_fd, write_fd = os.pipe()
        os.set_inheritable(read_fd, True)
        env = dict(os.environ)
        env["FAVLO_CONTROL_FD"] = str(read_fd)
        if self.stub:
            env["FAVLO_WORKER_STUB"] = "1"
        proc = subprocess.Popen(
            [sys.executable, "-m", "favlo.pipeline_worker"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            pass_fds=(read_fd,),
            start_new_session=True,
        )
        os.close(read_fd)
        self._proc = proc
        self._control = write_fd
        threading.Thread(target=self._read_stdout, name="favlo-pipeline-read", daemon=True).start()
        threading.Thread(target=self._reap, name="favlo-pipeline-reap", daemon=True).start()
        if proc.stderr is not None:
            threading.Thread(
                target=_drain,
                args=(proc.stderr,),
                name="favlo-pipeline-log",
                daemon=True,
            ).start()

    def _send(self, message: dict[str, Any]) -> None:
        proc = self._proc
        if proc is None or proc.stdin is None:
            raise ChildExited("pipeline process is gone")
        blob = (json.dumps(message) + "\n").encode()
        with self._write_lock:
            try:
                proc.stdin.write(blob)
                proc.stdin.flush()
            except (BrokenPipeError, OSError) as exc:
                raise ChildExited("pipeline process is gone") from exc

    def _signal(self, name: str) -> None:
        fd = self._control
        if fd is None:
            return
        blob = (json.dumps({"op": "signal", "name": name}) + "\n").encode()
        with self._write_lock:
            try:
                os.write(fd, blob)
            except OSError:
                return

    def _watch(self, reply: Any, state: dict[str, Any], stop: threading.Event) -> None:
        sent: set[str] = set()
        while not stop.is_set():
            if _flagged(state, "preempt"):
                self._signal("preempt")
            if "cancel" not in sent and _flagged(state, "cancel"):
                self._signal("cancel")
                sent.add("cancel")
            stopped = getattr(reply, "stopped", None)
            if callable(stopped) and stopped():
                self._signal("disconnect")
            stop.wait(0.05)

    def _read_stdout(self) -> None:
        proc = self._proc
        if proc is None or proc.stdout is None:
            return
        for line in proc.stdout:
            text = line.decode("utf-8", errors="replace").strip()
            if not text:
                continue
            try:
                message = json.loads(text)
            except json.JSONDecodeError:
                _log_line(text)
                continue
            if isinstance(message, dict) and message.get("op") == "console":
                self._apply_console(message.get("event"))
                continue
            if isinstance(message, dict):
                self._messages.put(message)
        self._messages.put({"op": "gone"})

    def _apply_console(self, event: Any) -> None:
        apply = getattr(self._ui, "apply_event", None)
        if isinstance(event, dict) and callable(apply):
            apply(event)

    def _reap(self) -> None:
        proc = self._proc
        if proc is None:
            return
        proc.wait()
        with self._write_lock:
            crashed = not self._shut
            self._shut = True
        self._forget()
        if crashed:
            hook = self.on_crash
            if callable(hook):
                hook()

    def _next(self, ops: set[str]) -> dict[str, Any]:
        message = self._messages.get()
        while message.get("op") not in ops:
            message = self._messages.get()
        return message

    def _note_resident(self) -> None:
        """Bytes this child took out of the reclaimable count."""
        from favlo.memory import reclaimable_bytes

        current = reclaimable_bytes()
        self._while_resident = current
        dropped = 0 if self._baseline is None else max(0, self._baseline - current)
        self._held = max(dropped, sum(self.counts.values()))

    def _hold_memory(self) -> None:
        hold = getattr(self.admission, "hold_memory", None)
        if callable(hold):
            hold()

    def _release_memory(self) -> None:
        release = getattr(self.admission, "release_memory", None)
        if callable(release):
            release()

    def _forget(self) -> None:
        with self._write_lock:
            if self._forgotten:
                return
            self._forgotten = True
            # Wait for the kernel gap measured while the process was resident.
            # The ledger's byte count can be larger than the pages the kernel
            # wired, and a count taken at close can already include a release.
            before = self._while_resident
            baseline = self._baseline
            if before is not None and baseline is not None:
                held = max(0, baseline - before)
            else:
                held = self._held
                if before is None:
                    before = self._release_from
            owners = tuple(self.counts)
            proc = self._proc
        from favlo.memory import memory

        self._hold_memory()
        try:
            code = proc.returncode if proc is not None else None
            if before is not None and code is not None:
                if not memory.confirm_released(before, held):
                    _log_line("reclaimable pages stopped short")
            memory.forget(owners)
        finally:
            self._release_memory()
            self._released.set()


def _homes(state: dict[str, Any]) -> dict[str, str]:
    found: dict[str, str] = {}
    for key in ("image_home", "video_home", "music_home", "voice_home"):
        value = state.get(key)
        if isinstance(value, Path):
            found[key] = str(value)
    return found


def _hold_paths(state: dict[str, Any]) -> dict[str, str]:
    found: dict[str, str] = {}
    for key in _HOLDS:
        value = state.get(key)
        if isinstance(value, Path):
            found[key] = str(value)
    return found


def _apply_holds(state: dict[str, Any], holds: Any) -> None:
    if not isinstance(holds, dict):
        return
    for key in _HOLDS:
        if key not in holds:
            continue
        value = holds.get(key)
        if isinstance(value, str) and value:
            state[key] = Path(value)
        else:
            state.pop(key, None)


def _apply_tables(state: dict[str, Any], tables: Any) -> None:
    if not isinstance(tables, dict):
        return
    for key in ("voice", "music", "videos"):
        value = tables.get(key)
        if isinstance(value, dict):
            state[key] = value


def _flagged(state: dict[str, Any], key: str) -> bool:
    flag = state.get(key)
    isset = getattr(flag, "is_set", None)
    return bool(callable(isset) and isset())


def _selection_payload(selection: Any) -> dict[str, Any]:
    parts = getattr(selection, "parts", None) or {}
    return {
        "folder": selection.folder,
        "path": str(selection.path),
        "pipeline": selection.pipeline,
        "alias": selection.alias,
        "requested_id": selection.requested_id,
        "parent": selection.parent,
        "variant": selection.variant,
        "sampling": selection.sampling,
        "kv_cache": selection.kv_cache,
        "resident_percent": selection.resident_percent,
        "max_prompt_tokens": selection.max_prompt_tokens,
        "context_length": selection.context_length,
        "variants": selection.variants,
        "default_variant": selection.default_variant,
        "runtime": selection.runtime,
        "custom": selection.custom,
        "parts": {str(key): str(path) for key, path in parts.items()},
    }


def _log_line(text: str) -> None:
    from favlo.listener import _log

    _log(text)


def _drain(stream: Any) -> None:
    for line in stream:
        text = line.decode("utf-8", errors="replace").rstrip()
        if text:
            _log_line(text)


def main() -> None:
    """Read one command stream. Stdout is the protocol. Prints go to stderr."""
    proto_fd = os.dup(1)
    os.dup2(2, 1)
    proto = os.fdopen(proto_fd, "wb", buffering=0)
    control = _control_events()
    runtime: Any = None
    module: Any = None
    selection: Any = None
    max_memory: str | None = None
    stub = os.environ.get("FAVLO_WORKER_STUB") == "1"
    console = _pipeline_console(proto)
    try:
        _serve_child(proto, control, runtime, module, selection, max_memory, stub, console)
    finally:
        _release_process()


def _serve_child(
    proto: Any,
    control: dict[str, threading.Event],
    runtime: Any,
    module: Any,
    selection: Any,
    max_memory: str | None,
    stub: bool,
    console: Any,
) -> None:
    for line in sys.stdin.buffer:
        try:
            message = json.loads(line.decode("utf-8"))
        except json.JSONDecodeError:
            continue
        if not isinstance(message, dict):
            continue
        op = message.get("op")
        try:
            if op == "open":
                runtime, module, selection, max_memory = _open(message, stub, console)
                _RUNTIME[:] = [runtime]
                _emit(
                    proto,
                    {
                        "op": "opened",
                        "owners": _owners(runtime),
                        "kinds": _served_kinds(module, stub),
                    },
                )
            elif op == "call":
                _call(
                    proto,
                    runtime,
                    module,
                    selection,
                    max_memory,
                    message,
                    control,
                    stub,
                    console,
                )
            elif op == "heartbeat":
                pump = getattr(module, "idle_heartbeat", None) if module is not None else None
                if callable(pump):
                    pump(runtime, selection)
            elif op == "close":
                _shutdown(runtime)
                console.close()
                break
        except Exception as exc:
            console.close()
            _emit(proto, {"op": "error", "message": str(exc), "type": type(exc).__name__})
            if op == "open":
                break


def _control_events() -> dict[str, threading.Event]:
    events = {
        "preempt": threading.Event(),
        "cancel": threading.Event(),
        "disconnect": threading.Event(),
        "halt": threading.Event(),
    }
    raw = os.environ.get("FAVLO_CONTROL_FD")
    if raw is None or not raw.isdigit():
        return events

    def read() -> None:
        stream = os.fdopen(int(raw), "rb", buffering=0)
        pending = b""
        while True:
            chunk = stream.read(4096)
            if not chunk:
                return
            pending += chunk
            while b"\n" in pending:
                line, pending = pending.split(b"\n", 1)
                _apply_signal(line, events)

    threading.Thread(target=read, name="favlo-pipeline-control", daemon=True).start()
    return events


def _apply_signal(line: bytes, events: dict[str, threading.Event]) -> None:
    try:
        message = json.loads(line.decode("utf-8"))
    except json.JSONDecodeError:
        return
    if not isinstance(message, dict):
        return
    name = message.get("name")
    if name == "close":
        events["halt"].set()
        _halt_runtime()
        return
    flag = events.get(name) if isinstance(name, str) else None
    if flag is not None:
        flag.set()
    if name == "halt":
        events["cancel"].set()
        _halt_runtime()


def _halt_runtime() -> None:
    runtime = _RUNTIME[0] if _RUNTIME else None
    halt = getattr(runtime, "halt", None)
    if callable(halt):
        halt()


def _open(
    message: dict[str, Any], stub: bool, console: Any
) -> tuple[Any, Any, Any, str | None]:
    if stub:
        return SimpleNamespace(halt=lambda: None, close=lambda: None), None, None, None
    from favlo.pipelines import open_runtime, pipeline_module
    from favlo.registry import Selection

    raw = message.get("selection")
    if not isinstance(raw, dict):
        message_text = "pipeline selection is missing"
        raise RuntimeError(message_text)
    parts = raw.get("parts") if isinstance(raw.get("parts"), dict) else {}
    max_memory = message.get("max_memory")
    limit = max_memory if isinstance(max_memory, str) else None
    resident = raw.get("resident_percent")
    prompt_limit = raw.get("max_prompt_tokens")
    variant = raw.get("default_variant")
    selection = Selection(
        folder=str(raw.get("folder") or ""),
        path=Path(str(raw.get("path") or "")),
        pipeline=str(raw.get("pipeline") or ""),
        alias=raw.get("alias") if isinstance(raw.get("alias"), str) else None,
        requested_id=str(raw.get("requested_id") or ""),
        parent=str(raw.get("parent") or ""),
        variant=raw.get("variant") if isinstance(raw.get("variant"), str) else None,
        sampling=raw.get("sampling") if isinstance(raw.get("sampling"), dict) else None,
        kv_cache=str(raw.get("kv_cache") or "e4m3"),
        resident_percent=resident if isinstance(resident, int | float) else None,
        max_prompt_tokens=prompt_limit if isinstance(prompt_limit, int) else None,
        context_length=int(raw.get("context_length") or 131072),
        variants=raw.get("variants") if isinstance(raw.get("variants"), dict) else None,
        default_variant=variant if isinstance(variant, str) else None,
        runtime=str(raw.get("runtime") or ""),
        custom=bool(raw.get("custom")),
        parts={str(key): Path(str(path)) for key, path in parts.items()},
    )
    module = pipeline_module(selection)
    if console is not None:
        console.pipeline = selection.pipeline
        console.load_detail = selection.alias or selection.folder
    runtime = open_runtime(selection, console, limit)
    return runtime, module, selection, limit


def _owners(runtime: Any) -> dict[str, int]:
    if os.environ.get("FAVLO_WORKER_STUB") == "1":
        return {"stub": 32}
    from favlo.memory import memory
    from favlo.pipelines import opening_owners

    counts = memory.owner_bytes()
    for owner in opening_owners(frozenset(), runtime):
        counts.setdefault(owner, 0)
    return counts


def _call(
    proto: Any,
    runtime: Any,
    module: Any,
    selection: Any,
    max_memory: str | None,
    message: dict[str, Any],
    control: dict[str, threading.Event],
    stub: bool,
    console: Any,
) -> None:
    body = message.get("body") if isinstance(message.get("body"), dict) else {}
    if stub:
        if body.get("crash"):
            os._exit(1)
        _emit(
            proto,
            {"op": "result", "status": 200, "payload": {"status": "completed", "stub": True}},
        )
        return
    handler_name = _KINDS.get(message.get("kind")) if isinstance(message.get("kind"), str) else None
    call = getattr(module, handler_name, None) if handler_name else None
    if not callable(call):
        text = "the pipeline does not handle this request"
        raise RuntimeError(text)
    for flag in control.values():
        flag.clear()
    if message.get("preempt"):
        control["preempt"].set()
    if message.get("disconnect"):
        control["disconnect"].set()
    if control["halt"].is_set():
        _halt_runtime()
    state = _child_state(message, control)
    reply = _ChildReply(proto, control["disconnect"])
    args = SimpleNamespace(max_memory=max_memory, console_ui=console, shutdown=None, lazy=True)
    call(reply, runtime, body, selection, args, state)
    _emit(
        proto,
        {
            "op": "result",
            "status": reply.status,
            "payload": reply.payload,
            "holds": _export_holds(state),
            "tables": {
                key: state.get(key)
                for key in ("voice", "music", "videos")
                if isinstance(state.get(key), dict)
            },
        },
    )


def _child_state(message: dict[str, Any], control: dict[str, threading.Event]) -> dict[str, Any]:
    state: dict[str, Any] = {
        "preempt": control["preempt"],
        "cancel": control["cancel"],
        "lock": threading.Lock(),
    }
    homes = message.get("homes")
    if isinstance(homes, dict):
        for key, value in homes.items():
            if isinstance(key, str) and isinstance(value, str) and value:
                state[key] = Path(value)
    holds = message.get("holds")
    if isinstance(holds, dict):
        for key, value in holds.items():
            if key in _HOLDS and isinstance(value, str) and value:
                state[key] = _load_hold(Path(value))
    return state


def _export_holds(state: dict[str, Any]) -> dict[str, str | None]:
    import pickle

    found: dict[str, str | None] = {}
    root = Path(os.environ.get("TMPDIR") or "/tmp")
    for key in _HOLDS:
        if key not in state:
            continue
        value = state.get(key)
        if isinstance(value, Path):
            found[key] = str(value)
            continue
        if value is None:
            found[key] = None
            continue
        path = root / f"favlo-{key}-{os.getpid()}.pkl"
        path.write_bytes(pickle.dumps(value))
        found[key] = str(path)
    return found


def _load_hold(path: Path) -> Any:
    import pickle

    return pickle.loads(path.read_bytes())


def _shutdown(runtime: Any) -> None:
    close = getattr(runtime, "close", None)
    if callable(close):
        close()


def _release_process() -> None:
    """Sweep every owner and block until the ledger says the heaps are gone."""
    from favlo.memory import memory

    for owner in tuple(memory.owners()):
        memory.sweep(owner)
    memory.wait_released()


_WRITE = threading.Lock()


def _emit(proto: Any, message: dict[str, Any]) -> None:
    blob = (json.dumps(message) + "\n").encode()
    with _WRITE:
        proto.write(blob)


def _pipeline_console(proto: Any) -> Any:
    """JSON console whose events travel on the protocol stream."""
    from favlo.console import Console

    ui = Console("json", pipeline="", stream=_ProtocolStream(proto), start_thread=True)
    ui.live_decode = True
    return ui


class _ProtocolStream:
    """Text stream that wraps each console line as a protocol message."""

    def __init__(self, proto: Any) -> None:
        self._proto = proto
        self._pending = ""

    def write(self, text: str) -> int:
        self._pending += text
        while "\n" in self._pending:
            line, self._pending = self._pending.split("\n", 1)
            _emit_console(self._proto, line)
        return len(text)

    def flush(self) -> None:
        return None


def _emit_console(proto: Any, line: str) -> None:
    if not line:
        return
    try:
        event = json.loads(line)
    except json.JSONDecodeError:
        event = {"type": "log", "line": line}
    if isinstance(event, dict):
        _emit(proto, {"op": "console", "event": event})


class _ChildReply:
    """A reply that writes protocol lines. The daemon copies them onto the HTTP reply."""

    def __init__(self, proto: Any, disconnected: threading.Event) -> None:
        self.model_id = ""
        self.started = False
        self.status = 200
        self.payload: dict[str, Any] | None = None
        self._proto = proto
        self._disconnected = disconnected
        self._stream = False

    def json(self, status: int, payload: dict[str, Any]) -> None:
        if self._stream:
            return
        self.started = True
        self.status = status
        self.payload = payload

    def begin_events(self) -> None:
        if self._disconnected.is_set():
            text = "client closed"
            raise OSError(text)
        self.started = True
        self._stream = True
        _emit(self._proto, {"op": "stream"})

    def event(self, payload: dict[str, Any]) -> None:
        if self._disconnected.is_set():
            text = "client closed"
            raise OSError(text)
        _emit(self._proto, {"op": "event", "payload": payload})

    def done(self) -> None:
        _emit(self._proto, {"op": "done"})

    def stopped(self) -> bool:
        return self._disconnected.is_set()


if __name__ == "__main__":
    main()
