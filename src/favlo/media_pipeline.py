"""Listener contract for one media request.

A served image, video, music, or voice pipeline inherits ``MediaPipeline``.
The class reads cancel and preempt, stores the host checkpoint, and writes
the job dict the daemon already reads. The subclass parses the body and
runs the forward. The class does not name stages or dispatch a kernel.
"""

from __future__ import annotations

import secrets
import time
from abc import abstractmethod
from pathlib import Path
from typing import Any

from favlo.pipeline import Pipeline


class MediaPaused(Exception):
    """The request stopped between stages. The job stays in progress."""

    def __init__(self, reason: str, hold: Any) -> None:
        super().__init__(reason)
        self.reason = reason
        self.hold = hold


class MediaRequestError(ValueError):
    """The body cannot be run. The reply is a 400."""

    def __init__(self, message: str, param: str | None = None) -> None:
        super().__init__(message)
        self.param = param


class MediaCall:
    """One invocation. The subclass hangs its progress row here."""

    def __init__(self) -> None:
        self.progress: Any = None
        self.home: Path | None = None
        self.table: dict[str, Any] = {}


class MediaPipeline(Pipeline):
    """What the listener calls for one media kind.

    The module binds ``handle_<kind> = Pipeline()``. ``__call__`` is
    ``handle``, which is the signature the listener already looks up.
    """

    kind: str
    id_prefix: str
    object_name: str
    table_name: str | None = None
    id_hex: int = 12
    seeds: bool = True

    def handle(
        self,
        reply: Any,
        runtime: Any,
        body: dict[str, Any],
        selection: Any,
        args: Any,
        state: dict[str, Any],
    ) -> None:
        """Parse, resume, generate, then pause, fail, or complete."""
        del args
        call = MediaCall()
        try:
            self._run(reply, runtime, body, selection, state, call)
        finally:
            self._hook(self.on_finish, runtime, call)

    def stop(self, state: dict[str, Any], runtime: Any) -> str | None:
        """``cancel`` beats preempt. ``halted`` counts as cancel."""
        cancel = state.get("cancel")
        if getattr(cancel, "is_set", lambda: False)():
            return "cancel"
        if self.halted(runtime):
            return "cancel"
        preempt = state.get("preempt")
        if getattr(preempt, "is_set", lambda: False)():
            return "preempt"
        return None

    def halted(self, runtime: Any) -> bool:
        """True when the runtime was asked to stop from outside the event."""
        del runtime
        return False

    def identity(self, request: Any) -> Any:
        """A value stored beside the hold. The default matches every resume."""
        del request
        return None

    def matches(self, request: Any, identity: Any) -> bool:
        """False drops the stored hold and starts a new job."""
        del request, identity
        return True

    def working(self, request: Any, job: dict[str, Any], hold: Any) -> dict[str, Any] | None:
        """A replacement in-progress payload. ``None`` keeps ``job``."""
        del request, job, hold
        return None

    def describe(self, request: Any) -> dict[str, Any]:
        """Extra keys merged into the job dict."""
        del request
        return {}

    def on_pause(self, runtime: Any, reason: str, call: MediaCall) -> None:
        del runtime, reason, call

    def on_fail(self, runtime: Any, exc: BaseException, call: MediaCall) -> None:
        del runtime, exc, call

    def on_complete(self, runtime: Any, call: MediaCall) -> None:
        del runtime, call

    def on_finish(self, runtime: Any, call: MediaCall) -> None:
        del runtime, call

    @abstractmethod
    def parse(self, body: dict[str, Any]) -> Any:
        """The pipeline's request fields."""

    @abstractmethod
    def generate(
        self,
        runtime: Any,
        request: Any,
        hold: Any,
        job: dict[str, Any],
        stop: Any,
        call: MediaCall,
    ) -> Any:
        """The forward. Raise ``MediaPaused`` to yield. Mutate ``job`` progress."""

    @abstractmethod
    def pack(
        self,
        runtime: Any,
        request: Any,
        result: Any,
        job: dict[str, Any],
        home: Path | None,
        table: dict[str, Any],
        call: MediaCall,
    ) -> dict[str, Any] | None:
        """Write the file. Return a replacement payload, or ``None`` to keep ``job``."""

    def _run(
        self,
        reply: Any,
        runtime: Any,
        body: dict[str, Any],
        selection: Any,
        state: dict[str, Any],
        call: MediaCall,
    ) -> None:
        self._assign_seed(body, state)
        try:
            request = self.parse(body)
        except MediaRequestError as exc:
            self.reject(reply, str(exc), exc.param)
            return
        record = self._load(state)
        if record is not None and not self.matches(request, record.get("identity")):
            self._clear(state)
            record = None
        if record is None:
            job = self._new_job(body, selection, request)
            hold = None
        else:
            job = record["job"]
            hold = record.get("hold")
        table = self._table(state)
        if isinstance(job, dict) and isinstance(job.get("id"), str):
            table[job["id"]] = job
        home = self._home(state)
        call.home = home
        call.table = table
        reply.json(200, job)
        stop = lambda: self.stop(state, runtime)
        try:
            result = self.generate(runtime, request, hold, job, stop, call)
        except MediaRequestError as exc:
            self._clear(state)
            self.reject(reply, str(exc), exc.param)
            return
        except MediaPaused as paused:
            self._paused(reply, runtime, request, job, state, paused, call)
            return
        except Exception as exc:
            self._failed(reply, runtime, job, state, exc, call)
            return
        try:
            replacement = self.pack(runtime, request, result, job, home, table, call)
        except MediaPaused as paused:
            self._paused(reply, runtime, request, job, state, paused, call)
            return
        except Exception as exc:
            self._failed(reply, runtime, job, state, exc, call)
            return
        self._clear(state)
        self._hook(self.on_complete, runtime, call)
        shown = job if replacement is None else replacement
        self._store_metadata(request, shown, job, home)
        if replacement is None:
            if isinstance(job, dict):
                job["completed_at"] = int(time.time())
                job["progress"] = 100
                job["eta"] = 0
                job["status"] = "completed"
            return
        reply.json(200, replacement)

    def _paused(
        self,
        reply: Any,
        runtime: Any,
        request: Any,
        job: dict[str, Any],
        state: dict[str, Any],
        paused: MediaPaused,
        call: MediaCall,
    ) -> None:
        if paused.reason == "cancel":
            self._clear(state)
        else:
            held: dict[str, Any] = {
                "job": job,
                "hold": paused.hold,
                "identity": self.identity(request),
            }
            seed = getattr(request, "seed", None)
            if isinstance(seed, int) and not isinstance(seed, bool):
                held["seed"] = seed
            state[self._hold_key] = held
        self._hook(self.on_pause, runtime, paused.reason, call)
        shown = self.working(request, job, paused.hold)
        if shown is not None:
            reply.json(200, shown)

    def _failed(
        self,
        reply: Any,
        runtime: Any,
        job: dict[str, Any],
        state: dict[str, Any],
        exc: BaseException,
        call: MediaCall,
    ) -> None:
        self._clear(state)
        if isinstance(job, dict):
            job["status"] = "failed"
            job["error"] = {"message": str(exc), "type": type(exc).__name__}
        self._report(runtime)
        self._hook(self.on_fail, runtime, exc, call)

    def _new_job(self, body: dict[str, Any], selection: Any, request: Any) -> dict[str, Any]:
        job = {
            "id": self.id_prefix + secrets.token_hex(self.id_hex),
            "object": self.object_name,
            "created_at": int(time.time()),
            "completed_at": None,
            "status": "in_progress",
            "model": self.model_id(body, selection),
            "progress": 0,
            "stage": None,
            "eta": None,
            "error": None,
        }
        extra = self.describe(request)
        if isinstance(extra, dict):
            job.update(extra)
        return job

    def _assign_seed(self, body: dict[str, Any], state: dict[str, Any]) -> None:
        """Draw a seed when the body has none, and reuse it after a pause."""
        if not self.seeds or _seed_present(body):
            return
        held = state.get(self._hold_key)
        if isinstance(held, dict):
            stored = held.get("seed")
            if isinstance(stored, int) and not isinstance(stored, bool):
                body["seed"] = stored
                return
        body["seed"] = secrets.randbits(31)

    def parameters(self, request: Any) -> dict[str, Any]:
        """JSON fields the forward used. Arrays and missing values stay out."""
        raw: dict[str, Any] = {}
        if isinstance(request, dict):
            raw.update(request)
        elif hasattr(request, "_asdict"):
            raw.update(request._asdict())
        elif hasattr(request, "__dataclass_fields__"):
            for key in request.__dataclass_fields__:
                raw[key] = getattr(request, key)
        seconds = getattr(request, "seconds", None)
        if "seconds" not in raw and seconds is not None:
            raw["seconds"] = seconds
        if self.kind == "voice" and isinstance(raw.get("text"), str):
            raw["input"] = raw["text"]
        found: dict[str, Any] = {}
        for key, value in raw.items():
            if not isinstance(key, str) or key.startswith("_"):
                continue
            encoded = _jsonable(value)
            if encoded is _SKIP:
                continue
            found[key] = encoded
        return found

    def _store_metadata(
        self,
        request: Any,
        payload: Any,
        job: dict[str, Any],
        home: Path | None,
    ) -> None:
        if not isinstance(payload, dict):
            return
        from favlo.logs import favlo_home
        from favlo.media_meta import save_media

        root = home if isinstance(home, Path) else favlo_home()
        fields = self.parameters(request)
        if isinstance(job, dict):
            for key in ("seconds", "size", "response_format"):
                if key in fields:
                    continue
                encoded = _jsonable(job.get(key))
                if encoded is _SKIP:
                    continue
                fields[key] = encoded
        model = ""
        if isinstance(job, dict) and isinstance(job.get("model"), str):
            model = job["model"]
        names = _media_names(payload)
        base = fields.get("seed")
        for index, name in enumerate(names):
            item = dict(fields)
            if isinstance(base, int) and not isinstance(base, bool) and index:
                item["seed"] = base + index
            try:
                save_media(root, self.kind, name, item, model)
            except ValueError:
                continue

    def _load(self, state: dict[str, Any]) -> dict[str, Any] | None:
        held = state.get(self._hold_key)
        if not isinstance(held, dict):
            return None
        job = held.get("job")
        if not isinstance(job, dict) or job.get("status") != "in_progress":
            self._clear(state)
            return None
        return held

    def _clear(self, state: dict[str, Any]) -> None:
        state[self._hold_key] = None

    def _table(self, state: dict[str, Any]) -> dict[str, Any]:
        name = self.table_name
        if not isinstance(name, str) or not name:
            return {}
        table = state.setdefault(name, {})
        if not isinstance(table, dict):
            table = {}
            state[name] = table
        return table

    def _home(self, state: dict[str, Any]) -> Path | None:
        home = state.get(f"{self.kind}_home")
        if isinstance(home, Path):
            return home
        return None

    @property
    def _hold_key(self) -> str:
        return f"{self.kind}_hold"

    def _report(self, runtime: Any) -> None:
        from favlo.listener import _report_failure

        _report_failure(getattr(runtime, "console", None))

    def _hook(self, fn: Any, *args: Any) -> None:
        try:
            fn(*args)
        except Exception:
            runtime = args[0] if args else None
            try:
                self._report(runtime)
            except Exception:
                return


def _seed_present(body: dict[str, Any]) -> bool:
    if "seed" not in body:
        return False
    value = body.get("seed")
    return value is not None and value != ""


def _media_names(payload: dict[str, Any]) -> list[str]:
    names: list[str] = []
    data = payload.get("data")
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict) and isinstance(item.get("id"), str) and item["id"]:
                names.append(item["id"])
    if names:
        return names
    item_id = payload.get("id")
    if isinstance(item_id, str) and item_id:
        return [item_id]
    return []


_SKIP = object()


def _jsonable(value: Any) -> Any:
    if value is None:
        return _SKIP
    if isinstance(value, bool) or isinstance(value, (str, int, float)):
        return value
    if isinstance(value, (bytes, bytearray, memoryview)):
        return _SKIP
    if type(value).__name__ == "ndarray":
        return _SKIP
    if isinstance(value, (list, tuple)):
        items = []
        for item in value:
            encoded = _jsonable(item)
            if encoded is _SKIP:
                return _SKIP
            items.append(encoded)
        return items
    if isinstance(value, dict):
        found = {}
        for key, item in value.items():
            encoded = _jsonable(item)
            if encoded is _SKIP:
                return _SKIP
            found[str(key)] = encoded
        return found
    return _SKIP
