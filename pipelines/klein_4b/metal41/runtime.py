"""Runtime shell. ``edit`` runs the forward on the packed record."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from favlo.media_memory import begin_release, finish_release, open_gate, stamp_owner
from favlo.memory import memory

from .constants import FORMAT, SCRATCH_BYTES
from .store import Store


class Runtime:
    """One packed record. ``close`` sweeps the owner."""

    def __init__(self, path: Path, console: Any = None, owner: str | None = None) -> None:
        self.path = Path(path)
        self.console = console
        self._owner = stamp_owner("klein_4b", owner, id(self))
        self._store: Store | None = None
        self._gpu: Any = None
        self._open = False

    @property
    def store(self) -> Store:
        if self._store is None:
            message = "runtime is not open"
            raise RuntimeError(message)
        return self._store

    def gpu(self) -> Any:
        if self._gpu is None:
            from .metal import Gpu

            self._gpu = Gpu(self._owner)
            if self._store is not None:
                self._gpu.bind_store(self._store)
        return self._gpu

    def ensure_open(self, request: Any) -> None:
        """Open the gate. Weights upload on the first projection."""
        del request
        if self._open and self._store is not None:
            return
        manifest = self.path / "manifest.json"
        if not manifest.is_file():
            message = f"{self.path.name} has no manifest.json"
            raise FileNotFoundError(message)
        body = json.loads(manifest.read_text(encoding="utf-8"))
        if body.get("format") != FORMAT or body.get("complete") is not True:
            message = f"{self.path.name} is not a finished {FORMAT} record"
            raise RuntimeError(message)
        self._store = Store(self.path)
        open_gate(self._need(body), detail=self.path.name)
        self._open = True

    def edit(self, request: Any, hold: Any, stop: Any, job: dict[str, Any] | None) -> Any:
        if not self._open:
            self.ensure_open(request)
        from .generate import edit as run_edit

        return run_edit(self, request, hold, stop, job)

    def close(self) -> None:
        before, held = begin_release(self._owner)
        gpu = self._gpu
        self._gpu = None
        if gpu is not None:
            gpu.close()
        if self._store is not None:
            self._store.close()
            self._store = None
        self._open = False
        finish_release(self._owner, before, held)

    def _need(self, body: dict[str, Any]) -> int:
        stages = body.get("stage_bytes")
        if isinstance(stages, dict):
            text = int(stages.get("text", 0) or 0)
            dit = int(stages.get("dit", 0) or 0)
            vae = int(stages.get("vae", 0) or 0)
            return max(text, dit) + vae + SCRATCH_BYTES
        nbytes = body.get("nbytes")
        if isinstance(nbytes, int):
            return nbytes + SCRATCH_BYTES
        return 6_000_000_000


def open_runtime(
    path: Path,
    *,
    kv_cache: str = "int8",
    console: Any = None,
    resident_percent: float | None = None,
    max_memory: str | int | None = None,
    owner: str | None = None,
) -> Runtime:
    del kv_cache, resident_percent
    memory.set_max_memory(max_memory)
    return Runtime(Path(path), console, owner=owner)
