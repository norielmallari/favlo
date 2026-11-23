"""Open and close one int4 record. The canvas is kept."""

from __future__ import annotations

from pathlib import Path

from favlo.host import device_budget
from favlo.media_memory import begin_release, finish_release, open_gate, stamp_owner
from favlo.media_pipeline import MediaRequestError
from favlo.pipeline import Runtime
from klein_4b.cuda.constants import opening_bytes
from klein_4b.cuda.ops import Ops
from klein_4b.cuda.request import Request
from klein_4b.cuda.store import Store


class KleinRuntime(Runtime):
    def __init__(self, path: Path, owner: str | None) -> None:
        self.path = path
        self._owner_arg = owner
        self._owner = ""
        self.ops: Ops | None = None
        self.store: Store | None = None

    def ensure_open(self, request: Request) -> None:
        if self.ops is not None and self.store is not None:
            return
        self._owner = stamp_owner("klein_4b", self._owner_arg, id(self))
        self.store = Store(self.path)
        refs = 0
        for image in request.images:
            refs += (image.shape[0] // 16) * (image.shape[1] // 16)
        need = opening_bytes(request.height, request.width, refs, self.store.stages())
        budget = device_budget()
        if budget is not None and need > budget:
            message = "the canvas does not fit in device memory"
            raise MediaRequestError(message, "size")
        open_gate(need, detail="klein_4b")
        self.ops = Ops()

    def close(self) -> None:
        if self.ops is None and self.store is None:
            return
        before, held = begin_release(self._owner) if self._owner else (0, 0)
        if self.ops is not None:
            self.ops.close()
            self.ops = None
        if self.store is not None:
            self.store.close()
            self.store = None
        if self._owner:
            finish_release(self._owner, before, held)
