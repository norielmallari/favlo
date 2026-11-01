"""Resident expert layers for Gemma 4.

A full fit keeps every layer file. A short fit keeps as many files as the
budget allows after the resident record, the scratch, and the key/value
cache. A layer that is not resident is read into the oldest slot.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable

from favlo.memory import memory
from gemma4_26b_a4b.geometry import (
    GLOBAL_DIM,
    GLOBAL_HEADS,
    GLOBAL_LAYERS,
    LAYERS,
    LOCAL_DIM,
    LOCAL_HEADS,
)


def kv_bytes(capacity: int) -> int:
    """Float16 key and value bytes for ``capacity`` tokens."""
    if capacity <= 0:
        return 0
    local_layers = LAYERS - len(GLOBAL_LAYERS)
    local = local_layers * capacity * LOCAL_HEADS * LOCAL_DIM * 2 * 2
    glob = len(GLOBAL_LAYERS) * capacity * GLOBAL_HEADS * GLOBAL_DIM * 2 * 2
    return local + glob


def kv_margin(capacity: int) -> int:
    """One local cache buffer. Growth allocates it before the old one is released."""
    if capacity <= 0:
        return 0
    return capacity * LOCAL_HEADS * LOCAL_DIM * 2


def layer_slots(room: int, layer_bytes: int, layers: int) -> int:
    """How many layer files fit in ``room``. Zero means one file does not fit."""
    if layer_bytes <= 0 or layers <= 0 or room < layer_bytes:
        return 0
    return min(layers, room // layer_bytes)


class ExpertAtlas:
    """Layer files kept on device. ``slots`` is how many are resident."""

    def __init__(
        self,
        owner: str,
        root: Path,
        layer_bytes: int,
        limit: int,
        reader: Callable[[object, Path], None],
    ) -> None:
        if limit < 1:
            message = "the expert atlas does not fit the memory budget"
            raise RuntimeError(message)
        self.owner = owner
        self.root = root
        self.layer_bytes = layer_bytes
        self.limit = min(limit, LAYERS)
        self.complete = self.limit >= LAYERS
        self._reader = reader
        self._files: dict[int, object] = {}
        self._order: list[int] = []
        self._alloc: dict[object, int] = {}
        self.sizes: list[int] | None = None

    @property
    def slots(self) -> int:
        return len(self._files)

    @property
    def nbytes(self) -> int:
        return sum(self._alloc[handle] for handle in self._files.values())

    def load(self, note: Callable[[int], None]) -> str:
        """Read the resident layers. ``note`` receives each layer's byte count."""
        started = time.perf_counter()
        total = 0
        for layer in range(self.limit):
            handle, size = self._open_layer(layer)
            self._files[layer] = handle
            self._order.append(layer)
            total += size
            note(size)
        elapsed = time.perf_counter() - started
        rate = 0.0 if elapsed <= 0.0 else total / elapsed / 1e9
        return f"experts {total / 1e9:.2f} GB in {elapsed:.2f}s ({rate:.2f} GB/s)"

    def bind(self, session: Any, layer: int) -> object:
        """The buffer for ``layer``. A miss reuses the oldest slot."""
        found = self._files.get(layer)
        if found is not None:
            self._touch(layer)
            return found
        if len(self._files) < self.limit:
            handle, _size = self._open_layer(layer)
            self._files[layer] = handle
            self._order.append(layer)
            return handle
        session.flush()
        old = self._order.pop(0)
        handle = self._files.pop(old)
        size = self._file_size(layer)
        if self._alloc.get(handle) != size:
            memory.release(handle)
            self._alloc.pop(handle, None)
            handle = memory.allocate(size, owner=self.owner)
            self._alloc[handle] = size
        self._read(handle, layer)
        self._files[layer] = handle
        self._order.append(layer)
        return handle

    def retune(self, session: Any, room: int) -> None:
        """Drop layer files until ``room`` holds what remains. A full fit stays."""
        limit = layer_slots(room, self.layer_bytes, LAYERS)
        if limit < 1:
            message = "the expert atlas does not fit the memory budget"
            raise RuntimeError(message)
        if self.sizes and room >= max(self.sizes) and room >= sum(self.sizes):
            limit = min(LAYERS, len(self.sizes))
        if self.complete and limit >= LAYERS:
            return
        if limit >= len(self._order):
            self.limit = limit
            return
        session.flush()
        while len(self._order) > limit:
            layer = self._order.pop(0)
            handle = self._files.pop(layer)
            self._alloc.pop(handle, None)
            memory.release(handle)
        self.limit = limit
        self.complete = False

    def _file_size(self, layer: int) -> int:
        path = self.root / "experts" / f"{layer}.bin"
        size = path.stat().st_size
        if size <= 0 or size > self.layer_bytes:
            message = "expert layer is larger than the atlas slot"
            raise RuntimeError(message)
        return size

    def _open_layer(self, layer: int) -> tuple[object, int]:
        size = self._file_size(layer)
        handle = memory.allocate(size, owner=self.owner)
        self._alloc[handle] = size
        self._read(handle, layer)
        return handle, size

    def _read(self, handle: object, layer: int) -> None:
        path = self.root / "experts" / f"{layer}.bin"
        self._reader(handle, path)

    def _touch(self, layer: int) -> None:
        if layer in self._order:
            self._order.remove(layer)
        self._order.append(layer)
