"""Resident expert layers for Gemma 4.

A full fit keeps every layer file. A short fit keeps as many files as the
budget allows after the resident record, the scratch, and the key/value
cache. A layer that is not resident is read into the oldest slot.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable

from favlo.memory import device_buffer, memory
from gemma4_26b_a4b.metal41.geometry import (
    GLOBAL_DIM,
    GLOBAL_HEADS,
    GLOBAL_LAYERS,
    LAYERS,
    LOCAL_DIM,
    LOCAL_HEADS,
)


KV_MODES = ("half", "fp8_e4m3")
_ALIASES = {
    "e4m3": "fp8_e4m3",
    "fp8": "fp8_e4m3",
    "f16": "half",
    "fp16": "half",
}


def kv_mode(name: str | None) -> str:
    """``half`` or ``fp8_e4m3``. Empty means ``fp8_e4m3``.

    ``e4m3`` and ``fp8`` are ``fp8_e4m3``. ``f16`` and ``fp16`` are ``half``.
    """
    text = "fp8_e4m3" if name is None or name == "" else str(name)
    text = _ALIASES.get(text, text)
    if text not in KV_MODES:
        message = f"kv cache {name} is not half or fp8_e4m3"
        raise ValueError(message)
    return text


def plain_cache(mode: str | None) -> bool:
    """Raw Metal half. That mode skips the E4M3 record."""
    return kv_mode(mode) == "half"


def kv_record_bytes(dim: int, mode: str | None = "fp8_e4m3") -> int:
    """Bytes of one head at one position."""
    if plain_cache(mode):
        return dim * 2
    if dim % 32:
        message = f"fp8 key/value dim {dim} is not a multiple of 32"
        raise ValueError(message)
    return dim + dim // 32


def plane_bytes(capacity: int, heads: int, dim: int, mode: str | None = "fp8_e4m3") -> int:
    """One key plane or one value plane."""
    if capacity <= 0 or heads <= 0 or dim <= 0:
        return 0
    return capacity * heads * kv_record_bytes(dim, mode)


def kv_bytes(capacity: int, mode: str | None = "fp8_e4m3") -> int:
    """Key and value bytes for ``capacity`` tokens."""
    if capacity <= 0:
        return 0
    local_layers = LAYERS - len(GLOBAL_LAYERS)
    local = local_layers * plane_bytes(capacity, LOCAL_HEADS, LOCAL_DIM, mode) * 2
    glob = len(GLOBAL_LAYERS) * plane_bytes(capacity, GLOBAL_HEADS, GLOBAL_DIM, mode) * 2
    return local + glob


def kv_margin(capacity: int, mode: str | None = "fp8_e4m3") -> int:
    """One local cache buffer. Growth allocates it before the old one is released."""
    return plane_bytes(capacity, LOCAL_HEADS, LOCAL_DIM, mode)


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
            handle = device_buffer(self.owner, size)
            self._alloc[handle] = size
        self._read(handle, layer)
        self._files[layer] = handle
        self._order.append(layer)
        return handle

    def admit(self, session: Any, room: int) -> None:
        """Load more layer files when ``room`` grew. A smaller room stays."""
        del session
        limit = self._fit_limit(room)
        if limit <= self.limit:
            return
        self.limit = limit
        for layer in range(LAYERS):
            if len(self._files) >= self.limit:
                break
            if layer in self._files:
                continue
            handle, _size = self._open_layer(layer)
            self._files[layer] = handle
            self._order.append(layer)
        if self.limit >= LAYERS and len(self._files) >= LAYERS:
            self.complete = True

    def retune(self, session: Any, room: int) -> None:
        """Drop layer files until ``room`` holds what remains. A full fit stays."""
        limit = self._fit_limit(room)
        if limit < 1:
            message = "the expert atlas does not fit the memory budget"
            raise RuntimeError(message)
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
        handle = device_buffer(self.owner, size)
        self._alloc[handle] = size
        self._read(handle, layer)
        return handle, size

    def _fit_limit(self, room: int) -> int:
        limit = layer_slots(room, self.layer_bytes, LAYERS)
        if self.sizes and room >= max(self.sizes) and room >= sum(self.sizes):
            return min(LAYERS, len(self.sizes))
        return limit

    def _read(self, handle: object, layer: int) -> None:
        path = self.root / "experts" / f"{layer}.bin"
        self._reader(handle, path)

    def _touch(self, layer: int) -> None:
        if layer in self._order:
            self._order.remove(layer)
        self._order.append(layer)
