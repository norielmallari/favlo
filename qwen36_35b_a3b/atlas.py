"""Resident expert atlas.

Language expert files stay on SSD. The runtime has a maximum byte size. The
atlas keeps as many whole layer files as fit under that maximum after the
resident record, the key/value cache, and the other buffers. A layer that does
not fit is read into the least recently used slot when that layer runs.
"""

from __future__ import annotations

import ctypes
import os
from pathlib import Path
from typing import Any

LAYERS = 40


def atlas_slots(max_memory: int, file_bytes: int, reserved: int, *, layers: int = LAYERS) -> int:
    """Layer files that fit in ``max_memory`` after ``reserved`` bytes."""
    if file_bytes < 1 or layers < 1 or max_memory < 1:
        message = "expert file is empty"
        raise ValueError(message)
    room = max_memory - reserved
    if room < file_bytes:
        message = (
            f"maximum memory {max_memory} bytes is below "
            f"{reserved + file_bytes} bytes for the runtime"
        )
        raise ValueError(message)
    return min(layers, room // file_bytes)


def available_bytes() -> int:
    """Reclaimable pages: free, inactive, and speculative."""
    libc = ctypes.CDLL("/usr/lib/libSystem.dylib", use_errno=True)

    class Stats(ctypes.Structure):
        _fields_ = [
            ("free_count", ctypes.c_uint),
            ("active_count", ctypes.c_uint),
            ("inactive_count", ctypes.c_uint),
            ("wire_count", ctypes.c_uint),
            ("zero_fill_count", ctypes.c_uint64),
            ("reactivations", ctypes.c_uint64),
            ("pageins", ctypes.c_uint64),
            ("pageouts", ctypes.c_uint64),
            ("faults", ctypes.c_uint64),
            ("cow_faults", ctypes.c_uint64),
            ("lookups", ctypes.c_uint64),
            ("hits", ctypes.c_uint64),
            ("purges", ctypes.c_uint64),
            ("purgeable_count", ctypes.c_uint),
            ("speculative_count", ctypes.c_uint),
        ]

    count = ctypes.c_uint(ctypes.sizeof(Stats) // 4)
    stats = Stats()
    kern = libc.host_statistics64(
        libc.mach_host_self(),
        ctypes.c_int(4),
        ctypes.byref(stats),
        ctypes.byref(count),
    )
    if kern != 0:
        page = os.sysconf("SC_PAGE_SIZE")
        pages = os.sysconf("SC_PHYS_PAGES")
        return int(page) * int(pages)
    page = int(os.sysconf("SC_PAGE_SIZE"))
    reclaimable = stats.free_count + stats.inactive_count + stats.speculative_count
    return int(reclaimable) * page


def _read_into(buffer: Any, path: Path, nbytes: int) -> None:
    view = memoryview(buffer.contents().as_buffer(nbytes))
    with path.open("rb") as handle:
        offset = 0
        while offset < nbytes:
            chunk = handle.read(8 << 20)
            if not chunk:
                break
            view[offset : offset + len(chunk)] = chunk
            offset += len(chunk)
    if offset != nbytes:
        message = f"{path.name} stopped at {offset} of {nbytes}"
        raise OSError(message)


class ExpertAtlas:
    """A pool of layer-file slots. ``buffer`` loads a miss before the caller encodes."""

    def __init__(
        self,
        session: Any,
        root: Path,
        *,
        max_memory: int,
        reserved: int,
    ) -> None:
        self.root = root
        first = root / "experts" / "0.bin"
        self.file_bytes = first.stat().st_size
        present = sum(1 for layer in range(LAYERS) if (root / "experts" / f"{layer}.bin").is_file())
        if present != LAYERS:
            message = f"{root.name} has {present} language expert files"
            raise FileNotFoundError(message)
        self.max_memory = max_memory
        wanted = atlas_slots(max_memory, self.file_bytes, reserved)
        self.buffers: list[Any] = []
        self.slots = wanted
        while self.slots >= 1:
            made = [session.allocate(self.file_bytes) for _ in range(self.slots)]
            if all(item is not None for item in made):
                self.buffers = made
                break
            self.slots -= 1
        if not self.buffers:
            message = f"expert atlas could not allocate one {self.file_bytes} byte slot"
            raise RuntimeError(message)
        self.where: dict[int, int] = {}
        self.layer_of = [-1] * self.slots
        self.last_use = [0] * self.slots
        self.clock = 1
        if self.slots == LAYERS:
            for layer in range(LAYERS):
                self._fill(layer, layer)

    def _fill(self, slot: int, layer: int) -> None:
        previous = self.layer_of[slot]
        if previous >= 0:
            self.where.pop(previous, None)
        path = self.root / "experts" / f"{layer}.bin"
        _read_into(self.buffers[slot], path, self.file_bytes)
        self.layer_of[slot] = layer
        self.where[layer] = slot
        self.last_use[slot] = self.clock
        self.clock += 1

    def cached(self, layer: int) -> Any | None:
        slot = self.where.get(layer)
        if slot is None:
            return None
        self.last_use[slot] = self.clock
        self.clock += 1
        return self.buffers[slot]

    def load(self, layer: int) -> Any:
        found = self.cached(layer)
        if found is not None:
            return found
        slot = min(range(self.slots), key=lambda index: self.last_use[index])
        self._fill(slot, layer)
        return self.buffers[slot]

    def close(self) -> None:
        self.buffers.clear()
        self.where.clear()
