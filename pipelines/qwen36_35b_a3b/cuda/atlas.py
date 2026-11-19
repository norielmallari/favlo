"""Expert atlas for the CUDA runtime.

The int8 experts stay in VRAM for the open. The int4 experts stay in host RAM
and are registered read-only, so a miss is a direct copy. The files are not
read again on the hot path.
"""

from __future__ import annotations

import ctypes
from pathlib import Path
from typing import Any

import numpy as np

from qwen36_35b_a3b.cuda.disk import alloc_pages, free_pages, read_file

from favlo.host import available_bytes
from qwen36_35b_a3b.cuda.layout import (
    EXPERTS,
    INT4_IMAGE,
    INT8_IMAGE,
    LAYERS,
    TOP_K,
    stream_spans,
)


_PAGE = 4096
_MARGIN = 32 << 20


class ExpertAtlas:
    """Resident int8 images, and an int4 host span with a device stage."""

    def __init__(self, root: Path, engine: Any) -> None:
        folder = root / "experts"
        widths_path = folder / "widths.bin"
        offsets_path = folder / "offsets.bin"
        raw = widths_path.read_bytes()
        if len(raw) != LAYERS * EXPERTS:
            message = f"expert widths are {len(raw)} bytes, not {LAYERS * EXPERTS}"
            raise RuntimeError(message)
        self.widths = np.frombuffer(raw, dtype=np.uint8).reshape(LAYERS, EXPERTS).copy()
        blob = offsets_path.read_bytes()
        if len(blob) != LAYERS * EXPERTS * 8:
            message = f"expert offsets are {len(blob)} bytes, not {LAYERS * EXPERTS * 8}"
            raise RuntimeError(message)
        self.offsets = np.frombuffer(blob, dtype="<u8").reshape(LAYERS, EXPERTS).copy()
        sizes = self._layer_sizes(folder)
        int8_count = int((self.widths == 8).sum())
        int4_count = int((self.widths == 4).sum())
        self.nbytes = int4_count * INT4_IMAGE
        if self.nbytes > available_bytes():
            message = "int4 expert atlas does not fit in host RAM"
            raise RuntimeError(message)
        self.engine = engine
        self._pinned = engine.alloc(int8_count * INT8_IMAGE) if int8_count else 0
        self._buffer = np.empty(max(self.nbytes, 1) + 2 * _PAGE, dtype=np.uint8)
        base = int(self._buffer.ctypes.data)
        self.host = (base + _PAGE - 1) & ~(_PAGE - 1)
        self._registered: list[int] = []
        self._device: dict[tuple[int, int], int] = {}
        self._host_at: dict[tuple[int, int], int] = {}
        self.stage = 0
        self._cache = 0
        self.slots = 0
        self._where: dict[tuple[int, int], tuple[dict[str, Any], int]] = {}
        self._pools: list[dict[str, Any]] = []
        self._clock = 1
        self.hits = 0
        self.misses = 0
        try:
            self._load(folder, sizes)
            self._lock_host()
            self.stage = engine.alloc(TOP_K * INT4_IMAGE)
            self._open_cache()
        except Exception:
            self.close()
            raise

    def close(self) -> None:
        engine = self.engine
        if engine is None:
            return
        if self._cache:
            engine.free(self._cache)
            self._cache = 0
        if self.stage:
            engine.free(self.stage)
        if self._pinned:
            engine.free(self._pinned)
            self._pinned = 0
        for pointer in self._registered:
            engine.unregister(pointer)
        self._registered.clear()
        self._buffer = None
        self.engine = None

    def bind(self, layer: int, expert: int) -> tuple[int, int, int]:
        """Device pointer, host source, and copy size. A resident image copies nothing."""
        layer = int(layer)
        expert = int(expert)
        width = int(self.widths[layer, expert])
        if width == 8:
            return self._device[(layer, expert)], 0, 0
        source = self.host + self._host_at[(layer, expert)]
        if not self.slots or not self._cache:
            return self.stage, source, INT4_IMAGE
        key = (layer, expert)
        found = self._where.get(key)
        if found is not None:
            self.hits += 1
            pool, slot = found
            pool["stamp"][slot] = self._clock
            self._clock += 1
            return pool["base"] + slot * pool["stride"], 0, 0
        self.misses += 1
        pool = self._pools[0]
        if not pool["key"]:
            return self.stage, source, INT4_IMAGE
        slot = self._victim(pool)
        previous = pool["key"][slot]
        if previous is not None:
            self._where.pop(previous, None)
        destination = pool["base"] + slot * pool["stride"]
        pool["key"][slot] = key
        pool["stamp"][slot] = self._clock
        self._where[key] = (pool, slot)
        self._clock += 1
        return destination, source, INT4_IMAGE

    def place(self, layer: int, expert: int, stage_slot: int) -> tuple[int, int]:
        """Device pointer and width. An int8 image is already resident."""
        layer = int(layer)
        expert = int(expert)
        width = int(self.widths[layer, expert])
        if width == 8:
            return self._device[(layer, expert)], 8
        if self.slots and self._cache:
            return self._cached(layer, expert), 4
        source = self.host + self._host_at[(layer, expert)]
        destination = self.stage + int(stage_slot) * INT4_IMAGE
        self.engine.h2d_ptr(destination, source, INT4_IMAGE)
        return destination, 4

    def stream(self, layer: int, experts: list[int]) -> list[int]:
        """Copy ``experts`` that are not resident. Returns the width of each."""
        return [self.place(layer, expert, slot)[1] for slot, expert in enumerate(experts)]

    def _layer_sizes(self, folder: Path) -> list[int]:
        sizes: list[int] = []
        for layer in range(LAYERS):
            path = folder / f"{layer}.bin"
            size = path.stat().st_size
            end = int(self.offsets[layer, -1]) + _image_bytes(int(self.widths[layer, -1]))
            if size != end:
                message = f"{path.name} is {size} bytes, not {end}"
                raise RuntimeError(message)
            sizes.append(size)
        return sizes

    def _load(self, folder: Path, sizes: list[int]) -> None:
        scratch = alloc_pages(max(sizes))
        pinned = self.engine.register(scratch, max(sizes))
        int8_index = 0
        host_cursor = 0
        try:
            for layer, size in enumerate(sizes):
                path = folder / f"{layer}.bin"
                read_file(path, scratch, size)
                for expert in range(EXPERTS):
                    width = int(self.widths[layer, expert])
                    off = int(self.offsets[layer, expert])
                    image = INT8_IMAGE if width == 8 else INT4_IMAGE
                    if off + image > size:
                        message = f"layer {layer} expert {expert} is short"
                        raise RuntimeError(message)
                    if width == 8:
                        dest = self._pinned + int8_index * INT8_IMAGE
                        self.engine.h2d_ptr(dest, scratch + off, image)
                        self._device[(layer, expert)] = dest
                        int8_index += 1
                        continue
                    ctypes.memmove(self.host + host_cursor, scratch + off, image)
                    self._host_at[(layer, expert)] = host_cursor
                    host_cursor += INT4_IMAGE
                self.engine.sync()
        finally:
            self.engine.sync()
            if pinned:
                self.engine.unregister(scratch)
            free_pages(scratch)
        if host_cursor != self.nbytes or int8_index != int((self.widths == 8).sum()):
            message = "expert atlas did not place every image"
            raise RuntimeError(message)

    def _lock_host(self) -> None:
        if self.nbytes < 1:
            return
        cover = (self.nbytes + _PAGE - 1) & ~(_PAGE - 1)
        if self.engine.register_ro(self.host, cover) or self.engine.register(self.host, cover):
            self._registered.append(self.host)
            return
        message = "int4 expert atlas could not be locked for a direct copy"
        raise RuntimeError(message)

    def release(self, nbytes: int) -> int:
        """Drop int4 slots until ``nbytes`` are free. Cached copies are discarded."""
        if nbytes < 1 or not self._cache or self.slots < 1:
            return 0
        held = self.slots * INT4_IMAGE
        keep = max(0, self.slots - (int(nbytes) + INT4_IMAGE - 1) // INT4_IMAGE)
        self.engine.free(self._cache)
        self._cache = 0
        self._where.clear()
        self._pools.clear()
        self.slots = 0
        if keep < 1:
            return held
        try:
            self._cache = self.engine.alloc(keep * INT4_IMAGE)
        except RuntimeError:
            return held
        self.slots = keep
        self._pools = [self._pool(self._cache, keep, INT4_IMAGE)]
        return held - keep * INT4_IMAGE

    def _open_cache(self) -> None:
        free, _total = self.engine.mem_info()
        room = free - _MARGIN
        while room >= INT4_IMAGE:
            slots = room // INT4_IMAGE
            nbytes = slots * INT4_IMAGE
            try:
                self._cache = self.engine.alloc(nbytes)
            except RuntimeError:
                room //= 2
                continue
            self._pools = [self._pool(self._cache, slots, INT4_IMAGE)]
            self.slots = slots
            return

    def _pool(self, base: int, slots: int, stride: int) -> dict[str, Any]:
        return {
            "base": base,
            "stride": stride,
            "key": [None] * slots,
            "stamp": [0] * slots,
        }

    def _cached(self, layer: int, expert: int) -> int:
        key = (layer, expert)
        found = self._where.get(key)
        if found is not None:
            self.hits += 1
            pool, slot = found
            pool["stamp"][slot] = self._clock
            self._clock += 1
            return pool["base"] + slot * pool["stride"]
        self.misses += 1
        pool = self._pools[0]
        if not pool["key"]:
            source = self.host + self._host_at[(layer, expert)]
            self.engine.h2d_ptr(self.stage, source, INT4_IMAGE)
            return self.stage
        slot = self._victim(pool)
        previous = pool["key"][slot]
        if previous is not None:
            self._where.pop(previous, None)
        source = self.host + self._host_at[(layer, expert)]
        destination = pool["base"] + slot * pool["stride"]
        self.engine.h2d_ptr(destination, source, INT4_IMAGE)
        pool["key"][slot] = key
        pool["stamp"][slot] = self._clock
        self._where[key] = (pool, slot)
        self._clock += 1
        return destination

    def _victim(self, pool: dict[str, Any]) -> int:
        keys = pool["key"]
        for slot, key in enumerate(keys):
            if key is None:
                return slot
        stamps = pool["stamp"]
        return min(range(len(keys)), key=stamps.__getitem__)

    def span(self, layer: int, experts: list[int]) -> list[tuple[int, int, int]]:
        row_w = [int(value) for value in self.widths[layer]]
        row_o = [int(value) for value in self.offsets[layer]]
        return stream_spans(row_w, row_o, experts)


def _image_bytes(width: int) -> int:
    if width == 8:
        return INT8_IMAGE
    if width == 4:
        return INT4_IMAGE
    message = f"expert width {width} is not 4 or 8"
    raise RuntimeError(message)
