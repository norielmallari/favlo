"""Expert atlas for the CUDA runtime.

Routed experts live in write-combined pinned memory, the fast host-to-device
source. The files are not read again on the hot path. Device slots are the
memory left after the key/value cache. S3-FIFO keeps a second request and
drops a one-hit.
"""

from __future__ import annotations

import ctypes
from collections import deque
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


_MARGIN = 32 << 20
_FREQ_CAP = 3
_Image = tuple[int, int]


class S3Fifo:
    """Small FIFO, main FIFO, and a ghost queue. One entry is ``span`` bytes.

    A second request moves into the main FIFO. A one-hit is dropped. This is
    the admission rule from Yang, Zhang, Qiu, Yue, and Rashmi, SOSP 2023.
    """

    def __init__(self, budget: int, span: int) -> None:
        if budget < span or span < 1:
            message = "expert cache is smaller than one image"
            raise ValueError(message)
        self.span = span
        self.small_cap = max(span, budget // 10)
        self._small: deque[_Image] = deque()
        self._main: deque[_Image] = deque()
        self._where: dict[_Image, str] = {}
        self._freq: dict[_Image, int] = {}
        self._nbytes: dict[_Image, int] = {}
        self._small_bytes = 0
        self._main_bytes = 0
        self._ghost: deque[_Image] = deque()
        self._ghost_set: set[_Image] = set()
        self._ghost_slots = max(8, (budget - self.small_cap) // (1 << 20))

    def place(self, key: _Image) -> str | None:
        return self._where.get(key)

    def hit(self, key: _Image) -> None:
        freq = self._freq.get(key)
        if freq is None or freq >= _FREQ_CAP:
            return
        self._freq[key] = freq + 1

    def insert(self, key: _Image, nbytes: int | None = None) -> None:
        size = self.span if nbytes is None else int(nbytes)
        if size < 1:
            message = "expert image is empty"
            raise ValueError(message)
        self._nbytes[key] = size
        if key in self._ghost_set:
            self._ghost_set.remove(key)
            self._main.append(key)
            self._where[key] = "m"
            self._main_bytes += size
        else:
            self._small.append(key)
            self._where[key] = "s"
            self._small_bytes += size
        self._freq[key] = 0

    def relieve(self, protect: set[_Image]) -> _Image | None:
        """Drop one one-hit once the small FIFO exceeds a tenth of the cache."""
        return self._pop_small(protect, 0)

    def victim(self, protect: set[_Image]) -> _Image | None:
        """Free one image so another image can be loaded."""
        chosen = self._pop_small(protect, self.span)
        if chosen is not None:
            return chosen
        return self._pop_main(protect)

    def _pop_small(self, protect: set[_Image], pending: int) -> _Image | None:
        pinned = 0
        while self._small and self._small_bytes + pending > self.small_cap:
            if pinned >= len(self._small):
                return None
            key = self._small.popleft()
            if key in protect:
                self._small.append(key)
                pinned += 1
                continue
            size = self._nbytes.get(key, self.span)
            self._small_bytes -= size
            if self._freq.get(key, 0) > 0:
                self._freq[key] = 0
                self._main.append(key)
                self._where[key] = "m"
                self._main_bytes += size
                continue
            self._nbytes.pop(key, None)
            self._where.pop(key, None)
            self._freq.pop(key, None)
            self._remember(key)
            return key
        return None

    def _pop_main(self, protect: set[_Image]) -> _Image | None:
        pinned = 0
        while self._main:
            if pinned >= len(self._main):
                return None
            key = self._main.popleft()
            if key in protect:
                self._main.append(key)
                pinned += 1
                continue
            if self._freq.get(key, 0) > 0:
                self._freq[key] -= 1
                self._main.append(key)
                continue
            self._main_bytes -= self._nbytes.pop(key, self.span)
            self._where.pop(key, None)
            self._freq.pop(key, None)
            return key
        return None

    def _remember(self, key: _Image) -> None:
        if key in self._ghost_set:
            return
        self._ghost.append(key)
        self._ghost_set.add(key)
        while len(self._ghost) > self._ghost_slots:
            old = self._ghost.popleft()
            self._ghost_set.discard(old)


def host_pointer(base: int, chunks: list[tuple[int, int, int]], offset: int) -> int:
    """Address one byte in a single pinned span, or in a later whole-image span."""
    if base:
        return base + offset
    for start, size, pointer in chunks:
        if start <= offset < start + size:
            return pointer + (offset - start)
    message = "expert image is outside the pinned span"
    raise RuntimeError(message)


class ExpertAtlas:
    """Resident int8 images, and a write-combined int4 span with a device stage."""

    def __init__(self, root: Path, engine: Any, *, reserve: int = 0) -> None:
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
        self.host = 0
        self._host_chunks: list[tuple[int, int, int]] = []
        self._page: np.ndarray | None = None
        self._page_ptr = 0
        self._page_at = 0
        self._registered: list[int] = []
        self._device: dict[tuple[int, int], int] = {}
        self._host_at: dict[tuple[int, int], int] = {}
        self.stage = 0
        self._cache = 0
        self.slots = 0
        self._where: dict[_Image, int] = {}
        self._free: list[int] = []
        self._s3: S3Fifo | None = None
        self._protect: set[_Image] | None = None
        self.hits = 0
        self.misses = 0
        self._reserve = max(0, int(reserve))
        try:
            self._alloc_host()
            self._load(folder, sizes)
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
        for _start, _size, pointer in self._host_chunks:
            engine.host_free(pointer)
        self._host_chunks = []
        self._page = None
        self._page_ptr = 0
        self._page_at = 0
        self.host = 0
        self.engine = None

    def begin(self, layer: int, experts: list[int]) -> None:
        """Count hits for this layer and protect those experts from eviction."""
        counts: dict[int, int] = {}
        for expert in experts:
            counts[int(expert)] = counts.get(int(expert), 0) + 1
        self._protect = {(int(layer), expert) for expert in counts}
        if self._s3 is None:
            return
        for expert, count in counts.items():
            key = (int(layer), expert)
            if key not in self._where:
                continue
            for _hit in range(min(count, _FREQ_CAP)):
                self._s3.hit(key)
            self.hits += 1

    def end(self) -> None:
        """Drop one-hits once the small FIFO is past a tenth of the cache."""
        protect = self._protect or set()
        policy = self._s3
        self._protect = None
        if policy is None:
            return
        while True:
            key = policy.relieve(protect)
            if key is None:
                return
            self._drop(key)

    def bind(self, layer: int, expert: int) -> tuple[int, int, int]:
        """Device pointer, host source, and copy size. A resident image copies nothing."""
        layer = int(layer)
        expert = int(expert)
        width = int(self.widths[layer, expert])
        if width == 8:
            return self._device[(layer, expert)], 0, 0
        source = self._at_host(self._host_at[(layer, expert)])
        if not self.slots or not self._cache or self._s3 is None:
            return self.stage, source, INT4_IMAGE
        key = (layer, expert)
        found = self._where.get(key)
        if found is not None:
            if self._protect is None or key not in self._protect:
                self._s3.hit(key)
                self.hits += 1
            return self._cache + found * INT4_IMAGE, 0, 0
        self.misses += 1
        slot = self._claim()
        self._where[key] = slot
        self._s3.insert(key, INT4_IMAGE)
        return self._cache + slot * INT4_IMAGE, source, INT4_IMAGE

    def place(self, layer: int, expert: int, stage_slot: int) -> tuple[int, int]:
        """Device pointer and width. An int8 image is already resident."""
        layer = int(layer)
        expert = int(expert)
        width = int(self.widths[layer, expert])
        pointer, source, nbytes = self.bind(layer, expert)
        if nbytes and pointer == self.stage:
            destination = self.stage + int(stage_slot) * INT4_IMAGE
            self.engine.h2d_ptr(destination, source, nbytes)
            return destination, 4 if width == 4 else 8
        if nbytes:
            self.engine.h2d_ptr(pointer, source, nbytes)
        return pointer, 8 if width == 8 else 4

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
                    self._copy_host(host_cursor, scratch + off, image)
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

    def _alloc_host(self) -> None:
        """Pin the int4 span write-combined. A refused tail stays pageable."""
        if self.nbytes < 1:
            return
        images = self.nbytes // INT4_IMAGE
        pointer = self.engine.host_alloc(self.nbytes)
        pinned = images if pointer else 0
        if not pointer:
            trim = 64
            while trim < images:
                pointer = self.engine.host_alloc((images - trim) * INT4_IMAGE)
                if pointer:
                    pinned = images - trim
                    break
                trim *= 2
        if pointer:
            self._host_chunks = [(0, pinned * INT4_IMAGE, pointer)]
            if pinned == images:
                self.host = pointer
        tail = self.nbytes - pinned * INT4_IMAGE
        if tail:
            self._page = np.empty(tail, dtype=np.uint8)
            self._page_ptr = int(self._page.ctypes.data)
            self._page_at = pinned * INT4_IMAGE

    def _at_host(self, offset: int) -> int:
        if self._page_ptr and offset >= self._page_at:
            return self._page_ptr + (offset - self._page_at)
        return host_pointer(self.host, self._host_chunks, offset)

    def _copy_host(self, offset: int, source: int, nbytes: int) -> None:
        copied = 0
        while copied < nbytes:
            dest = self._at_host(offset + copied)
            room = self._room(offset + copied, nbytes - copied)
            ctypes.memmove(dest, source + copied, room)
            copied += room

    def _room(self, offset: int, nbytes: int) -> int:
        if self.host or (self._page_ptr and offset >= self._page_at):
            return nbytes
        for start, size, _pointer in self._host_chunks:
            if start <= offset < start + size:
                return min(nbytes, start + size - offset)
        return nbytes

    def release(self, nbytes: int) -> int:
        """Drop int4 slots until ``nbytes`` are free. The key/value cache wins."""
        if nbytes < 1 or not self._cache or self.slots < 1:
            return 0
        held = self.slots * INT4_IMAGE
        keep = max(0, self.slots - (int(nbytes) + INT4_IMAGE - 1) // INT4_IMAGE)
        self._reset_cache(0)
        if keep < 1:
            return held
        try:
            cache = self.engine.alloc(keep * INT4_IMAGE)
        except RuntimeError:
            return held
        self._install_cache(cache, keep)
        return held - keep * INT4_IMAGE

    def _open_cache(self) -> None:
        free, _total = self.engine.mem_info()
        room = free - _MARGIN - self._reserve
        if room < INT4_IMAGE:
            room = free - _MARGIN
        while room >= INT4_IMAGE:
            slots = room // INT4_IMAGE
            try:
                cache = self.engine.alloc(slots * INT4_IMAGE)
            except RuntimeError:
                if slots <= 8:
                    return
                room = max(INT4_IMAGE * 8, room * 7 // 8)
                continue
            self._install_cache(cache, slots)
            return

    def _install_cache(self, cache: int, slots: int) -> None:
        self._cache = cache
        self.slots = slots
        self._where = {}
        self._free = list(range(slots))
        self._s3 = S3Fifo(slots * INT4_IMAGE, INT4_IMAGE)
        self._protect = None

    def _reset_cache(self, _kept: int) -> None:
        if self._cache:
            self.engine.free(self._cache)
        self._cache = 0
        self.slots = 0
        self._where = {}
        self._free = []
        self._s3 = None
        self._protect = None

    def _claim(self) -> int:
        if self._free:
            return self._free.pop()
        policy = self._s3
        if policy is None:
            message = "expert atlas has no eviction queue"
            raise RuntimeError(message)
        key = policy.victim(self._protect or set())
        if key is None:
            message = "expert atlas cannot hold the routed images"
            raise RuntimeError(message)
        return self._where.pop(key)

    def _drop(self, key: _Image) -> None:
        slot = self._where.pop(key, None)
        if slot is not None:
            self._free.append(slot)

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
