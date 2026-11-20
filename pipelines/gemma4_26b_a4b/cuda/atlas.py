"""Expert atlas for the CUDA runtime.

Every routed expert is int4. The images live in one write-combined pinned
span, which is the fast host-to-device source. Leftover VRAM is an S3-FIFO
cache: a new image enters the small queue, a second use promotes it, and a
ghost key that returns goes straight to the main queue.
"""

from __future__ import annotations

import ctypes
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np

from gemma4_26b_a4b.cuda.disk import alloc_pages, free_pages, read_file
from gemma4_26b_a4b.cuda.layout import EXPERTS, INT4_IMAGE, LAYERS


_MARGIN = 32 << 20


class _S3:
    """S3-FIFO over a fixed number of device slots. Ghost entries store keys only."""

    def __init__(self, slots: int) -> None:
        self.slots = slots
        self.small_limit = max(1, slots // 10)
        self.ghost_limit = max(1, (slots * 9) // 10)
        self.key: list[tuple[int, int] | None] = [None] * slots
        self.freq = [0] * slots
        self.free = list(range(slots))
        self.small: deque[int] = deque()
        self.main: deque[int] = deque()
        self.ghost: deque[tuple[int, int]] = deque()
        self.ghost_set: set[tuple[int, int]] = set()
        self.where: dict[tuple[int, int], int] = {}

    def hit(self, key: tuple[int, int]) -> int | None:
        slot = self.where.get(key)
        if slot is None:
            return None
        self.freq[slot] = min(self.freq[slot] + 1, 3)
        return slot

    def admit(self, key: tuple[int, int]) -> int:
        while not self.free:
            self._evict()
        slot = self.free.pop()
        self.key[slot] = key
        self.freq[slot] = 0
        self.where[key] = slot
        if key in self.ghost_set:
            self._ghost_drop(key)
            self.main.appendleft(slot)
        else:
            self.small.appendleft(slot)
        return slot

    def _evict(self) -> None:
        if len(self.small) >= self.small_limit and self.small:
            self._evict_small()
            return
        self._evict_main()

    def _evict_small(self) -> None:
        while self.small:
            slot = self.small.pop()
            if self.freq[slot] > 1:
                self.freq[slot] = 0
                if len(self.main) >= max(1, self.slots - self.small_limit):
                    self._evict_main()
                self.main.appendleft(slot)
                continue
            remembered = self.key[slot]
            if remembered is not None:
                self._ghost_add(remembered)
            self._drop(slot)
            return

    def _evict_main(self) -> None:
        guard = 0
        limit = max(1, len(self.main)) * 4
        while self.main and guard < limit:
            slot = self.main.pop()
            guard += 1
            if self.freq[slot] > 0:
                self.freq[slot] -= 1
                self.main.appendleft(slot)
                continue
            self._drop(slot)
            return
        if self.main:
            self._drop(self.main.pop())
            return
        if self.small:
            self._drop(self.small.pop())

    def _drop(self, slot: int) -> None:
        key = self.key[slot]
        if key is not None:
            self.where.pop(key, None)
        self.key[slot] = None
        self.freq[slot] = 0
        self.free.append(slot)

    def _ghost_add(self, key: tuple[int, int]) -> None:
        if key in self.ghost_set:
            self._ghost_drop(key)
        self.ghost.append(key)
        self.ghost_set.add(key)
        while len(self.ghost) > self.ghost_limit:
            oldest = self.ghost.popleft()
            self.ghost_set.discard(oldest)

    def _ghost_drop(self, key: tuple[int, int]) -> None:
        self.ghost_set.discard(key)
        self.ghost = deque(item for item in self.ghost if item != key)


class ExpertAtlas:
    """Pinned int4 experts and an S3-FIFO device cache."""

    def __init__(self, root: Path, engine: Any) -> None:
        folder = root / "experts"
        raw = (folder / "widths.bin").read_bytes()
        if len(raw) != LAYERS * EXPERTS:
            message = f"expert widths are {len(raw)} bytes, not {LAYERS * EXPERTS}"
            raise RuntimeError(message)
        self.widths = np.frombuffer(raw, dtype=np.uint8).reshape(LAYERS, EXPERTS).copy()
        if int((self.widths == 4).sum()) != LAYERS * EXPERTS:
            message = "routed experts are not all int4"
            raise RuntimeError(message)
        self.nbytes = LAYERS * EXPERTS * INT4_IMAGE
        self.engine = engine
        self.host = 0
        self.stage = 0
        self._cache = 0
        self.slots = 0
        self._s3: _S3 | None = None
        self.hits = 0
        self.misses = 0
        try:
            self.host = engine.host_alloc(self.nbytes)
            self._load(folder)
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
            self.stage = 0
        if self.host:
            engine.host_free(self.host)
            self.host = 0
        self.engine = None

    def bind(self, layer: int, expert: int) -> tuple[int, int, int]:
        """Device pointer, pinned source, and copy size. A hit copies nothing."""
        key = (int(layer), int(expert))
        source = self.host + (key[0] * EXPERTS + key[1]) * INT4_IMAGE
        if self._s3 is None or not self._cache or self.slots < EXPERTS:
            message = "expert cache does not hold one layer"
            raise RuntimeError(message)
        found = self._s3.hit(key)
        if found is not None:
            self.hits += 1
            return self._cache + found * INT4_IMAGE, 0, 0
        self.misses += 1
        slot = self._s3.admit(key)
        return self._cache + slot * INT4_IMAGE, source, INT4_IMAGE

    def release(self, nbytes: int) -> int:
        """Drop device slots until ``nbytes`` are free. Cached copies are discarded."""
        if nbytes < 1 or not self._cache or self.slots < 1:
            return 0
        held = self.slots * INT4_IMAGE
        keep = max(EXPERTS, self.slots - (int(nbytes) + INT4_IMAGE - 1) // INT4_IMAGE)
        if keep >= self.slots:
            return 0
        self.engine.free(self._cache)
        self._cache = 0
        self._s3 = None
        self.slots = 0
        try:
            self._cache = self.engine.alloc(keep * INT4_IMAGE)
        except RuntimeError:
            return held
        self.slots = keep
        self._s3 = _S3(keep)
        return held - keep * INT4_IMAGE

    def _load(self, folder: Path) -> None:
        scratch = alloc_pages(EXPERTS * INT4_IMAGE + 65536)
        try:
            cursor = 0
            for layer in range(LAYERS):
                path = folder / f"{layer}.bin"
                size = path.stat().st_size
                expect = EXPERTS * INT4_IMAGE
                if size != expect:
                    message = f"{path.name} is {size} bytes, not {expect}"
                    raise RuntimeError(message)
                read_file(path, scratch, size)
                ctypes.memmove(self.host + cursor, scratch, size)
                cursor += size
        finally:
            free_pages(scratch)
        if cursor != self.nbytes:
            message = "expert atlas did not place every image"
            raise RuntimeError(message)

    def _open_cache(self) -> None:
        free, _total = self.engine.mem_info()
        room = free - _MARGIN
        slots = max(0, room // INT4_IMAGE)
        while slots >= EXPERTS:
            try:
                self._cache = self.engine.alloc(slots * INT4_IMAGE)
            except RuntimeError:
                slots = (slots * 3) // 4
                continue
            self.slots = slots
            self._s3 = _S3(slots)
            return
        message = f"expert cache needs {EXPERTS} int4 slots"
        raise RuntimeError(message)
