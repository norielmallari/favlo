"""Resident expert images.

Language expert files stay on SSD. Each file is 256 images of fixed size, and
the atlas keeps as many of those images as fit under the runtime maximum after
the resident record, the key/value cache, and the other buffers. A routed image
that is not resident is read from its layer file. S3-FIFO (Yang, Zhang, Qiu,
Yue, and Rashmi, SOSP 2023) keeps a second request and drops a one-hit. This
pipeline does not promote or demote an expert to another quant.
"""

from __future__ import annotations

import ctypes
import os
from collections import deque
from pathlib import Path
from typing import Any

from qwen36_35b_a3b.formats import EXPERT_IMAGE

LAYERS = 40
EXPERTS = 256
_Image = tuple[int, int]
_FREQ_CAP = 3


def atlas_slots(max_memory: int, file_bytes: int, reserved: int, *, layers: int = LAYERS) -> int:
    """How many ``file_bytes`` units fit in ``max_memory`` after ``reserved`` bytes."""
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


class S3Fifo:
    """Small FIFO, main FIFO, and a ghost queue. One entry is ``span`` bytes."""

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

    def insert(self, key: _Image) -> None:
        if key in self._ghost_set:
            self._ghost_set.remove(key)
            self._main.append(key)
            self._where[key] = "m"
            self._main_bytes += self.span
        else:
            self._small.append(key)
            self._where[key] = "s"
            self._small_bytes += self.span
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
            self._small_bytes -= self.span
            if self._freq.get(key, 0) > 0:
                self._freq[key] = 0
                self._main.append(key)
                self._where[key] = "m"
                self._main_bytes += self.span
                continue
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
            self._main_bytes -= self.span
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


class ExpertAtlas:
    """A pool of expert images. ``stage`` loads a miss before the caller encodes."""

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
        if self.file_bytes != EXPERTS * EXPERT_IMAGE:
            message = f"{first.name} is {self.file_bytes} bytes, not {EXPERTS} expert images"
            raise RuntimeError(message)
        present = sum(1 for layer in range(LAYERS) if (root / "experts" / f"{layer}.bin").is_file())
        if present != LAYERS:
            message = f"{root.name} has {present} language expert files"
            raise FileNotFoundError(message)
        self.max_memory = max_memory
        self.pack = session.allocate(EXPERTS * EXPERT_IMAGE)
        if self.pack is None:
            message = f"expert pack could not allocate {EXPERTS * EXPERT_IMAGE} bytes"
            raise RuntimeError(message)
        pack_bytes = EXPERTS * EXPERT_IMAGE
        self.slots = atlas_slots(
            max_memory,
            EXPERT_IMAGE,
            reserved + pack_bytes,
            layers=LAYERS * EXPERTS,
        )
        if self.slots < EXPERTS:
            message = (
                f"maximum memory {max_memory} bytes holds {self.slots} expert images, "
                f"below one layer of {EXPERTS}"
            )
            raise RuntimeError(message)
        self.buffer = None
        while self.slots >= EXPERTS:
            made = session.allocate(self.slots * EXPERT_IMAGE)
            if made is not None:
                self.buffer = made
                break
            if self.slots == EXPERTS:
                break
            self.slots = max(EXPERTS, self.slots * 7 // 8)
        if self.buffer is None:
            message = f"expert atlas could not allocate {EXPERTS} images"
            raise RuntimeError(message)
        self.buffers = [self.buffer, self.pack]
        self._atlas_view = memoryview(self.buffer.contents().as_buffer(self.slots * EXPERT_IMAGE))
        self._pack_view = memoryview(self.pack.contents().as_buffer(pack_bytes))
        self._resident: dict[_Image, int] = {}
        self._free = list(range(self.slots))
        self._eviction = S3Fifo(self.slots * EXPERT_IMAGE, EXPERT_IMAGE)
        self._fds = [
            os.open(self.root / "experts" / f"{layer}.bin", os.O_RDONLY) for layer in range(LAYERS)
        ]

    @property
    def bytes(self) -> int:
        return self.slots * EXPERT_IMAGE + EXPERTS * EXPERT_IMAGE

    def stage(self, layer: int, file_slots: list[int]) -> list[int]:
        """Load each file slot and return its atlas index."""
        self._ensure(layer, file_slots)
        return [self._resident[(layer, expert)] for expert in file_slots]

    def stage_pack(self, layer: int, file_slots: list[int]) -> None:
        """Load each file slot and copy it to the same slot in the pack buffer."""
        self._ensure(layer, file_slots)
        seen: set[int] = set()
        for expert in file_slots:
            if expert in seen:
                continue
            seen.add(expert)
            src = self._resident[(layer, expert)] * EXPERT_IMAGE
            dest = expert * EXPERT_IMAGE
            self._pack_view[dest : dest + EXPERT_IMAGE] = self._atlas_view[src : src + EXPERT_IMAGE]

    def _ensure(self, layer: int, file_slots: list[int]) -> None:
        counts: dict[int, int] = {}
        for expert in file_slots:
            if expert < 0 or expert >= EXPERTS:
                message = f"expert slot {expert} is outside 0..{EXPERTS - 1}"
                raise RuntimeError(message)
            counts[expert] = counts.get(expert, 0) + 1
        protect = {(layer, expert) for expert in counts}
        for expert, count in counts.items():
            key = (layer, expert)
            if key in self._resident:
                for _hit in range(min(count, _FREQ_CAP)):
                    self._eviction.hit(key)
        for expert in counts:
            if (layer, expert) not in self._resident:
                self._load(layer, expert, protect)
        self._shed(protect)

    def _load(self, layer: int, expert: int, protect: set[_Image]) -> None:
        slot = self._claim(protect)
        blob = os.pread(self._fds[layer], EXPERT_IMAGE, expert * EXPERT_IMAGE)
        if len(blob) != EXPERT_IMAGE:
            message = f"expert {layer}:{expert} stopped at {len(blob)} of {EXPERT_IMAGE}"
            raise OSError(message)
        offset = slot * EXPERT_IMAGE
        self._atlas_view[offset : offset + EXPERT_IMAGE] = blob
        self._resident[(layer, expert)] = slot
        self._eviction.insert((layer, expert))

    def _claim(self, protect: set[_Image]) -> int:
        if self._free:
            return self._free.pop()
        key = self._eviction.victim(protect)
        if key is None:
            message = "expert atlas cannot hold the routed images"
            raise RuntimeError(message)
        return self._resident.pop(key)

    def _shed(self, protect: set[_Image]) -> None:
        while True:
            key = self._eviction.relieve(protect)
            if key is None:
                return
            self._free.append(self._resident.pop(key))

    def close(self) -> None:
        for fd in self._fds:
            os.close(fd)
        self._fds.clear()
        self.buffers.clear()
        self.buffer = None
        self.pack = None
        self._resident.clear()
        self._free.clear()
