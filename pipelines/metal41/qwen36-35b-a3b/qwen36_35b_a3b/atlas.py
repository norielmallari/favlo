"""Resident expert images.

Language expert files stay on SSD. A uniform file is 256 fp4 images. A widths
file stores each expert at its own length. The atlas keeps as many of those
images as fit under the runtime maximum after the resident record, the
key/value cache, and the other buffers. The cache grows ahead of the atlas:
the atlas can shrink to eight images. When every image fits, each layer file
is read once. A routed image that is not resident is read from its layer
file. S3-FIFO (Yang, Zhang, Qiu, Yue, and Rashmi, SOSP 2023) keeps a second
request and drops a one-hit. This pipeline does not promote or demote an
expert to another quant.
"""

from __future__ import annotations

import ctypes
import os
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from qwen36_35b_a3b.formats import (
    EXPERT_FP8,
    EXPERT_IMAGE,
    expert_bytes,
    expert_file_bytes,
    expert_offsets,
)

LAYERS = 40
EXPERTS = 256
# One token routes eight experts. The atlas can shrink to that set.
MIN_IMAGES = 8
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


@dataclass(frozen=True)
class ExpertLayout:
    """Width byte and file offset for every routed expert."""

    widths: np.ndarray
    offsets: np.ndarray
    mixed: bool
    file_bytes: tuple[int, ...]

    def has(self, layer: int, width: int) -> bool:
        return bool(np.any(self.widths[layer] == np.uint8(width)))

    def count(self, width: int) -> int:
        return int(np.count_nonzero(self.widths == np.uint8(width)))

    @property
    def total_bytes(self) -> int:
        return int(sum(self.file_bytes))


def uniform_expert_layout() -> ExpertLayout:
    """Every routed expert is the fp4 image at ``slot × EXPERT_IMAGE``."""
    widths = np.full((LAYERS, EXPERTS), 4, dtype=np.uint8)
    row = np.arange(EXPERTS, dtype=np.uint64) * np.uint64(EXPERT_IMAGE)
    offsets = np.broadcast_to(row, (LAYERS, EXPERTS)).copy()
    size = EXPERTS * EXPERT_IMAGE
    return ExpertLayout(widths, offsets, False, tuple(size for _layer in range(LAYERS)))


def read_expert_layout(root: Path) -> ExpertLayout:
    """Read ``experts/widths.bin``. Absent, or all fours, keeps the fp4 stride."""
    path = root / "experts" / "widths.bin"
    if not path.is_file():
        return uniform_expert_layout()
    raw = path.read_bytes()
    if len(raw) != LAYERS * EXPERTS:
        message = f"expert widths are {len(raw)} bytes, not {LAYERS * EXPERTS}"
        raise RuntimeError(message)
    widths = np.frombuffer(raw, dtype=np.uint8).reshape(LAYERS, EXPERTS).copy()
    if not np.isin(widths, (4, 8)).all():
        message = "expert width is not 4 or 8"
        raise RuntimeError(message)
    if not np.any(widths == 8):
        return uniform_expert_layout()
    offsets_path = root / "experts" / "offsets.bin"
    if offsets_path.is_file():
        blob = offsets_path.read_bytes()
        if len(blob) != LAYERS * EXPERTS * 8:
            message = f"expert offsets are {len(blob)} bytes, not {LAYERS * EXPERTS * 8}"
            raise RuntimeError(message)
        offsets = np.frombuffer(blob, dtype="<u8").reshape(LAYERS, EXPERTS).copy()
    else:
        offsets = expert_offsets(widths)
    try:
        sizes = expert_file_bytes(widths, offsets)
    except ValueError as exc:
        raise RuntimeError(str(exc)) from exc
    for layer, size in enumerate(sizes):
        file = root / "experts" / f"{layer}.bin"
        got = file.stat().st_size
        if got != size:
            message = f"{file.name} is {got} bytes, not {size}"
            raise RuntimeError(message)
    return ExpertLayout(widths, offsets, True, tuple(sizes))


class ByteArena:
    """First-fit byte ranges. Every allocation starts on a 128-byte boundary."""

    def __init__(self, nbytes: int) -> None:
        self.nbytes = int(nbytes)
        self._free: list[tuple[int, int]] = [(0, self.nbytes)]

    def free_bytes(self) -> int:
        return sum(span for _off, span in self._free)

    def alloc(self, size: int) -> int | None:
        need = (int(size) + 127) & ~127
        for index, (off, span) in enumerate(self._free):
            aligned = (off + 127) & ~127
            pad = aligned - off
            if span < pad + need:
                continue
            rest_off = aligned + need
            rest = off + span - rest_off
            replacement: list[tuple[int, int]] = []
            if pad:
                replacement.append((off, pad))
            if rest:
                replacement.append((rest_off, rest))
            self._free[index : index + 1] = replacement
            return aligned
        return None

    def free(self, off: int, size: int) -> None:
        need = (int(size) + 127) & ~127
        self._free.append((int(off), need))
        self._free.sort()
        merged: list[tuple[int, int]] = []
        for start, span in self._free:
            if merged and merged[-1][0] + merged[-1][1] == start:
                prev_off, prev_span = merged[-1]
                merged[-1] = (prev_off, prev_span + span)
            else:
                merged.append((start, span))
        self._free = merged

    def occupy_prefix(self, used: int) -> None:
        if used < 0 or used > self.nbytes:
            message = "expert arena prefix is outside the buffer"
            raise ValueError(message)
        self._free = [] if used == self.nbytes else [(used, self.nbytes - used)]


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


def _read_file(buffer: Any, path: Path, nbytes: int, note: Any = None) -> None:
    view = memoryview(buffer.contents().as_buffer(nbytes))
    with path.open("rb") as handle:
        offset = 0
        while offset < nbytes:
            chunk = handle.read(8 << 20)
            if not chunk:
                break
            view[offset : offset + len(chunk)] = chunk
            offset += len(chunk)
            if note is not None:
                note(len(chunk))
    if offset != nbytes:
        message = f"{path.name} stopped at {offset} of {nbytes}"
        raise OSError(message)


class ExpertAtlas:
    """Expert images. A full fit keeps every layer file. A short fit loads one image."""

    def __init__(
        self,
        session: Any,
        root: Path,
        *,
        max_memory: int,
        reserved: int,
        layout: ExpertLayout | None = None,
    ) -> None:
        self.root = root
        self.layout = layout if layout is not None else read_expert_layout(root)
        self.max_memory = max_memory
        self.complete = False
        self.layers: list[Any] = []
        self.pack = None
        self.buffer = None
        self.buffers: list[Any] = []
        self._fds: list[int] = []
        self._resident: dict[_Image, int] = {}
        self._free: list[int] = []
        self._arena: ByteArena | None = None
        self._atlas_view: memoryview | None = None
        self._eviction: S3Fifo | None = None
        self.slots = 0
        self.file_bytes = 0
        if self.layout.mixed:
            self._open_mixed(session, reserved)
            return
        self._open_uniform(session, reserved)

    def retune(self, session: Any, reserved: int) -> None:
        """Fit the atlas to the bytes left after ``reserved``. A full atlas stays put."""
        if not self._rebuilds(reserved):
            return
        self._release_buffers(session)
        root = self.root
        memory = self.max_memory
        layout = self.layout
        self.close()
        self.__init__(session, root, max_memory=memory, reserved=reserved, layout=layout)

    def _rebuilds(self, reserved: int) -> bool:
        if self.layout.mixed:
            room = self.max_memory - reserved
            images = atlas_slots(self.max_memory, EXPERT_FP8, reserved, layers=LAYERS * EXPERTS)
            full = room >= self.layout.total_bytes
            if self.complete and full:
                return False
            if not self.complete and not full and images == self.slots:
                return False
            return True
        images = atlas_slots(
            self.max_memory,
            EXPERT_IMAGE,
            reserved,
            layers=LAYERS * EXPERTS,
        )
        if self.complete and images >= LAYERS * EXPERTS:
            return False
        if not self.complete and images == self.slots:
            return False
        return True

    def _release_buffers(self, session: Any) -> None:
        from favlo.memory import memory

        drop = {id(buf) for buf in self.buffers}
        for buf in self.buffers:
            memory.release(buf)
        owned = getattr(session, "_owned", None)
        if owned is not None:
            session._owned = [item for item in owned if id(item) not in drop]
        self.buffers.clear()
        self.layers.clear()
        self.buffer = None

    def _open_uniform(self, session: Any, reserved: int) -> None:
        first = self.root / "experts" / "0.bin"
        self.file_bytes = first.stat().st_size
        if self.file_bytes != EXPERTS * EXPERT_IMAGE:
            message = f"{first.name} is {self.file_bytes} bytes, not {EXPERTS} expert images"
            raise RuntimeError(message)
        present = sum(
            1 for layer in range(LAYERS) if (self.root / "experts" / f"{layer}.bin").is_file()
        )
        if present != LAYERS:
            message = f"{self.root.name} has {present} language expert files"
            raise FileNotFoundError(message)
        images = atlas_slots(
            self.max_memory,
            EXPERT_IMAGE,
            reserved,
            layers=LAYERS * EXPERTS,
        )
        if images >= LAYERS * EXPERTS and self._load_layers(session, self.file_bytes):
            return
        self.slots = images
        if self.slots < MIN_IMAGES:
            message = (
                f"maximum memory {self.max_memory} bytes holds {self.slots} expert images, "
                f"below {MIN_IMAGES}"
            )
            raise RuntimeError(message)
        self.buffer = None
        while self.slots >= MIN_IMAGES:
            made = session.allocate(self.slots * EXPERT_IMAGE)
            if made is not None:
                self.buffer = made
                break
            if self.slots == MIN_IMAGES:
                break
            self.slots = max(MIN_IMAGES, self.slots * 7 // 8)
        if self.buffer is None:
            message = f"expert atlas could not allocate {MIN_IMAGES} images"
            raise RuntimeError(message)
        nbytes = self.slots * EXPERT_IMAGE
        self.buffers = [self.buffer]
        self._atlas_view = memoryview(self.buffer.contents().as_buffer(nbytes))
        self._free = list(range(self.slots))
        self._eviction = S3Fifo(nbytes, EXPERT_IMAGE)
        self._open_fds()

    def _open_mixed(self, session: Any, reserved: int) -> None:
        present = sum(
            1 for layer in range(LAYERS) if (self.root / "experts" / f"{layer}.bin").is_file()
        )
        if present != LAYERS:
            message = f"{self.root.name} has {present} language expert files"
            raise FileNotFoundError(message)
        room = self.max_memory - reserved
        if room >= self.layout.total_bytes and self._load_mixed_layers(session):
            return
        self.slots = atlas_slots(
            self.max_memory, EXPERT_FP8, reserved, layers=LAYERS * EXPERTS
        )
        if self.slots < MIN_IMAGES:
            message = (
                f"maximum memory {self.max_memory} bytes holds {self.slots} expert images, "
                f"below {MIN_IMAGES}"
            )
            raise RuntimeError(message)
        self.buffer = None
        while self.slots >= MIN_IMAGES:
            made = session.allocate(self.slots * EXPERT_FP8)
            if made is not None:
                self.buffer = made
                break
            if self.slots == MIN_IMAGES:
                break
            self.slots = max(MIN_IMAGES, self.slots * 7 // 8)
        if self.buffer is None:
            message = f"expert atlas could not allocate {MIN_IMAGES} images"
            raise RuntimeError(message)
        nbytes = self.slots * EXPERT_FP8
        self.buffers = [self.buffer]
        self._atlas_view = memoryview(self.buffer.contents().as_buffer(nbytes))
        self._arena = ByteArena(nbytes)
        self._eviction = S3Fifo(nbytes, EXPERT_IMAGE)
        self._open_fds()

    def _open_fds(self) -> None:
        self._fds = [
            os.open(self.root / "experts" / f"{layer}.bin", os.O_RDONLY) for layer in range(LAYERS)
        ]

    def _load_layers(self, session: Any, nbytes: int) -> bool:
        """Read every layer file. The kernels index that buffer by file slot."""
        made: list[Any] = []
        for layer in range(LAYERS):
            buffer = session.allocate(nbytes)
            if buffer is None:
                self.layers = []
                return False
            _read_file(
                buffer,
                self.root / "experts" / f"{layer}.bin",
                nbytes,
                getattr(session, "_load_note", None),
            )
            made.append(buffer)
        self.layers = made
        self.buffers = made
        self.complete = True
        self.slots = LAYERS * EXPERTS
        return True

    def _load_mixed_layers(self, session: Any) -> bool:
        made: list[Any] = []
        for layer, nbytes in enumerate(self.layout.file_bytes):
            buffer = session.allocate(nbytes)
            if buffer is None:
                self.layers = []
                return False
            _read_file(
                buffer,
                self.root / "experts" / f"{layer}.bin",
                nbytes,
                getattr(session, "_load_note", None),
            )
            made.append(buffer)
        self.layers = made
        self.buffers = made
        self.complete = True
        self.slots = LAYERS * EXPERTS
        return True

    @property
    def bytes(self) -> int:
        if self.layout.mixed:
            if self.complete:
                return self.layout.total_bytes
            return self.slots * EXPERT_FP8
        return self.slots * EXPERT_IMAGE

    def stage(self, layer: int, file_slots: list[int]) -> list[int]:
        """Load each file slot and return its atlas index."""
        if self.layout.mixed:
            message = "a mixed atlas stages byte offsets"
            raise RuntimeError(message)
        if self.complete:
            message = "a resident layer does not stage images"
            raise RuntimeError(message)
        self._ensure(layer, file_slots)
        return [self._resident[(layer, expert)] for expert in file_slots]

    def stage_offsets(self, layer: int, file_slots: list[int]) -> list[int]:
        """Load each file slot and return its byte offset in the atlas."""
        if not self.layout.mixed:
            message = "a uniform atlas stages slot indexes"
            raise RuntimeError(message)
        if self.complete:
            message = "a resident layer does not stage images"
            raise RuntimeError(message)
        self._ensure(layer, file_slots)
        return [int(self._resident[(layer, expert)]) for expert in file_slots]

    def _ensure(self, layer: int, file_slots: list[int]) -> None:
        counts: dict[int, int] = {}
        for expert in file_slots:
            if expert < 0 or expert >= EXPERTS:
                message = f"expert slot {expert} is outside 0..{EXPERTS - 1}"
                raise RuntimeError(message)
            counts[expert] = counts.get(expert, 0) + 1
        protect = {(layer, expert) for expert in counts}
        eviction = self._eviction
        if eviction is None:
            message = "expert atlas has no eviction queue"
            raise RuntimeError(message)
        for expert, count in counts.items():
            key = (layer, expert)
            if key in self._resident:
                for _hit in range(min(count, _FREQ_CAP)):
                    eviction.hit(key)
        for expert in counts:
            if (layer, expert) not in self._resident:
                self._load(layer, expert, protect)
        self._shed(protect)

    def _load(self, layer: int, expert: int, protect: set[_Image]) -> None:
        if self.layout.mixed:
            self._load_mixed(layer, expert, protect)
            return
        slot = self._claim(protect)
        blob = os.pread(self._fds[layer], EXPERT_IMAGE, expert * EXPERT_IMAGE)
        if len(blob) != EXPERT_IMAGE:
            message = f"expert {layer}:{expert} stopped at {len(blob)} of {EXPERT_IMAGE}"
            raise OSError(message)
        view = self._atlas_view
        if view is None:
            message = "expert atlas has no buffer"
            raise RuntimeError(message)
        offset = slot * EXPERT_IMAGE
        view[offset : offset + EXPERT_IMAGE] = blob
        self._resident[(layer, expert)] = slot
        eviction = self._eviction
        if eviction is None:
            message = "expert atlas has no eviction queue"
            raise RuntimeError(message)
        eviction.insert((layer, expert))

    def _load_mixed(self, layer: int, expert: int, protect: set[_Image]) -> None:
        size = expert_bytes(int(self.layout.widths[layer, expert]))
        file_off = int(self.layout.offsets[layer, expert])
        arena = self._arena
        eviction = self._eviction
        view = self._atlas_view
        if arena is None or eviction is None or view is None:
            message = "expert atlas has no buffer"
            raise RuntimeError(message)
        while True:
            at = arena.alloc(size)
            if at is None and arena.free_bytes() >= size:
                self._compact()
                at = arena.alloc(size)
            if at is not None:
                break
            key = eviction.victim(protect)
            if key is None:
                message = "expert atlas cannot hold the routed images"
                raise RuntimeError(message)
            self._release(key)
        blob = os.pread(self._fds[layer], size, file_off)
        if len(blob) != size:
            message = f"expert {layer}:{expert} stopped at {len(blob)} of {size}"
            raise OSError(message)
        view[at : at + size] = blob
        self._resident[(layer, expert)] = at
        eviction.insert((layer, expert), size)

    def _compact(self) -> None:
        view = self._atlas_view
        arena = self._arena
        if view is None or arena is None:
            message = "expert atlas has no buffer"
            raise RuntimeError(message)
        order = sorted(self._resident.items(), key=lambda item: item[1])
        cursor = 0
        for key, off in order:
            size = self._image_bytes(key)
            if off != cursor:
                blob = bytes(view[off : off + size])
                view[cursor : cursor + size] = blob
            self._resident[key] = cursor
            cursor += size
        arena.occupy_prefix(cursor)

    def _image_bytes(self, key: _Image) -> int:
        layer, expert = key
        return expert_bytes(int(self.layout.widths[layer, expert]))

    def _release(self, key: _Image) -> None:
        at = self._resident.pop(key)
        arena = self._arena
        if arena is None:
            message = "expert atlas has no buffer"
            raise RuntimeError(message)
        arena.free(at, self._image_bytes(key))

    def _claim(self, protect: set[_Image]) -> int:
        if self._free:
            return self._free.pop()
        eviction = self._eviction
        if eviction is None:
            message = "expert atlas has no eviction queue"
            raise RuntimeError(message)
        key = eviction.victim(protect)
        if key is None:
            message = "expert atlas cannot hold the routed images"
            raise RuntimeError(message)
        return self._resident.pop(key)

    def _shed(self, protect: set[_Image]) -> None:
        eviction = self._eviction
        if eviction is None:
            message = "expert atlas has no eviction queue"
            raise RuntimeError(message)
        if self.layout.mixed:
            while True:
                key = eviction.relieve(protect)
                if key is None:
                    return
                self._release(key)
        while True:
            key = eviction.relieve(protect)
            if key is None:
                return
            self._free.append(self._resident.pop(key))

    def close(self) -> None:
        for fd in self._fds:
            os.close(fd)
        self._fds.clear()
        self.buffers.clear()
        self.layers.clear()
        self.buffer = None
        self.pack = None
        self._atlas_view = None
        self._arena = None
        self._eviction = None
        self._resident.clear()
        self._free.clear()
