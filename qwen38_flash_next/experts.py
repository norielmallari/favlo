"""Resident expert images. Consecutive ids stream as one sequential load."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Sequence
from typing import Any

import numpy as np

from qwen38_flash_next.formats import PAGE
from qwen38_flash_next.session import _buffer_view, _fill_buffer, _log

RESIDENT_CANDIDATES = (20 << 30, 16 << 30, 12 << 30, 8 << 30)


def group_assignments(
    ids: np.ndarray, count: int, topk: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Group router choices so one expert's weights are read once.

    The inverse maps ``token * topk + slot`` back to that group-local row.
    Scatter walks slots in the original order, so the weighted sum matches
    the per-slot accumulation.
    """
    flat = np.asarray(ids, dtype=np.uint32).reshape(count, topk)
    chosen: dict[int, list[tuple[int, int]]] = {}
    for token in range(count):
        for slot in range(topk):
            chosen.setdefault(int(flat[token, slot]), []).append((token, slot))
    tokens: list[int] = []
    inverse = np.empty(count * topk, dtype=np.uint32)
    offsets = [0]
    experts = sorted(chosen)
    for expert in experts:
        for token, slot in chosen[expert]:
            inverse[token * topk + slot] = len(tokens)
            tokens.append(token)
        offsets.append(len(tokens))
    return (
        np.asarray(experts, dtype=np.uint32),
        np.asarray(offsets, dtype=np.uint32),
        np.asarray(tokens, dtype=np.uint32),
        inverse,
    )


def _consecutive(ids: Sequence[int]) -> list[tuple[int, int]]:
    if not ids:
        return []
    runs: list[tuple[int, int]] = []
    first = prev = int(ids[0])
    for expert in ids[1:]:
        expert = int(expert)
        if expert == prev + 1:
            prev = expert
            continue
        runs.append((first, prev))
        first = prev = expert
    runs.append((first, prev))
    return runs


class ExpertAtlas:
    """A bounded device copy of expert images, indexed by a 64-bit offset table."""

    def __init__(self, gpu: Any, files: Any, layers: int, experts: int) -> None:
        self.gpu = gpu
        self.files = files
        self.layers = layers
        self.experts = experts
        self.live = int(files.live)
        self.slot = int(files.slot)
        self.atlas, self.budget = _allocate(gpu, RESIDENT_CANDIDATES)
        self.table = gpu.alloc(layers * experts * 8)
        _fill_buffer(self.table, layers * experts * 8, 0xFF)
        self.miss = gpu.zeros(layers * 4)
        self.resident: dict[tuple[int, int], int] = {}
        self.expert_run: dict[tuple[int, int], tuple[int, int, int]] = {}
        self.runs: OrderedDict[tuple[int, int, int], tuple[int, int]] = OrderedDict()
        self.holes: list[tuple[int, int]] = []
        self.bump = 0
        self.streamed = 0
        _log(f"expert atlas {self.budget >> 20} MiB, slot {self.slot} live {self.live}")

    @property
    def resident_count(self) -> int:
        return len(self.resident)

    def stage(
        self,
        layer: int,
        ids: Sequence[int],
        protect: set[tuple[int, int]] | None = None,
    ) -> int:
        """Load missing experts. The returned event is 0 when nothing was read."""
        wanted = sorted({int(expert) for expert in ids})
        held = {(layer, expert) for expert in wanted}
        if protect:
            held.update(protect)
        protect = held
        for expert in wanted:
            self._touch(layer, expert)
        missing = [expert for expert in wanted if (layer, expert) not in self.resident]
        if not missing:
            return 0
        loads: list[tuple[Any, int, int, int]] = []
        for first, last in _consecutive(missing):
            span = (last - first) * self.slot + self.live
            offset = self._alloc_span(span, protect)
            if offset % PAGE or span % PAGE:
                message = f"expert load {offset}+{span} is not a multiple of {PAGE}"
                raise RuntimeError(message)
            for index, expert in enumerate(range(first, last + 1)):
                self._publish(layer, expert, offset + index * self.slot, first, last, offset, span)
            loads.append((self.atlas, offset, span, first * self.slot))
            self.streamed += span
        return self.gpu.io_submit(self.files.expert_io[layer], loads)

    def _touch(self, layer: int, expert: int) -> None:
        key = self.expert_run.get((layer, expert))
        if key is not None:
            self.runs.move_to_end(key)

    def _publish(
        self,
        layer: int,
        expert: int,
        image: int,
        first: int,
        last: int,
        offset: int,
        span: int,
    ) -> None:
        key = (layer, first, last)
        if key not in self.runs:
            self.runs[key] = (offset, span)
        self.runs.move_to_end(key)
        self.resident[(layer, expert)] = image
        self.expert_run[(layer, expert)] = key
        self._write_table(layer, expert, image)

    def _alloc_span(self, span: int, protect: set[tuple[int, int]]) -> int:
        for _ in range(len(self.runs) + 1):
            found = self._try_alloc(span)
            if found >= 0:
                return found
            if not self._evict(protect):
                break
        message = "resident expert budget cannot hold the experts for this layer"
        raise RuntimeError(message)

    def _try_alloc(self, span: int) -> int:
        for index, (offset, size) in enumerate(self.holes):
            if size < span:
                continue
            self.holes.pop(index)
            if size > span:
                self.holes.append((offset + span, size - span))
            return offset
        if self.bump + span <= self.budget:
            offset = self.bump
            self.bump += span
            return offset
        return -1

    def _evict(self, protect: set[tuple[int, int]]) -> bool:
        for key in list(self.runs):
            layer, first, last = key
            if any((layer, expert) in protect for expert in range(first, last + 1)):
                continue
            self._free_run(key)
            return True
        return False

    def _free_run(self, key: tuple[int, int, int]) -> None:
        offset, span = self.runs.pop(key)
        layer, first, last = key
        for expert in range(first, last + 1):
            self.resident.pop((layer, expert), None)
            self.expert_run.pop((layer, expert), None)
            self._write_table(layer, expert, -1)
        self.holes.append((offset, span))
        self._coalesce()

    def _coalesce(self) -> None:
        if not self.holes:
            return
        merged: list[list[int]] = []
        for offset, size in sorted(self.holes):
            if merged and offset == merged[-1][0] + merged[-1][1]:
                merged[-1][1] += size
            elif not merged or offset > merged[-1][0] + merged[-1][1]:
                merged.append([offset, size])
        if merged and merged[-1][0] + merged[-1][1] == self.bump:
            self.bump = merged[-1][0]
            merged.pop()
        self.holes = [(offset, size) for offset, size in merged]

    def _write_table(self, layer: int, expert: int, value: int) -> None:
        index = layer * self.experts + expert
        start = index * 8
        _buffer_view(self.table, start + 8)[start : start + 8] = int(value).to_bytes(
            8, "little", signed=True
        )


def _allocate(gpu: Any, candidates: tuple[int, ...]) -> tuple[Any, int]:
    for nbytes in candidates:
        buffer = gpu.device.newBufferWithLength_options_(
            nbytes, gpu.metal.MTLResourceStorageModeShared
        )
        if buffer is not None:
            return buffer, nbytes
    message = "unable to allocate a resident expert atlas"
    raise RuntimeError(message)
