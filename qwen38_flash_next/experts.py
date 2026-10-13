"""Resident expert images. Consecutive rows in one tier stream as one load."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Sequence
from typing import Any

import numpy as np

from qwen38_flash_next.formats import PAGE
from qwen38_flash_next.session import _buffer_view, _fill_buffer, _log

RESIDENT_CANDIDATES = (24 << 30, 20 << 30, 16 << 30, 12 << 30, 8 << 30)


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


def group_tier(
    ids: np.ndarray, count: int, topk: int, members: set[int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Group one tier. Slots outside ``members`` are ``0xFFFFFFFF`` in the inverse."""
    flat = np.asarray(ids, dtype=np.uint32).reshape(count, topk)
    chosen: dict[int, list[tuple[int, int]]] = {}
    for token in range(count):
        for slot in range(topk):
            expert = int(flat[token, slot])
            if expert not in members:
                continue
            chosen.setdefault(expert, []).append((token, slot))
    tokens: list[int] = []
    inverse = np.full(count * topk, np.uint32(0xFFFFFFFF), dtype=np.uint32)
    offsets = [0]
    experts = sorted(chosen)
    for expert in experts:
        for token, slot in chosen[expert]:
            inverse[token * topk + slot] = len(tokens)
            tokens.append(token)
        offsets.append(len(tokens))
    if not experts:
        return (
            np.zeros(0, dtype=np.uint32),
            np.zeros(1, dtype=np.uint32),
            np.zeros(0, dtype=np.uint32),
            inverse,
        )
    return (
        np.asarray(experts, dtype=np.uint32),
        np.asarray(offsets, dtype=np.uint32),
        np.asarray(tokens, dtype=np.uint32),
        inverse,
    )


def _consecutive_rows(indexed: Sequence[tuple[int, int]]) -> list[list[tuple[int, int]]]:
    if not indexed:
        return []
    runs: list[list[tuple[int, int]]] = []
    current = [indexed[0]]
    for item in indexed[1:]:
        if item[0] == current[-1][0] + 1:
            current.append(item)
            continue
        runs.append(current)
        current = [item]
    runs.append(current)
    return runs


class ExpertAtlas:
    """A bounded device copy of expert images, indexed by a 64-bit offset table."""

    def __init__(self, gpu: Any, files: Any, layers: int, experts: int) -> None:
        self.gpu = gpu
        self.files = files
        self.layers = layers
        self.experts = experts
        self.atlas, self.budget = _allocate(gpu, RESIDENT_CANDIDATES)
        self.table = gpu.alloc(layers * experts * 8)
        _fill_buffer(self.table, layers * experts * 8, 0xFF)
        self.miss = gpu.zeros(layers * 4)
        self.resident: dict[tuple[int, int], int] = {}
        self.expert_run: dict[tuple[int, int], tuple[int, int, int, int]] = {}
        self.runs: OrderedDict[tuple[int, int, int, int], tuple[int, int]] = OrderedDict()
        self.run_members: dict[tuple[int, int, int, int], tuple[int, ...]] = {}
        self.holes: list[tuple[int, int]] = []
        self.bump = 0
        self.streamed = 0
        _log(
            f"expert atlas {self.budget >> 20} MiB, "
            f"hot {files.stride(0)} cold {files.stride(1)}"
        )

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
        missing = []
        for expert in wanted:
            if self.files.tier_row(layer, expert)[0] == 2:
                continue
            if (layer, expert) not in self.resident:
                missing.append(expert)
        if not missing:
            return 0
        groups: list[tuple[Any, list[tuple[Any, int, int, int]]]] = []
        for tier in (0, 1):
            indexed = []
            for expert in missing:
                expert_tier, row = self.files.tier_row(layer, expert)
                if expert_tier == tier:
                    indexed.append((row, expert))
            indexed.sort()
            slot = int(self.files.stride(tier))
            loads: list[tuple[Any, int, int, int]] = []
            for run in _consecutive_rows(indexed):
                first_row = run[0][0]
                last_row = run[-1][0]
                span = (last_row - first_row + 1) * slot
                offset = self._alloc_span(span, protect)
                if offset % PAGE or span % PAGE:
                    message = f"expert load {offset}+{span} is not a multiple of {PAGE}"
                    raise RuntimeError(message)
                members = tuple(expert for _row, expert in run)
                key = (layer, tier, first_row, last_row)
                for index, (_row, expert) in enumerate(run):
                    self._publish(
                        layer, expert, offset + index * slot, key, offset, span, members
                    )
                loads.append((self.atlas, offset, span, first_row * slot))
                self.streamed += span
            if loads:
                groups.append((self.files.io_handle(layer, tier), loads))
        return self.gpu.io_submit_mixed(groups)

    def _touch(self, layer: int, expert: int) -> None:
        key = self.expert_run.get((layer, expert))
        if key is not None:
            self.runs.move_to_end(key)

    def _publish(
        self,
        layer: int,
        expert: int,
        image: int,
        key: tuple[int, int, int, int],
        offset: int,
        span: int,
        members: tuple[int, ...],
    ) -> None:
        if key not in self.runs:
            self.runs[key] = (offset, span)
            self.run_members[key] = members
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
            layer = key[0]
            if any((layer, expert) in protect for expert in self.run_members[key]):
                continue
            self._free_run(key)
            return True
        return False

    def _free_run(self, key: tuple[int, int, int, int]) -> None:
        offset, span = self.runs.pop(key)
        layer = key[0]
        for expert in self.run_members.pop(key):
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
