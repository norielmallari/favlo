"""Resident expert images. Consecutive rows in one tier stream as one load."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Sequence
from typing import Any

import numpy as np

from qwen38_flash_next.eviction import DEFAULT_EVICTION, make_eviction
from qwen38_flash_next.formats import PAGE
from qwen38_flash_next.session import (
    _buffer_view,
    _fill_buffer,
    commit_pages,
    loosen_buffer,
    release_pages,
)

RESIDENT_CANDIDATES = (24 << 30, 20 << 30, 16 << 30, 12 << 30, 8 << 30)


def _pin_file_tier(bits: int) -> int:
    """File tier of a pin. The image stays in the file for that width."""
    tier = {8: 0, 4: 1, 2: 3}.get(int(bits))
    if tier is None:
        message = f"pin width {bits} is not 2, 4, or 8"
        raise RuntimeError(message)
    return tier


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

    def __init__(
        self,
        gpu: Any,
        files: Any,
        layers: int,
        experts: int,
        candidates: tuple[int, ...] | None = None,
        strategy: str = DEFAULT_EVICTION,
    ) -> None:
        self.gpu = gpu
        self.files = files
        self.layers = layers
        self.experts = experts
        chosen = RESIDENT_CANDIDATES if candidates is None else candidates
        self.atlas, self.budget = _allocate(gpu, chosen)
        self.eviction = make_eviction(strategy, self.budget)
        self.table = gpu.alloc(layers * experts * 8)
        _fill_buffer(self.table, layers * experts * 8, 0xFF)
        self.miss = gpu.zeros(layers * 4)
        self.pins: set[tuple[int, int]] = set()
        self.resident: dict[tuple[int, int], int] = {}
        self.expert_run: dict[tuple[int, int], tuple[int, int, int, int]] = {}
        self.runs: OrderedDict[tuple[int, int, int, int], tuple[int, int]] = OrderedDict()
        self.run_members: dict[tuple[int, int, int, int], tuple[int, ...]] = {}
        self.holes: list[tuple[int, int]] = []
        self.bump = 0
        self.streamed = 0

    @property
    def resident_count(self) -> int:
        return len(self.resident)

    def stage(
        self,
        layer: int,
        ids: Sequence[int],
        protect: set[tuple[int, int]] | None = None,
    ) -> int:
        """Load missing experts. The returned event is 0 when nothing was read.

        Repeated ids are separate token selections. A repeated expert is marked
        reused so S3-FIFO keeps it; a one-hit stays in the small FIFO.
        """
        counts: dict[int, int] = {}
        for expert in ids:
            value = int(expert)
            counts[value] = counts.get(value, 0) + 1
        wanted = sorted(counts)
        if protect:
            for layer_id, expert in protect:
                self._touch(int(layer_id), int(expert))
        held = {(layer, expert) for expert in wanted}
        if protect:
            held.update(protect)
        return self._stage_missing(layer, counts, wanted, held)

    def _stage_missing(
        self,
        layer: int,
        counts: dict[int, int],
        wanted: list[int],
        protect: set[tuple[int, int]],
    ) -> int:
        for expert in wanted:
            for _ in range(min(counts[expert], 3)):
                self._touch(layer, expert)
        missing = []
        for expert in wanted:
            if (layer, expert) not in self.resident:
                missing.append(expert)
        if not missing:
            return 0
        groups: list[tuple[Any, list[tuple[Any, int, int, int]]]] = []
        for tier in (0, 1, 3):
            indexed = []
            for expert in missing:
                expert_tier, row = self.files.tier_row(layer, expert)
                if expert_tier == 2:
                    expert_tier = _pin_file_tier(self.files.expert_bits(layer, expert))
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
                commit_pages(self.atlas, offset, span)
                members = tuple(expert for _row, expert in run)
                key = (layer, tier, first_row, last_row)
                reuse = min(max(counts[expert] for expert in members) - 1, 3)
                for index, (_row, expert) in enumerate(run):
                    self._publish(
                        layer,
                        expert,
                        offset + index * slot,
                        key,
                        offset,
                        span,
                        members,
                        reuse=reuse if index == 0 else 0,
                    )
                loads.append((self.atlas, offset, span, first_row * slot))
                self.streamed += span
            if loads:
                groups.append((self.files.io_handle(layer, tier), loads))
        return self.gpu.io_submit_mixed(groups)

    def evict_bytes(
        self, nbytes: int, protect: set[tuple[int, int]] | None = None
    ) -> int:
        """Discard victim runs until ``nbytes`` of expert images are released.

        The atlas allocation stays. ``protect`` is skipped while another victim
        exists. The caller has finished GPU use of this buffer.
        """
        if nbytes <= 0 or self.atlas is None:
            return 0
        held = self._guard(protect)
        freed = 0
        for _ in range(len(self.runs) + 1):
            if freed >= nbytes:
                break
            key = self.eviction.victim(held, self.run_members, self.budget)
            if key is None or key not in self.runs:
                break
            offset, span = self.runs[key]
            self._free_run(key)
            if offset % PAGE == 0 and span % PAGE == 0:
                release_pages(self.atlas, offset, span)
            freed += span
        self.release_unused()
        return freed

    def release_unused(self) -> None:
        """Discard atlas pages that do not hold a live expert image.

        The GPU is finished with this buffer before this runs. Evicted runs
        and the never-filled tail are a cache of the expert files, so they
        are not kept as swap.
        """
        for offset, size in self.holes:
            release_pages(self.atlas, offset, size)
        tail = self.budget - self.bump
        if tail:
            release_pages(self.atlas, self.bump, tail)

    def _touch(self, layer: int, expert: int) -> None:
        key = self.expert_run.get((layer, expert))
        if key is not None:
            self.eviction.hit(key)

    def _publish(
        self,
        layer: int,
        expert: int,
        image: int,
        key: tuple[int, int, int, int],
        offset: int,
        span: int,
        members: tuple[int, ...],
        reuse: int = 0,
    ) -> None:
        if key not in self.runs:
            self.runs[key] = (offset, span)
            self.run_members[key] = members
            self.eviction.insert(key, span)
            for _ in range(max(0, reuse)):
                self.eviction.hit(key)
            self._relieve()
            if key not in self.runs:
                return
        else:
            self.eviction.hit(key)
        self.resident[(layer, expert)] = image
        self.expert_run[(layer, expert)] = key
        self._write_table(layer, expert, image)

    def _alloc_span(self, span: int, protect: set[tuple[int, int]]) -> int:
        for _ in range(len(self.runs) + 1):
            found = self._try_alloc(span)
            if found >= 0:
                return found
            if not self._evict(protect, span):
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

    def _guard(self, protect: set[tuple[int, int]] | None) -> set[tuple[int, int]]:
        """Pinned experts stay. A caller can protect more for this load."""
        held = set(self.pins)
        if protect:
            held.update(protect)
        return held

    def _evict(self, protect: set[tuple[int, int]], span: int) -> bool:
        key = self.eviction.victim(self._guard(protect), self.run_members, span)
        if key is None or key not in self.runs:
            return False
        self._free_run(key)
        return True

    def _relieve(self) -> None:
        """Apply quick demotion while the byte budget still has room."""
        for _ in range(len(self.runs) + 1):
            key = self.eviction.relieve(self._guard(None), self.run_members)
            if key is None or key not in self.runs:
                return
            self._free_run(key)

    def _free_run(self, key: tuple[int, int, int, int]) -> None:
        self.eviction.remove(key)
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
            loosen_buffer(buffer)
            return buffer, nbytes
    message = "unable to allocate a resident expert atlas"
    raise RuntimeError(message)
