"""SSD prefix adapter for Inkling 2 Mini.

A record holds one ``conv`` region with every short-convolution state and, for
each of the 48 layer visits, key and value rows in position order (half, 1024
channels). A full-attention visit stores positions ``0 .. count-1``. A
sliding-window visit stores only the last ``WINDOW`` of them, because no later
query reads further back. On the device that tail lives in a ring of
``min(WINDOW, context)`` slots. The record continues a prompt only when the
prompt is longer: the next prefill steps at least one token, so an exact-length
file is never sampled.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from favlo.memory import memory
from favlo.prefix import (
    DEFAULT_BUDGET,
    PrefixHeader,
    PrefixSession,
    PrefixStore,
    Region,
    checkpoint_digest,
    default_prefix_dir,
    read_regions,
)
from inkling_2_mini.metal41.layout import FULL_LAYERS, KV_DIM, WINDOW

PIPELINE = "inkling_2_mini"
ROW_BYTES = KV_DIM * 2
VISIT_LAYERS = (*range(0, 6), *range(6, 18), *range(6, 18), *range(6, 18), *range(18, 24))
VISITS = len(VISIT_LAYERS)


def _names() -> list[str]:
    names = ["conv"]
    for visit in range(VISITS):
        names.extend((f"k{visit:02d}", f"v{visit:02d}"))
    return names


def cache_slots(layer: int, context: int) -> int:
    """Device rows for one visit. A local visit keeps its window, not the context."""
    if layer in FULL_LAYERS:
        return context
    return min(WINDOW, context)


def template_boundary(tokens: Sequence[int], marker: int) -> bool:
    """True when ``tokens`` ends on the first user marker and no earlier one."""
    return bool(tokens) and tokens[-1] == marker and marker not in tokens[:-1]


def visit_rows(visit: int, count: int) -> tuple[int, int]:
    """First stored position and stored row count of one visit."""
    if VISIT_LAYERS[visit] in FULL_LAYERS:
        return 0, count
    first = max(0, count - WINDOW)
    return first, count - first


def _gather(raw: bytes, slots: int, first: int, rows: int) -> bytes:
    """Chronological rows from a ring. Slot ``p`` holds position ``p``."""
    row = ROW_BYTES
    out = bytearray(rows * row)
    mask = slots - 1
    for index in range(rows):
        slot = (first + index) & mask
        start = slot * row
        out[index * row : (index + 1) * row] = raw[start : start + row]
    return bytes(out)


def _scatter(blob: bytes, slots: int, first: int) -> bytes:
    """Place chronological rows onto a ring. Untouched slots stay zero."""
    row = ROW_BYTES
    rows = len(blob) // row
    ring = bytearray(slots * row)
    mask = slots - 1
    for index in range(rows):
        slot = (first + index) & mask
        start = slot * row
        ring[start : start + row] = blob[index * row : (index + 1) * row]
    return bytes(ring)


class InklingPrefix:
    """Copies the live sequence between the device buffers and one record."""

    def __init__(self, runtime: Any) -> None:
        self.runtime = runtime

    def capture(self) -> list[Region]:
        runtime = self.runtime
        count = runtime._filled
        regions = [Region("conv", memory.load(runtime.state_all, 0, runtime.state_bytes))]
        for visit in range(VISITS):
            layer = VISIT_LAYERS[visit]
            slots = cache_slots(layer, runtime.context)
            first, rows = visit_rows(visit, count)
            if layer in FULL_LAYERS:
                offset = first * ROW_BYTES
                nbytes = rows * ROW_BYTES
                k = memory.load(runtime.k_cache[visit], offset, nbytes)
                v = memory.load(runtime.v_cache[visit], offset, nbytes)
            else:
                raw_k = memory.load(runtime.k_cache[visit], 0, slots * ROW_BYTES)
                raw_v = memory.load(runtime.v_cache[visit], 0, slots * ROW_BYTES)
                k = _gather(raw_k, slots, first, rows)
                v = _gather(raw_v, slots, first, rows)
            regions.append(Region(f"k{visit:02d}", k))
            regions.append(Region(f"v{visit:02d}", v))
        return regions

    def install(self, header: PrefixHeader) -> bool:
        runtime = self.runtime
        count = len(header.tokens)
        if count <= 0 or count >= runtime.context:
            return False
        names = _names()
        for name in names:
            ref = header.region(name)
            if name == "conv":
                want = runtime.state_bytes
            else:
                want = visit_rows(int(name[1:]), count)[1] * ROW_BYTES
            if ref is None or ref.length != want:
                return False
        blobs = read_regions(header, names)
        if blobs is None:
            return False
        memory.store(runtime.state_all, 0, blobs["conv"])
        for visit in range(VISITS):
            layer = VISIT_LAYERS[visit]
            first, _rows = visit_rows(visit, count)
            if layer in FULL_LAYERS:
                offset = first * ROW_BYTES
                memory.store(runtime.k_cache[visit], offset, blobs[f"k{visit:02d}"])
                memory.store(runtime.v_cache[visit], offset, blobs[f"v{visit:02d}"])
            else:
                slots = cache_slots(layer, runtime.context)
                memory.store(
                    runtime.k_cache[visit], 0, _scatter(blobs[f"k{visit:02d}"], slots, first)
                )
                memory.store(
                    runtime.v_cache[visit], 0, _scatter(blobs[f"v{visit:02d}"], slots, first)
                )
        runtime._ids = [int(token) for token in header.tokens]
        runtime._filled = count
        return True

    def reset(self) -> None:
        self.runtime.reset()

    def accept(self, header: PrefixHeader, prompt: list[int]) -> bool:
        count = len(header.tokens)
        if not (0 < count < len(prompt) and count < self.runtime.context):
            return False
        # The first user marker is only the chat template. Every one-turn
        # prompt shares it, so reloading that file is not a resumed message.
        return not template_boundary(header.tokens, self.runtime._user_marker())


def open_prefix(runtime: Any) -> PrefixSession:
    store = PrefixStore(
        default_prefix_dir(),
        model=PIPELINE,
        digest=checkpoint_digest(runtime.root),
        tag=f"{runtime.kv_cache}-{runtime.weights.dense_tag}",
        budget=DEFAULT_BUDGET,
    )
    return PrefixSession(store, InklingPrefix(runtime))
