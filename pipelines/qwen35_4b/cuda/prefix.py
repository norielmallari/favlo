"""SSD prefix adapter for the CUDA language opening.

The shared store in ``favlo.prefix`` owns the file. This module packs the used
key and value positions, the Gated DeltaNet state, and the convolution history.
A stored file is accepted when it is a strict prefix of the new prompt.
"""

from __future__ import annotations

import struct
from typing import Any

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
from qwen35_4b.cuda.layout import (
    BLOCK,
    CONV_BYTES,
    FULL,
    GDN_STATE_BYTES,
    HEAD_DIM,
    KV_HEADS,
    STEM,
)

_SLOTS = FULL
_ROPE_BYTES = 4


def open_session(runtime: Any) -> PrefixSession:
    return PrefixSession(
        PrefixStore(
            default_prefix_dir(),
            model=STEM,
            digest=checkpoint_digest(runtime.root),
            tag=str(runtime.kv_cache),
            budget=DEFAULT_BUDGET,
        ),
        CacheAdapter(runtime),
    )


class CacheAdapter:
    """Pack and restore this opening's live sequence."""

    def __init__(self, runtime: Any) -> None:
        self.runtime = runtime

    def capture(self) -> list[Region]:
        count = int(self.runtime.pos)
        regions = [
            Region("history", self.runtime._device_bytes(self.runtime.conv_hist, CONV_BYTES)),
            Region("state", self.runtime._device_bytes(self.runtime.gdn_state, GDN_STATE_BYTES)),
            Region("rope", struct.pack("<i", int(self.runtime._rope_delta))),
        ]
        for slot in range(_SLOTS):
            regions.append(Region(f"k.{slot}", _pack_slot(self.runtime, slot, count, "key")))
            regions.append(Region(f"v.{slot}", _pack_slot(self.runtime, slot, count, "value")))
        return regions

    def accept(self, header: PrefixHeader, prompt: list[int]) -> bool:
        count = len(header.tokens)
        if count < 1 or count >= len(prompt) or count > int(self.runtime.max_pos):
            return False
        return _fits(header, count)

    def install(self, header: PrefixHeader) -> bool:
        count = len(header.tokens)
        if not _fits(header, count):
            return False
        names = ["history", "state", "rope"]
        names.extend(f"k.{slot}" for slot in range(_SLOTS))
        names.extend(f"v.{slot}" for slot in range(_SLOTS))
        blobs = read_regions(header, names)
        if blobs is None:
            return False
        self.runtime._host_bytes(self.runtime.conv_hist, blobs["history"])
        self.runtime._host_bytes(self.runtime.gdn_state, blobs["state"])
        self.runtime._rope_delta = int(struct.unpack("<i", blobs["rope"])[0])
        for slot in range(_SLOTS):
            _unpack_slot(self.runtime, slot, count, "key", blobs[f"k.{slot}"])
            _unpack_slot(self.runtime, slot, count, "value", blobs[f"v.{slot}"])
        self.runtime.pos = count
        return True

    def reset(self) -> None:
        self.runtime._reset_state()
        self.runtime._rope_delta = 0


def _data_row() -> int:
    return KV_HEADS * HEAD_DIM


def _scale_row() -> int:
    return KV_HEADS * (HEAD_DIM // BLOCK)


def _fits(header: PrefixHeader, count: int) -> bool:
    history = header.region("history")
    state = header.region("state")
    rope = header.region("rope")
    if history is None or history.length != CONV_BYTES:
        return False
    if state is None or state.length != GDN_STATE_BYTES:
        return False
    if rope is None or rope.length != _ROPE_BYTES:
        return False
    length = count * _data_row() + count * _scale_row()
    for slot in range(_SLOTS):
        key = header.region(f"k.{slot}")
        value = header.region(f"v.{slot}")
        if key is None or value is None or key.length != length or value.length != length:
            return False
    return True


def _pack_slot(runtime: Any, slot: int, count: int, kind: str) -> bytes:
    data = runtime.k_data if kind == "key" else runtime.v_data
    scale = runtime.k_scale if kind == "key" else runtime.v_scale
    data_off = slot * runtime.max_pos * _data_row()
    scale_off = slot * runtime.max_pos * _scale_row()
    packed = runtime._device_bytes(data + data_off, count * _data_row())
    return packed + runtime._device_bytes(scale + scale_off, count * _scale_row())


def _unpack_slot(runtime: Any, slot: int, count: int, kind: str, packed: bytes) -> None:
    data = runtime.k_data if kind == "key" else runtime.v_data
    scale = runtime.k_scale if kind == "key" else runtime.v_scale
    data_off = slot * runtime.max_pos * _data_row()
    scale_off = slot * runtime.max_pos * _scale_row()
    row = count * _data_row()
    runtime._host_bytes(data + data_off, packed[:row])
    runtime._host_bytes(scale + scale_off, packed[row:])
