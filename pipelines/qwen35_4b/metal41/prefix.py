"""SSD prefix adapter for this pipeline.

The shared store in ``favlo.prefix`` owns the file. This module packs the used
key and value positions in the live device layout (token-major
``[position, kv_head, channel]``), plus the Gated DeltaNet state, the
convolution history, and the M-RoPE position delta. A stored file is accepted
when it is a strict prefix of the new prompt, so the next token is still
stepped.
"""

from __future__ import annotations

import struct
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
from qwen35_4b.metal41.layout import (
    BLOCK,
    CONV_KERNEL,
    FULL,
    GDN_QKV,
    GDN_STATE_BYTES,
    HEAD_DIM,
    KV_HEADS,
    LAYERS,
    STEM,
    kv_mode,
)

_SLOTS = len(FULL)
_HISTORY_BYTES = (LAYERS - len(FULL)) * GDN_QKV * (CONV_KERNEL - 1) * 4
_STATE_BYTES = GDN_STATE_BYTES
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
    """Pack and restore this pipeline's live sequence."""

    def __init__(self, runtime: Any) -> None:
        self.runtime = runtime

    def capture(self) -> list[Region]:
        count = int(self.runtime.pos)
        mode = kv_mode(self.runtime.kv_cache)
        regions = [
            Region("history", memory.load(self.runtime.conv_hist, 0, _HISTORY_BYTES)),
            Region("state", memory.load(self.runtime.gdn_state, 0, _STATE_BYTES)),
            Region("rope", struct.pack("<i", int(self.runtime._rope_delta))),
        ]
        for slot in range(_SLOTS):
            regions.append(Region(f"k.{slot}", _pack_slot(self.runtime, slot, count, mode, "key")))
            regions.append(Region(f"v.{slot}", _pack_slot(self.runtime, slot, count, mode, "value")))
        return regions

    def accept(self, header: PrefixHeader, prompt: list[int]) -> bool:
        count = len(header.tokens)
        if count < 1 or count >= len(prompt):
            return False
        return _fits(header, count, self.runtime.kv_cache)

    def install(self, header: PrefixHeader) -> bool:
        count = len(header.tokens)
        mode = kv_mode(self.runtime.kv_cache)
        if not _fits(header, count, mode):
            return False
        names = ["history", "state", "rope"]
        names.extend(f"k.{slot}" for slot in range(_SLOTS))
        names.extend(f"v.{slot}" for slot in range(_SLOTS))
        blobs = read_regions(header, names)
        if blobs is None:
            return False
        memory.store(self.runtime.conv_hist, 0, blobs["history"])
        memory.store(self.runtime.gdn_state, 0, blobs["state"])
        self.runtime._rope_delta = int(struct.unpack("<i", blobs["rope"])[0])
        for slot in range(_SLOTS):
            _unpack_slot(self.runtime, slot, count, mode, "key", blobs[f"k.{slot}"])
            _unpack_slot(self.runtime, slot, count, mode, "value", blobs[f"v.{slot}"])
        self.runtime.pos = count
        return True

    def reset(self) -> None:
        self.runtime._reset_state()
        self.runtime._rope_delta = 0


def pack_plane(raw: bytes, max_pos: int, count: int, mode: str) -> bytes:
    """Used positions of one live plane (data, then scales when ``fp8_e4m3``)."""
    if count < 1:
        return b""
    mode = kv_mode(mode)
    data_stride = _data_stride(max_pos, mode)
    data = raw[:data_stride]
    row = _data_row(mode)
    packed = data[: count * row]
    if mode == "half":
        return packed
    scale_row = _scale_row()
    scales = raw[data_stride : data_stride + max_pos * scale_row]
    return packed + scales[: count * scale_row]


def unpack_plane(raw: bytearray, max_pos: int, count: int, mode: str, packed: bytes) -> None:
    """Write ``packed`` onto positions ``0 .. count-1`` of a live plane."""
    if count < 1:
        return
    mode = kv_mode(mode)
    row = _data_row(mode)
    data_bytes = count * row
    data_stride = _data_stride(max_pos, mode)
    raw[:data_bytes] = packed[:data_bytes]
    if mode == "half":
        return
    scale_row = _scale_row()
    scale_bytes = count * scale_row
    scale_off = data_stride
    raw[scale_off : scale_off + scale_bytes] = packed[data_bytes : data_bytes + scale_bytes]


def _fits(header: PrefixHeader, count: int, mode: str) -> bool:
    history = header.region("history")
    state = header.region("state")
    rope = header.region("rope")
    if history is None or history.length != _HISTORY_BYTES:
        return False
    if state is None or state.length != _STATE_BYTES:
        return False
    if rope is None or rope.length != _ROPE_BYTES:
        return False
    length = _packed_length(count, mode)
    for slot in range(_SLOTS):
        key = header.region(f"k.{slot}")
        value = header.region(f"v.{slot}")
        if key is None or value is None or key.length != length or value.length != length:
            return False
    return True


def _packed_length(count: int, mode: str) -> int:
    mode = kv_mode(mode)
    length = count * _data_row(mode)
    if mode == "fp8_e4m3":
        length += count * _scale_row()
    return length


def _data_row(mode: str) -> int:
    elem = 2 if kv_mode(mode) == "half" else 1
    return KV_HEADS * HEAD_DIM * elem


def _scale_row() -> int:
    return KV_HEADS * (HEAD_DIM // BLOCK)


def _data_stride(max_pos: int, mode: str) -> int:
    return max_pos * _data_row(mode)


def _scale_stride(max_pos: int) -> int:
    return max_pos * _scale_row()


def _pack_slot(runtime: Any, slot: int, count: int, mode: str, kind: str) -> bytes:
    handle = runtime.k_data if kind == "key" else runtime.v_data
    data_off = slot * _data_stride(runtime.max_pos, mode)
    packed = memory.load(handle, data_off, count * _data_row(mode))
    if mode == "half":
        return packed
    scale = runtime.k_scale if kind == "key" else runtime.v_scale
    scale_off = slot * _scale_stride(runtime.max_pos)
    return packed + memory.load(scale, scale_off, count * _scale_row())


def _unpack_slot(
    runtime: Any, slot: int, count: int, mode: str, kind: str, packed: bytes
) -> None:
    handle = runtime.k_data if kind == "key" else runtime.v_data
    data_off = slot * _data_stride(runtime.max_pos, mode)
    row = _data_row(mode)
    memory.store(handle, data_off, packed[: count * row])
    if mode == "half":
        return
    scale = runtime.k_scale if kind == "key" else runtime.v_scale
    scale_off = slot * _scale_stride(runtime.max_pos)
    memory.store(scale, scale_off, packed[count * row :])
