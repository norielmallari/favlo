"""SSD prefix adapter for this pipeline.

The shared store in ``madmetal.prefix`` owns the file. This module packs the
used key and value positions off the context stride, plus the DeltaNet state
and the convolution history. A stored file is accepted when it is a strict
prefix of the new prompt, so the next token is still stepped.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from madmetal.prefix import (
    DEFAULT_BUDGET,
    PrefixHeader,
    PrefixSession,
    PrefixStore,
    Region,
    checkpoint_digest,
    default_prefix_dir,
    read_regions,
)
from qwen38_27b.cache import plane_bytes
from qwen38_27b.layout import FULL_LAYERS, HEAD, KV_HEADS, LINEAR_LAYERS, LINEAR_V, MIX

_HISTORY_BYTES = LINEAR_LAYERS * 3 * MIX * 2
_STATE_BYTES = LINEAR_LAYERS * LINEAR_V * 128 * 128 * 4
_SCALE = HEAD // 32


def open_session(runtime: Any) -> PrefixSession:
    return PrefixSession(
        PrefixStore(
            default_prefix_dir(),
            model="qwen38-27b",
            digest=checkpoint_digest(runtime.root),
            tag=runtime.kv_cache,
            budget=DEFAULT_BUDGET,
        ),
        CacheAdapter(runtime),
    )


class CacheAdapter:
    """Pack and restore this pipeline's live sequence."""

    def __init__(self, runtime: Any) -> None:
        self.runtime = runtime

    def capture(self) -> list[Region]:
        count = int(self.runtime.filled)
        regions = [
            Region("history", _read(self.runtime.history, _HISTORY_BYTES)),
            Region("state", _read(self.runtime.state, _STATE_BYTES)),
        ]
        for slot in range(FULL_LAYERS):
            regions.append(
                Region(
                    f"k.{slot}",
                    _pack(self.runtime.k_cache[slot], self.runtime.max_pos, count, kind="key"),
                )
            )
            regions.append(
                Region(
                    f"v.{slot}",
                    _pack(self.runtime.v_cache[slot], self.runtime.max_pos, count, kind="value"),
                )
            )
        return regions

    def accept(self, header: PrefixHeader, prompt: list[int]) -> bool:
        count = len(header.tokens)
        if count < 1 or count >= len(prompt):
            return False
        return _fits(header, count)

    def install(self, header: PrefixHeader) -> bool:
        count = len(header.tokens)
        if not _fits(header, count):
            return False
        names = ["history", "state"]
        names.extend(f"k.{slot}" for slot in range(FULL_LAYERS))
        names.extend(f"v.{slot}" for slot in range(FULL_LAYERS))
        blobs = read_regions(header, names)
        if blobs is None:
            return False
        for slot in range(FULL_LAYERS):
            _unpack(
                self.runtime.k_cache[slot],
                self.runtime.max_pos,
                count,
                blobs[f"k.{slot}"],
                kind="key",
            )
            _unpack(
                self.runtime.v_cache[slot],
                self.runtime.max_pos,
                count,
                blobs[f"v.{slot}"],
                kind="value",
            )
        _write(self.runtime.history, blobs["history"])
        _write(self.runtime.state, blobs["state"])
        self.runtime.filled = count
        return True

    def reset(self) -> None:
        self.runtime.reset()


def pack_plane(buffer: np.ndarray, max_pos: int, count: int, *, kind: str) -> bytes:
    """Used positions of one key or value plane, packed off the ``max_pos`` stride."""
    if count < 1:
        return b""
    array = np.frombuffer(buffer, dtype=np.uint8) if not isinstance(buffer, np.ndarray) else buffer
    data, scales = _slices(array, max_pos, count, kind)
    return np.ascontiguousarray(data).tobytes() + np.ascontiguousarray(scales).tobytes()


def unpack_plane(buffer: np.ndarray, max_pos: int, count: int, packed: bytes, *, kind: str) -> None:
    """Write ``packed`` back onto the ``max_pos`` stride. Other positions stay."""
    if count < 1:
        return
    array = np.frombuffer(buffer, dtype=np.uint8) if not isinstance(buffer, np.ndarray) else buffer
    data, scales = _slices(array, max_pos, count, kind)
    data_bytes = int(np.prod(data.shape))
    data[:] = np.frombuffer(packed[:data_bytes], dtype=np.uint8).reshape(data.shape)
    scales[:] = np.frombuffer(packed[data_bytes:], dtype=np.uint8).reshape(scales.shape)


def _fits(header: PrefixHeader, count: int) -> bool:
    history = header.region("history")
    state = header.region("state")
    if history is None or history.length != _HISTORY_BYTES:
        return False
    if state is None or state.length != _STATE_BYTES:
        return False
    for slot in range(FULL_LAYERS):
        key = header.region(f"k.{slot}")
        value = header.region(f"v.{slot}")
        if key is None or value is None:
            return False
        if key.length != _packed_length(count, "key"):
            return False
        if value.length != _packed_length(count, "value"):
            return False
    return True


def _packed_length(count: int, kind: str) -> int:
    if kind == "key":
        return count * KV_HEADS * (HEAD + _SCALE)
    blocks = (count + 31) // 32
    return KV_HEADS * HEAD * count + KV_HEADS * HEAD * blocks


def _pack(buffer: Any, max_pos: int, count: int, *, kind: str) -> bytes:
    view = np.frombuffer(
        memoryview(buffer.contents().as_buffer(plane_bytes(max_pos))), dtype=np.uint8
    )
    return pack_plane(view, max_pos, count, kind=kind)


def _unpack(buffer: Any, max_pos: int, count: int, packed: bytes, *, kind: str) -> None:
    view = np.frombuffer(
        memoryview(buffer.contents().as_buffer(plane_bytes(max_pos))), dtype=np.uint8
    )
    unpack_plane(view, max_pos, count, packed, kind=kind)


def _read(buffer: Any, nbytes: int) -> bytes:
    return bytes(memoryview(buffer.contents().as_buffer(nbytes)))


def _write(buffer: Any, payload: bytes) -> None:
    view = memoryview(buffer.contents().as_buffer(len(payload)))
    view[:] = payload


def _slices(
    array: np.ndarray, max_pos: int, count: int, kind: str
) -> tuple[np.ndarray, np.ndarray]:
    scale_base = KV_HEADS * max_pos * HEAD
    if kind == "key":
        data = array[:scale_base].reshape(KV_HEADS, max_pos, HEAD)[:, :count, :]
        scales = array[scale_base:].reshape(KV_HEADS, max_pos, _SCALE)[:, :count, :]
        return data, scales
    if kind != "value":
        message = f"unknown plane {kind}"
        raise ValueError(message)
    blocks = (count + 31) // 32
    data = array[:scale_base].reshape(KV_HEADS, HEAD, max_pos)[:, :, :count]
    scales = array[scale_base:].reshape(KV_HEADS, HEAD, max_pos // 32)[:, :, :blocks]
    return data, scales
