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
from qwen36_35b_a3b.cache import kv_buffer_bytes

_SLOTS = 10
_KV = 2
_HEAD = 256
_HISTORY_BYTES = 30 * 3 * 8192 * 2
_STATE_BYTES = 30 * 32 * 128 * 128 * 4


def open_session(runtime: Any) -> PrefixSession:
    return PrefixSession(
        PrefixStore(
            default_prefix_dir(),
            model="qwen36-35b-a3b",
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
        mode = self.runtime.kv_cache
        regions = [
            Region("history", _read(self.runtime.history, _HISTORY_BYTES)),
            Region("state", _read(self.runtime.state, _STATE_BYTES)),
        ]
        for slot in range(_SLOTS):
            regions.append(
                Region(
                    f"k.{slot}",
                    _pack(self.runtime.k_cache[slot], self.runtime.max_pos, count, mode, "key"),
                )
            )
            regions.append(
                Region(
                    f"v.{slot}",
                    _pack(self.runtime.v_cache[slot], self.runtime.max_pos, count, mode, "value"),
                )
            )
        return regions

    def accept(self, header: PrefixHeader, prompt: list[int]) -> bool:
        count = len(header.tokens)
        if count < 1 or count >= len(prompt):
            return False
        return _fits(header, count, self.runtime.kv_cache)

    def install(self, header: PrefixHeader) -> bool:
        count = len(header.tokens)
        mode = self.runtime.kv_cache
        if not _fits(header, count, mode):
            return False
        names = ["history", "state"]
        names.extend(f"k.{slot}" for slot in range(_SLOTS))
        names.extend(f"v.{slot}" for slot in range(_SLOTS))
        blobs = read_regions(header, names)
        if blobs is None:
            return False
        for slot in range(_SLOTS):
            _unpack(
                self.runtime.k_cache[slot],
                self.runtime.max_pos,
                count,
                mode,
                "key",
                blobs[f"k.{slot}"],
            )
            _unpack(
                self.runtime.v_cache[slot],
                self.runtime.max_pos,
                count,
                mode,
                "value",
                blobs[f"v.{slot}"],
            )
        _write(self.runtime.history, blobs["history"])
        _write(self.runtime.state, blobs["state"])
        self.runtime.filled = count
        return True

    def reset(self) -> None:
        self.runtime.reset()


def pack_plane(
    buffer: np.ndarray, max_pos: int, count: int, mode: str, kind: str
) -> bytes:
    """Used positions of one key or value plane, packed off the context stride."""
    if count < 1:
        return b""
    array = np.asarray(buffer, dtype=np.uint8).reshape(-1)
    data, scales = _slices(array, max_pos, count, mode, kind)
    packed = np.ascontiguousarray(data).tobytes()
    if scales is None:
        return packed
    return packed + np.ascontiguousarray(scales).tobytes()


def unpack_plane(
    buffer: np.ndarray, max_pos: int, count: int, mode: str, kind: str, packed: bytes
) -> None:
    """Write ``packed`` back onto the context stride. Other positions stay."""
    if count < 1:
        return
    array = np.asarray(buffer, dtype=np.uint8).reshape(-1)
    data, scales = _slices(array, max_pos, count, mode, kind)
    data_bytes = int(np.prod(data.shape))
    data[:] = np.frombuffer(packed[:data_bytes], dtype=np.uint8).reshape(data.shape)
    if scales is not None:
        scales[:] = np.frombuffer(packed[data_bytes:], dtype=np.uint8).reshape(scales.shape)


def _fits(header: PrefixHeader, count: int, mode: str) -> bool:
    history = header.region("history")
    state = header.region("state")
    if history is None or history.length != _HISTORY_BYTES:
        return False
    if state is None or state.length != _STATE_BYTES:
        return False
    key_len = _packed_length(count, mode, "key")
    value_len = _packed_length(count, mode, "value")
    for slot in range(_SLOTS):
        key = header.region(f"k.{slot}")
        value = header.region(f"v.{slot}")
        if key is None or value is None or key.length != key_len or value.length != value_len:
            return False
    return True


def _packed_length(count: int, mode: str, kind: str) -> int:
    if mode == "f32":
        return _KV * count * _HEAD * 4
    width = 256 if mode == "e4m3" else 128
    if kind == "key":
        return _KV * count * (width + 8)
    blocks = (count + 31) // 32
    if mode == "e4m3":
        return _KV * _HEAD * count + _KV * _HEAD * blocks
    return _KV * _HEAD * ((count + 1) // 2) + _KV * _HEAD * blocks


def _slices(
    array: np.ndarray, max_pos: int, count: int, mode: str, kind: str
) -> tuple[np.ndarray, np.ndarray | None]:
    if mode == "f32":
        row = _HEAD * 4
        return array.reshape(_KV, max_pos, row)[:, :count, :], None
    width = 256 if mode == "e4m3" else 128
    scale_base = _KV * max_pos * width
    blocks = (count + 31) // 32
    if kind == "key":
        data = array[:scale_base].reshape(_KV, max_pos, width)[:, :count, :]
        scales = array[scale_base:].reshape(_KV, max_pos, 8)[:, :count, :]
        return data, scales
    if kind != "value":
        message = f"unknown plane {kind}"
        raise ValueError(message)
    if mode == "e4m3":
        data = array[:scale_base].reshape(_KV, _HEAD, max_pos)[:, :, :count]
    else:
        data = array[:scale_base].reshape(_KV, _HEAD, max_pos // 2)[:, :, : (count + 1) // 2]
    scales = array[scale_base:].reshape(_KV, _HEAD, max_pos // 32)[:, :, :blocks]
    return data, scales


def _pack(buffer: Any, max_pos: int, count: int, mode: str, kind: str) -> bytes:
    view = np.frombuffer(
        memoryview(buffer.contents().as_buffer(kv_buffer_bytes(mode, max_pos))),
        dtype=np.uint8,
    )
    return pack_plane(view, max_pos, count, mode, kind)


def _unpack(
    buffer: Any, max_pos: int, count: int, mode: str, kind: str, packed: bytes
) -> None:
    view = np.frombuffer(
        memoryview(buffer.contents().as_buffer(kv_buffer_bytes(mode, max_pos))),
        dtype=np.uint8,
    )
    unpack_plane(view, max_pos, count, mode, kind, packed)


def _read(buffer: Any, nbytes: int) -> bytes:
    return bytes(memoryview(buffer.contents().as_buffer(nbytes)))


def _write(buffer: Any, payload: bytes) -> None:
    view = memoryview(buffer.contents().as_buffer(len(payload)))
    view[:] = payload
