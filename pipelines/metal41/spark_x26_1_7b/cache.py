"""fp8 key and value planes. Full attention grows. Sliding attention keeps a ring of 512."""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Any

from favlo.memory import memory

from spark_x26_1_7b.layout import HEAD_DIM, KV_HEADS, LAYERS, WINDOW, is_full
from spark_x26_1_7b.session import Session, u32


@dataclass
class Slot:
    full: bool
    capacity: int
    key: Any
    key_scale: Any
    value: Any
    value_scale: Any
    scratch: Any


def _slot_bytes(capacity: int, *, full: bool) -> tuple[int, int, int, int, int]:
    key = KV_HEADS * capacity * HEAD_DIM
    key_scale = KV_HEADS * capacity * (HEAD_DIM // 32)
    value = KV_HEADS * HEAD_DIM * capacity
    value_scale = KV_HEADS * HEAD_DIM * (capacity // 32)
    scratch_tokens = 32 if full else capacity
    scratch = KV_HEADS * HEAD_DIM * scratch_tokens * 4
    return key, key_scale, value, value_scale, scratch


def take_token_major(blob: memoryview, heads: int, channels: int, length: int, capacity: int) -> bytes:
    """Copy ``length`` tokens out of a ``[head, token, channel]`` plane."""
    parts: list[bytes] = []
    for head in range(heads):
        start = (head * capacity) * channels
        parts.append(bytes(blob[start : start + length * channels]))
    return b"".join(parts)


def take_channel_major(blob: memoryview, heads: int, channels: int, length: int, capacity: int) -> bytes:
    """Copy ``length`` tokens out of a ``[head, channel, token]`` plane."""
    parts: list[bytes] = []
    for head in range(heads):
        for channel in range(channels):
            start = (head * channels + channel) * capacity
            parts.append(bytes(blob[start : start + length]))
    return b"".join(parts)


class Kv:
    """Device planes for every layer. ``length`` is the number of tokens applied."""

    def __init__(self, session: Session, capacity: int) -> None:
        if capacity < 32 or capacity % 32:
            message = f"full-attention capacity {capacity} is not a multiple of 32"
            raise ValueError(message)
        self.session = session
        self.capacity = capacity
        self.length = 0
        self.slots = [self._make(is_full(layer), capacity if is_full(layer) else WINDOW) for layer in range(LAYERS)]

    def _make(self, full: bool, capacity: int) -> Slot:
        key, key_scale, value, value_scale, scratch = _slot_bytes(capacity, full=full)
        session = self.session
        return Slot(
            full=full,
            capacity=capacity,
            key=session.allocate(key, zero=True),
            key_scale=session.allocate(key_scale, zero=True),
            value=session.allocate(value, zero=True),
            value_scale=session.allocate(value_scale, zero=True),
            scratch=session.allocate(scratch, zero=True),
        )

    def ensure(self, pos: int) -> None:
        """Grow the full-attention planes so position ``pos`` fits."""
        needed = pos + 1
        if needed <= self.capacity:
            return
        capacity = self.capacity
        while capacity < needed:
            capacity *= 2
        self._grow(capacity)

    def _grow(self, capacity: int) -> None:
        length = self.length
        blocks = (length + 31) // 32
        for slot in self.slots:
            if not slot.full:
                continue
            fresh = self._make(True, capacity)
            self._copy_token(slot.key, fresh.key, HEAD_DIM, length, slot.capacity, capacity)
            self._copy_token(slot.key_scale, fresh.key_scale, HEAD_DIM // 32, length, slot.capacity, capacity)
            self._copy_channel(slot.value, fresh.value, HEAD_DIM, length, slot.capacity, capacity)
            self._copy_channel(
                slot.value_scale, fresh.value_scale, HEAD_DIM, blocks, slot.capacity // 32, capacity // 32
            )
            memory.release(slot.key)
            memory.release(slot.key_scale)
            memory.release(slot.value)
            memory.release(slot.value_scale)
            memory.release(slot.scratch)
            slot.key = fresh.key
            slot.key_scale = fresh.key_scale
            slot.value = fresh.value
            slot.value_scale = fresh.value_scale
            slot.scratch = fresh.scratch
            slot.capacity = capacity
        self.capacity = capacity
        self.rebuild()

    def _copy_token(self, source: Any, dest: Any, channels: int, length: int, old: int, new: int) -> None:
        if length <= 0:
            return
        self.session.launch(
            "copy_token_major",
            [source, dest],
            [u32(KV_HEADS), u32(channels), u32(length), u32(old), u32(new)],
            (KV_HEADS * length, 1, 1),
            (1, 1, 1),
        )

    def _copy_channel(self, source: Any, dest: Any, channels: int, length: int, old: int, new: int) -> None:
        if length <= 0:
            return
        self.session.launch(
            "copy_channel_major",
            [source, dest],
            [u32(KV_HEADS), u32(channels), u32(length), u32(old), u32(new)],
            (KV_HEADS * channels, 1, 1),
            (1, 1, 1),
        )

    def rebuild(self) -> None:
        """Fill the value scratch from the planes so the next token requantizes a partial block."""
        for slot in self.slots:
            name = "rebuild_full" if slot.full else "rebuild_slide"
            cap = slot.capacity if slot.full else WINDOW
            self.session.launch(
                name,
                [slot.value, slot.value_scale, slot.scratch],
                [struct.pack("<II", self.length, cap)],
                (1, 1, 1),
                (32, 1, 1),
            )
