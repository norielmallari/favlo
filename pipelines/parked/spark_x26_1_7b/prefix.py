"""Prefix files for the key and value planes. An exact-length file is not sampled."""

from __future__ import annotations

from typing import Any

from favlo.memory import memory
from favlo.prefix import PrefixHeader, Region, read_regions
from spark_x26_1_7b.cache import take_channel_major, take_token_major
from spark_x26_1_7b.layout import HEAD_DIM, KV_HEADS, LAYERS, WINDOW, is_full, kind_ordinal


def region_names(layer: int, *, kind: str = "e4m3") -> tuple[str, ...]:
    kind_letter = "f" if is_full(layer) else "s"
    ordinal = kind_ordinal(layer)
    stem = f"{kind_letter}{ordinal}"
    if kind == "f16":
        return (f"{stem}k", f"{stem}v")
    return (f"{stem}k", f"{stem}ks", f"{stem}v", f"{stem}vs")


def scatter_token_major(
    dest: bytearray, blob: bytes, heads: int, channels: int, length: int, capacity: int
) -> None:
    for head in range(heads):
        src = head * length * channels
        dst = head * capacity * channels
        dest[dst : dst + length * channels] = blob[src : src + length * channels]


def scatter_channel_major(
    dest: bytearray, blob: bytes, heads: int, channels: int, length: int, capacity: int
) -> None:
    for head in range(heads):
        for channel in range(channels):
            src = (head * channels + channel) * length
            dst = (head * channels + channel) * capacity
            dest[dst : dst + length] = blob[src : src + length]


class CacheAdapter:
    """Device side of ``PrefixSession``. ``accept`` refuses a file as long as the prompt."""

    def __init__(self, runtime: Any) -> None:
        self.runtime = runtime

    @property
    def kind(self) -> str:
        return str(getattr(self.runtime, "kv_cache", "e4m3"))

    def capture(self) -> list[Region]:
        length = int(self.runtime.kv.length)
        regions: list[Region] = []
        for layer in range(LAYERS):
            slot = self.runtime.kv.slots[layer]
            names = region_names(layer, kind=self.kind)
            if slot.full:
                regions.extend(self._tight(slot, names, length))
            elif length >= WINDOW:
                regions.extend(self._ring(slot, names))
            else:
                regions.extend(self._tight(slot, names, length, capacity=WINDOW))
        return regions

    def install(self, header: PrefixHeader) -> bool:
        count = len(header.tokens)
        runtime = self.runtime
        if count < 1 or count > runtime.limit:
            return False
        if not runtime._ensure(count - 1):
            return False
        names = [name for layer in range(LAYERS) for name in region_names(layer, kind=self.kind)]
        blobs = read_regions(header, names)
        if blobs is None:
            return False
        for layer in range(LAYERS):
            slot = runtime.kv.slots[layer]
            names_here = region_names(layer, kind=self.kind)
            if self.kind == "f16":
                key_name, value_name = names_here
                if slot.full or count < WINDOW:
                    self._place_f16(slot, blobs[key_name], blobs[value_name], count)
                else:
                    memory.store(slot.key, 0, blobs[key_name])
                    memory.store(slot.value, 0, blobs[value_name])
                continue
            key_name, key_scale, value_name, value_scale = names_here
            if slot.full or count < WINDOW:
                self._place(
                    slot,
                    blobs[key_name],
                    blobs[key_scale],
                    blobs[value_name],
                    blobs[value_scale],
                    count,
                )
            else:
                memory.store(slot.key, 0, blobs[key_name])
                memory.store(slot.key_scale, 0, blobs[key_scale])
                memory.store(slot.value, 0, blobs[value_name])
                memory.store(slot.value_scale, 0, blobs[value_scale])
        runtime.kv.length = count
        runtime.kv.rebuild()
        return True

    def reset(self) -> None:
        self.runtime.kv.length = 0

    def accept(self, header: PrefixHeader, prompt: list[int]) -> bool:
        count = len(header.tokens)
        if count < 1 or count >= len(prompt) or count > self.runtime.limit:
            return False
        return header.region("f0k") is not None

    def _tight(
        self, slot: Any, names: tuple[str, ...], length: int, capacity: int | None = None
    ) -> list[Region]:
        cap = slot.capacity if capacity is None else capacity
        if self.kind == "f16":
            key = take_token_major(
                memory.map(slot.key, 0, memory.nbytes(slot.key)),
                KV_HEADS,
                HEAD_DIM * 2,
                length,
                cap,
            )
            value = take_channel_major(
                memory.map(slot.value, 0, memory.nbytes(slot.value)),
                KV_HEADS,
                HEAD_DIM,
                length * 2,
                cap * 2,
            )
            return [Region(names[0], key), Region(names[1], value)]
        blocks = (length + 31) // 32
        key = take_token_major(
            memory.map(slot.key, 0, memory.nbytes(slot.key)),
            KV_HEADS,
            HEAD_DIM,
            length,
            cap,
        )
        key_scale = take_token_major(
            memory.map(slot.key_scale, 0, memory.nbytes(slot.key_scale)),
            KV_HEADS,
            HEAD_DIM // 32,
            length,
            cap
        )
        value = take_channel_major(
            memory.map(slot.value, 0, memory.nbytes(slot.value)), KV_HEADS, HEAD_DIM, length, cap
        )
        value_scale = take_channel_major(
            memory.map(slot.value_scale, 0, memory.nbytes(slot.value_scale)),
            KV_HEADS,
            HEAD_DIM,
            blocks,
            cap // 32,
        )
        return [
            Region(names[0], key),
            Region(names[1], key_scale),
            Region(names[2], value),
            Region(names[3], value_scale),
        ]

    def _ring(self, slot: Any, names: tuple[str, ...]) -> list[Region]:
        if self.kind == "f16":
            key = KV_HEADS * WINDOW * HEAD_DIM * 2
            return [
                Region(names[0], bytes(memory.map(slot.key, 0, key))),
                Region(names[1], bytes(memory.map(slot.value, 0, key))),
            ]
        key = KV_HEADS * WINDOW * HEAD_DIM
        key_scale = KV_HEADS * WINDOW * (HEAD_DIM // 32)
        value_scale = KV_HEADS * HEAD_DIM * (WINDOW // 32)
        return [
            Region(names[0], bytes(memory.map(slot.key, 0, key))),
            Region(names[1], bytes(memory.map(slot.key_scale, 0, key_scale))),
            Region(names[2], bytes(memory.map(slot.value, 0, key))),
            Region(names[3], bytes(memory.map(slot.value_scale, 0, value_scale))),
        ]

    def _place(
        self,
        slot: Any,
        key: bytes,
        key_scale: bytes,
        value: bytes,
        value_scale: bytes,
        length: int,
    ) -> None:
        cap = slot.capacity
        blocks = (length + 31) // 32
        self._scatter_token(slot.key, key, HEAD_DIM, length, cap)
        self._scatter_token(slot.key_scale, key_scale, HEAD_DIM // 32, length, cap)
        self._scatter_channel(slot.value, value, HEAD_DIM, length, cap)
        self._scatter_channel(slot.value_scale, value_scale, HEAD_DIM, blocks, cap // 32)

    def _place_f16(self, slot: Any, key: bytes, value: bytes, length: int) -> None:
        cap = slot.capacity
        self._scatter_token(slot.key, key, HEAD_DIM * 2, length, cap)
        self._scatter_channel(slot.value, value, HEAD_DIM, length * 2, cap * 2)

    def _scatter_token(
        self, handle: Any, blob: bytes, channels: int, length: int, capacity: int
    ) -> None:
        for head in range(KV_HEADS):
            src = head * length * channels
            dst = head * capacity * channels
            memory.store(handle, dst, blob[src : src + length * channels])

    def _scatter_channel(
        self, handle: Any, blob: bytes, channels: int, length: int, capacity: int
    ) -> None:
        for head in range(KV_HEADS):
            for channel in range(channels):
                src = (head * channels + channel) * length
                dst = (head * channels + channel) * capacity
                memory.store(handle, dst, blob[src : src + length])
