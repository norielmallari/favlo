"""The quantized record the request path reads."""

from __future__ import annotations

import json
import mmap
import os
from pathlib import Path
from typing import Any

import numpy as np

from qwen38_flash_next.convert import ShardStore, _bypass_cache, pread_exact
from qwen38_flash_next.formats import (
    NGRAM_SLOT,
    NGRAM_WIDTH,
    PAGE,
    RECORD_VERSION,
    dequant_int4,
    int4_plane_sizes,
    stored_k_int4,
)
from qwen38_flash_next.session import MetalSession


def prefill_image_table(
    hot: list[int],
    supers: list[int],
    experts: int,
    hot_slot: int,
    cold_slot: int,
) -> tuple[int, int, np.ndarray]:
    """Byte offset of each expert in a hot-file-then-cold-file staging buffer.

    Hot rows keep the hot-file order. Cold rows are the ids in neither list,
    ascending, which is the order ``NN-cold.bin`` was packed. A super id is -1.
    """
    if hot_slot % PAGE or cold_slot % PAGE:
        message = f"prefill stride {hot_slot}/{cold_slot} is not a multiple of {PAGE}"
        raise RuntimeError(message)
    placed: dict[int, tuple[int, int]] = {}
    if set(hot) & set(supers):
        message = "a super expert is still in the hot file"
        raise RuntimeError(message)
    for row, expert in enumerate(hot):
        placed[int(expert)] = (0, row)
    for row, expert in enumerate(supers):
        placed[int(expert)] = (2, row)
    table = np.full(experts, -1, dtype=np.int64)
    hot_bytes = len(hot) * hot_slot
    cold_row = 0
    for expert in range(experts):
        found = placed.get(expert)
        if found is None:
            table[expert] = hot_bytes + cold_row * cold_slot
            cold_row += 1
            continue
        tier, row = found
        if tier == 0:
            table[expert] = row * hot_slot
    cold_bytes = cold_row * cold_slot
    return hot_bytes, cold_bytes, table


class QuantRecord:
    """The file the shaders read. It does not contain BF16 weights."""

    def __init__(self, root: Path) -> None:
        manifest_path = root / "manifest.json"
        if not manifest_path.is_file():
            message = (
                f"quantized record is missing: {manifest_path}\n"
                "madmetal --qwen38-flash-next quantize writes it. The request does not quantize."
            )
            raise FileNotFoundError(message)
        self.root = root
        self.manifest = json.loads(manifest_path.read_text())
        if self.manifest.get("version") != RECORD_VERSION or not self.manifest.get("complete"):
            message = f"quantized record is incomplete: {root}"
            raise RuntimeError(message)
        layers = int(self.manifest["layers"])
        experts = int(self.manifest["experts"])
        self.hot = [[int(expert) for expert in row] for row in self.manifest["hot"]]
        raw_super = self.manifest.get("super") or [[] for _ in range(layers)]
        self.super = [[int(expert) for expert in row] for row in raw_super]
        if len(self.hot) != layers or len(self.super) != layers:
            message = f"tier lists do not match {layers} layers"
            raise RuntimeError(message)
        self._hot_sets = [set(row) for row in self.hot]
        self._super_sets = [set(row) for row in self.super]
        self._cold_sets: list[set[int]] = []
        self._tier_row: list[dict[int, tuple[int, int]]] = []
        for hot, pin in zip(self.hot, self.super, strict=True):
            mapping: dict[int, tuple[int, int]] = {}
            for row, expert in enumerate(hot):
                mapping[expert] = (0, row)
            for row, expert in enumerate(pin):
                mapping[expert] = (2, row)
            cold: set[int] = set()
            cold_row = 0
            for expert in range(experts):
                if expert in mapping:
                    continue
                mapping[expert] = (1, cold_row)
                cold.add(expert)
                cold_row += 1
            self._tier_row.append(mapping)
            self._cold_sets.append(cold)
        self.hot_fds = [
            os.open(root / "experts" / f"{layer:02d}-hot.bin", os.O_RDONLY)
            for layer in range(layers)
        ]
        self.cold_fds = [
            os.open(root / "experts" / f"{layer:02d}-cold.bin", os.O_RDONLY)
            for layer in range(layers)
        ]
        self.super_fds: list[int | None] = []
        for layer, pin in enumerate(self.super):
            if not pin:
                self.super_fds.append(None)
                continue
            fd = os.open(root / "experts" / f"{layer:02d}-super.bin", os.O_RDONLY)
            _bypass_cache(fd)
            self.super_fds.append(fd)
        for fd in (*self.hot_fds, *self.cold_fds):
            _bypass_cache(fd)
        self.ngram_fd = os.open(root / "ngram.bin", os.O_RDONLY)
        _bypass_cache(self.ngram_fd)
        self.live = int(self.manifest["expert_live"])
        self.slot = int(self.manifest["expert_slot"])
        self.cold_live = int(self.manifest["cold_live"])
        self.cold_slot = int(self.manifest["cold_slot"])
        self.super_live = int(self.manifest.get("super_live") or 0)
        self.super_slot = int(self.manifest.get("super_slot") or 0)
        self.expert_scratch = mmap.mmap(-1, max(self.live, self.cold_live, self.super_live))
        self.page_scratch = mmap.mmap(-1, PAGE)
        self.pages: dict[int, bytes] = {}
        self.gate_up_bytes = sum(int4_plane_sizes(1280, stored_k_int4(2560, align_k=True)))

    def open_io(self, gpu: MetalSession) -> None:
        """File handles for loads that land in the buffer a shader reads."""
        import Foundation

        gpu.ensure_io()

        def handle(path: Path) -> Any:
            url = Foundation.NSURL.fileURLWithPath_(str(path))
            opened = gpu.device.newUncachedIOHandleWithURL_error_(url, None)
            if opened is None:
                message = f"Metal IO handle failed for {path.name}"
                raise RuntimeError(message)
            return opened

        self.resident_io = handle(self.root / "resident.bin")
        self.ngram_io = handle(self.root / "ngram.bin")
        layers = int(self.manifest["layers"])
        self.hot_io = [
            handle(self.root / "experts" / f"{layer:02d}-hot.bin") for layer in range(layers)
        ]
        self.cold_io = [
            handle(self.root / "experts" / f"{layer:02d}-cold.bin") for layer in range(layers)
        ]

    def hot_set(self, layer: int) -> set[int]:
        return self._hot_sets[layer]

    def cold_set(self, layer: int) -> set[int]:
        return self._cold_sets[layer]

    def super_set(self, layer: int) -> set[int]:
        return self._super_sets[layer]

    def tier_row(self, layer: int, expert: int) -> tuple[int, int]:
        return self._tier_row[layer][int(expert)]

    def stride(self, tier: int) -> int:
        if tier == 0:
            return self.slot
        if tier == 2:
            return self.super_slot
        return self.cold_slot

    def payload(self, tier: int) -> int:
        if tier == 0:
            return self.live
        if tier == 2:
            return self.super_live
        return self.cold_live

    def io_handle(self, layer: int, tier: int) -> Any:
        if tier == 2:
            message = "a super expert is resident and is not staged from SSD"
            raise RuntimeError(message)
        return self.hot_io[layer] if tier == 0 else self.cold_io[layer]

    def expert_image(self, layer: int, expert: int) -> bytes:
        tier, row = self.tier_row(layer, expert)
        if tier == 2:
            fd = self.super_fds[layer]
        elif tier == 0:
            fd = self.hot_fds[layer]
        else:
            fd = self.cold_fds[layer]
        if fd is None:
            message = f"layer {layer} has no super file"
            raise RuntimeError(message)
        view = pread_exact(fd, self.payload(tier), row * self.stride(tier), self.expert_scratch)
        blob = bytes(view)
        del view
        return blob

    def ngram_vector(self, row: int) -> np.ndarray:
        if row < 0 or row >= int(self.manifest["ngram_rows"]):
            message = f"n-gram row {row} is outside the table"
            raise IndexError(message)
        slot_off = row * NGRAM_SLOT
        page = (slot_off // PAGE) * PAGE
        cached = self.pages.get(page)
        if cached is None:
            cached = bytes(pread_exact(self.ngram_fd, PAGE, page, self.page_scratch))
            if len(self.pages) > 4096:
                self.pages.clear()
            self.pages[page] = cached
        local = slot_off - page
        slot = cached[local : local + NGRAM_SLOT]
        return dequant_int4(slot[:80], slot[80:85], 1, NGRAM_WIDTH)[0]


class _Bf16Ngram:
    """Gather one BF16 n-gram row from the checkpoint shards."""

    def __init__(self, store: ShardStore) -> None:
        self.store = store

    def ngram_vector(self, row: int) -> np.ndarray:
        return self.store.ngram_row(row, 2_500_012, NGRAM_WIDTH)
