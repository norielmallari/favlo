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
        self.expert_fds = [
            os.open(root / "experts" / f"{layer:02d}.bin", os.O_RDONLY)
            for layer in range(int(self.manifest["layers"]))
        ]
        for fd in self.expert_fds:
            _bypass_cache(fd)
        self.ngram_fd = os.open(root / "ngram.bin", os.O_RDONLY)
        _bypass_cache(self.ngram_fd)
        self.live = int(self.manifest["expert_live"])
        self.slot = int(self.manifest["expert_slot"])
        self.expert_scratch = mmap.mmap(-1, self.live)
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
        self.expert_io = [
            handle(self.root / "experts" / f"{layer:02d}.bin")
            for layer in range(int(self.manifest["layers"]))
        ]

    def expert_image(self, layer: int, expert: int) -> bytes:
        view = pread_exact(
            self.expert_fds[layer], self.live, expert * self.slot, self.expert_scratch
        )
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
