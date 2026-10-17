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
    QUANT_LAYOUT_VERSION,
    RECORD_VERSION,
    dequant_int4,
    int4_plane_sizes,
    quant_file,
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


def routed_tier(bits: int, widths: list[int]) -> int:
    """Plane of one routed expert. Three widths keep int2 on tier 3."""
    if len(widths) == 3:
        plane = {8: 0, 4: 1, 2: 3}.get(int(bits))
        if plane is None:
            message = f"expert quant {bits} is not 2, 4, or 8"
            raise RuntimeError(message)
        return plane
    if len(widths) == 2 and int(bits) == int(widths[0]):
        return 0
    return 1


class QuantRecord:
    """The file the shaders read. It does not contain BF16 weights."""

    def __init__(self, root: Path) -> None:
        self._closed = False
        self.hot_fds: list[int] = []
        self.cold_fds: list[int] = []
        self.narrow_fds: list[int | None] = []
        self.super_fds: list[int | None] = []
        self.narrow_bits = 0
        self.narrow_live = 0
        self.narrow_slot = 0
        self._narrow_sets: list[set[int]] = []
        self.ngram_fd: int | None = None
        self.expert_scratch: mmap.mmap | None = None
        self.page_scratch: mmap.mmap | None = None
        self.pages: dict[int, bytes] = {}
        self.hot_io: list[Any] = []
        self.cold_io: list[Any] = []
        self.narrow_io: list[Any] = []
        self.resident_io: Any = None
        self.ngram_io: Any = None
        self.mtp_io: Any = None
        try:
            self._open(root)
        except BaseException:
            self.close()
            raise

    def _open(self, root: Path) -> None:
        manifest_path = root / "manifest.json"
        if not manifest_path.is_file():
            message = (
                f"quantized record is missing: {manifest_path}\n"
                "madmetal --qwen38-flash-next quantize writes it. The request does not quantize."
            )
            raise FileNotFoundError(message)
        self.root = root
        self.manifest = json.loads(manifest_path.read_text())
        if not self.manifest.get("complete"):
            message = f"quantized record is incomplete: {root}"
            raise RuntimeError(message)
        self._quant_grid = None
        version = int(self.manifest.get("version") or 0)
        if version == QUANT_LAYOUT_VERSION:
            self._open_quant_files()
            return
        if version != RECORD_VERSION:
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
        self._narrow_sets = [set() for _ in range(layers)]
        for layer in range(layers):
            self.hot_fds.append(os.open(root / "experts" / f"{layer:02d}-hot.bin", os.O_RDONLY))
        for layer in range(layers):
            self.cold_fds.append(os.open(root / "experts" / f"{layer:02d}-cold.bin", os.O_RDONLY))
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
        self.hot_bits = int(self.manifest.get("hot_bits") or 4)
        self.cold_bits = int(self.manifest.get("cold_bits") or 2)
        self.super_bits = int(self.manifest.get("super_bits") or 8)
        self.live = int(self.manifest["expert_live"])
        self.slot = int(self.manifest["expert_slot"])
        self.cold_live = int(self.manifest["cold_live"])
        self.cold_slot = int(self.manifest["cold_slot"])
        self.super_live = int(self.manifest.get("super_live") or 0)
        self.super_slot = int(self.manifest.get("super_slot") or 0)
        self.expert_scratch = mmap.mmap(-1, max(self.live, self.cold_live, self.super_live))
        self.page_scratch = mmap.mmap(-1, PAGE)
        self.gate_up_bytes = sum(int4_plane_sizes(1280, stored_k_int4(2560, align_k=True)))

    def _open_quant_files(self) -> None:
        """One file per quant. Expert ``e`` of that quant sits at ``e × stride``."""
        layers = int(self.manifest["layers"])
        experts = int(self.manifest["experts"])
        quant = self.manifest.get("expert_quant")
        if isinstance(quant, list):
            self._open_quant_grid(quant)
            return
        if isinstance(quant, bool) or not isinstance(quant, int) or quant not in (2, 4, 8):
            message = f"expert quant {quant} is not 2, 4, or 8"
            raise RuntimeError(message)
        raw_super = self.manifest.get("super") or [[] for _ in range(layers)]
        self.super = [[int(expert) for expert in row] for row in raw_super]
        if len(self.super) != layers:
            message = f"super list does not match {layers} layers"
            raise RuntimeError(message)
        self.hot = [[] for _ in range(layers)]
        self._hot_sets = [set() for _ in range(layers)]
        self._super_sets = []
        self._cold_sets = []
        self._narrow_sets = [set() for _ in range(layers)]
        self._tier_row = []
        for pin in self.super:
            if len(pin) != len(set(pin)) or any(expert < 0 or expert >= experts for expert in pin):
                message = f"super pin is not ids in 0..{experts - 1}"
                raise RuntimeError(message)
            pin_set = set(pin)
            mapping: dict[int, tuple[int, int]] = {}
            cold: set[int] = set()
            for expert in range(experts):
                mapping[expert] = (1, expert)
                cold.add(expert)
            self._super_sets.append(pin_set)
            self._tier_row.append(mapping)
            self._cold_sets.append(cold)
        slot = int(self.manifest["expert_slot"])
        live = int(self.manifest["expert_live"])
        if slot < live or slot % PAGE:
            message = f"expert stride {slot} does not hold a {live}-byte image"
            raise RuntimeError(message)
        self.super_fds = [None] * layers
        for layer in range(layers):
            path = self.root / "experts" / quant_file(layer, quant)
            fd = os.open(path, os.O_RDONLY)
            self.cold_fds.append(fd)
            if os.fstat(fd).st_size < experts * slot:
                message = f"{path.name} is shorter than {experts} experts"
                raise RuntimeError(message)
        for fd in self.cold_fds:
            _bypass_cache(fd)
        self.ngram_fd = os.open(self.root / "ngram.bin", os.O_RDONLY)
        _bypass_cache(self.ngram_fd)
        pin_quant = int(self.manifest.get("super_bits") or quant)
        if pin_quant != quant:
            message = "a super pin quant differs from the expert file"
            raise RuntimeError(message)
        self.hot_bits = quant
        self.cold_bits = quant
        self.super_bits = pin_quant
        self.live = live
        self.slot = slot
        self.cold_live = live
        self.cold_slot = slot
        self.super_live = live
        self.super_slot = slot
        self.expert_scratch = mmap.mmap(-1, live)
        self.page_scratch = mmap.mmap(-1, PAGE)
        self.gate_up_bytes = sum(int4_plane_sizes(1280, stored_k_int4(2560, align_k=True)))

    def _open_quant_grid(self, quant: list[Any]) -> None:
        """Up to three widths. fp8 is tier 0, fp4 is tier 1, and int2 is tier 3."""
        layers = int(self.manifest["layers"])
        experts = int(self.manifest["experts"])
        if len(quant) != layers or any(len(row) != experts for row in quant):
            message = "expert quant does not match the checkpoint"
            raise RuntimeError(message)
        grid = [[int(bits) for bits in row] for row in quant]
        described = self.manifest.get("quants") or {}
        widths = sorted({bits for row in grid for bits in row}, reverse=True)
        if len(widths) > 3 or any(bits not in (2, 4, 8) for bits in widths):
            message = "expert quant uses more than three widths"
            raise RuntimeError(message)
        raw_super = self.manifest.get("super") or [[] for _ in range(layers)]
        self.super = [[int(expert) for expert in row] for row in raw_super]
        if len(self.super) != layers:
            message = f"super list does not match {layers} layers"
            raise RuntimeError(message)
        self.hot = [[] for _ in range(layers)]
        self._hot_sets = []
        self._cold_sets = []
        self._narrow_sets = []
        self._super_sets = []
        self._tier_row = []
        self._quant_grid = grid
        self._quant_row = []
        self._quant_live = {}
        self._quant_slot = {}
        self._quant_fds: dict[int, list[int | None]] = {bits: [] for bits in widths}
        for bits in widths:
            meta = described.get(str(bits)) or {}
            live = int(meta.get("live") or 0)
            slot = int(meta.get("slot") or 0)
            if live < 1 or slot < live or slot % PAGE:
                message = f"quant {bits} stride {slot} does not hold a {live}-byte image"
                raise RuntimeError(message)
            self._quant_live[bits] = live
            self._quant_slot[bits] = slot
        high = widths[0]
        low = widths[-1]
        for row, pin in zip(grid, self.super, strict=True):
            if len(pin) != len(set(pin)) or any(expert < 0 or expert >= experts for expert in pin):
                message = f"super pin is not ids in 0..{experts - 1}"
                raise RuntimeError(message)
            pin_set = set(pin)
            ranks = {bits: 0 for bits in widths}
            mapping: dict[int, tuple[int, int]] = {}
            placed: list[int] = []
            hot: set[int] = set()
            cold: set[int] = set()
            narrow: set[int] = set()
            for expert, bits in enumerate(row):
                row_index = ranks[bits]
                ranks[bits] += 1
                placed.append(row_index)
                if expert in pin_set:
                    mapping[expert] = (2, row_index)
                    continue
                plane = routed_tier(bits, widths)
                mapping[expert] = (plane, row_index)
                if plane == 0:
                    hot.add(expert)
                elif plane == 3:
                    narrow.add(expert)
                else:
                    cold.add(expert)
            self._quant_row.append(placed)
            self._tier_row.append(mapping)
            self._hot_sets.append(hot)
            self._cold_sets.append(cold)
            self._narrow_sets.append(narrow)
            self._super_sets.append(pin_set)
        for bits in widths:
            for layer, row in enumerate(grid):
                count = sum(1 for item in row if item == bits)
                path = self.root / "experts" / quant_file(layer, bits)
                if count == 0:
                    self._quant_fds[bits].append(None)
                    continue
                fd = os.open(path, os.O_RDONLY)
                if os.fstat(fd).st_size < count * self._quant_slot[bits]:
                    os.close(fd)
                    message = f"{path.name} is shorter than {count} experts"
                    raise RuntimeError(message)
                _bypass_cache(fd)
                self._quant_fds[bits].append(fd)
        if len(widths) == 3:
            self.hot_bits = 8
            self.cold_bits = 4
            self.narrow_bits = 2
            self.hot_fds = list(self._quant_fds[8])
            self.cold_fds = list(self._quant_fds[4])
            self.narrow_fds = list(self._quant_fds[2])
        elif len(widths) == 2:
            self.hot_bits = high
            self.cold_bits = low
            self.narrow_bits = 0
            self.hot_fds = list(self._quant_fds[high])
            self.cold_fds = list(self._quant_fds[low])
            self.narrow_fds = []
        else:
            self.hot_bits = low
            self.cold_bits = low
            self.narrow_bits = 0
            self.hot_fds = []
            self.cold_fds = list(self._quant_fds[low])
            self.narrow_fds = []
        self.super_fds = [None] * layers
        self.ngram_fd = os.open(self.root / "ngram.bin", os.O_RDONLY)
        _bypass_cache(self.ngram_fd)
        pin_quant = int(self.manifest.get("super_bits") or self.cold_bits)
        self.super_bits = pin_quant
        self.live = self._quant_live[self.hot_bits]
        self.slot = self._quant_slot[self.hot_bits]
        self.cold_live = self._quant_live[self.cold_bits]
        self.cold_slot = self._quant_slot[self.cold_bits]
        self.narrow_live = self._quant_live.get(self.narrow_bits, 0)
        self.narrow_slot = self._quant_slot.get(self.narrow_bits, 0)
        self.super_live = self._quant_live.get(pin_quant, self.cold_live)
        self.super_slot = self._quant_slot.get(pin_quant, self.cold_slot)
        self.expert_scratch = mmap.mmap(-1, max(self._quant_live.values()))
        self.page_scratch = mmap.mmap(-1, PAGE)
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
        mtp_path = self.root / "mtp-experts.bin"
        self.mtp_io = handle(mtp_path) if mtp_path.is_file() else None
        layers = int(self.manifest["layers"])
        if int(self.manifest.get("version") or 0) == QUANT_LAYOUT_VERSION:
            quant = self.manifest["expert_quant"]
            if isinstance(quant, list):
                self.hot_io = [
                    self._io_or_none(handle, layer, self.hot_bits) for layer in range(layers)
                ]
                self.cold_io = [
                    self._io_or_none(handle, layer, self.cold_bits) for layer in range(layers)
                ]
                self.narrow_io = [
                    self._io_or_none(handle, layer, self.narrow_bits) if self.narrow_bits else None
                    for layer in range(layers)
                ]
                return
            self.hot_io = []
            self.narrow_io = []
            self.cold_io = [
                handle(self.root / "experts" / quant_file(layer, int(quant)))
                for layer in range(layers)
            ]
            return
        self.hot_io = [
            handle(self.root / "experts" / f"{layer:02d}-hot.bin") for layer in range(layers)
        ]
        self.narrow_io = []
        self.cold_io = [
            handle(self.root / "experts" / f"{layer:02d}-cold.bin") for layer in range(layers)
        ]

    def _io_or_none(self, handle: Any, layer: int, bits: int) -> Any:
        path = self.root / "experts" / quant_file(layer, bits)
        if not path.is_file():
            return None
        return handle(path)

    def hot_set(self, layer: int) -> set[int]:
        return self._hot_sets[layer]

    def cold_set(self, layer: int) -> set[int]:
        return self._cold_sets[layer]

    def narrow_set(self, layer: int) -> set[int]:
        return self._narrow_sets[layer]

    def super_set(self, layer: int) -> set[int]:
        return self._super_sets[layer]

    def expert_bits(self, layer: int, expert: int) -> int:
        """Width of one expert. A uniform record uses the single stored width."""
        if self._quant_grid is not None:
            return int(self._quant_grid[layer][int(expert)])
        return int(self.cold_bits)

    def routed_ids(self, layer: int) -> list[tuple[int, set[int]]]:
        """Ids on each width, a pin included with the file that stores it."""
        groups: dict[int, set[int]] = {}

        def add(bits: int, ids: set[int]) -> None:
            if not bits:
                return
            groups.setdefault(int(bits), set()).update(ids)

        add(self.hot_bits, self.hot_set(layer))
        add(self.cold_bits, self.cold_set(layer))
        add(self.narrow_bits, self.narrow_set(layer))
        covered: set[int] = set()
        for ids in groups.values():
            covered |= ids
        for expert in self.super_set(layer) - covered:
            add(self.expert_bits(layer, expert), {expert})
        order: list[int] = []
        for bits in (self.hot_bits, self.cold_bits, self.narrow_bits, 8, 4, 2):
            width = int(bits)
            if width and width not in order and groups.get(width):
                order.append(width)
        return [(bits, groups[bits]) for bits in order]

    def tier_row(self, layer: int, expert: int) -> tuple[int, int]:
        return self._tier_row[layer][int(expert)]

    def stride(self, tier: int) -> int:
        if tier == 0:
            return self.slot
        if tier == 2:
            return self.super_slot
        if tier == 3:
            return self.narrow_slot
        return self.cold_slot

    def payload(self, tier: int) -> int:
        if tier == 0:
            return self.live
        if tier == 2:
            return self.super_live
        if tier == 3:
            return self.narrow_live
        return self.cold_live

    def io_handle(self, layer: int, tier: int) -> Any:
        if tier == 2:
            message = "a super expert is resident and is not staged from SSD"
            raise RuntimeError(message)
        if tier == 0:
            return self.hot_io[layer]
        if tier == 3:
            return self.narrow_io[layer]
        return self.cold_io[layer]

    def _image_fd(self, layer: int, tier: int) -> int:
        if tier == 2 and self.super_fds[layer] is not None:
            return int(self.super_fds[layer])
        if tier == 0:
            return self.hot_fds[layer]
        if tier == 3:
            fd = self.narrow_fds[layer]
            if fd is None:
                message = f"layer {layer} has no int2 file"
                raise RuntimeError(message)
            return fd
        return self.cold_fds[layer]

    def expert_image(self, layer: int, expert: int) -> bytes:
        if self._quant_grid is not None:
            bits = self._quant_grid[layer][int(expert)]
            row = self._quant_row[layer][int(expert)]
            fd = self._quant_fds[bits][layer]
            if fd is None:
                message = f"layer {layer} has no quant {bits} file"
                raise RuntimeError(message)
            view = pread_exact(
                fd,
                self._quant_live[bits],
                row * self._quant_slot[bits],
                self.expert_scratch,
            )
            blob = bytes(view)
            del view
            return blob
        tier, row = self.tier_row(layer, expert)
        fd = self._image_fd(layer, tier)
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

    def release_io(self) -> None:
        """Drop Metal IO handles while the device is still alive."""
        self.hot_io = []
        self.cold_io = []
        self.narrow_io = []
        self.resident_io = None
        self.ngram_io = None
        self.mtp_io = None

    def close(self) -> None:
        """Close record fds and scratch mappings. A second call does nothing."""
        if self._closed:
            return
        self._closed = True
        self.release_io()
        extra: list[int | None] = []
        for group in getattr(self, "_quant_fds", {}).values():
            extra.extend(group)
        fds: list[int] = []
        seen: set[int] = set()
        for fd in (*self.hot_fds, *self.cold_fds, *self.super_fds, self.ngram_fd, *extra):
            if isinstance(fd, int) and fd not in seen:
                seen.add(fd)
                fds.append(fd)
        self.hot_fds = []
        self.cold_fds = []
        self.narrow_fds = []
        self.super_fds = []
        self._quant_fds = {}
        self.ngram_fd = None
        for fd in fds:
            try:
                os.close(fd)
            except OSError:
                pass
        for name in ("expert_scratch", "page_scratch"):
            scratch = getattr(self, name, None)
            if scratch is not None:
                scratch.close()
                setattr(self, name, None)
        self.pages.clear()


class _Bf16Ngram:
    """Gather one BF16 n-gram row from the checkpoint shards."""

    def __init__(self, store: ShardStore) -> None:
        self.store = store

    def ngram_vector(self, row: int) -> np.ndarray:
        return self.store.ngram_row(row, 2_500_012, NGRAM_WIDTH)
