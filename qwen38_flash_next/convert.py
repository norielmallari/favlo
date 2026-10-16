"""Write the Metal record from Hugging Face BF16 shards."""

from __future__ import annotations

import ctypes
import fcntl
import json
import mmap
import os
import shutil
import struct
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np

from qwen38_flash_next.allocate import preallocate, round_preferred, seal_contiguous
from qwen38_flash_next.formats import (
    NGRAM_SLOT,
    NGRAM_WIDTH,
    PAGE,
    QUANT_LAYOUT_VERSION,
    RECORD_VERSION,
    int4_plane_sizes,
    ngram_slots,
    quant_file,
    quantize_int2_image,
    quantize_int4_image,
    quantize_int8_image,
    tier_image_bytes,
)
from qwen38_flash_next.session import _log, bf16_to_f32


def _plane_format(bits: int) -> str:
    if bits == 8:
        return "fp8_e4m3"
    if bits == 4:
        return "fp4_e2m1"
    if bits == 2:
        return "int2"
    message = f"expert width {bits} is not 2, 4, or 8"
    raise ValueError(message)


class ShardStore:
    def __init__(self, root: Path) -> None:
        self.root = root
        index = json.loads((root / "model.safetensors.index.json").read_text())
        self.weight_map: dict[str, str] = index["weight_map"]
        self._headers: dict[str, tuple[int, dict[str, Any]]] = {}
        self._fds: dict[str, int] = {}

    def close(self) -> None:
        for fd in self._fds.values():
            os.close(fd)
        self._fds.clear()
        self._headers.clear()

    def _locate(self, key: str) -> tuple[int, int, str, list[int]]:
        shard = self.weight_map[key]
        if shard not in self._headers:
            path = self.root / shard
            with path.open("rb") as handle:
                header_len = struct.unpack("<Q", handle.read(8))[0]
                header = json.loads(handle.read(header_len))
            self._headers[shard] = (8 + header_len, header)
            fd = os.open(path, os.O_RDONLY)
            _bypass_cache(fd)
            self._fds[shard] = fd
        base, header = self._headers[shard]
        meta = header[key]
        start, _end = meta["data_offsets"]
        return self._fds[shard], base + start, meta["dtype"], meta["shape"]

    def read_bf16(
        self, key: str, elem_offset: int, elem_count: int, shape: tuple[int, ...]
    ) -> np.ndarray:
        fd, start, dtype, _shape = self._locate(key)
        if dtype != "BF16":
            message = f"{key} is {dtype}, expected BF16"
            raise RuntimeError(message)
        raw = os.pread(fd, elem_count * 2, start + elem_offset * 2)
        return bf16_to_f32(raw).reshape(shape)

    def matrix(self, key: str) -> np.ndarray:
        _fd, _start, _dtype, shape = self._locate(key)
        count = int(np.prod(shape))
        return self.read_bf16(key, 0, count, tuple(shape))

    def i64(self, key: str) -> np.ndarray:
        fd, start, dtype, shape = self._locate(key)
        if dtype != "I64":
            message = f"{key} is {dtype}, expected I64"
            raise RuntimeError(message)
        count = int(np.prod(shape))
        raw = os.pread(fd, count * 8, start)
        return np.frombuffer(raw, dtype=np.int64).reshape(shape).copy()

    def expert(self, layer: int, name: str, expert: int) -> np.ndarray:
        key = f"model.language_model.layers.{layer}.mlp.experts.{name}"
        _fd, _start, _dtype, shape = self._locate(key)
        _experts, out, inner = shape
        count = out * inner
        return self.read_bf16(key, expert * count, count, (out, inner))

    def ngram_row(self, row: int, rows_per_shard: int, width: int) -> np.ndarray:
        shard = row // rows_per_shard
        local = row % rows_per_shard
        key = (
            "model.language_model.layers.1.ple.ple_embedding."
            f"ngram_embedding.shard_{shard}.weight"
        )
        return self.read_bf16(key, local * width, width, (width,))


_LIBC = ctypes.CDLL(None, use_errno=True)
_LIBC.pread.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int64]
_LIBC.pread.restype = ctypes.c_ssize_t


def _bypass_cache(fd: int) -> None:
    flag = getattr(fcntl, "F_NOCACHE", None)
    if flag is not None:
        fcntl.fcntl(fd, flag, 1)


def pread_exact(fd: int, nbytes: int, offset: int, scratch: mmap.mmap) -> memoryview:
    if len(scratch) < nbytes:
        message = f"scratch {len(scratch)} is smaller than {nbytes}"
        raise RuntimeError(message)
    holder = ctypes.c_char.from_buffer(scratch)
    addr = ctypes.addressof(holder)
    done = 0
    while done < nbytes:
        got = _LIBC.pread(fd, addr + done, nbytes - done, offset + done)
        if got < 0:
            err = ctypes.get_errno()
            raise OSError(err, os.strerror(err))
        if got == 0:
            message = f"short read at {offset + done}"
            raise OSError(message)
        done += int(got)
    del holder
    return memoryview(scratch)[:nbytes]


def _expert_matrix(store: ShardStore, key: str, expert: int) -> np.ndarray:
    _fd, _start, _dtype, shape = store._locate(key)
    _count, out, inner = shape
    return store.read_bf16(key, expert * out * inner, out * inner, (out, inner))


class ResidentWriter:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o644)
        os.ftruncate(self.fd, 0)
        self.offset = 0
        self.int4: dict[str, dict[str, int]] = {}
        self.f16: dict[str, dict[str, int]] = {}
        self.f32: dict[str, dict[str, Any]] = {}
        self.i64: dict[str, dict[str, Any]] = {}

    def _write(self, blob: bytes) -> int:
        start = self.offset
        os.pwrite(self.fd, blob, start)
        self.offset = start + len(blob)
        pad = (128 - self.offset % 128) % 128
        if pad:
            os.pwrite(self.fd, b"\x00" * pad, self.offset)
            self.offset += pad
        return start

    def add_int4(self, key: str, matrix: np.ndarray) -> None:
        image, stored = quantize_int4_image(matrix, align_k=True)
        rows, logical = int(matrix.shape[0]), int(matrix.shape[1])
        data_n, scale_n = int4_plane_sizes(rows, stored)
        self.int4[key] = {
            "offset": self._write(image),
            "rows": rows,
            "logical_inner": logical,
            "stored_inner": stored,
            "data": data_n,
            "scales": scale_n,
        }

    def add_f16(self, key: str, matrix: np.ndarray) -> None:
        half = np.clip(np.ascontiguousarray(matrix, dtype=np.float32), -65504.0, 65504.0)
        blob = np.ascontiguousarray(half, dtype=np.float16).tobytes()
        self.f16[key] = {
            "offset": self._write(blob),
            "rows": int(matrix.shape[0]),
            "inner": int(matrix.shape[1]),
        }

    def add_f32(self, key: str, array: np.ndarray) -> None:
        blob = np.ascontiguousarray(array, dtype=np.float32).tobytes()
        self.f32[key] = {"offset": self._write(blob), "shape": list(array.shape)}

    def add_i64(self, key: str, array: np.ndarray) -> None:
        blob = np.ascontiguousarray(array, dtype=np.int64).tobytes()
        self.i64[key] = {"offset": self._write(blob), "shape": list(array.shape)}

    def close(self) -> None:
        padded = round_preferred(self.offset)
        if padded > self.offset:
            os.pwrite(self.fd, b"\x00" * (padded - self.offset), self.offset)
            self.offset = padded
        os.fsync(self.fd)
        os.close(self.fd)
        seal_contiguous(self.path, self.offset)


def _progress_path(dest: Path) -> Path:
    return dest / "quantize_progress.json"


LAYER_WORKERS = 4
NGRAM_ROWS_PER_SHARD = 2_500_012
NGRAM_SHARD_COUNT = 128


def pending_indices(done: Sequence[int], total: int) -> list[int]:
    """Indices that are not in the finished set."""
    finished = {int(item) for item in done}
    return [index for index in range(total) if index not in finished]


def ngram_byte_offset(shard: int, local: int) -> int:
    """Byte offset of one n-gram row. Shard tensors do not share a row range."""
    return (shard * NGRAM_ROWS_PER_SHARD + local) * NGRAM_SLOT


def load_quant_map(
    path: Path, layers: int, experts: int
) -> tuple[list[list[int]], list[list[int]]]:
    """Read a quant map. One width per expert, and the super pin. At most two widths."""
    raw = json.loads(path.read_text())
    if not isinstance(raw, dict) or "experts" not in raw or "super" not in raw:
        message = "quant map lists experts and super"
        raise RuntimeError(message)
    rows = raw["experts"]
    if not isinstance(rows, list) or len(rows) != layers:
        found = len(rows) if isinstance(rows, list) else "no"
        message = f"quant map has {found} layers, the checkpoint has {layers}"
        raise RuntimeError(message)
    grid: list[list[int]] = []
    for layer, row in enumerate(rows):
        if not isinstance(row, list) or len(row) != experts:
            message = f"quant map layer {layer} does not list {experts} experts"
            raise RuntimeError(message)
        widths: list[int] = []
        for expert, bits in enumerate(row):
            if isinstance(bits, bool) or int(bits) not in (2, 4, 8):
                message = f"quant map {layer}:{expert} is not a width in 2, 4, 8"
                raise RuntimeError(message)
            widths.append(int(bits))
        grid.append(widths)
    used = {bits for row in grid for bits in row}
    if len(used) > 2:
        message = "a quant map uses at most two widths"
        raise RuntimeError(message)
    return grid, _super_rows(raw["super"], layers, experts)


def _super_rows(items: Any, layers: int, experts: int) -> list[list[int]]:
    """Layer index to the super pin, in the order the map lists them."""
    if not isinstance(items, list):
        message = "quant map super is not a list"
        raise RuntimeError(message)
    rows: list[list[int]] = [[] for _ in range(layers)]
    for item in items:
        if not isinstance(item, dict):
            message = "quant map super entry is not a layer and expert"
            raise RuntimeError(message)
        layer = item.get("layer")
        expert = item.get("expert")
        if (
            isinstance(layer, bool)
            or isinstance(expert, bool)
            or not isinstance(layer, int)
            or not isinstance(expert, int)
            or layer < 0
            or layer >= layers
            or expert < 0
            or expert >= experts
        ):
            message = f"super expert {layer}:{expert} is outside the model"
            raise RuntimeError(message)
        if expert in rows[layer]:
            message = f"super expert {expert} is repeated on layer {layer}"
            raise RuntimeError(message)
        rows[layer].append(expert)
    return rows


def quant_groups(widths: Sequence[int]) -> dict[int, list[int]]:
    """Expert ids of each width, in id order. That order is the file row."""
    groups: dict[int, list[int]] = {}
    for expert, bits in enumerate(widths):
        width = int(bits)
        if width not in (2, 4, 8):
            message = f"expert quant {width} is not 2, 4, or 8"
            raise RuntimeError(message)
        groups.setdefault(width, []).append(expert)
    return groups


def _load_progress(dest: Path) -> dict[str, Any]:
    path = _progress_path(dest)
    if not path.is_file():
        return {
            "version": QUANT_LAYOUT_VERSION,
            "resident": False,
            "layers": [],
            "mtp": False,
            "ngram_shards": [],
        }
    progress = json.loads(path.read_text())
    if int(progress.get("version") or 0) not in (RECORD_VERSION, QUANT_LAYOUT_VERSION):
        message = f"delete {dest}; it was started with record version {progress.get('version')}"
        raise RuntimeError(message)
    return progress


def _save_progress(dest: Path, progress: dict[str, Any]) -> None:
    path = _progress_path(dest)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(progress))
    fd = os.open(tmp, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    os.replace(tmp, path)
    directory = os.open(dest, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def ngram_resume_row(saved: int, file_size: int) -> int:
    """Keep the checkpoint only for rows that are already in the file."""
    durable = max(file_size, 0) // NGRAM_SLOT
    if saved > durable:
        return durable
    return max(saved, 0)


def _expert_start(
    progress: dict[str, Any],
    *,
    done: bool,
    inflight_key: int | str,
    path: Path,
    experts: int,
    slot: int,
) -> int:
    """First expert still to write. ``experts`` means the file is finished."""
    full = experts * slot
    size = path.stat().st_size if path.is_file() else 0
    if done and size >= full:
        return experts
    if progress.get("inflight_key") == inflight_key and size >= full:
        nxt = int(progress.get("inflight_expert", -1)) + 1
        if nxt >= experts:
            return experts
        if nxt > 0:
            return nxt
    return 0


def _clear_inflight(progress: dict[str, Any]) -> None:
    progress.pop("inflight_key", None)
    progress.pop("inflight_expert", None)


def _layer_int4_keys(prefix: str, kind: str) -> list[str]:
    keys = [
        f"{prefix}.attn_hyper_connection.input_mix_weight_down.weight",
        f"{prefix}.attn_hyper_connection.input_mix_weight_up.weight",
        f"{prefix}.attn_hyper_connection.block_inject_weight.weight",
        f"{prefix}.mlp_hyper_connection.input_mix_weight_down.weight",
        f"{prefix}.mlp_hyper_connection.input_mix_weight_up.weight",
        f"{prefix}.mlp_hyper_connection.block_inject_weight.weight",
        f"{prefix}.mlp.shared_expert.gate_proj.weight",
        f"{prefix}.mlp.shared_expert.up_proj.weight",
        f"{prefix}.mlp.shared_expert.down_proj.weight",
        f"{prefix}.mlp.shared_expert_gate.weight",
    ]
    if kind == "linear_attention":
        keys.extend(
            [
                f"{prefix}.linear_attn.in_proj_qkv.weight",
                f"{prefix}.linear_attn.in_proj_z.weight",
                f"{prefix}.linear_attn.in_proj_a.weight",
                f"{prefix}.linear_attn.in_proj_b.weight",
                f"{prefix}.linear_attn.out_proj.weight",
            ]
        )
    elif kind == "full_attention":
        keys.extend(
            [
                f"{prefix}.self_attn.q_proj.weight",
                f"{prefix}.self_attn.k_proj.weight",
                f"{prefix}.self_attn.v_proj.weight",
                f"{prefix}.self_attn.o_proj.weight",
                f"{prefix}.self_attn.indexer.index_qk_proj.weight",
            ]
        )
    else:
        message = f"unknown layer type {kind}"
        raise RuntimeError(message)
    return keys


def _layer_f32_keys(prefix: str, kind: str) -> list[str]:
    keys = [
        f"{prefix}.attn_hyper_connection.hc_norm.weight",
        f"{prefix}.mlp_hyper_connection.hc_norm.weight",
    ]
    if kind == "linear_attention":
        keys.extend(
            [
                f"{prefix}.linear_attn.conv1d.weight",
                f"{prefix}.linear_attn.A_log",
                f"{prefix}.linear_attn.dt_bias",
                f"{prefix}.linear_attn.norm.weight",
            ]
        )
    else:
        keys.extend(
            [
                f"{prefix}.self_attn.q_norm.weight",
                f"{prefix}.self_attn.k_norm.weight",
            ]
        )
    return keys


def write_tier_file(path: Path, images: Sequence[bytes], slot: int) -> None:
    """Preallocate one tier file and write each image at ``row * slot``."""
    if slot % PAGE:
        message = f"tier stride {slot} is not a multiple of {PAGE}"
        raise RuntimeError(message)
    if not images:
        message = f"{path.name} has no images"
        raise RuntimeError(message)
    fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        preallocate(fd, len(images) * slot)
        for row, image in enumerate(images):
            if len(image) > slot:
                message = f"{path.name} image {row} is longer than the stride"
                raise RuntimeError(message)
            os.pwrite(fd, image, row * slot)
        os.fsync(fd)
    finally:
        os.close(fd)


def _open_tier(path: Path, count: int, slot: int) -> int:
    if count < 1 or slot % PAGE:
        message = f"{path.name} stride {slot} count {count} is not page-aligned"
        raise RuntimeError(message)
    fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        preallocate(fd, count * slot)
    except Exception:
        os.close(fd)
        raise
    return fd


def _expert_image(gate: np.ndarray, down: np.ndarray, bits: int) -> bytes:
    if bits == 8:
        pack = quantize_int8_image
    elif bits == 4:
        pack = quantize_int4_image
    elif bits == 2:
        pack = quantize_int2_image
    else:
        message = f"expert width {bits} is not 2, 4, or 8"
        raise ValueError(message)
    return pack(gate)[0] + pack(down)[0]


def _layer_ready(dest: Path, layer: int, widths: Sequence[int]) -> bool:
    root = dest / "experts"
    for bits, ids in quant_groups(widths).items():
        _live, slot = tier_image_bytes(bits)
        path = root / quant_file(layer, bits)
        if not path.is_file() or path.stat().st_size < len(ids) * slot:
            return False
    return True


def pack_expert_layer(source: str, dest: str, layer: int, widths: list[int]) -> int:
    """Pack one layer. Each width is its own file, experts of that width in id order."""
    store = ShardStore(Path(source))
    prefix = f"model.language_model.layers.{layer}.mlp.experts"
    gate_key = f"{prefix}.gate_up_proj"
    down_key = f"{prefix}.down_proj"
    root = Path(dest) / "experts"
    groups = quant_groups(widths)
    opened: list[tuple[int, dict[int, int], int, int]] = []
    try:
        _log(f"quantize experts layer {layer}")
        for bits, ids in groups.items():
            live, slot = tier_image_bytes(bits)
            fd = _open_tier(root / quant_file(layer, bits), len(ids), slot)
            opened.append((fd, {expert: row for row, expert in enumerate(ids)}, live, slot))
        for expert, bits in enumerate(widths):
            gate = _expert_matrix(store, gate_key, expert)
            down = _expert_matrix(store, down_key, expert)
            if gate.shape != (1280, 2560) or down.shape != (2560, 640):
                message = f"{gate_key} shapes {gate.shape} {down.shape}"
                raise RuntimeError(message)
            image = _expert_image(gate, down, int(bits))
            fd, rows, live, slot = next(item for item in opened if expert in item[1])
            if len(image) != live:
                message = f"expert image is {len(image)} bytes, expected {live}"
                raise RuntimeError(message)
            os.pwrite(fd, image, rows[expert] * slot)
            if expert % 64 == 0:
                _log(f"layer {layer} expert {expert}")
        for fd, _rows, _live, _slot in opened:
            os.fsync(fd)
    finally:
        for fd, _rows, _live, _slot in opened:
            os.close(fd)
    return layer


def _ngram_key(shard: int) -> str:
    return (
        "model.language_model.layers.1.ple.ple_embedding."
        f"ngram_embedding.shard_{shard}.weight"
    )


def ngram_file_groups(weight_map: dict[str, str], shards: Sequence[int]) -> list[list[int]]:
    """One group per safetensors file, so a file has one reader."""
    groups: dict[str, list[int]] = {}
    order: list[str] = []
    for shard in shards:
        file = weight_map[_ngram_key(int(shard))]
        if file not in groups:
            groups[file] = []
            order.append(file)
        groups[file].append(int(shard))
    return [groups[file] for file in order]


def pack_ngram_shards(source: str, dest: str, shards: list[int]) -> list[int]:
    """Write finished n-gram tensors at their fixed offsets. Does not truncate the file."""
    store = ShardStore(Path(source))
    path = Path(dest) / "ngram.bin"
    fd = os.open(path, os.O_RDWR)
    try:
        for shard in shards:
            key = _ngram_key(shard)
            _fd, _start, _dtype, shape = store._locate(key)
            if list(shape) != [NGRAM_ROWS_PER_SHARD, NGRAM_WIDTH]:
                message = f"{key} shape {shape}"
                raise RuntimeError(message)
            local = 0
            while local < NGRAM_ROWS_PER_SHARD:
                count = min(4096, NGRAM_ROWS_PER_SHARD - local)
                matrix = store.read_bf16(
                    key, local * NGRAM_WIDTH, count * NGRAM_WIDTH, (count, NGRAM_WIDTH)
                )
                os.pwrite(fd, ngram_slots(matrix), ngram_byte_offset(shard, local))
                local += count
            _log(f"n-gram shard {shard}")
        os.fsync(fd)
    finally:
        os.close(fd)
    return shards


def _write_expert_file(
    store: ShardStore,
    path: Path,
    gate_key: str,
    down_key: str,
    experts: int,
    start: int = 0,
    commit: Callable[[int], None] | None = None,
    bits: int = 4,
) -> None:
    live, slot = tier_image_bytes(bits)
    fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        full = experts * slot
        if start == 0 or os.fstat(fd).st_size < full:
            preallocate(fd, full)
        for expert in range(start, experts):
            gate = _expert_matrix(store, gate_key, expert)
            down = _expert_matrix(store, down_key, expert)
            if gate.shape != (1280, 2560) or down.shape != (2560, 640):
                message = f"{gate_key} shapes {gate.shape} {down.shape}"
                raise RuntimeError(message)
            image = _expert_image(gate, down, bits)
            if len(image) != live:
                message = f"expert image is {len(image)} bytes, expected {live}"
                raise RuntimeError(message)
            os.pwrite(fd, image, expert * slot)
            if expert % 64 == 0:
                _log(f"{path.name} expert {expert}")
            if commit is not None and (expert % 64 == 63 or expert == experts - 1):
                os.fsync(fd)
                commit(expert)
    finally:
        os.close(fd)


def quantize_checkpoint(
    source: Path,
    dest: Path,
    quants: list[list[int]],
    supers: list[list[int]],
    mtp_bits: int = 4,
) -> bool:
    """Write the Metal record once. A later run continues from quantize_progress.json.

    Returns true when this call wrote the manifest. A record that is already complete
    returns false and leaves the packed files in place. ``quants`` is the width of
    every expert. ``supers`` is the pin, and those images stay in the file for their width.
    """
    if not (source / "config.json").is_file():
        message = f"checkpoint config is missing: {source}"
        raise FileNotFoundError(message)
    dest.mkdir(parents=True, exist_ok=True)
    manifest_path = dest / "manifest.json"
    if manifest_path.is_file():
        done = json.loads(manifest_path.read_text())
        if done.get("complete") and int(done.get("version") or 0) == QUANT_LAYOUT_VERSION:
            _log(f"record already complete: {dest}")
            return False
        if done.get("complete"):
            message = f"delete {dest}; it is record version {done.get('version')}"
            raise RuntimeError(message)
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json"):
        shutil.copyfile(source / name, dest / name)
    progress = _load_progress(dest)
    if int(progress.get("version") or 0) != QUANT_LAYOUT_VERSION:
        message = f"delete {dest}; it was started with record version {progress.get('version')}"
        raise RuntimeError(message)
    store = ShardStore(source)
    config = json.loads((source / "config.json").read_text())
    text = config["text_config"]
    kinds = list(text["layer_types"])
    experts = int(text["num_experts"])
    if len(quants) != len(kinds) or any(len(row) != experts for row in quants):
        message = "quant map does not match the checkpoint"
        raise RuntimeError(message)
    if len(supers) != len(kinds):
        message = "quant map super does not match the checkpoint"
        raise RuntimeError(message)
    super_ids = supers
    pin_widths = {
        int(quants[layer][expert])
        for layer, row in enumerate(super_ids)
        for expert in row
    }
    if len(pin_widths) > 1:
        message = "super experts do not share one quant"
        raise RuntimeError(message)
    _plane_format(mtp_bits)
    _mtp_live, mtp_slot = tier_image_bytes(mtp_bits)
    saved_quant = progress.get("quant")
    if saved_quant is None:
        progress["quant"] = quants
        progress["super"] = super_ids
        progress["mtp_bits"] = mtp_bits
        _save_progress(dest, progress)
    elif saved_quant != quants or progress.get("super") != super_ids or int(
        progress.get("mtp_bits") or 4
    ) != mtp_bits:
        message = "quant map does not match this partial record"
        raise RuntimeError(message)
    if progress["resident"] and not (dest / "resident.bin").is_file():
        progress["resident"] = False
    if not progress["resident"]:
        _log("quantize resident matrices")
        writer = ResidentWriter(dest / "resident.bin")
        try:
            writer.add_int4("lm_head.weight", store.matrix("lm_head.weight"))
            embed_key = "model.language_model.embed_tokens.weight"
            writer.add_int4(embed_key, store.matrix(embed_key))
            for layer, kind in enumerate(kinds):
                prefix = f"model.language_model.layers.{layer}"
                for key in _layer_int4_keys(prefix, kind):
                    writer.add_int4(key, store.matrix(key))
                gate_key = f"{prefix}.mlp.gate.weight"
                writer.add_f16(gate_key, store.matrix(gate_key))
                for key in _layer_f32_keys(prefix, kind):
                    writer.add_f32(key, store.matrix(key))
                _log(f"resident layer {layer}")
            ple = "model.language_model.layers.1.ple"
            table = f"{ple}.ple_embedding"
            for key in (f"{ple}.key_proj.weight", f"{ple}.value_proj.weight"):
                writer.add_int4(key, store.matrix(key))
            for key in (
                f"{ple}.conv1d.weight",
                f"{ple}.norm_conv.weight",
                f"{ple}.norm_key.weight",
                f"{ple}.norm_query.weight",
            ):
                writer.add_f32(key, store.matrix(key))
            for key in (
                f"{table}.layer_multipliers",
                f"{table}.ngram_heads_offsets",
                f"{table}.ngram_heads_vocab_sizes",
            ):
                writer.add_i64(key, store.i64(key))
            mixer = "model.language_model.hyper_connection_mixer"
            writer.add_f32(f"{mixer}.hc_norm.weight", store.matrix(f"{mixer}.hc_norm.weight"))
            writer.add_int4(
                f"{mixer}.input_mix_weight_down.weight",
                store.matrix(f"{mixer}.input_mix_weight_down.weight"),
            )
            writer.add_int4(
                f"{mixer}.input_mix_weight_up.weight",
                store.matrix(f"{mixer}.input_mix_weight_up.weight"),
            )
            mtp = "mtp.layers.0"
            for key in _layer_int4_keys(mtp, "full_attention"):
                writer.add_int4(key, store.matrix(key))
            writer.add_f16(f"{mtp}.mlp.gate.weight", store.matrix(f"{mtp}.mlp.gate.weight"))
            for key in _layer_f32_keys(mtp, "full_attention"):
                writer.add_f32(key, store.matrix(key))
            for key in ("mtp.fc_embedding.weight", "mtp.fc_hidden.weight"):
                writer.add_int4(key, store.matrix(key))
            mtp_mix = "mtp.hyper_connection_mixer"
            writer.add_f32(f"{mtp_mix}.hc_norm.weight", store.matrix(f"{mtp_mix}.hc_norm.weight"))
            writer.add_int4(
                f"{mtp_mix}.input_mix_weight_down.weight",
                store.matrix(f"{mtp_mix}.input_mix_weight_down.weight"),
            )
            writer.add_int4(
                f"{mtp_mix}.input_mix_weight_up.weight",
                store.matrix(f"{mtp_mix}.input_mix_weight_up.weight"),
            )
            for key in ("mtp.pre_fc_norm_embedding.weight", "mtp.pre_fc_norm_hidden.weight"):
                writer.add_f32(key, store.matrix(key))
        finally:
            writer.close()
        progress["resident"] = True
        progress["catalog"] = {
            "int4": writer.int4,
            "f16": writer.f16,
            "f32": writer.f32,
            "i64": writer.i64,
        }
        _save_progress(dest, progress)
        _log("resident record written")
    catalog = progress["catalog"]
    expert_dir = dest / "experts"
    expert_dir.mkdir(exist_ok=True)
    done_layers = {int(layer) for layer in progress["layers"]}
    pending_layers: list[int] = []
    for layer in range(len(kinds)):
        if layer in done_layers and _layer_ready(dest, layer, quants[layer]):
            continue
        if layer in done_layers:
            progress["layers"] = [item for item in progress["layers"] if int(item) != layer]
            done_layers.discard(layer)
        pending_layers.append(layer)
    if pending_layers:
        store.close()
        _log(f"quantize experts, {len(pending_layers)} layers, {LAYER_WORKERS} workers")
        with ProcessPoolExecutor(max_workers=LAYER_WORKERS) as pool:
            futures = [
                pool.submit(pack_expert_layer, str(source), str(dest), layer, quants[layer])
                for layer in pending_layers
            ]
            for future in as_completed(futures):
                layer = int(future.result())
                progress["layers"].append(layer)
                done_layers.add(layer)
                _save_progress(dest, progress)
                _log(f"experts layer {layer} done")
    mtp_path = dest / "mtp-experts.bin"
    mtp_start = _expert_start(
        progress,
        done=bool(progress["mtp"]),
        inflight_key="mtp",
        path=mtp_path,
        experts=experts,
        slot=mtp_slot,
    )
    if mtp_start >= experts:
        if not progress["mtp"]:
            progress["mtp"] = True
            _clear_inflight(progress)
            _save_progress(dest, progress)
    else:
        if progress["mtp"]:
            progress["mtp"] = False
        if mtp_start:
            _log(f"resume mtp experts at {mtp_start}")
        else:
            _log("quantize mtp experts")

        def commit_mtp(expert: int) -> None:
            progress["inflight_key"] = "mtp"
            progress["inflight_expert"] = expert
            _save_progress(dest, progress)

        _write_expert_file(
            store,
            mtp_path,
            "mtp.layers.0.mlp.experts.gate_up_proj",
            "mtp.layers.0.mlp.experts.down_proj",
            experts,
            mtp_start,
            commit_mtp,
            mtp_bits,
        )
        progress["mtp"] = True
        _clear_inflight(progress)
        _save_progress(dest, progress)
    _write_ngram(store, dest, progress)
    widths = sorted({bits for row in quants for bits in row}, reverse=True)
    pin_bits = next(iter(pin_widths)) if pin_widths else widths[-1]
    pin_live, pin_slot = tier_image_bytes(pin_bits)
    manifest: dict[str, Any] = {
        "version": QUANT_LAYOUT_VERSION,
        "complete": True,
        "super": super_ids,
        "super_bits": pin_bits,
        "super_format": _plane_format(pin_bits),
        "super_live": pin_live,
        "super_slot": pin_slot,
        "mtp_bits": mtp_bits,
        "mtp_format": _plane_format(mtp_bits),
        "mtp_slot": mtp_slot,
        "experts": experts,
        "layers": len(kinds),
        "ngram_rows": NGRAM_SHARD_COUNT * NGRAM_ROWS_PER_SHARD,
        "ngram_slot": NGRAM_SLOT,
        "ngram_width": NGRAM_WIDTH,
        "embed_key": "model.language_model.embed_tokens.weight",
        "head_key": "lm_head.weight",
        **catalog,
    }
    if len(widths) == 1:
        live, slot = tier_image_bytes(widths[0])
        manifest["expert_quant"] = widths[0]
        manifest["expert_format"] = _plane_format(widths[0])
        manifest["expert_live"] = live
        manifest["expert_slot"] = slot
    else:
        described: dict[str, Any] = {}
        for bits in widths:
            live, slot = tier_image_bytes(bits)
            described[str(bits)] = {
                "format": _plane_format(bits),
                "live": live,
                "slot": slot,
            }
        manifest["expert_quant"] = quants
        manifest["quants"] = described
    tmp = manifest_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(manifest))
    manifest_fd = os.open(tmp, os.O_RDONLY)
    try:
        os.fsync(manifest_fd)
    finally:
        os.close(manifest_fd)
    os.replace(tmp, manifest_path)
    _log(f"record complete: {dest}")
    return True


def _write_ngram(store: ShardStore, dest: Path, progress: dict[str, Any]) -> None:
    path = dest / "ngram.bin"
    total_bytes = NGRAM_SHARD_COUNT * NGRAM_ROWS_PER_SHARD * NGRAM_SLOT
    done = [int(shard) for shard in progress.get("ngram_shards", [])]
    pending = pending_indices(done, NGRAM_SHARD_COUNT)
    fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        size = os.fstat(fd).st_size
        if not done and size < total_bytes:
            preallocate(fd, total_bytes)
        elif size < total_bytes:
            message = f"n-gram file is {size} bytes, expected {total_bytes}"
            raise RuntimeError(message)
    finally:
        os.close(fd)
    if not pending:
        return
    groups = ngram_file_groups(store.weight_map, pending)
    _log(f"quantize n-gram, {len(pending)} tensors, {len(groups)} files")
    with ProcessPoolExecutor(max_workers=LAYER_WORKERS) as pool:
        futures = [
            pool.submit(pack_ngram_shards, str(store.root), str(dest), group) for group in groups
        ]
        for future in as_completed(futures):
            finished = [int(shard) for shard in future.result()]
            progress.setdefault("ngram_shards", [])
            progress["ngram_shards"].extend(finished)
            _save_progress(dest, progress)
            _log(f"n-gram shards {len(progress['ngram_shards'])}/{NGRAM_SHARD_COUNT}")
