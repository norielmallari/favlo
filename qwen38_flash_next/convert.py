"""Write the Metal record from Hugging Face BF16 shards."""

from __future__ import annotations

import ctypes
import fcntl
import json
import mmap
import os
import shutil
import struct
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from qwen38_flash_next.allocate import preallocate, round_preferred, seal_contiguous
from qwen38_flash_next.formats import (
    NGRAM_SLOT,
    NGRAM_WIDTH,
    RECORD_VERSION,
    expert_live_bytes,
    expert_slot_bytes,
    int4_plane_sizes,
    ngram_slots,
    quantize_int4_image,
)
from qwen38_flash_next.session import _log, bf16_to_f32


class ShardStore:
    def __init__(self, root: Path) -> None:
        self.root = root
        index = json.loads((root / "model.safetensors.index.json").read_text())
        self.weight_map: dict[str, str] = index["weight_map"]
        self._headers: dict[str, tuple[int, dict[str, Any]]] = {}
        self._fds: dict[str, int] = {}

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


def _load_progress(dest: Path) -> dict[str, Any]:
    path = _progress_path(dest)
    if not path.is_file():
        return {
            "version": RECORD_VERSION,
            "resident": False,
            "layers": [],
            "mtp": False,
            "ngram_rows": 0,
        }
    progress = json.loads(path.read_text())
    if progress.get("version") != RECORD_VERSION:
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
) -> int:
    """First expert still to write. ``experts`` means the file is finished."""
    full = experts * expert_slot_bytes()
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


def _write_expert_file(
    store: ShardStore,
    path: Path,
    gate_key: str,
    down_key: str,
    experts: int,
    start: int = 0,
    commit: Callable[[int], None] | None = None,
) -> None:
    live = expert_live_bytes()
    slot = expert_slot_bytes()
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
            image = quantize_int4_image(gate)[0] + quantize_int4_image(down)[0]
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


def quantize_checkpoint(source: Path, dest: Path) -> None:
    """Write the Metal record once. A later run continues from quantize_progress.json."""
    if not (source / "config.json").is_file():
        message = f"checkpoint config is missing: {source}"
        raise FileNotFoundError(message)
    dest.mkdir(parents=True, exist_ok=True)
    manifest_path = dest / "manifest.json"
    if manifest_path.is_file():
        done = json.loads(manifest_path.read_text())
        if done.get("complete") and done.get("version") == RECORD_VERSION:
            _log(f"record already complete: {dest}")
            return
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json"):
        shutil.copyfile(source / name, dest / name)
    progress = _load_progress(dest)
    store = ShardStore(source)
    config = json.loads((source / "config.json").read_text())
    text = config["text_config"]
    kinds = list(text["layer_types"])
    experts = int(text["num_experts"])
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
    done_layers = set(progress["layers"])
    for layer in range(len(kinds)):
        path = expert_dir / f"{layer:02d}.bin"
        prefix = f"model.language_model.layers.{layer}.mlp.experts"
        start = _expert_start(
            progress,
            done=layer in done_layers,
            inflight_key=layer,
            path=path,
            experts=experts,
        )
        if start >= experts:
            if layer not in done_layers:
                progress["layers"].append(layer)
                done_layers.add(layer)
                _clear_inflight(progress)
                _save_progress(dest, progress)
            continue
        if layer in done_layers:
            progress["layers"] = [item for item in progress["layers"] if item != layer]
            done_layers.discard(layer)
        if start:
            _log(f"resume experts layer {layer} at {start}")
        else:
            _log(f"quantize experts layer {layer}")

        def commit_expert(expert: int, layer: int = layer) -> None:
            progress["inflight_key"] = layer
            progress["inflight_expert"] = expert
            _save_progress(dest, progress)

        _write_expert_file(
            store,
            path,
            f"{prefix}.gate_up_proj",
            f"{prefix}.down_proj",
            experts,
            start,
            commit_expert,
        )
        progress["layers"].append(layer)
        done_layers.add(layer)
        _clear_inflight(progress)
        _save_progress(dest, progress)
    mtp_path = dest / "mtp-experts.bin"
    mtp_start = _expert_start(
        progress,
        done=bool(progress["mtp"]),
        inflight_key="mtp",
        path=mtp_path,
        experts=experts,
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
        )
        progress["mtp"] = True
        _clear_inflight(progress)
        _save_progress(dest, progress)
    _write_ngram(store, dest, progress)
    manifest = {
        "version": RECORD_VERSION,
        "complete": True,
        "expert_slot": expert_slot_bytes(),
        "expert_live": expert_live_bytes(),
        "experts": experts,
        "layers": len(kinds),
        "ngram_rows": 128 * 2_500_012,
        "ngram_slot": NGRAM_SLOT,
        "ngram_width": NGRAM_WIDTH,
        "embed_key": "model.language_model.embed_tokens.weight",
        "head_key": "lm_head.weight",
        **catalog,
    }
    tmp = manifest_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(manifest))
    manifest_fd = os.open(tmp, os.O_RDONLY)
    try:
        os.fsync(manifest_fd)
    finally:
        os.close(manifest_fd)
    os.replace(tmp, manifest_path)
    _log(f"record complete: {dest}")


def _write_ngram(store: ShardStore, dest: Path, progress: dict[str, Any]) -> None:
    path = dest / "ngram.bin"
    fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        rows_per = 2_500_012
        shards = 128
        total = shards * rows_per
        total_bytes = total * NGRAM_SLOT
        saved = int(progress["ngram_rows"])
        written = ngram_resume_row(saved, os.fstat(fd).st_size)
        if written != saved:
            progress["ngram_rows"] = written
            _save_progress(dest, progress)
            _log(f"n-gram cursor rewound to {written}")
        if written == 0 and os.fstat(fd).st_size < total_bytes:
            preallocate(fd, total_bytes)
        if written >= total:
            return
        if written:
            _log(f"resume n-gram at row {written}/{total}")
        shard = written // rows_per
        local = written % rows_per
        while shard < shards:
            key = (
                "model.language_model.layers.1.ple.ple_embedding."
                f"ngram_embedding.shard_{shard}.weight"
            )
            _fd, _start, _dtype, shape = store._locate(key)
            if list(shape) != [rows_per, NGRAM_WIDTH]:
                message = f"{key} shape {shape}"
                raise RuntimeError(message)
            while local < rows_per:
                count = min(4096, rows_per - local)
                matrix = store.read_bf16(
                    key, local * NGRAM_WIDTH, count * NGRAM_WIDTH, (count, NGRAM_WIDTH)
                )
                blob = ngram_slots(matrix)
                os.pwrite(fd, blob, (shard * rows_per + local) * NGRAM_SLOT)
                local += count
                written += count
                if local % (4096 * 16) == 0 or local == rows_per:
                    os.fsync(fd)
                    progress["ngram_rows"] = written
                    _save_progress(dest, progress)
                    _log(f"n-gram rows {written}/{total}")
            shard += 1
            local = 0
        os.fsync(fd)
        progress["ngram_rows"] = total
        _save_progress(dest, progress)
    finally:
        os.close(fd)
