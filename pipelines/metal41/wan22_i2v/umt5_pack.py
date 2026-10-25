"""Packed umT5 encoder.

Linears are per-channel int8. The token embedding, RMSNorm weights, and
relative buckets stay half. The pack lives under ``models/google`` so other
pipelines can share it.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

from wan22_i2v.formats import TILE_M, LinearPack, pack_linear, pad_to, relative_error, restore_linear
from wan22_i2v.metal import Gpu
from wan22_i2v.text import (
    _EPS,
    _FFN,
    _HEADS,
    _LAYERS,
    _PREFIX,
    _WIDTH,
    _float,
    _metadata,
    encoder_path,
    relative_buckets,
)

_MATRICES = (
    "attn.q.weight",
    "attn.k.weight",
    "attn.v.weight",
    "attn.o.weight",
    "ffn.gate.0.weight",
    "ffn.fc1.weight",
    "ffn.fc2.weight",
)


def pack_dir(record: Path) -> Path:
    """Packed encoder beside one DiT record."""
    return Path(record) / "umt5"


def shared_pack_dir(root: Path | None = None) -> Path:
    """Int8 encoder under ``models/google``, shared by every pipeline that uses umT5."""
    from wan22_i2v.source import models_dir

    return models_dir(root) / "google" / "umt5-xxl-int8"


def quantize_umt5(src: Path | None = None, dest: Path | None = None) -> float:
    """Pack the bf16 encoder as int8. Returns the worst relative RMS."""
    weights = src or encoder_path()
    if dest is None:
        message = "umT5 pack directory is missing"
        raise ValueError(message)
    dest.mkdir(parents=True, exist_ok=True)
    archive = zipfile.ZipFile(weights)
    try:
        meta = _metadata(archive)
        _write_half(archive, meta, "token_embedding.weight", dest / "embed.npy")
        _write_half(archive, meta, "norm.weight", dest / "norm.npy")
        for layer in range(_LAYERS):
            prefix = f"blocks.{layer}"
            _write_half(archive, meta, f"{prefix}.norm1.weight", dest / f"{prefix}.norm1.npy")
            _write_half(archive, meta, f"{prefix}.norm2.weight", dest / f"{prefix}.norm2.npy")
            _write_half(
                archive,
                meta,
                f"{prefix}.pos_embedding.embedding.weight",
                dest / f"{prefix}.pos.npy",
            )
        worst = 0.0
        worst_name = ""
        for layer in range(_LAYERS):
            for suffix in _MATRICES:
                name = f"blocks.{layer}.{suffix}"
                path = dest / f"{name}.npz"
                if _stored_kind(path) == "int8":
                    continue
                weight = _matrix(archive, meta, name)
                pack = pack_linear(weight, None, "int8")
                error = relative_error(restore_linear(pack), weight)
                _save(path, pack)
                del weight, pack
                print(f"int8 {name} {error:.6f}", flush=True)
                if error >= worst:
                    worst = error
                    worst_name = name
    finally:
        archive.close()
    (dest / "manifest.json").write_text(
        json.dumps({"weight": "int8", "worst": worst, "worst_name": worst_name}),
        encoding="utf-8",
    )
    print(f"umt5 int8 worst {worst_name} {worst:.6f}", flush=True)
    return worst


class PackedEncoder:
    """One packed umT5. Matrices are uploaded for the multiply, then dropped."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        manifest = json.loads((self.path / "manifest.json").read_text(encoding="utf-8"))
        self.weight = str(manifest.get("weight", "fp4"))

    def run(self, ids: np.ndarray, on_layer: Any, gpu: Gpu) -> np.ndarray:
        rows = int(ids.shape[0])
        padded = pad_to(rows, TILE_M)
        table = np.zeros((padded, _WIDTH), dtype=np.float32)
        embed = np.load(self.path / "embed.npy", mmap_mode="r")
        table[:rows] = np.asarray(embed[ids], dtype=np.float32)
        hidden = gpu.upload(table)
        buckets = gpu.upload(relative_buckets(rows, rows).astype(np.int32))
        half = gpu.empty(padded * pad_to(_FFN, 256) * 2)
        try:
            for layer in range(_LAYERS):
                hidden = self._block(gpu, hidden, rows, padded, layer, buckets, half)
                if on_layer is not None:
                    on_layer()
            normed = self._norm(gpu, hidden, padded, self.path / "norm.npy")
            vectors = gpu.read_f32(normed, rows * _WIDTH).reshape(rows, _WIDTH)
            gpu.drop(normed)
            return vectors
        finally:
            gpu.drop(hidden)
            gpu.drop(buckets)
            gpu.drop(half)

    def _block(
        self,
        gpu: Gpu,
        hidden: Any,
        rows: int,
        padded: int,
        layer: int,
        buckets: Any,
        half: Any,
    ) -> Any:
        prefix = f"blocks.{layer}"
        normed = self._norm(gpu, hidden, padded, self.path / f"{prefix}.norm1.npy")
        query = self._linear(gpu, normed, padded, f"{prefix}.attn.q.weight", half)
        key = self._linear(gpu, normed, padded, f"{prefix}.attn.k.weight", half)
        value = self._linear(gpu, normed, padded, f"{prefix}.attn.v.weight", half)
        gpu.drop(normed)
        pos = np.load(self.path / f"{prefix}.pos.npy").astype(np.float32)
        table = gpu.upload(np.ascontiguousarray(pos))
        attn = gpu.alloc_f32(rows * _WIDTH)
        gpu.attend_bias(
            query, key, value, attn, buckets, table, rows, rows, _HEADS, _WIDTH // _HEADS
        )
        gpu.end()
        gpu.drop_many([query, key, value, table])
        gpu.add_f32(hidden, attn, rows * _WIDTH)
        gpu.end()
        gpu.drop(attn)
        normed = self._norm(gpu, hidden, padded, self.path / f"{prefix}.norm2.npy")
        gate = self._linear(gpu, normed, padded, f"{prefix}.ffn.gate.0.weight", half)
        gpu.pointwise("gelu_tanh", gate, rows, _FFN, _FFN)
        inner = self._linear(gpu, normed, padded, f"{prefix}.ffn.fc1.weight", half)
        gpu.drop(normed)
        gpu.mul_f32(gate, inner, rows * _FFN)
        gpu.end()
        gpu.drop(inner)
        out = self._linear(gpu, gate, padded, f"{prefix}.ffn.fc2.weight", half)
        gpu.drop(gate)
        gpu.add_f32(hidden, out, rows * _WIDTH)
        gpu.end()
        gpu.drop(out)
        return hidden

    def _linear(self, gpu: Gpu, source: Any, rows: int, name: str, half: Any) -> Any:
        pack = _load(self.path / f"{name}.npz")
        dest = gpu.alloc_f32(rows * pack.padded_n)
        uploaded = {
            "packed": gpu.upload(pack.packed),
            "scales": gpu.upload(pack.scales),
            "bias": gpu.upload(pack.bias),
            "perm": gpu.upload(pack.perm),
        }
        gpu.project(source, pack.logical_k, rows, pack, dest, uploaded, half)
        gpu.end()
        gpu.drop_many(list(uploaded.values()))
        return dest

    def _norm(self, gpu: Gpu, source: Any, rows: int, path: Path) -> Any:
        weight = np.load(path).astype(np.float16)
        scale = gpu.upload(np.ascontiguousarray(weight))
        dest = gpu.alloc_f32(rows * _WIDTH)
        gpu.rmsnorm(source, scale, dest, rows, _WIDTH, float(_EPS))
        gpu.end()
        gpu.drop(scale)
        return dest


def _tensor(archive: zipfile.ZipFile, meta: dict[str, tuple], name: str) -> np.ndarray:
    key, offset, shape, stride = meta[name]
    raw = archive.read(f"{_PREFIX}/{key}")
    storage = np.frombuffer(raw, dtype="<u2")
    return np.ascontiguousarray(_float(storage, offset, shape, stride), dtype=np.float32)


def _matrix(archive: zipfile.ZipFile, meta: dict[str, tuple], name: str) -> np.ndarray:
    array = _tensor(archive, meta, name)
    if array.ndim != 2:
        message = f"{name} is not a matrix"
        raise RuntimeError(message)
    return array


def _write_half(
    archive: zipfile.ZipFile, meta: dict[str, tuple], name: str, dest: Path
) -> None:
    if dest.is_file():
        return
    if name == "token_embedding.weight":
        _write_embed(archive, meta, dest)
        return
    np.save(dest, np.ascontiguousarray(_tensor(archive, meta, name), dtype=np.float16))


def _write_embed(archive: zipfile.ZipFile, meta: dict[str, tuple], dest: Path) -> None:
    key, offset, shape, stride = meta["token_embedding.weight"]
    if tuple(stride) != (shape[1], 1):
        message = "token embedding is not row major"
        raise RuntimeError(message)
    rows, width = int(shape[0]), int(shape[1])
    out = np.lib.format.open_memmap(dest, mode="w+", dtype=np.float16, shape=(rows, width))
    member = archive.open(f"{_PREFIX}/{key}")
    try:
        step = 512
        for start in range(0, rows, step):
            count = min(step, rows - start)
            member.seek((int(offset) + start * width) * 2)
            raw = np.frombuffer(member.read(count * width * 2), dtype="<u2").copy()
            bits = raw.astype(np.uint32) << np.uint32(16)
            out[start : start + count] = bits.view(np.float32).reshape(count, width)
    finally:
        member.close()
        out.flush()


def _save(path: Path, pack: LinearPack) -> None:
    np.savez(
        path,
        kind=np.array(pack.kind),
        packed=pack.packed,
        scales=pack.scales,
        bias=pack.bias,
        perm=pack.perm,
        logical_n=np.int32(pack.logical_n),
        logical_k=np.int32(pack.logical_k),
        padded_n=np.int32(pack.padded_n),
        stored_k=np.int32(pack.stored_k),
    )


def _load(path: Path) -> LinearPack:
    blob = np.load(path)
    return LinearPack(
        kind=str(blob["kind"].item()),
        packed=np.ascontiguousarray(blob["packed"]),
        scales=np.ascontiguousarray(blob["scales"]),
        bias=np.ascontiguousarray(blob["bias"]),
        perm=np.ascontiguousarray(blob["perm"]),
        logical_n=int(blob["logical_n"]),
        logical_k=int(blob["logical_k"]),
        padded_n=int(blob["padded_n"]),
        stored_k=int(blob["stored_k"]),
    )


def _stored_kind(path: Path) -> str | None:
    if not path.is_file():
        return None
    try:
        return str(np.load(path)["kind"].item())
    except (OSError, ValueError, EOFError, zipfile.BadZipFile):
        path.unlink(missing_ok=True)
        return None
