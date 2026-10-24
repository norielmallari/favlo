"""Packed umT5 encoder.

Linears are fp4, except the four layers with the highest fp4 error, which are
fp8. The token embedding, RMSNorm weights, and relative buckets stay half.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

from wan22_i2v.formats import TILE_M, LinearPack, pack_linear, pad_to, relative_error, restore_linear
from wan22_i2v.metal import Gpu
from wan22_i2v.schedule import promote_count
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


def quantize_umt5(src: Path | None = None, dest: Path | None = None) -> list[int]:
    """Pack the bf16 encoder. Returns the fp8 layer indices."""
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
        scores = _score(archive, meta, dest)
    finally:
        archive.close()
    winners = block_winners_local(scores)
    archive = zipfile.ZipFile(weights)
    try:
        meta = _metadata(archive)
        chosen = set(winners)
        for layer in range(_LAYERS):
            kind = "fp8" if layer in chosen else "fp4"
            for suffix in _MATRICES:
                name = f"blocks.{layer}.{suffix}"
                path = dest / f"{name}.npz"
                if _stored_kind(path) == kind:
                    continue
                _save(path, pack_linear(_matrix(archive, meta, name), None, kind))
                print(f"write {name} {kind}", flush=True)
    finally:
        archive.close()
    (dest / "manifest.json").write_text(
        json.dumps({"fp8": winners}),
        encoding="utf-8",
    )
    print(f"umt5 fp8 {winners}", flush=True)
    return winners


class PackedEncoder:
    """One packed umT5. Matrices are uploaded for the multiply, then dropped."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        manifest = json.loads((self.path / "manifest.json").read_text(encoding="utf-8"))
        self.fp8 = [int(layer) for layer in manifest["fp8"]]

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


def _score(archive: zipfile.ZipFile, meta: dict[str, tuple], dest: Path) -> dict[int, float]:
    path = dest / "scores.json"
    found: dict[str, float] = {}
    if path.is_file():
        found = json.loads(path.read_text(encoding="utf-8"))
    scores: dict[int, float] = {}
    for layer in range(_LAYERS):
        key = str(layer)
        ready = all(_stored_kind(dest / f"blocks.{layer}.{suffix}.npz") == "fp4" for suffix in _MATRICES)
        if key in found and ready:
            scores[layer] = float(found[key])
            continue
        worst = 0.0
        for suffix in _MATRICES:
            name = f"blocks.{layer}.{suffix}"
            weight = _matrix(archive, meta, name)
            pack = pack_linear(weight, None, "fp4")
            worst = max(worst, relative_error(restore_linear(pack), weight))
            _save(dest / f"{name}.npz", pack)
            del weight, pack
        scores[layer] = worst
        found[key] = worst
        path.write_text(json.dumps(found), encoding="utf-8")
        print(f"score umt5 {layer} {worst:.6f}", flush=True)
    return scores


def block_winners_local(scores: dict[int, float]) -> list[int]:
    from wan22_i2v.formats import block_winners

    return block_winners(scores, promote_count(_LAYERS))


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
