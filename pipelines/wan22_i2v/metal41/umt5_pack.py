"""Packed umT5 encoder for a clip.

Linears are per-channel int8. The token embedding, RMSNorm weights, and
relative buckets stay half. ``favlo --umt5_xxl quantize`` writes the shared
pack under ``models/google``. The tokenizer lives in that pack.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from wan22_i2v.metal41.formats import TILE_M, LinearPack, pad_to
from wan22_i2v.metal41.metal import Gpu
from wan22_i2v.metal41.text import _EPS, _FFN, _HEADS, _LAYERS, _WIDTH, relative_buckets


def pack_dir(record: Path) -> Path:
    """Packed encoder beside one DiT record."""
    return Path(record) / "umt5"


def shared_pack_dir(root: Path | None = None) -> Path:
    """Int8 encoder under ``models/google``, shared by every pipeline that uses umT5."""
    from wan22_i2v.metal41.source import models_dir

    return models_dir(root) / "google" / "umt5-xxl-int8"


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
