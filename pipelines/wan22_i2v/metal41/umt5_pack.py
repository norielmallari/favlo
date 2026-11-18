"""Packed umT5 encoder for a clip.

Linears are per-channel int8. The token embedding, RMSNorm weights, and
relative buckets stay half. ``favlo --wan22_i2v umt5 quantize`` writes the
shared pack under ``models/google``. The tokenizer lives in that pack.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from wan22_i2v.metal41.blob import Plane, blob_ready, fill_span, load_index
from wan22_i2v.metal41.formats import TILE_M, LinearPack, pad_to
from wan22_i2v.metal41.metal import BufferView, Gpu
from wan22_i2v.metal41.text import _EPS, _FFN, _HEADS, _LAYERS, _WIDTH, relative_buckets


def pack_dir(record: Path) -> Path:
    """Packed encoder beside one DiT record."""
    return Path(record) / "umt5"


def shared_pack_dir(root: Path | None = None) -> Path:
    """Int8 encoder under ``models/google``, shared by every pipeline that uses umT5."""
    from wan22_i2v.metal41.source import models_dir

    return models_dir(root) / "google" / "umt5-xxl-int8"


class PackedEncoder:
    """One packed umT5. Each linear is dropped after the multiply.

    A ``weights.bin`` span cannot free one plane on its own. Views are dropped
    and the span stays until ``run`` returns, then it is released. The DiT
    expert is not open during that encode.
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        manifest = json.loads((self.path / "manifest.json").read_text(encoding="utf-8"))
        self.weight = str(manifest.get("weight", "fp4"))

    def run(self, ids: np.ndarray, on_layer: Any, gpu: Gpu) -> np.ndarray:
        rows = int(ids.shape[0])
        padded = pad_to(rows, TILE_M)
        source = _open_source(self.path, gpu)
        hidden = None
        buckets = None
        half = None
        try:
            table = np.zeros((padded, _WIDTH), dtype=np.float32)
            source.gather(table, ids, rows)
            hidden = gpu.upload(table)
            buckets = gpu.upload(relative_buckets(rows, rows).astype(np.int32))
            half = gpu.empty(padded * pad_to(_FFN, 256) * 2)
            for layer in range(_LAYERS):
                hidden = self._block(gpu, hidden, rows, padded, layer, buckets, half, source)
                if on_layer is not None:
                    on_layer()
            normed = self._norm(gpu, hidden, padded, source.norm(gpu, "norm"))
            vectors = gpu.read_f32(normed, rows * _WIDTH).reshape(rows, _WIDTH)
            gpu.drop(normed)
            return vectors
        finally:
            if hidden is not None:
                gpu.drop(hidden)
            if buckets is not None:
                gpu.drop(buckets)
            if half is not None:
                gpu.drop(half)
            source.close(gpu)

    def _block(
        self,
        gpu: Gpu,
        hidden: Any,
        rows: int,
        padded: int,
        layer: int,
        buckets: Any,
        half: Any,
        source: _WeightSource,
    ) -> Any:
        prefix = f"blocks.{layer}"
        normed = self._norm(gpu, hidden, padded, source.norm(gpu, f"{prefix}.norm1"))
        query = self._linear(gpu, normed, padded, f"{prefix}.attn.q.weight", half, source)
        key = self._linear(gpu, normed, padded, f"{prefix}.attn.k.weight", half, source)
        value = self._linear(gpu, normed, padded, f"{prefix}.attn.v.weight", half, source)
        gpu.drop(normed)
        table = source.pos(gpu, layer)
        attn = gpu.alloc_f32(rows * _WIDTH)
        gpu.attend_bias(
            query, key, value, attn, buckets, table, rows, rows, _HEADS, _WIDTH // _HEADS
        )
        gpu.end()
        gpu.drop_many([query, key, value, table])
        gpu.add_f32(hidden, attn, rows * _WIDTH)
        gpu.end()
        gpu.drop(attn)
        normed = self._norm(gpu, hidden, padded, source.norm(gpu, f"{prefix}.norm2"))
        gate = self._linear(gpu, normed, padded, f"{prefix}.ffn.gate.0.weight", half, source)
        gpu.pointwise("gelu_tanh", gate, rows, _FFN, _FFN)
        inner = self._linear(gpu, normed, padded, f"{prefix}.ffn.fc1.weight", half, source)
        gpu.drop(normed)
        gpu.mul_f32(gate, inner, rows * _FFN)
        gpu.end()
        gpu.drop(inner)
        out = self._linear(gpu, gate, padded, f"{prefix}.ffn.fc2.weight", half, source)
        gpu.drop(gate)
        gpu.add_f32(hidden, out, rows * _WIDTH)
        gpu.end()
        gpu.drop(out)
        return hidden

    def _linear(
        self, gpu: Gpu, source: Any, rows: int, name: str, half: Any, weights: _WeightSource
    ) -> Any:
        pack, uploaded = weights.linear(gpu, name)
        dest = gpu.alloc_f32(rows * pack.padded_n)
        gpu.project(source, pack.logical_k, rows, pack, dest, uploaded, half)
        gpu.end()
        gpu.drop_many(list(uploaded.values()))
        return dest

    def _norm(self, gpu: Gpu, source: Any, rows: int, scale: Any) -> Any:
        dest = gpu.alloc_f32(rows * _WIDTH)
        gpu.rmsnorm(source, scale, dest, rows, _WIDTH, float(_EPS))
        gpu.end()
        gpu.drop(scale)
        return dest


class _WeightSource:
    def gather(self, table: np.ndarray, ids: np.ndarray, rows: int) -> None:
        raise NotImplementedError

    def linear(self, gpu: Gpu, name: str) -> tuple[LinearPack, dict[str, Any]]:
        raise NotImplementedError

    def norm(self, gpu: Gpu, name: str) -> Any:
        raise NotImplementedError

    def pos(self, gpu: Gpu, layer: int) -> Any:
        raise NotImplementedError

    def close(self, gpu: Gpu) -> None:
        return None


class _FileSource(_WeightSource):
    """The installed pack: one ``.npz`` or ``.npy`` per plane."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def gather(self, table: np.ndarray, ids: np.ndarray, rows: int) -> None:
        embed = np.load(self.path / "embed.npy", mmap_mode="r")
        table[:rows] = np.asarray(embed[ids], dtype=np.float32)

    def linear(self, gpu: Gpu, name: str) -> tuple[LinearPack, dict[str, Any]]:
        pack = _load(self.path / f"{name}.npz")
        uploaded = {
            "packed": gpu.upload(pack.packed),
            "scales": gpu.upload(pack.scales),
            "bias": gpu.upload(pack.bias),
            "perm": gpu.upload(pack.perm),
        }
        return pack, uploaded

    def norm(self, gpu: Gpu, name: str) -> Any:
        weight = np.load(self.path / f"{name}.npy").astype(np.float16)
        return gpu.upload(np.ascontiguousarray(weight))

    def pos(self, gpu: Gpu, layer: int) -> Any:
        pos = np.load(self.path / f"blocks.{layer}.pos.npy").astype(np.float32)
        return gpu.upload(np.ascontiguousarray(pos))


class _BlobSource(_WeightSource):
    """One encoder span. Views bind planes. The span is released at ``close``."""

    def __init__(self, buf: Any, index: dict[str, Plane]) -> None:
        self.buf = buf
        self.index = index

    def gather(self, table: np.ndarray, ids: np.ndarray, rows: int) -> None:
        plane = _pick(self.index, "embed", "embed.weight")
        embed = _span_array(self.buf, plane)
        table[:rows] = np.asarray(embed[ids], dtype=np.float32)

    def linear(self, gpu: Gpu, name: str) -> tuple[LinearPack, dict[str, Any]]:
        del gpu
        stem = name[: -len(".weight")] if name.endswith(".weight") else name
        packed = _pick(self.index, f"{stem}.packed", f"{name}.packed")
        scales = _pick(self.index, f"{stem}.scales", f"{name}.scales")
        bias = _pick(self.index, f"{stem}.bias", f"{name}.bias")
        uploaded: dict[str, Any] = {
            "packed": BufferView(self.buf, packed.offset),
            "scales": BufferView(self.buf, scales.offset),
            "bias": BufferView(self.buf, bias.offset),
        }
        perm = self.index.get(f"{stem}.perm")
        if perm is None:
            perm = self.index.get(f"{name}.perm")
        if perm is not None:
            uploaded["perm"] = BufferView(self.buf, perm.offset)
        return _umt5_pack(stem, packed, scales, bias), uploaded

    def norm(self, gpu: Gpu, name: str) -> Any:
        del gpu
        if name == "norm":
            plane = _pick(self.index, "norm", "norm.weight", "final.norm")
        else:
            plane = _pick(self.index, name, f"{name}.weight")
        if plane.dtype != "float16":
            message = f"{plane.name} is {plane.dtype}, expected float16"
            raise ValueError(message)
        return BufferView(self.buf, plane.offset)

    def pos(self, gpu: Gpu, layer: int) -> Any:
        plane = _pick(
            self.index,
            f"blocks.{layer}.pos",
            f"blocks.{layer}.pos.weight",
        )
        if plane.dtype == "float32":
            return BufferView(self.buf, plane.offset)
        if plane.dtype != "float16":
            message = f"{plane.name} is {plane.dtype}, expected float16 or float32"
            raise ValueError(message)
        raw = np.array(_span_array(self.buf, plane), dtype=np.float32, copy=True)
        return gpu.upload(np.ascontiguousarray(raw))

    def close(self, gpu: Gpu) -> None:
        gpu.drop(self.buf)


def _open_source(path: Path, gpu: Gpu) -> _WeightSource:
    if not blob_ready(path):
        return _FileSource(path)
    index = load_index(path)
    if index is None:
        return _FileSource(path)
    _require_encoder(index)
    blob = path / "weights.bin"
    buf = gpu.keep(gpu.span(blob.stat().st_size))
    try:
        fill_span(buf, blob)
    except Exception:
        gpu.unkeep(buf)
        gpu.drop(buf)
        raise
    return _BlobSource(buf, index)


def _require_encoder(index: dict[str, Plane]) -> None:
    _pick(index, "embed", "embed.weight")
    _pick(index, "norm", "final.norm", "norm.weight")
    for layer in range(_LAYERS):
        _pick(index, f"blocks.{layer}.norm1", f"blocks.{layer}.norm1.weight")
        _pick(index, f"blocks.{layer}.norm2", f"blocks.{layer}.norm2.weight")
        _pick(index, f"blocks.{layer}.pos", f"blocks.{layer}.pos.weight")
        for stem in _stems(layer):
            _pick(index, f"{stem}.packed", f"{stem}.weight.packed")
            _pick(index, f"{stem}.scales", f"{stem}.weight.scales")
            _pick(index, f"{stem}.bias", f"{stem}.weight.bias")


def _stems(layer: int) -> tuple[str, ...]:
    base = f"blocks.{layer}"
    return (
        f"{base}.attn.q",
        f"{base}.attn.k",
        f"{base}.attn.v",
        f"{base}.attn.o",
        f"{base}.ffn.gate.0",
        f"{base}.ffn.fc1",
        f"{base}.ffn.fc2",
    )


def _pick(index: dict[str, Plane], *names: str) -> Plane:
    for name in names:
        plane = index.get(name)
        if plane is not None:
            return plane
    message = f"plane {' or '.join(names)} is missing"
    raise KeyError(message)


def _umt5_pack(stem: str, packed: Plane, scales: Plane, bias: Plane) -> LinearPack:
    if packed.dtype != "int8" or scales.dtype != "float32" or bias.dtype != "float16":
        message = f"{stem} planes are {packed.dtype}, {scales.dtype}, {bias.dtype}"
        raise ValueError(message)
    if len(packed.shape) != 2 or len(bias.shape) != 1:
        message = f"{stem} plane rank is not a matrix and a bias"
        raise ValueError(message)
    logical_k = _FFN if stem.endswith("ffn.fc2") else _WIDTH
    return LinearPack(
        kind="int8",
        packed=np.empty(0, dtype=np.int8),
        scales=np.empty(0, dtype=np.float32),
        bias=np.empty(0, dtype=np.float16),
        perm=np.empty(0, dtype=np.uint32),
        logical_n=int(bias.shape[0]),
        logical_k=logical_k,
        padded_n=int(packed.shape[0]),
        stored_k=int(packed.shape[1]),
    )


def _span_array(buf: Any, plane: Plane) -> np.ndarray:
    from favlo.memory import memory

    view = memory.map(buf, plane.offset, plane.nbytes)
    return np.ndarray(plane.shape, np.dtype(plane.dtype), buffer=view)


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
