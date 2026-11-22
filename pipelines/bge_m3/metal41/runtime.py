"""Load ``models/bge-m3-int8`` and return one dense vector.

This module is the Metal session. The serve shell is ``bge_m3.portable``.
This module does not quantize and does not open the float32 checkout.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from tokenizers import Tokenizer

from bge_m3.metal41.gpu import Bound, Gpu, fit_length
from bge_m3.metal41.text import (
    HIDDEN,
    INTERMEDIATE,
    LAYERS,
    key_mask,
    l2_normalize,
    pad_ids,
    padded_length,
    position_ids,
)

_PROJECTIONS = ("q", "k", "v", "o", "up", "down")


@dataclass(frozen=True)
class Embedding:
    """One dense vector and the token count that produced it."""

    vector: list[float]
    tokens: int


class Runtime:
    """The int8 weight image and one sequence workspace."""

    def __init__(
        self,
        path: Path,
        *,
        owner: str | None = None,
        max_memory: int | None = None,
    ) -> None:
        manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        if not manifest.get("complete") or manifest.get("quant") != "int8":
            message = f"{path.name} is not a finished record"
            raise FileNotFoundError(message)
        blob = path / "weights.bin"
        index = path / "weights.json"
        weights_path = path / "weights.npz"
        if blob.is_file():
            if not index.is_file():
                message = f"{path.name} has no weight index"
                raise FileNotFoundError(message)
        elif not weights_path.is_file():
            message = f"{path.name} has no weights"
            raise FileNotFoundError(message)
        self.path = path
        self.tokenizer = Tokenizer.from_file(str(path / "tokenizer.json"))
        self._owner = owner or f"bge_m3:{id(self)}"
        self.gpu = Gpu(self._owner)
        try:
            if blob.is_file():
                self._weights = _load_blob(self.gpu, blob, index)
            else:
                packed = np.load(weights_path)
                self._weights = _upload(self.gpu, packed)
            self.limit = fit_length(0, max_memory)
            self._work = _workspace(self.gpu, self.limit)
            self.gpu.bind_workspace(self._work)
        except Exception:
            self.gpu.close()
            raise

    def close(self) -> None:
        self.gpu.close()

    def embed(self, text: str) -> Embedding:
        ids = list(self.tokenizer.encode(text).ids)
        if len(ids) > self.limit:
            message = f"input is {len(ids)} tokens and the limit is {self.limit}"
            raise ValueError(message)
        length = padded_length(len(ids))
        if length > self.limit:
            message = f"input is {len(ids)} tokens and the limit is {self.limit}"
            raise ValueError(message)
        padded = pad_ids(ids, length)
        hidden = self.gpu.encode(
            self._weights,
            np.asarray(padded, dtype=np.int32),
            np.asarray(position_ids(padded), dtype=np.int32),
            key_mask(padded).reshape(-1),
            self._work,
        )
        vector = l2_normalize(hidden)
        return Embedding(vector=vector.tolist(), tokens=len(ids))


def _load_blob(gpu: Gpu, blob: Path, index: Path) -> dict[str, Any]:
    """One shared span. Each tensor binds at the offset in ``weights.json``."""
    from bge_m3.metal41.quantize import _load_weight_index

    rows = _load_weight_index(index)
    total = int(rows[-1]["offset"]) + int(rows[-1]["nbytes"])
    size = blob.stat().st_size
    if size != total:
        message = f"{blob.name} is {size} bytes and the index is {total}"
        raise RuntimeError(message)
    handle = gpu.read_shared(blob, total)
    by_name = {str(row["name"]): row for row in rows}

    def bind(name: str) -> Bound:
        return Bound(handle, int(by_name[name]["offset"]))

    weights: dict[str, Any] = {
        "word": bind("word"),
        "position": bind("position"),
        "type": bind("type"),
        "embed_w": bind("embed_w"),
        "embed_b": bind("embed_b"),
        "layers": [],
    }
    for layer in range(LAYERS):
        item: dict[str, Any] = {}
        for name in _PROJECTIONS:
            prefix = f"l{layer}_{name}"
            codes = by_name[f"{prefix}_codes"]
            item[name] = {
                "n": int(codes["shape"][0]),
                "codes": Bound(handle, int(codes["offset"])),
                "scale": bind(f"{prefix}_scale"),
                "bias": bind(f"{prefix}_bias"),
            }
        item["attn_w"] = bind(f"l{layer}_attn_w")
        item["attn_b"] = bind(f"l{layer}_attn_b")
        item["ffn_w"] = bind(f"l{layer}_ffn_w")
        item["ffn_b"] = bind(f"l{layer}_ffn_b")
        weights["layers"].append(item)
    return weights


def _upload(gpu: Gpu, packed: np.lib.npyio.NpzFile) -> dict[str, Any]:
    weights: dict[str, Any] = {
        "word": gpu.upload(packed["word"]),
        "position": gpu.upload(packed["position"]),
        "type": gpu.upload(packed["type"]),
        "embed_w": gpu.upload(packed["embed_w"]),
        "embed_b": gpu.upload(packed["embed_b"]),
        "layers": [],
    }
    for layer in range(LAYERS):
        item: dict[str, Any] = {}
        for name in _PROJECTIONS:
            prefix = f"l{layer}_{name}"
            codes = packed[f"{prefix}_codes"]
            item[name] = {
                "n": int(codes.shape[0]),
                "codes": gpu.upload(codes),
                "scale": gpu.upload(packed[f"{prefix}_scale"]),
                "bias": gpu.upload(packed[f"{prefix}_bias"]),
            }
        item["attn_w"] = gpu.upload(packed[f"l{layer}_attn_w"])
        item["attn_b"] = gpu.upload(packed[f"l{layer}_attn_b"])
        item["ffn_w"] = gpu.upload(packed[f"l{layer}_ffn_w"])
        item["ffn_b"] = gpu.upload(packed[f"l{layer}_ffn_b"])
        weights["layers"].append(item)
    return weights


def _workspace(gpu: Gpu, length: int) -> dict[str, Any]:
    rows = int(length)
    narrow = rows * HIDDEN * 4
    wide = rows * INTERMEDIATE * 4

    def f32(count: int) -> Any:
        return gpu.scratch(count, zero=True)

    work = {
        "ids": gpu.scratch(rows * 4),
        "pos": gpu.scratch(rows * 4),
        "mask": f32(rows * 4),
        "cur": f32(narrow),
        "mix": f32(narrow),
        "nxt": f32(narrow),
        "q": f32(narrow),
        "k": f32(narrow),
        "v": f32(narrow),
        "attn": f32(narrow),
        "wide": f32(wide),
        "codes": gpu.scratch(rows * INTERMEDIATE),
        "row_scale": f32(rows * 4),
        "acc": gpu.scratch(wide),
    }
    return work


def open_device(
    path: str | Path,
    kv_cache: str | None = None,
    console: Any = None,
    resident_percent: float | None = None,
    max_memory: int | None = None,
    owner: str | None = None,
) -> Runtime:
    """Open the packed record. The extra arguments are the listener's."""
    del kv_cache, console, resident_percent
    return Runtime(Path(path), owner=owner, max_memory=max_memory)