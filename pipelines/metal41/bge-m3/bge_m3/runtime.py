"""Load ``models/bge-m3-int8`` and return one dense vector.

This module does not quantize and does not open the float32 checkout.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from tokenizers import Tokenizer

from favlo.embeddings_pipeline import EmbeddingsPipeline
from bge_m3.gpu import Gpu, fit_length
from bge_m3.text import (
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
        weights_path = path / "weights.npz"
        if not weights_path.is_file():
            message = f"{path.name} has no weights"
            raise FileNotFoundError(message)
        self.path = path
        self.tokenizer = Tokenizer.from_file(str(path / "tokenizer.json"))
        self._owner = owner or f"bge-m3:{id(self)}"
        self.gpu = Gpu(self._owner)
        try:
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


def open_runtime(
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


class Pipeline(EmbeddingsPipeline):
    """``POST /v1/embeddings``. One string in, one dense vector out."""

    def embed(self, runtime: Runtime, text: str) -> tuple[list[float], int]:
        result = runtime.embed(text)
        return result.vector, result.tokens


handle_embed = Pipeline()
