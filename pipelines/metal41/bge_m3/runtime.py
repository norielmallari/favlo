"""Load ``models/bge-m3-w8a8`` and return one dense vector.

This module does not quantize and does not open the float32 checkout.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from tokenizers import Tokenizer

from bge_m3.text import (
    HIDDEN,
    bucket_for,
    key_mask,
    l2_normalize,
    lengths_from_names,
    pad_ids,
    position_ids,
)


@dataclass(frozen=True)
class Embedding:
    """One dense vector and the token count that produced it."""

    vector: list[float]
    tokens: int


class Runtime:
    """The packed multifunction program. One function per bucket."""

    def __init__(self, path: Path) -> None:
        manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        if not manifest.get("complete") or manifest.get("quant") != "w8a8":
            message = f"{path.name} is not a finished record"
            raise FileNotFoundError(message)
        names = manifest.get("functions")
        if not isinstance(names, list):
            message = f"{path.name} has no encode functions"
            raise FileNotFoundError(message)
        try:
            self.buckets = lengths_from_names([str(name) for name in names])
        except ValueError as exc:
            raise FileNotFoundError(str(exc)) from exc
        self.path = path
        self.tokenizer = Tokenizer.from_file(str(path / "tokenizer.json"))
        self._package = path / "model.mlpackage"
        self._compiled = path / "model.mlmodelc"
        self._functions: dict[int, Any] = {}

    def close(self) -> None:
        self._functions.clear()

    def embed(self, text: str) -> Embedding:
        ids = list(self.tokenizer.encode(text).ids)
        length = bucket_for(len(ids), self.buckets)
        padded = pad_ids(ids, length)
        hidden = self._predict(
            length,
            {
                "input_ids": np.asarray([padded], dtype=np.int32),
                "position_ids": np.asarray([position_ids(padded)], dtype=np.int32),
                "key_mask": key_mask(padded),
            },
        ).reshape(length, HIDDEN)
        vector = l2_normalize(hidden[0])
        return Embedding(vector=vector.tolist(), tokens=len(ids))

    def _predict(self, length: int, features: dict[str, np.ndarray]) -> np.ndarray:
        from bge_m3.program import compiled_model, load_function, predict_hidden

        loaded = self._functions.get(length)
        if loaded is None:
            compiled = compiled_model(self._package, self._compiled)
            loaded = load_function(compiled, f"encode_{length}")
            self._functions[length] = loaded
        hidden = predict_hidden(loaded, features)
        return np.asarray(hidden, dtype=np.float32)


def open_runtime(
    path: str | Path,
    kv_cache: str | None = None,
    console: Any = None,
    resident_percent: float | None = None,
    max_memory: int | None = None,
) -> Runtime:
    """Open the packed program. The extra arguments are the listener's."""
    del kv_cache, console, resident_percent, max_memory
    return Runtime(Path(path))


def handle_embed(
    reply: Any,
    runtime: Runtime,
    body: dict[str, Any],
    selection: Any,
    args: Any,
    state: dict[str, Any],
) -> None:
    """``POST /v1/embeddings``. One string in, one dense vector out."""
    del args, state
    text = body.get("input")
    if not isinstance(text, str) or not text.strip():
        reply.json(
            400,
            {
                "error": {
                    "message": "input must be a string",
                    "type": "invalid_request_error",
                    "param": "input",
                }
            },
        )
        return
    try:
        result = runtime.embed(text)
    except ValueError as exc:
        reply.json(400, {"error": {"message": str(exc), "type": "invalid_request_error"}})
        return
    model = ""
    if selection is not None:
        model = str(getattr(selection, "requested_id", "") or "")
    reply.json(
        200,
        {
            "object": "list",
            "model": model,
            "data": [
                {
                    "object": "embedding",
                    "index": 0,
                    "embedding": result.vector,
                }
            ],
            "usage": {"prompt_tokens": result.tokens, "total_tokens": result.tokens},
        },
    )
