"""Pack the float32 checkout as int8 projections and score the GPU forward."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from bge_m3.pack import pack_int8
from bge_m3.passages import passages
from bge_m3.text import HIDDEN, l2_normalize, pad_ids, padded_length, position_ids

RECORD = "bge-m3-int8"
_COSINE_MIN = 0.96
_MATCH_MIN = 0.999


def quantize(source: Path, dest: Path) -> None:
    """Write ``dest`` and register it. A finished record is left in place."""
    if _finished(dest):
        from bge_m3.runtime import Runtime

        runtime = Runtime(dest)
        try:
            limit = runtime.limit
        finally:
            runtime.close()
        _register(dest, source, limit)
        return
    _require_torch()
    from bge_m3.model import load_encoder

    encoder = load_encoder(source)
    dest.mkdir(parents=True, exist_ok=True)
    tokenizer = source / "tokenizer.json"
    if not tokenizer.is_file():
        message = "tokenizer.json is not in the checkout"
        raise FileNotFoundError(message)
    shutil.copyfile(tokenizer, dest / "tokenizer.json")
    print("packing int8", flush=True)
    _write_weights(dest, encoder)
    print("scoring", flush=True)
    _write_manifest(dest, complete=True)
    try:
        scores = _score(encoder, dest)
    except Exception:
        _write_manifest(dest, complete=False)
        raise
    if scores["min"] < _COSINE_MIN:
        _write_manifest(dest, complete=False)
        message = f"cosine min {scores['min']} is below {_COSINE_MIN}"
        raise RuntimeError(message)
    _write_json(dest / "cosine.json", scores)
    print(
        f"cosine passages {scores['passages']} min {scores['min']} mean {scores['mean']}",
        flush=True,
    )
    match = _match_pads(dest)
    print(
        f"match pad passages {match['passages']} min {match['min']} mean {match['mean']}",
        flush=True,
    )
    if match["min"] < _MATCH_MIN:
        _write_manifest(dest, complete=False)
        message = f"pad match min {match['min']} is below {_MATCH_MIN}"
        raise RuntimeError(message)
    from bge_m3.runtime import Runtime

    runtime = Runtime(dest)
    try:
        limit = runtime.limit
    finally:
        runtime.close()
    _register(dest, source, limit)


def _write_weights(dest: Path, encoder: Any) -> None:
    import torch

    arrays: dict[str, np.ndarray] = {
        "word": _half(encoder.word.weight),
        "position": _half(encoder.position.weight),
        "type": _half(encoder.type_row.reshape(-1)),
        "embed_w": _half(encoder.embed_norm.weight),
        "embed_b": _half(encoder.embed_norm.bias),
    }
    for index, (attention, feed) in enumerate(
        zip(encoder.attention, encoder.feed_forward, strict=True)
    ):
        layers = {
            "q": attention.query,
            "k": attention.key,
            "v": attention.value,
            "o": attention.out,
            "up": feed.up,
            "down": feed.down,
        }
        for name, layer in layers.items():
            weight = layer.conv.weight.detach().to(torch.float32).numpy()[:, :, 0, 0]
            bias = layer.conv.bias.detach().to(torch.float32).numpy()
            codes, scale = pack_int8(weight)
            prefix = f"l{index}_{name}"
            arrays[f"{prefix}_codes"] = codes
            arrays[f"{prefix}_scale"] = scale
            arrays[f"{prefix}_bias"] = np.ascontiguousarray(bias, dtype=np.float16)
        arrays[f"l{index}_attn_w"] = _half(attention.norm.weight)
        arrays[f"l{index}_attn_b"] = _half(attention.norm.bias)
        arrays[f"l{index}_ffn_w"] = _half(feed.norm.weight)
        arrays[f"l{index}_ffn_b"] = _half(feed.norm.bias)
    np.savez(dest / "weights.npz", **arrays)


def _score(encoder: Any, dest: Path) -> dict[str, float]:
    import torch
    from tokenizers import Tokenizer

    from bge_m3.runtime import Runtime

    tokenizer = Tokenizer.from_file(str(dest / "tokenizer.json"))
    runtime = Runtime(dest)
    scores: list[float] = []
    try:
        encoder.eval()
        with torch.no_grad():
            for text in passages():
                ids = list(tokenizer.encode(text).ids)
                length = padded_length(len(ids))
                padded = pad_ids(ids, length)
                array = np.asarray([padded], dtype=np.int64)
                positions = np.asarray([position_ids(padded)], dtype=np.int64)
                mask = np.asarray(
                    [0.0 if token != 1 else -40000.0 for token in padded], dtype=np.float32
                )
                reference = encoder(
                    torch.from_numpy(array),
                    torch.from_numpy(positions),
                    torch.from_numpy(mask.reshape(1, 1, length)),
                )[0]
                reference = l2_normalize(reference[0, 0].to(torch.float32).numpy())
                predicted = np.asarray(runtime.embed(text).vector, dtype=np.float32)
                scores.append(float(np.dot(reference, predicted)))
                if len(scores) % 16 == 0:
                    print(f"cosine {len(scores)} {min(scores):.6f}", flush=True)
    finally:
        runtime.close()
    return {"passages": len(scores), "min": min(scores), "mean": sum(scores) / len(scores)}


def _match_pads(dest: Path) -> dict[str, float]:
    """The same tokens at the next tile must keep the vector."""
    from tokenizers import Tokenizer

    from bge_m3.runtime import Runtime

    tokenizer = Tokenizer.from_file(str(dest / "tokenizer.json"))
    runtime = Runtime(dest)
    scores: list[float] = []
    try:
        for text in passages():
            ids = list(tokenizer.encode(text).ids)
            short = padded_length(len(ids))
            long = short + 64
            if long > runtime.limit:
                continue
            first = _vector_at(runtime, ids, short)
            second = _vector_at(runtime, ids, long)
            scores.append(float(np.dot(first, second)))
    finally:
        runtime.close()
    if not scores:
        message = "pad match has no passage"
        raise RuntimeError(message)
    return {"passages": len(scores), "min": min(scores), "mean": sum(scores) / len(scores)}


def _vector_at(runtime: Any, ids: list[int], length: int) -> np.ndarray:
    from bge_m3.text import key_mask as mask_of

    padded = pad_ids(ids, length)
    hidden = runtime.gpu.encode(
        runtime._weights,
        np.asarray(padded, dtype=np.int32),
        np.asarray(position_ids(padded), dtype=np.int32),
        mask_of(padded).reshape(-1),
        runtime._work,
    )
    return l2_normalize(hidden)


def _half(tensor: Any) -> np.ndarray:
    return np.ascontiguousarray(tensor.detach().half().cpu().numpy())


def _finished(dest: Path) -> bool:
    if not (dest / "weights.npz").is_file() or not (dest / "tokenizer.json").is_file():
        return False
    cosine = _read_cosine(dest / "cosine.json")
    if cosine is None:
        return False
    try:
        body = json.loads((dest / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        bool(body.get("complete"))
        and body.get("quant") == "int8"
        and body.get("hidden") == HIDDEN
    )


def _read_cosine(path: Path) -> dict[str, float] | None:
    if not path.is_file():
        return None
    try:
        body = json.loads(path.read_text(encoding="utf-8"))
        scores = {
            "passages": int(body["passages"]),
            "min": float(body["min"]),
            "mean": float(body["mean"]),
        }
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None
    if scores["passages"] != 128 or scores["min"] < _COSINE_MIN:
        return None
    return scores


def _write_manifest(dest: Path, *, complete: bool) -> None:
    _write_json(
        dest / "manifest.json",
        {"complete": complete, "quant": "int8", "hidden": HIDDEN, "context": 4096},
    )


def _write_json(path: Path, body: dict[str, Any]) -> None:
    path.write_text(json.dumps(body), encoding="utf-8")


def _register(dest: Path, source: Path, limit: int) -> None:
    from favlo.registry import register_quantized

    register_quantized(
        dest.name,
        source=source,
        alias=None,
        directory=dest,
        pipeline="bge-m3",
        runtime="metal41",
        max_prompt_tokens=limit,
    )


def _require_torch() -> None:
    try:
        import torch  # noqa: F401
    except ImportError as exc:
        message = "quantize needs torch installed for this pipeline"
        raise RuntimeError(message) from exc
