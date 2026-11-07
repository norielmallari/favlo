"""BF16 Diffusers checkpoint to fp8 planes, with a mixed DiT MLP.

Every linear and convolution is fp8 E4M3 with an E8M0 scale per 32 columns,
except the DiT MLP. Twenty percent of those layers stay on the highest fp4
relative error and are stored as fp8. The rest of the MLP stays fp4. Norms,
the silence latent, and Snake parameters are float32. The VAE encoder, the
null condition, and the unused timbre token are left out.
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ace_step_15_turbo.pack import pack_fp4, pack_fp8, promote_count, promoted_ids, relative_error

PACKED_NAME = "acestep-v15-xl-turbo-fp8-fp4"
MLP_FP8_PERCENT = 20.0
DIT_LAYERS = 32
SOURCE_NAME = "acestep-v15-xl-turbo"
HUB = "ACE-Step/acestep-v15-xl-turbo-diffusers"


@dataclass
class Plane:
    data: Any
    scale: Any | None
    rows: int
    cols: int
    inner: int
    kind: str = "f32"
    bias: Any | None = None


def _keep(key: str) -> bool:
    if "null_condition" in key or "special_token" in key:
        return False
    if key.startswith("encoder.") or ".encoder." in key:
        return False
    if "fsq" in key or "quantizer" in key or "audio_decoder" in key or key.startswith("lm_head"):
        return False
    return True


def _header(path: Path) -> tuple[int, dict[str, Any]]:
    with path.open("rb") as handle:
        size = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(size))
    return size, header


def _array(path: Path, header_size: int, info: dict[str, Any]) -> np.ndarray:
    start, end = info["data_offsets"]
    shape = tuple(int(dim) for dim in info["shape"])
    with path.open("rb") as handle:
        handle.seek(8 + header_size + int(start))
        raw = handle.read(int(end) - int(start))
    dtype = info["dtype"]
    if dtype == "BF16":
        bits = np.frombuffer(raw, dtype="<u2").astype(np.uint32) << 16
        return bits.view(np.float32).reshape(shape).copy()
    if dtype == "F16":
        return np.frombuffer(raw, dtype="<f2").reshape(shape).astype(np.float32, copy=True)
    if dtype == "F32":
        return np.frombuffer(raw, dtype="<f4").reshape(shape).copy()
    message = f"unsupported dtype {dtype}"
    raise TypeError(message)


def _fuse(direction: np.ndarray, scale: np.ndarray) -> np.ndarray:
    vector = np.ascontiguousarray(direction, dtype=np.float32)
    gain = np.ascontiguousarray(scale, dtype=np.float32).reshape((vector.shape[0],) + (1,) * (vector.ndim - 1))
    axes = tuple(range(1, vector.ndim))
    norm = np.sqrt(np.sum(vector * vector, axis=axes, keepdims=True))
    return gain * vector / np.maximum(norm, 1e-12)


def _write(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(np.ascontiguousarray(array).tobytes())
    temporary.replace(path)


def quantize(source: Path, dest: Path) -> None:
    """Pack one Diffusers checkout. ``manifest.json`` is written last."""
    if (dest / "manifest.json").is_file():
        body = json.loads((dest / "manifest.json").read_text(encoding="utf-8"))
        if body.get("complete") is True:
            if float(body.get("mlp_fp8_percent", -1)) != MLP_FP8_PERCENT:
                message = "record percentage does not match the MLP fp8 budget"
                raise RuntimeError(message)
            return
    shards = sorted(source.rglob("*.safetensors"))
    if not shards:
        message = f"no safetensors under {source}"
        raise FileNotFoundError(message)
    dest.mkdir(parents=True, exist_ok=True)
    weight_dir = dest / "weights"
    index: dict[str, Any] = {"tensors": {}}
    grouped: dict[str, dict[str, Any]] = {}
    for shard in shards:
        header_size, header = _header(shard)
        stem = _stem(shard)
        for key, info in header.items():
            if key == "__metadata__" or not _keep(key):
                continue
            stored = f"{stem}{key}" if stem else key
            grouped.setdefault(stored, {"path": shard, "header": header_size, "info": info})
    names = sorted(grouped)
    promoted = _mlp_budget(grouped, dest)
    fused: set[str] = set()
    for key in names:
        if key.endswith(".weight_g"):
            stem = key[: -len(".weight_g")]
            direction = stem + ".weight_v"
            if direction not in grouped:
                message = f"{key} has no weight_v"
                raise KeyError(message)
            values = _fuse(
                _load(grouped[direction]),
                _load(grouped[key]),
            )
            _store_weight(index, weight_dir, stem + ".weight", values, "fp8")
            print(stem + ".weight", flush=True)
            fused.add(key)
            fused.add(direction)
    for key in names:
        if key in fused:
            continue
        if key.endswith(".weight_v") or key.endswith(".weight_g"):
            continue
        values = _load(grouped[key])
        if key.endswith(".weight") and values.ndim >= 2:
            kind = "fp8" if _fp8_weight(key, promoted) else "fp4"
            _store_weight(index, weight_dir, key, values, kind)
        else:
            _store_vector(index, weight_dir, key, values)
        print(key, flush=True)
    tokenizer = source / "tokenizer" / "tokenizer.json"
    if not tokenizer.is_file():
        message = "tokenizer.json is missing"
        raise FileNotFoundError(message)
    (dest / "tokenizer.json").write_bytes(tokenizer.read_bytes())
    _write_json(dest / "index.json", index)
    _write_json(
        dest / "manifest.json",
        {
            "complete": True,
            "pipeline": "ace-step-15-turbo",
            "base_model": "ACE-Step/acestep-v15-xl-turbo",
            "mlp_fp8_percent": MLP_FP8_PERCENT,
            "mlp_fp8_layers": sorted(promoted),
        },
    )


def _stem(path: Path) -> str:
    """Keep the text tower and the DiT from sharing ``layers.*`` names."""
    parent = path.parent.name
    if parent == "text_encoder":
        return "text."
    if parent == "transformer":
        return "dit."
    return ""


def _load(item: dict[str, Any]) -> np.ndarray:
    return _array(item["path"], item["header"], item["info"])


def _mlp_part(key: str) -> int | None:
    prefix = "dit.layers."
    if not key.startswith(prefix) or not key.endswith(".weight"):
        return None
    layer_text, _, tail = key[len(prefix) :].partition(".")
    if tail not in ("mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight"):
        return None
    if not layer_text.isdigit():
        return None
    return int(layer_text)


def _fp8_weight(key: str, promoted: set[int]) -> bool:
    layer = _mlp_part(key)
    if layer is None:
        return True
    return layer in promoted


def _mlp_budget(grouped: dict[str, dict[str, Any]], dest: Path) -> set[int]:
    """Score each DiT MLP by its worst fp4 matrix and promote the top 20%."""
    path = dest / "mlp_error.json"
    count = promote_count(MLP_FP8_PERCENT, DIT_LAYERS)
    if path.is_file():
        saved = json.loads(path.read_text(encoding="utf-8"))
        if float(saved.get("percent", -1)) == MLP_FP8_PERCENT and int(saved.get("count", -1)) == count:
            return {int(layer) for layer in saved["promoted"]}
    parts = ("gate", "up", "down")
    scores: dict[int, dict[str, float]] = {}
    for layer in range(DIT_LAYERS):
        layer_scores: dict[str, float] = {}
        for part in parts:
            key = f"dit.layers.{layer}.mlp.{part}_proj.weight"
            if key not in grouped:
                message = f"record is missing {key}"
                raise KeyError(message)
            layer_scores[part] = relative_error(_load(grouped[key]))
            print(f"mlp {layer} {part} {layer_scores[part]:.6f}", flush=True)
        layer_scores["layer"] = max(layer_scores[part] for part in parts)
        scores[layer] = layer_scores
    chosen = promoted_ids({layer: item["layer"] for layer, item in scores.items()}, count)
    _write_json(
        path,
        {
            "percent": MLP_FP8_PERCENT,
            "count": count,
            "layers": DIT_LAYERS,
            "scores": {str(layer): scores[layer] for layer in range(DIT_LAYERS)},
            "promoted": chosen,
        },
    )
    print(f"mlp fp8 layers {chosen}", flush=True)
    return set(chosen)


def _store_weight(index: dict[str, Any], folder: Path, key: str, values: np.ndarray, kind: str) -> None:
    matrix, transpose = _as_matrix(key, values)
    packed, scales, padded_rows, padded_cols = pack_fp8(matrix) if kind == "fp8" else pack_fp4(matrix)
    stem = f"{len(index['tensors']):05d}"
    suffix = "fp8" if kind == "fp8" else "fp4"
    _write(folder / f"{stem}.{suffix}", packed)
    _write(folder / f"{stem}.e8m0", scales)
    index["tensors"][key] = {
        "kind": kind,
        "rows": int(matrix.shape[0]),
        "cols": int(matrix.shape[1]),
        "padded_rows": int(padded_rows),
        "padded_cols": int(padded_cols),
        "transpose": transpose,
        "file": f"weights/{stem}.{suffix}",
        "scale": f"weights/{stem}.e8m0",
    }


def _as_matrix(key: str, values: np.ndarray) -> tuple[np.ndarray, bool]:
    if values.ndim == 2:
        return np.ascontiguousarray(values, dtype=np.float32), False
    if values.ndim != 3:
        message = f"{key} is not a matrix or a convolution"
        raise ValueError(message)
    transpose = "conv_t" in key or "proj_out_conv" in key
    out_in, kernel = values.shape[0] * values.shape[1], values.shape[2]
    del out_in
    matrix = np.ascontiguousarray(values.reshape(values.shape[0], values.shape[1] * kernel), dtype=np.float32)
    return matrix, transpose


def _store_vector(index: dict[str, Any], folder: Path, key: str, values: np.ndarray) -> None:
    flat = np.ascontiguousarray(values, dtype=np.float32).reshape(-1)
    stem = f"{len(index['tensors']):05d}"
    _write(folder / f"{stem}.f32", flat)
    index["tensors"][key] = {
        "kind": "f32",
        "rows": int(flat.shape[0]),
        "cols": 1,
        "padded_rows": int(flat.shape[0]),
        "padded_cols": 1,
        "shape": [int(dim) for dim in values.shape],
        "file": f"weights/{stem}.f32",
    }


def _write_json(path: Path, body: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(body), encoding="utf-8")
    temporary.replace(path)


def weight_bytes(folder: Path) -> int:
    """Bytes ``load`` uploads from a packed record."""
    index = json.loads((folder / "index.json").read_text(encoding="utf-8"))
    total = 0
    seen: set[str] = set()
    for meta in index["tensors"].values():
        for key in ("file", "scale"):
            name = meta.get(key)
            if not isinstance(name, str) or name in seen:
                continue
            seen.add(name)
            total += (folder / name).stat().st_size
    return total


def load(gpu: Any, folder: Path) -> dict[str, Plane]:
    """Upload the packed record. Weights stay resident."""
    index = json.loads((folder / "index.json").read_text(encoding="utf-8"))
    planes: dict[str, Plane] = {}
    for key, meta in index["tensors"].items():
        kind = meta["kind"]
        if kind in ("fp4", "fp8"):
            packed = np.fromfile(folder / meta["file"], dtype=np.uint8)
            scales = np.fromfile(folder / meta["scale"], dtype=np.uint8)
            data = gpu.resident(gpu.upload(packed))
            scale = gpu.resident(gpu.upload(scales))
            planes[key] = Plane(
                data, scale, int(meta["rows"]), int(meta["cols"]), int(meta["padded_cols"]), kind
            )
        else:
            values = np.fromfile(folder / meta["file"], dtype=np.float32)
            data = gpu.resident(gpu.upload(values))
            planes[key] = Plane(data, None, int(meta["rows"]), 1, 1, "f32")
    for key, plane in list(planes.items()):
        if not key.endswith(".weight"):
            continue
        bias = planes.get(key[: -len(".weight")] + ".bias")
        if bias is not None:
            plane.bias = bias.data
    return planes


def plane(weights: dict[str, Plane], *keys: str) -> Plane:
    for key in keys:
        found = weights.get(key)
        if found is not None:
            return found
    message = f"record is missing {keys[0]}"
    raise KeyError(message)
