"""Score the 24 MoE layers and pack ``models/ZONOS2-fp8-fp4``."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from zonos2.checkpoint import Tensor, list_tensors, read_tensor
from zonos2.formats import choose_promoted, quantize_fp4, quantize_fp8, relative_error
from zonos2.layout import EXPERTS, FFN, FP8_PERCENT, moe_layers

_RECORD = Path("models/ZONOS2-fp8-fp4")


def quantize(src: Path, dest: Path | None = None) -> Path:
    """Read a BF16 checkout and write the packed record. Scores land before the binaries."""
    weights = src / "model.pth"
    if not weights.is_file():
        message = "model.pth is not in the source directory"
        raise FileNotFoundError(message)
    record = dest or _RECORD
    record.mkdir(parents=True, exist_ok=True)
    catalog = {item.name: item for item in list_tensors(weights)}
    scores = _scores(weights, catalog, record / "ffn_error.json")
    promoted = choose_promoted(scores, FP8_PERCENT)
    _pack(weights, catalog, record, promoted)
    manifest = {
        "complete": True,
        "fp8_percent": FP8_PERCENT,
        "fp8_layers": list(promoted),
        "half_planes": ["attention", "dense", "head"],
        "source": src.name,
    }
    _replace(record / "manifest.json", json.dumps(manifest, indent=2).encode())
    return record


def _scores(weights: Path, catalog: dict[str, Tensor], path: Path) -> dict[int, float]:
    if path.is_file():
        stored = json.loads(path.read_text(encoding="utf-8"))
        return {int(key): float(value) for key, value in stored["scores"].items()}
    scores: dict[int, float] = {}
    for layer in moe_layers():
        gate_up = read_tensor(weights, _need(catalog, f"layers.{layer}.feed_forward.experts.w13"))
        down = read_tensor(weights, _need(catalog, f"layers.{layer}.feed_forward.experts.w2"))
        worst = 0.0
        for expert in range(EXPERTS):
            gate, up = split_w13(gate_up[expert])
            score = max(
                relative_error(gate),
                relative_error(up),
                relative_error(down[expert]),
            )
            worst = max(worst, score)
        scores[layer] = worst
        print(f"layer {layer} {worst:.6f}", flush=True)
    payload = {
        "scores": {str(layer): scores[layer] for layer in sorted(scores)},
        "unit": "moe-layer",
        "percent": FP8_PERCENT,
    }
    _replace(path, json.dumps(payload, indent=2).encode())
    return scores


def split_w13(fused: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Even rows are the SiLU gate. Odd rows are the multiply branch."""
    if fused.ndim != 2 or fused.shape[0] != FFN * 2:
        message = "w13 is the interleaved expert gate and up projections"
        raise ValueError(message)
    return fused[0::2], fused[1::2]


def matrix_plane(name: str) -> str:
    """Attention, the dense FFNs, and the audio head are float16. Other matmuls stay fp4."""
    widened = (
        name == "multi_output.weight"
        or ".attention." in name
        or ".feed_forward.w_in" in name
        or ".feed_forward.w_out" in name
    )
    if widened and _matmul_name(name):
        return "f16"
    return "fp4"


def _pack(weights: Path, catalog: dict[str, Tensor], record: Path, promoted: tuple[int, ...]) -> None:
    promoted_set = set(promoted)
    for layer in moe_layers():
        plane = "fp8" if layer in promoted_set else "fp4"
        names = [
            f"layers.{layer}.expert.{expert}.{part}"
            for expert in range(EXPERTS)
            for part in ("gate", "up", "down")
        ]
        if all(_ready(record, name, plane) for name in names):
            print(f"packed layer {layer} {plane}", flush=True)
            continue
        gate_up = read_tensor(weights, _need(catalog, f"layers.{layer}.feed_forward.experts.w13"))
        down = read_tensor(weights, _need(catalog, f"layers.{layer}.feed_forward.experts.w2"))
        for expert in range(EXPERTS):
            gate, up = split_w13(gate_up[expert])
            _store_matrix(record, f"layers.{layer}.expert.{expert}.gate", gate, plane)
            _store_matrix(record, f"layers.{layer}.expert.{expert}.up", up, plane)
            _store_matrix(record, f"layers.{layer}.expert.{expert}.down", down[expert], plane)
        print(f"packed layer {layer} {plane}", flush=True)
    for name, tensor in catalog.items():
        if ".experts." in name:
            continue
        shape = tensor.shape
        if _matmul_name(name) and shape and shape[-1] % 32 == 0 and len(shape) in (2, 3):
            plane = matrix_plane(name)
            stored = [name] if len(shape) == 2 else [f"{name}.{index}" for index in range(shape[0])]
            if all(_ready(record, item, plane) for item in stored):
                continue
            array = read_tensor(weights, tensor)
            if array.ndim == 2:
                _store_matrix(record, name, array, plane)
            else:
                for index in range(array.shape[0]):
                    _store_matrix(record, f"{name}.{index}", array[index], plane)
            print(f"packed {name} {plane}", flush=True)
            continue
        if (record / "weights" / f"{name}.f16").is_file():
            continue
        _store_half(record, name, read_tensor(weights, tensor))


def _matmul_name(name: str) -> bool:
    return name.endswith(".weight") and "norm" not in name and "embed" not in name and "gater" not in name and "projection" not in name and "temp" not in name and "router" not in name


def _ready(record: Path, name: str, plane: str) -> bool:
    folder = record / "weights"
    if plane == "f16":
        return (folder / f"{name}.f16").is_file() and not _quantized_sibling(folder, name)
    other = "fp4" if plane == "fp8" else "fp8"
    return (
        (folder / f"{name}.{plane}").is_file()
        and (folder / f"{name}.scale").is_file()
        and not (folder / f"{name}.{other}").is_file()
        and not (folder / f"{name}.f16").is_file()
    )


def _store_matrix(record: Path, name: str, matrix: np.ndarray, plane: str) -> None:
    folder = record / "weights"
    if plane == "f16":
        _replace(folder / f"{name}.f16", np.ascontiguousarray(matrix, dtype=np.float16).tobytes())
        _drop_quantized(folder, name)
        return
    data_path = folder / f"{name}.{plane}"
    scale_path = folder / f"{name}.scale"
    other = "fp4" if plane == "fp8" else "fp8"
    sibling = folder / f"{name}.{other}"
    if data_path.is_file() and scale_path.is_file() and not sibling.is_file() and not (folder / f"{name}.f16").is_file():
        return
    data, scales = quantize_fp8(matrix) if plane == "fp8" else quantize_fp4(matrix)
    _replace(data_path, np.ascontiguousarray(data).tobytes())
    _replace(scale_path, np.ascontiguousarray(scales).tobytes())
    if sibling.is_file():
        sibling.unlink()
    half = folder / f"{name}.f16"
    if half.is_file():
        half.unlink()


def _quantized_sibling(folder: Path, name: str) -> bool:
    return any((folder / f"{name}.{suffix}").is_file() for suffix in ("fp4", "fp8", "scale"))


def _drop_quantized(folder: Path, name: str) -> None:
    for suffix in ("fp4", "fp8", "scale"):
        path = folder / f"{name}.{suffix}"
        if path.is_file():
            path.unlink()


def _store_half(record: Path, name: str, array: np.ndarray) -> None:
    path = record / "weights" / f"{name}.f16"
    if path.is_file():
        return
    _replace(path, np.ascontiguousarray(array, dtype=np.float16).tobytes())


def _need(catalog: dict[str, Tensor], name: str) -> Tensor:
    tensor = catalog.get(name)
    if tensor is None:
        message = f"checkpoint is missing {name}"
        raise KeyError(message)
    return tensor


def _replace(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)
