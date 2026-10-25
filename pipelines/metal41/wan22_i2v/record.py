"""Quantize one expert pair and load the packed record."""

from __future__ import annotations

import json
import zipfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from safetensors import safe_open
from safetensors.numpy import save_file

from wan22_i2v.dit import Expert, Resident, upload_half, upload_linear
from wan22_i2v.formats import (
    BLOCK_MATRICES,
    LinearPack,
    as_float32,
    block_winners,
    pack_linear,
    relative_error,
    restore_linear,
)
from wan22_i2v.schedule import GRAPH, PROMOTE_PERCENT, promote_count


def linear_names(layers: int) -> list[str]:
    names = [
        "text_embedding.0.weight",
        "text_embedding.2.weight",
        "time_embedding.0.weight",
        "time_embedding.2.weight",
        "time_projection.1.weight",
    ]
    for layer in range(layers):
        for part in ("self_attn", "cross_attn"):
            for proj in ("q", "k", "v", "o"):
                names.append(f"blocks.{layer}.{part}.{proj}.weight")
        names.append(f"blocks.{layer}.ffn.0.weight")
        names.append(f"blocks.{layer}.ffn.2.weight")
    return names


def half_names(layers: int) -> list[str]:
    names = [
        "patch_embedding.weight",
        "patch_embedding.bias",
        "head.modulation",
        "head.head.weight",
        "head.head.bias",
    ]
    for layer in range(layers):
        names.append(f"blocks.{layer}.modulation")
        names.append(f"blocks.{layer}.norm3.weight")
        names.append(f"blocks.{layer}.norm3.bias")
        for part in ("self_attn", "cross_attn"):
            names.append(f"blocks.{layer}.{part}.norm_q.weight")
            names.append(f"blocks.{layer}.{part}.norm_k.weight")
    return names


def write_graph_source(dest: Path, seed: int = 0) -> Path:
    """Small random experts with the Wan key names, for the 320p graph smoke."""
    dest.mkdir(parents=True, exist_ok=True)
    high_rng = np.random.default_rng(seed)
    low_rng = np.random.default_rng(seed + 1)
    embed_rng = np.random.default_rng(seed + 2)
    high = _expert_tensors(high_rng, GRAPH)
    low = _expert_tensors(low_rng, GRAPH)
    high["text_embed.weight"] = embed_rng.normal(0.0, 0.02, size=(256, GRAPH["text_dim"])).astype(
        np.float32
    )
    save_file(high, dest / "high.safetensors")
    save_file(low, dest / "low.safetensors")
    (dest / "config.json").write_text(json.dumps(GRAPH), encoding="utf-8")
    return dest


def quantize(
    high: Path,
    low: Path,
    dest: Path,
    config: dict[str, Any],
    *,
    high_dir: Path | None = None,
    low_dir: Path | None = None,
    percent: int = PROMOTE_PERCENT,
) -> dict[str, Any]:
    """Pack both experts. ``percent`` of the blocks are stored as fp8."""
    if percent < 0 or percent > 100:
        message = f"fp8 percent is 0 through 100, got {percent}"
        raise ValueError(message)
    dest.mkdir(parents=True, exist_ok=True)
    high_promoted = _pack_expert(high, high_dir or dest / "high", config, percent)
    low_promoted = _pack_expert(low, low_dir or dest / "low", config, percent)
    _copy_embed(high, dest / "embed.npy", int(config["text_dim"]))
    manifest = {
        "config": config,
        "fp8_percent": percent,
        "high_promoted": high_promoted,
        "low_promoted": low_promoted,
    }
    (dest / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return manifest


def quantize_int8(
    high: Path,
    low: Path,
    dest: Path,
    config: dict[str, Any],
    *,
    high_dir: Path | None = None,
    low_dir: Path | None = None,
) -> dict[str, Any]:
    """Pack both experts as per-channel int8 from the original weights."""
    dest.mkdir(parents=True, exist_ok=True)
    high_worst = pack_int8_expert(high, high_dir or dest / "high", config)
    low_worst = pack_int8_expert(low, low_dir or dest / "low", config)
    _copy_embed(high, dest / "embed.npy", int(config["text_dim"]))
    manifest = {
        "config": config,
        "weight": "int8",
        "high_promoted": [],
        "low_promoted": [],
        "high_worst": high_worst,
        "low_worst": low_worst,
    }
    (dest / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return manifest


def pack_int8_expert(path: Path, dest: Path, config: dict[str, Any]) -> float:
    """Pack one expert. Returns the worst relative RMS against the source matrix."""
    dest.mkdir(parents=True, exist_ok=True)
    layers = int(config["layers"])
    worst = 0.0
    worst_name = ""
    with safe_open(path, framework="numpy") as handle:
        keys = set(handle.keys())
        for name in linear_names(layers):
            dest_path = dest / f"{name}.npz"
            if _stored_int8(dest_path):
                continue
            weight = as_float32(handle.get_tensor(name))
            bias_name = name.replace(".weight", ".bias")
            bias = as_float32(handle.get_tensor(bias_name)) if bias_name in keys else None
            pack = pack_linear(weight, bias, "int8")
            error = relative_error(restore_linear(pack), weight)
            _save_pack(dest_path, pack)
            del weight
            del pack
            print(f"int8 {name} {error:.6f}", flush=True)
            if error >= worst:
                worst = error
                worst_name = name
        _write_halves(handle, keys, dest, config, layers)
    print(f"int8 worst {path.name} {worst_name} {worst:.6f}", flush=True)
    return worst


def load_manifest(path: Path) -> dict[str, Any]:
    return json.loads((path / "manifest.json").read_text(encoding="utf-8"))


def load_embed(path: Path) -> np.ndarray:
    return np.load(path / "embed.npy")


def load_linear(path: Path, name: str) -> LinearPack:
    """Load one packed matrix and close the file before the caller uploads it."""
    with np.load(path / f"{name}.npz") as blob:
        pack = LinearPack(
            kind=str(blob["kind"].item()),
            packed=np.array(blob["packed"], copy=True),
            scales=np.array(blob["scales"], copy=True),
            bias=np.array(blob["bias"], copy=True),
            perm=np.array(blob["perm"], copy=True),
            logical_n=int(blob["logical_n"]),
            logical_k=int(blob["logical_k"]),
            padded_n=int(blob["padded_n"]),
            stored_k=int(blob["stored_k"]),
        )
    return pack


def load_half(path: Path, name: str) -> np.ndarray:
    """Load one float16 weight into an array the caller can drop after upload."""
    return np.array(_half_array(path, name), copy=True)


def load_expert(
    path: Path, layers: int, on_file: Callable[[], None] | None = None
) -> Expert:
    packs: dict[str, LinearPack] = {}
    for name in linear_names(layers):
        packs[name] = load_linear(path, name)
        if on_file is not None:
            on_file()
    halves: dict[str, np.ndarray] = {}
    for name in half_names(layers):
        halves[name] = load_half(path, name)
        if on_file is not None:
            on_file()
    return Expert(packs=packs, halves=halves)


def stream_expert(
    gpu: Any,
    path: Path,
    layers: int,
    on_file: Callable[[], None] | None = None,
) -> Resident:
    """Upload one matrix at a time so the host copy is never the whole expert."""
    packs: dict[str, LinearPack] = {}
    weights: dict[str, dict[str, Any]] = {}
    owned: list[Any] = []
    for name in linear_names(layers):
        pack = load_linear(path, name)
        slim, buffers = upload_linear(gpu, pack)
        del pack
        packs[name] = slim
        weights[name] = buffers
        owned.extend(buffers.values())
        if on_file is not None:
            on_file()
    halves: dict[str, Any] = {}
    for name in half_names(layers):
        array = load_half(path, name)
        buf = upload_half(gpu, array)
        del array
        halves[name] = buf
        owned.append(buf)
        if on_file is not None:
            on_file()
    return Resident(packs=packs, weights=weights, halves=halves, host={}, owned=owned)


def _half_array(path: Path, name: str) -> np.ndarray:
    npy = path / f"{name}.npy"
    if npy.is_file():
        return np.load(npy)
    if name == "head.head.weight":
        return np.ascontiguousarray(_restored_head(path).astype(np.float16))
    if name == "head.head.bias":
        blob = np.load(path / "head.head.weight.npz")
        return np.ascontiguousarray(blob["bias"])
    message = f"{name} is missing"
    raise FileNotFoundError(message)


def _restored_head(path: Path) -> np.ndarray:
    blob = np.load(path / "head.head.weight.npz")
    pack = LinearPack(
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
    return restore_linear(pack)


def _pack_expert(
    path: Path, dest: Path, config: dict[str, Any], percent: int
) -> list[int]:
    dest.mkdir(parents=True, exist_ok=True)
    layers = int(config["layers"])
    scores = _score_blocks(path, layers)
    winners = block_winners(scores, promote_count(layers, percent))
    print(f"pack {path.name} fp8 {winners}", flush=True)
    chosen = set(winners)
    with safe_open(path, framework="numpy") as handle:
        keys = set(handle.keys())
        for name in linear_names(layers):
            kind = "fp8" if _block_index(name) in chosen and _is_block_matrix(name) else "fp4"
            dest_path = dest / f"{name}.npz"
            if _stored_kind(dest_path) == kind:
                continue
            weight = as_float32(handle.get_tensor(name))
            bias_name = name.replace(".weight", ".bias")
            bias = as_float32(handle.get_tensor(bias_name)) if bias_name in keys else None
            _save_pack(dest_path, pack_linear(weight, bias, kind))
            print(f"write {name}", flush=True)
        _write_halves(handle, keys, dest, config, layers)
    return winners


def _write_halves(
    handle: Any, keys: set[str], dest: Path, config: dict[str, Any], layers: int
) -> None:
    for name in half_names(layers):
        dest_path = dest / f"{name}.npy"
        (dest / f"{name}.npz").unlink(missing_ok=True)
        if _half_ready(dest_path):
            continue
        if name not in keys:
            if name != "head.head.bias":
                message = f"{name} is missing"
                raise RuntimeError(message)
            array = np.zeros(int(config["out_dim"]) * 4, dtype=np.float16)
        else:
            array = as_float32(handle.get_tensor(name)).astype(np.float16)
        np.save(dest_path, np.ascontiguousarray(array))
        print(f"write {name}", flush=True)


def _score_blocks(path: Path, layers: int) -> dict[int, float]:
    scores: dict[int, float] = {}
    with safe_open(path, framework="numpy") as handle:
        for layer in range(layers):
            worst = 0.0
            for suffix in BLOCK_MATRICES:
                weight = as_float32(handle.get_tensor(f"blocks.{layer}.{suffix}"))
                pack = pack_linear(weight, None, "fp4")
                worst = max(worst, relative_error(restore_linear(pack), weight))
            scores[layer] = worst
            print(f"score {path.name} {layer} {worst:.6f}", flush=True)
    return scores


def _copy_embed(path: Path, dest: Path, text_dim: int) -> None:
    with safe_open(path, framework="numpy") as handle:
        if "text_embed.weight" in handle.keys():
            array = as_float32(handle.get_tensor("text_embed.weight")).astype(np.float16)
        else:
            array = np.zeros((256, text_dim), dtype=np.float16)
    np.save(dest, array)


def _stored_int8(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        kind = str(np.load(path)["kind"].item())
    except (OSError, ValueError, EOFError, zipfile.BadZipFile):
        path.unlink(missing_ok=True)
        return False
    return kind == "int8"


def _stored_kind(path: Path) -> str | None:
    if not path.is_file():
        return None
    try:
        kind = str(np.load(path)["kind"].item())
    except (OSError, ValueError, EOFError, zipfile.BadZipFile):
        path.unlink(missing_ok=True)
        return None
    if kind not in {"fp4", "fp8"}:
        path.unlink(missing_ok=True)
        return None
    return kind


def _half_ready(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        np.load(path)
    except (OSError, ValueError, EOFError, zipfile.BadZipFile):
        path.unlink(missing_ok=True)
        return False
    return True


def _save_pack(path: Path, pack: LinearPack) -> None:
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


def _expert_tensors(rng: np.random.Generator, config: dict[str, Any]) -> dict[str, np.ndarray]:
    dim = int(config["dim"])
    ffn = int(config["ffn"])
    tensors: dict[str, np.ndarray] = {
        "patch_embedding.weight": rng.normal(0.0, 0.02, size=(dim, 36, 1, 2, 2)).astype(np.float32),
        "patch_embedding.bias": np.zeros(dim, dtype=np.float32),
        "head.modulation": rng.normal(0.0, 0.01, size=(1, 2, dim)).astype(np.float32),
        "text_embedding.0.weight": _matrix(rng, dim, int(config["text_dim"])),
        "text_embedding.0.bias": np.zeros(dim, dtype=np.float32),
        "text_embedding.2.weight": _matrix(rng, dim, dim),
        "text_embedding.2.bias": np.zeros(dim, dtype=np.float32),
        "time_embedding.0.weight": _matrix(rng, dim, int(config["freq_dim"])),
        "time_embedding.0.bias": np.zeros(dim, dtype=np.float32),
        "time_embedding.2.weight": _matrix(rng, dim, dim),
        "time_embedding.2.bias": np.zeros(dim, dtype=np.float32),
        "time_projection.1.weight": _matrix(rng, dim * 6, dim),
        "time_projection.1.bias": np.zeros(dim * 6, dtype=np.float32),
        "head.head.weight": _matrix(rng, int(config["out_dim"]) * 4, dim),
        "head.head.bias": np.zeros(int(config["out_dim"]) * 4, dtype=np.float32),
    }
    for layer in range(int(config["layers"])):
        tensors[f"blocks.{layer}.modulation"] = rng.normal(0.0, 0.01, size=(1, 6, dim)).astype(
            np.float32
        )
        tensors[f"blocks.{layer}.norm3.weight"] = np.ones(dim, dtype=np.float32)
        tensors[f"blocks.{layer}.norm3.bias"] = np.zeros(dim, dtype=np.float32)
        for part in ("self_attn", "cross_attn"):
            for proj in ("q", "k", "v", "o"):
                tensors[f"blocks.{layer}.{part}.{proj}.weight"] = _matrix(rng, dim, dim)
                tensors[f"blocks.{layer}.{part}.{proj}.bias"] = np.zeros(dim, dtype=np.float32)
            tensors[f"blocks.{layer}.{part}.norm_q.weight"] = np.ones(dim, dtype=np.float32)
            tensors[f"blocks.{layer}.{part}.norm_k.weight"] = np.ones(dim, dtype=np.float32)
        tensors[f"blocks.{layer}.ffn.0.weight"] = _matrix(rng, ffn, dim)
        tensors[f"blocks.{layer}.ffn.0.bias"] = np.zeros(ffn, dtype=np.float32)
        tensors[f"blocks.{layer}.ffn.2.weight"] = _matrix(rng, dim, ffn)
        tensors[f"blocks.{layer}.ffn.2.bias"] = np.zeros(dim, dtype=np.float32)
    return tensors


def _matrix(rng: np.random.Generator, rows: int, cols: int) -> np.ndarray:
    return rng.normal(0.0, 0.02, size=(rows, cols)).astype(np.float32)


def _is_block_matrix(name: str) -> bool:
    return any(name.endswith(suffix) for suffix in BLOCK_MATRICES)


def _block_index(name: str) -> int | None:
    if not name.startswith("blocks."):
        return None
    return int(name.split(".", 2)[1])
