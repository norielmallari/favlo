"""Pack the Turbo checkpoint and load the fp4 record onto the device."""

from __future__ import annotations

import json
import shutil
import struct
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from z_image_turbo.formats import as_float32, pack_fp4
from z_image_turbo.metal import Gpu

_PAD = {"cap_pad_token", "x_pad_token"}


class Plane:
    """One packed tensor. ``data`` is the device buffer."""

    def __init__(
        self,
        name: str,
        plane: str,
        shape: tuple[int, ...],
        logical_k: int,
        stored_k: int,
        data: Any,
        scale: Any | None,
    ) -> None:
        self.name = name
        self.plane = plane
        self.shape = shape
        self.logical_k = logical_k
        self.stored_k = stored_k
        self.data = data
        self.scale = scale

    def release(self) -> None:
        self.data = None
        self.scale = None


def quantize(
    weights: Path,
    dest: Path,
    *,
    tokenizer_source: Path | None = None,
) -> Path:
    """Pack generate-path weights. The VAE encoder is left unpacked."""
    manifest_path = dest / "manifest.json"
    if manifest_path.is_file():
        body = json.loads(manifest_path.read_text(encoding="utf-8"))
        if body.get("complete") is True:
            return dest
    if dest.exists():
        shutil.rmtree(dest)
    root = _tokenizer_root(weights, tokenizer_source)
    weights = weights.resolve()
    out_weights = dest / "weights"
    out_weights.mkdir(parents=True)
    tensors: dict[str, Any] = {}
    seen: set[str] = set()
    for shard, header, data_start in _shards(weights):
        print(f"quantize {shard.name}", flush=True)
        for key, info in header.items():
            if key == "__metadata__" or key.startswith("encoder."):
                continue
            if key in seen:
                message = f"tensor {key} appears in more than one safetensors file"
                raise ValueError(message)
            seen.add(key)
            array = _read(shard, info, data_start)
            kind = _kind(key, array.shape)
            entry: dict[str, Any] = {
                "plane": kind,
                "shape": list(array.shape),
            }
            if kind == "f16":
                raw = np.ascontiguousarray(array, dtype=np.float32).astype(np.float16)
                file_name = f"{key}.f16"
                (out_weights / file_name).write_bytes(raw.tobytes())
                entry["file"] = f"weights/{file_name}"
            else:
                matrix = array.reshape(array.shape[0], -1)
                packed, scales, stored = pack_fp4(matrix)
                data_name = f"{key}.fp4"
                scale_name = f"{key}.scale"
                (out_weights / data_name).write_bytes(np.ascontiguousarray(packed).tobytes())
                (out_weights / scale_name).write_bytes(np.ascontiguousarray(scales).tobytes())
                entry["logical_k"] = int(matrix.shape[1])
                entry["stored_k"] = int(stored)
                entry["file"] = f"weights/{data_name}"
                entry["scale"] = f"weights/{scale_name}"
            tensors[key] = entry
            del array
    body = {
        "complete": True,
        "format": "z-image-turbo-fp4",
        "source_repo": "Tongyi-MAI/Z-Image-Turbo",
        "source": str(root.resolve()),
        "tile_m": 64,
        "tile_n": 32,
        "tensors": tensors,
    }
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(body), encoding="utf-8")
    temporary.replace(manifest_path)
    return dest


def weight_bytes(path: Path) -> int:
    """Bytes ``load`` will read from a finished record."""
    total = 0
    for entry in _manifest(path)["tensors"].values():
        total += (path / entry["file"]).stat().st_size
        scale = entry.get("scale")
        if isinstance(scale, str):
            total += (path / scale).stat().st_size
    return total


def load(
    path: Path,
    gpu: Gpu,
    on_bytes: Callable[[int], None] | None = None,
) -> dict[str, Plane]:
    planes: dict[str, Plane] = {}
    items = _manifest(path)["tensors"]
    for index, (name, entry) in enumerate(items.items()):
        raw = (path / entry["file"]).read_bytes()
        scale = None
        scale_bytes = 0
        stored = int(entry.get("stored_k") or 0)
        logical = int(entry.get("logical_k") or 0)
        if entry["plane"] == "fp4":
            scale_raw = (path / entry["scale"]).read_bytes()
            scale_bytes = len(scale_raw)
            data = gpu.keep(gpu.upload(np.frombuffer(raw, dtype=np.uint8)))
            scale = gpu.keep(gpu.upload(np.frombuffer(scale_raw, dtype=np.uint8)))
        elif entry["plane"] == "f16":
            data = gpu.keep(gpu.upload(np.frombuffer(raw, dtype=np.float16)))
        else:
            message = f"unknown plane {entry['plane']} on {name}"
            raise RuntimeError(message)
        planes[name] = Plane(
            name,
            entry["plane"],
            tuple(int(dim) for dim in entry["shape"]),
            logical,
            stored,
            data,
            scale,
        )
        if on_bytes is not None:
            on_bytes(len(raw) + scale_bytes)
        elif index % 100 == 0:
            print(f"load {index}/{len(items)}", flush=True)
    if on_bytes is None:
        print(f"load {len(items)} tensors", flush=True)
    return planes


def _manifest(path: Path) -> dict[str, Any]:
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("complete") is not True:
        message = f"{path} is not a finished record"
        raise RuntimeError(message)
    return manifest


def _kind(key: str, shape: tuple[int, ...]) -> str:
    if key in _PAD or len(shape) == 1:
        return "f16"
    if len(shape) in {2, 4}:
        return "fp4"
    message = f"{key} has shape {shape}"
    raise ValueError(message)


def _tokenizer_root(weights: Path, tokenizer_source: Path | None) -> Path:
    if tokenizer_source is not None:
        return tokenizer_source.resolve()
    if weights.is_dir() and _is_diffusers_layout(weights):
        return weights.resolve()
    from z_image_turbo.source import source_dir

    fallback = source_dir()
    if (fallback / "tokenizer" / "tokenizer.json").is_file():
        return fallback
    message = (
        "tokenizer source is required when weights are not a full Diffusers checkout; "
        "pass --tokenizer-source"
    )
    raise FileNotFoundError(message)


def _is_diffusers_layout(source: Path) -> bool:
    return (
        (source / "text_encoder" / "model.safetensors.index.json").is_file()
        and (source / "transformer" / "diffusion_pytorch_model.safetensors.index.json").is_file()
        and (source / "vae" / "diffusion_pytorch_model.safetensors").is_file()
    )


def _shards(weights: Path) -> list[tuple[Path, dict[str, Any], int]]:
    if weights.is_file():
        if weights.suffix != ".safetensors":
            message = f"weights must be a .safetensors file, got {weights.name}"
            raise ValueError(message)
        header, data_start = _header(weights)
        return [(weights, header, data_start)]
    if not weights.is_dir():
        message = f"weights path is missing: {weights}"
        raise FileNotFoundError(message)
    if _is_diffusers_layout(weights):
        return _shards_diffusers(weights)
    shards = sorted(weights.rglob("*.safetensors"))
    if not shards:
        message = f"no safetensors under {weights}"
        raise FileNotFoundError(message)
    return [_shard_entry(shard) for shard in shards]


def _shard_entry(path: Path) -> tuple[Path, dict[str, Any], int]:
    header, data_start = _header(path)
    return path, header, data_start


def _shards_diffusers(source: Path) -> list[tuple[Path, dict[str, Any], int]]:
    found: list[tuple[Path, dict[str, Any], int]] = []
    text_index = source / "text_encoder" / "model.safetensors.index.json"
    dit_index = source / "transformer" / "diffusion_pytorch_model.safetensors.index.json"
    vae = source / "vae" / "diffusion_pytorch_model.safetensors"
    for index_path, folder in (
        (text_index, source / "text_encoder"),
        (dit_index, source / "transformer"),
    ):
        weight_map = json.loads(index_path.read_text(encoding="utf-8"))["weight_map"]
        by_file: dict[str, list[str]] = {}
        for key, file_name in weight_map.items():
            by_file.setdefault(file_name, []).append(key)
        for file_name in sorted(by_file):
            path = folder / file_name
            header, data_start = _header(path)
            found.append((path, header, data_start))
    header, data_start = _header(vae)
    found.append((vae, header, data_start))
    return found


def _header(path: Path) -> tuple[dict[str, Any], int]:
    with path.open("rb") as handle:
        size = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(size))
    return header, 8 + size


def _read(path: Path, info: dict[str, Any], data_start: int) -> np.ndarray:
    shape = tuple(int(dim) for dim in info["shape"])
    count = 1
    for dim in shape:
        count *= dim
    start, end = info["data_offsets"]
    width = {"F32": 4, "F16": 2, "BF16": 2}[info["dtype"]]
    if int(end) - int(start) != count * width:
        message = f"{path.name} tensor byte count does not match {shape}"
        raise RuntimeError(message)
    with path.open("rb") as handle:
        handle.seek(data_start + int(start))
        raw = handle.read(count * width)
    if info["dtype"] == "BF16":
        view = np.frombuffer(raw, dtype=np.uint16, count=count)
    elif info["dtype"] == "F16":
        view = np.frombuffer(raw, dtype=np.float16, count=count)
    else:
        view = np.frombuffer(raw, dtype=np.float32, count=count)
    return as_float32(view, info["dtype"]).reshape(shape)
