"""Pack the Turbo checkpoint and load the fp4 record onto the device."""

from __future__ import annotations

import json
import shutil
import struct
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


def quantize(source: Path, dest: Path) -> Path:
    """Pack generate-path weights. The VAE encoder is left unpacked."""
    manifest_path = dest / "manifest.json"
    if manifest_path.is_file():
        body = json.loads(manifest_path.read_text(encoding="utf-8"))
        if body.get("complete") is True:
            return dest
    if dest.exists():
        shutil.rmtree(dest)
    weights = dest / "weights"
    weights.mkdir(parents=True)
    tensors: dict[str, Any] = {}
    for shard, header, data_start in _shards(source):
        print(f"quantize {shard.name}", flush=True)
        for key, info in header.items():
            if key == "__metadata__" or key.startswith("encoder."):
                continue
            array = _read(shard, info, data_start)
            kind = _kind(key, array.shape)
            entry: dict[str, Any] = {
                "plane": kind,
                "shape": list(array.shape),
            }
            if kind == "f16":
                raw = np.ascontiguousarray(array, dtype=np.float32).astype(np.float16)
                file_name = f"{key}.f16"
                (weights / file_name).write_bytes(raw.tobytes())
                entry["file"] = f"weights/{file_name}"
            else:
                matrix = array.reshape(array.shape[0], -1)
                packed, scales, stored = pack_fp4(matrix)
                data_name = f"{key}.fp4"
                scale_name = f"{key}.scale"
                (weights / data_name).write_bytes(np.ascontiguousarray(packed).tobytes())
                (weights / scale_name).write_bytes(np.ascontiguousarray(scales).tobytes())
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
        "source": str(source.resolve()),
        "tile_m": 64,
        "tile_n": 32,
        "tensors": tensors,
    }
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(body), encoding="utf-8")
    temporary.replace(manifest_path)
    return dest


def load(path: Path, gpu: Gpu) -> dict[str, Plane]:
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("complete") is not True:
        message = f"{path} is not a finished record"
        raise RuntimeError(message)
    planes: dict[str, Plane] = {}
    items = manifest["tensors"]
    for index, (name, entry) in enumerate(items.items()):
        raw = (path / entry["file"]).read_bytes()
        scale = None
        stored = int(entry.get("stored_k") or 0)
        logical = int(entry.get("logical_k") or 0)
        if entry["plane"] == "fp4":
            data = gpu.keep(gpu.upload(np.frombuffer(raw, dtype=np.uint8)))
            scale = gpu.keep(
                gpu.upload(np.frombuffer((path / entry["scale"]).read_bytes(), dtype=np.uint8))
            )
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
        if index % 100 == 0:
            print(f"load {index}/{len(items)}", flush=True)
    print(f"load {len(items)} tensors", flush=True)
    return planes


def _kind(key: str, shape: tuple[int, ...]) -> str:
    if key in _PAD or len(shape) == 1:
        return "f16"
    if len(shape) in {2, 4}:
        return "fp4"
    message = f"{key} has shape {shape}"
    raise ValueError(message)


def _shards(source: Path) -> list[tuple[Path, dict[str, Any], int]]:
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
