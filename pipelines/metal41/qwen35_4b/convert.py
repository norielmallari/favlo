"""Download / pack the BF16 checkout into the fp8 record. Serving does not import this."""

from __future__ import annotations

import json
import os
import shutil
import struct
import urllib.error
import urllib.request
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any
from urllib.parse import quote

import numpy as np

from qwen35_4b.formats import bf16_to_f32, half_bytes, quantize_fp8
from qwen35_4b.layout import (
    BLOCK,
    FOLDER,
    HIDDEN,
    INTERMEDIATE,
    LAYERS,
    MERGER_DIM,
    PATCH_DIM,
    REPO,
    VISION_DEPTH,
    VISION_HIDDEN,
    VISION_INTERMEDIATE,
    VISION_OUT,
    VISION_POSITIONS,
    VOCAB,
    is_full,
)

_HOST = "https://huggingface.co"
_CHUNK = 1024 * 1024
_AGENT = "favlo"
Progress = Callable[[int, int, str], None]

_SIDECARS = (
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "processor_config.json",
)


def checkout_dir(models: Path) -> Path:
    return models / "Qwen3.5-4B-bf16"


def record_dir(models: Path) -> Path:
    return models / FOLDER


def quantize(src: Path, dest: Path) -> Path:
    """Pack language and vision tensors from ``src`` into ``dest/weights.bin``."""
    src = Path(src)
    dest = Path(dest)
    manifest_path = dest / "manifest.json"
    if manifest_path.is_file():
        try:
            current = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            current = {}
        stored = current.get("tensors")
        expected = {logical for _raw, logical, _kind, _shape in _order()}
        if (
            current.get("complete") is True
            and isinstance(stored, dict)
            and set(stored) == expected
            and (dest / "weights.bin").is_file()
        ):
            return dest
        manifest_path.unlink(missing_ok=True)
        (dest / "weights.bin").unlink(missing_ok=True)
    dest.mkdir(parents=True, exist_ok=True)
    tensors, cursor = _pack(dest / "weights.bin", _sources(src))
    for name in _SIDECARS:
        source = src / name
        if source.is_file():
            shutil.copy2(source, dest / name)
    if not (dest / "chat_template.jinja").is_file() or not (dest / "tokenizer.json").is_file():
        message = f"{src} is missing the chat template or the tokenizer"
        raise RuntimeError(message)
    manifest_path.write_text(
        json.dumps(
            {
                "complete": True,
                "weight": "fp8",
                "kv": "fp8_e4m3",
                "source": str(src.name),
                "official": REPO,
                "tensors": tensors,
                "bytes": cursor,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return dest


def list_files(host: str = _HOST, repo: str = REPO) -> list[tuple[str, int]]:
    """File names and sizes published for the official checkpoint."""
    url = f"{host}/api/models/{repo}"
    with _request(url) as response:
        body = json.loads(response.read().decode())
    siblings = body.get("siblings") if isinstance(body, dict) else None
    if not isinstance(siblings, list) or not siblings:
        message = f"{repo} published no files"
        raise RuntimeError(message)
    files: list[tuple[str, int]] = []
    for item in siblings:
        if not isinstance(item, dict):
            continue
        name = item.get("rfilename")
        if not isinstance(name, str) or not name or name.startswith(".."):
            continue
        size = item.get("size")
        files.append((name, size if isinstance(size, int) and size >= 0 else 0))
    files.sort(key=lambda item: (item[1], item[0]))
    return files


def download(dest: Path, files: list[tuple[str, int]], on_progress: Progress) -> None:
    """Fetch the official checkpoint into ``dest``. A finished file is skipped."""
    dest.mkdir(parents=True, exist_ok=True)
    total = sum(size for _name, size in files)
    finished = 0
    for name, size in files:
        path = dest / name
        if path.is_symlink() or ".." in Path(name).parts:
            message = f"refusing {name}"
            raise RuntimeError(message)
        path.parent.mkdir(parents=True, exist_ok=True)
        url = f"{_HOST}/{REPO}/resolve/main/{quote(name, safe='/')}"
        have = path.stat().st_size if path.is_file() else 0
        if size and have == size:
            finished += size
            on_progress(finished, total, name)
            continue
        if have and size and have > size:
            path.unlink()
            have = 0
        _stream(url, path, have if size else 0)
        final = path.stat().st_size
        if size and final != size:
            message = f"{name} is {final} bytes, expected {size}"
            raise RuntimeError(message)
        finished += final if not size else size
        on_progress(finished, total or finished, name)
    (dest / "fetch_complete.json").write_text(
        json.dumps({"repo": REPO, "files": len(files)}) + "\n",
        encoding="utf-8",
    )


def _sources(src: Path) -> dict[str, tuple[dict[str, Any], int, Path]]:
    """Each checkpoint tensor mapped to its safetensors header and file."""
    single = src / "model.safetensors"
    if single.is_file():
        header, data_start = _file_header(single)
        return {
            name: (header, data_start, single)
            for name in header
            if name != "__metadata__"
        }
    index_path = src / "model.safetensors.index.json"
    if not index_path.is_file():
        message = f"{src} has no model.safetensors"
        raise RuntimeError(message)
    body = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = body.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        message = f"{index_path} has no weight map"
        raise RuntimeError(message)
    headers: dict[str, tuple[dict[str, Any], int]] = {}
    sources: dict[str, tuple[dict[str, Any], int, Path]] = {}
    for name, file_name in weight_map.items():
        if not isinstance(name, str) or not isinstance(file_name, str):
            message = f"{index_path} has a bad weight map entry"
            raise RuntimeError(message)
        if ".." in Path(file_name).parts:
            message = f"refusing {file_name}"
            raise RuntimeError(message)
        if file_name not in headers:
            headers[file_name] = _file_header(src / file_name)
        header, data_start = headers[file_name]
        sources[name] = (header, data_start, src / file_name)
    return sources


def _file_header(path: Path) -> tuple[dict[str, Any], int]:
    with path.open("rb") as handle:
        header_len = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(header_len))
    return header, 8 + header_len


def _workers() -> int:
    count = os.process_cpu_count() or os.cpu_count() or 1
    return max(1, count)


def _order() -> list[tuple[str, str, str, tuple[int, ...]]]:
    """(checkpoint name, logical name, kind, shape)."""
    rows: list[tuple[str, str, str, tuple[int, ...]]] = [
        ("model.language_model.embed_tokens.weight", "embedding", "fp8", (VOCAB, HIDDEN)),
    ]
    for index in range(LAYERS):
        stem = f"model.language_model.layers.{index}."
        if is_full(index):
            rows.extend(
                (
                    (stem + "self_attn.q_proj.weight", f"layer.{index}.q", "fp8", (8192, HIDDEN)),
                    (stem + "self_attn.k_proj.weight", f"layer.{index}.k", "fp8", (1024, HIDDEN)),
                    (stem + "self_attn.v_proj.weight", f"layer.{index}.v", "fp8", (1024, HIDDEN)),
                    (stem + "self_attn.o_proj.weight", f"layer.{index}.o", "fp8", (HIDDEN, 4096)),
                    (stem + "self_attn.q_norm.weight", f"layer.{index}.q_norm", "half", (256,)),
                    (stem + "self_attn.k_norm.weight", f"layer.{index}.k_norm", "half", (256,)),
                )
            )
        else:
            rows.extend(
                (
                    (
                        stem + "linear_attn.in_proj_qkv.weight",
                        f"layer.{index}.qkv",
                        "fp8",
                        (8192, HIDDEN),
                    ),
                    (
                        stem + "linear_attn.in_proj_z.weight",
                        f"layer.{index}.z",
                        "fp8",
                        (4096, HIDDEN),
                    ),
                    (
                        stem + "linear_attn.in_proj_a.weight",
                        f"layer.{index}.a",
                        "fp8",
                        (32, HIDDEN),
                    ),
                    (
                        stem + "linear_attn.in_proj_b.weight",
                        f"layer.{index}.b",
                        "fp8",
                        (32, HIDDEN),
                    ),
                    (
                        stem + "linear_attn.out_proj.weight",
                        f"layer.{index}.out",
                        "fp8",
                        (HIDDEN, 4096),
                    ),
                    (
                        stem + "linear_attn.conv1d.weight",
                        f"layer.{index}.conv",
                        "half",
                        (8192, 1, 4),
                    ),
                    (stem + "linear_attn.A_log", f"layer.{index}.A_log", "half", (32,)),
                    (stem + "linear_attn.dt_bias", f"layer.{index}.dt_bias", "half", (32,)),
                    (stem + "linear_attn.norm.weight", f"layer.{index}.gdn_norm", "half", (128,)),
                )
            )
        rows.extend(
            (
                (stem + "mlp.gate_proj.weight", f"layer.{index}.gate", "fp8", (INTERMEDIATE, HIDDEN)),
                (stem + "mlp.up_proj.weight", f"layer.{index}.up", "fp8", (INTERMEDIATE, HIDDEN)),
                (stem + "mlp.down_proj.weight", f"layer.{index}.down", "fp8", (HIDDEN, INTERMEDIATE)),
                (stem + "input_layernorm.weight", f"layer.{index}.in_norm", "half", (HIDDEN,)),
                (
                    stem + "post_attention_layernorm.weight",
                    f"layer.{index}.post_norm",
                    "half",
                    (HIDDEN,),
                ),
            )
        )
    rows.append(("model.language_model.norm.weight", "final_norm", "half", (HIDDEN,)))
    rows.extend(_vision())
    return rows


def _vision() -> list[tuple[str, str, str, tuple[int, ...]]]:
    """Vision matrices are fp8. LayerNorm weights, LayerNorm biases, and other biases stay float16."""
    root = "model.visual."
    rows: list[tuple[str, str, str, tuple[int, ...]]] = [
        (root + "patch_embed.proj.weight", "visual.patch", "fp8", (VISION_HIDDEN, PATCH_DIM)),
        (root + "patch_embed.proj.bias", "visual.patch_bias", "half", (VISION_HIDDEN,)),
        (root + "pos_embed.weight", "visual.pos", "fp8", (VISION_POSITIONS, VISION_HIDDEN)),
    ]
    for index in range(VISION_DEPTH):
        stem = f"{root}blocks.{index}."
        name = f"visual.{index}."
        rows.extend(
            (
                (stem + "norm1.weight", name + "norm1", "half", (VISION_HIDDEN,)),
                (stem + "norm1.bias", name + "norm1_bias", "half", (VISION_HIDDEN,)),
                (
                    stem + "attn.qkv.weight",
                    name + "qkv",
                    "fp8",
                    (VISION_HIDDEN * 3, VISION_HIDDEN),
                ),
                (stem + "attn.qkv.bias", name + "qkv_bias", "half", (VISION_HIDDEN * 3,)),
                (
                    stem + "attn.proj.weight",
                    name + "proj",
                    "fp8",
                    (VISION_HIDDEN, VISION_HIDDEN),
                ),
                (stem + "attn.proj.bias", name + "proj_bias", "half", (VISION_HIDDEN,)),
                (stem + "norm2.weight", name + "norm2", "half", (VISION_HIDDEN,)),
                (stem + "norm2.bias", name + "norm2_bias", "half", (VISION_HIDDEN,)),
                (
                    stem + "mlp.linear_fc1.weight",
                    name + "fc1",
                    "fp8",
                    (VISION_INTERMEDIATE, VISION_HIDDEN),
                ),
                (stem + "mlp.linear_fc1.bias", name + "fc1_bias", "half", (VISION_INTERMEDIATE,)),
                (
                    stem + "mlp.linear_fc2.weight",
                    name + "fc2",
                    "fp8",
                    (VISION_HIDDEN, VISION_INTERMEDIATE),
                ),
                (stem + "mlp.linear_fc2.bias", name + "fc2_bias", "half", (VISION_HIDDEN,)),
            )
        )
    rows.extend(
        (
            (root + "merger.norm.weight", "visual.merger_norm", "half", (VISION_HIDDEN,)),
            (root + "merger.norm.bias", "visual.merger_norm_bias", "half", (VISION_HIDDEN,)),
            (
                root + "merger.linear_fc1.weight",
                "visual.merger_fc1",
                "fp8",
                (MERGER_DIM, MERGER_DIM),
            ),
            (root + "merger.linear_fc1.bias", "visual.merger_fc1_bias", "half", (MERGER_DIM,)),
            (
                root + "merger.linear_fc2.weight",
                "visual.merger_fc2",
                "fp8",
                (VISION_OUT, MERGER_DIM),
            ),
            (root + "merger.linear_fc2.bias", "visual.merger_fc2_bias", "half", (VISION_OUT,)),
        )
    )
    return rows


def _placements() -> list[tuple[str, str, str, tuple[int, ...], int, int, int]]:
    placed: list[tuple[str, str, str, tuple[int, ...], int, int, int]] = []
    cursor = 0
    for raw, logical, kind, shape in _order():
        if kind == "fp8":
            rows, inner = shape
            data_bytes = rows * inner
            scale_bytes = rows * (inner // BLOCK)
        else:
            count = 1
            for dim in shape:
                count *= dim
            data_bytes = count * 2
            scale_bytes = 0
        placed.append((raw, logical, kind, shape, cursor, data_bytes, scale_bytes))
        cursor += data_bytes + scale_bytes
    return placed


def _pack(
    path: Path, sources: dict[str, tuple[dict[str, Any], int, Path]]
) -> tuple[dict[str, Any], int]:
    expected = {raw for raw, *_ in _order()}
    missing = sorted(expected - set(sources))
    if missing:
        message = f"missing tensors: {', '.join(missing[:8])}"
        raise RuntimeError(message)
    placed = _placements()
    total = placed[-1][4] + placed[-1][5] + placed[-1][6]
    fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_TRUNC, 0o644)
    try:
        os.ftruncate(fd, total)
        tasks = [
            (
                fd,
                sources[raw][2],
                sources[raw][0],
                sources[raw][1],
                raw,
                logical,
                kind,
                shape,
                cursor,
                data_bytes,
                scale_bytes,
            )
            for raw, logical, kind, shape, cursor, data_bytes, scale_bytes in placed
        ]
        with ThreadPoolExecutor(max_workers=_workers()) as pool:
            packed = list(pool.map(_pack_tensor, tasks))
    finally:
        os.close(fd)
    return dict(packed), total


def _pack_tensor(task: tuple[Any, ...]) -> tuple[str, dict[str, Any]]:
    (
        fd,
        file_path,
        header,
        data_start,
        raw,
        logical,
        kind,
        shape,
        cursor,
        data_bytes,
        scale_bytes,
    ) = task
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    array = _load(file_path, header, data_start, raw)
    if tuple(array.shape) != tuple(shape):
        if array.size == _numel(shape):
            array = array.reshape(shape)
        else:
            message = f"{raw} has shape {array.shape}, expected {shape}"
            raise RuntimeError(message)
    if kind == "fp8":
        rows, inner = shape
        data, scale = quantize_fp8(array.reshape(rows, inner))
        if len(data) != data_bytes or len(scale) != scale_bytes:
            message = f"{raw} packed {len(data)} + {len(scale)} bytes"
            raise RuntimeError(message)
        os.pwrite(fd, data, cursor)
        os.pwrite(fd, scale, cursor + data_bytes)
        spec = {
            "kind": "fp8",
            "shape": list(shape),
            "data": cursor,
            "data_bytes": data_bytes,
            "scale": cursor + data_bytes,
            "scale_bytes": scale_bytes,
        }
    else:
        data = half_bytes(array)
        if len(data) != data_bytes:
            message = f"{raw} packed {len(data)} bytes"
            raise RuntimeError(message)
        os.pwrite(fd, data, cursor)
        spec = {
            "kind": "half",
            "shape": list(shape),
            "data": cursor,
            "data_bytes": data_bytes,
            "scale": 0,
            "scale_bytes": 0,
        }
    return logical, spec


def _numel(shape: tuple[int, ...]) -> int:
    count = 1
    for dim in shape:
        count *= dim
    return count


def _load(path: Path, header: dict[str, Any], data_start: int, name: str) -> np.ndarray:
    info = header[name]
    start, end = info["data_offsets"]
    with path.open("rb") as handle:
        handle.seek(data_start + int(start))
        raw = handle.read(int(end) - int(start))
    dtype = info.get("dtype")
    shape = tuple(info["shape"])
    if dtype == "BF16":
        values = np.frombuffer(raw, dtype=np.uint16)
        return bf16_to_f32(values).reshape(shape)
    if dtype == "F16":
        return np.frombuffer(raw, dtype=np.float16).astype(np.float32).reshape(shape)
    if dtype == "F32":
        return np.frombuffer(raw, dtype=np.float32).reshape(shape)
    message = f"{name} has unsupported dtype {dtype}"
    raise RuntimeError(message)


def _stream(url: str, path: Path, have: int) -> None:
    headers = {"Range": f"bytes={have}-"} if have else None
    try:
        response = _request(url, headers)
    except urllib.error.HTTPError as exc:
        message = f"download failed: {exc.code} {url}"
        raise RuntimeError(message) from exc
    with response:
        status = getattr(response, "status", None) or response.getcode()
        mode = "ab" if status == 206 else "wb"
        with path.open(mode) as handle:
            while True:
                chunk = response.read(_CHUNK)
                if not chunk:
                    break
                handle.write(chunk)


def _request(url: str, headers: dict[str, str] | None = None) -> Any:
    merged = {"User-Agent": _AGENT}
    if headers:
        merged.update(headers)
    request = urllib.request.Request(url, headers=merged)
    try:
        return urllib.request.urlopen(request, timeout=120)
    except urllib.error.URLError as exc:
        if isinstance(exc, urllib.error.HTTPError):
            raise
        message = f"download failed: {exc.reason}"
        raise RuntimeError(message) from exc
