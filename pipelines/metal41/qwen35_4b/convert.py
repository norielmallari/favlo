"""Download / pack the BF16 checkout into the fp8 record. Serving does not import this."""

from __future__ import annotations

import json
import os
import shutil
import struct
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from qwen35_4b.formats import bf16_to_f32, f16_bytes, quantize_fp8
from qwen35_4b.layout import (
    BLOCK,
    FOLDER,
    HIDDEN,
    INTERMEDIATE,
    LAYERS,
    REPO,
    VOCAB,
    is_full,
)

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
    """Pack language tensors from ``src`` into ``dest/weights.bin``."""
    src = Path(src)
    dest = Path(dest)
    manifest_path = dest / "manifest.json"
    if manifest_path.is_file():
        try:
            current = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            current = {}
        if current.get("complete") is True and (dest / "weights.bin").is_file():
            return dest
        manifest_path.unlink(missing_ok=True)
        (dest / "weights.bin").unlink(missing_ok=True)
    dest.mkdir(parents=True, exist_ok=True)
    header, data_start = _header(src)
    tensors, cursor = _pack(dest / "weights.bin", src, header, data_start)
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
                "kv": "e4m3",
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


def _header(src: Path) -> tuple[dict[str, Any], int]:
    single = src / "model.safetensors"
    index = src / "model.safetensors.index.json"
    if single.is_file():
        with single.open("rb") as handle:
            header_len = struct.unpack("<Q", handle.read(8))[0]
            header = json.loads(handle.read(header_len))
        return header, 8 + header_len
    if index.is_file():
        message = "sharded checkouts are not packed by this step yet; merge to model.safetensors"
        raise RuntimeError(message)
    message = f"{src} has no model.safetensors"
    raise RuntimeError(message)


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
                    (stem + "self_attn.q_norm.weight", f"layer.{index}.q_norm", "f16", (256,)),
                    (stem + "self_attn.k_norm.weight", f"layer.{index}.k_norm", "f16", (256,)),
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
                        "f16",
                        (8192, 1, 4),
                    ),
                    (stem + "linear_attn.A_log", f"layer.{index}.A_log", "f16", (32,)),
                    (stem + "linear_attn.dt_bias", f"layer.{index}.dt_bias", "f16", (32,)),
                    (stem + "linear_attn.norm.weight", f"layer.{index}.gdn_norm", "f16", (128,)),
                )
            )
        rows.extend(
            (
                (stem + "mlp.gate_proj.weight", f"layer.{index}.gate", "fp8", (INTERMEDIATE, HIDDEN)),
                (stem + "mlp.up_proj.weight", f"layer.{index}.up", "fp8", (INTERMEDIATE, HIDDEN)),
                (stem + "mlp.down_proj.weight", f"layer.{index}.down", "fp8", (HIDDEN, INTERMEDIATE)),
                (stem + "input_layernorm.weight", f"layer.{index}.in_norm", "f16", (HIDDEN,)),
                (
                    stem + "post_attention_layernorm.weight",
                    f"layer.{index}.post_norm",
                    "f16",
                    (HIDDEN,),
                ),
            )
        )
    rows.append(("model.language_model.norm.weight", "final_norm", "f16", (HIDDEN,)))
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
    path: Path, src: Path, header: dict[str, Any], data_start: int
) -> tuple[dict[str, Any], int]:
    expected = {raw for raw, *_ in _order()}
    names = {key for key in header if key != "__metadata__"}
    missing = sorted(expected - names)
    if missing:
        message = f"missing tensors: {', '.join(missing[:8])}"
        raise RuntimeError(message)
    placed = _placements()
    total = placed[-1][4] + placed[-1][5] + placed[-1][6]
    fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_TRUNC, 0o644)
    try:
        os.ftruncate(fd, total)
        tasks = [
            (fd, src, header, data_start, raw, logical, kind, shape, cursor, data_bytes, scale_bytes)
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
        src,
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
    array = _load(src, header, data_start, raw)
    if tuple(array.shape) != tuple(shape):
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
        data = f16_bytes(array)
        if len(data) != data_bytes:
            message = f"{raw} packed {len(data)} bytes"
            raise RuntimeError(message)
        os.pwrite(fd, data, cursor)
        spec = {
            "kind": "f16",
            "shape": list(shape),
            "data": cursor,
            "data_bytes": data_bytes,
            "scale": 0,
            "scale_bytes": 0,
        }
    return logical, spec


def _load(src: Path, header: dict[str, Any], data_start: int, name: str) -> np.ndarray:
    info = header[name]
    start, end = info["data_offsets"]
    with (src / "model.safetensors").open("rb") as handle:
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
