"""Download the BF16 checkout and pack the fp4 record. Serving does not import this module."""

from __future__ import annotations

import json
import shutil
import struct
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any
from urllib.parse import quote

import numpy as np

from spark_x26_1_7b.formats import bf16_to_f32, f16_bytes, quantize_fp4
from spark_x26_1_7b.layout import (
    FOLDER,
    HEADS,
    HIDDEN,
    INTERMEDIATE,
    KV_DIM,
    LAYERS,
    Q_DIM,
    REPO,
    VOCAB,
    record_bytes,
)

_HOST = "https://huggingface.co"
_CHUNK = 1024 * 1024
_AGENT = "favlo"
_SIDECARS = (
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "special_tokens_map.json",
)
_QKV = Q_DIM + 2 * KV_DIM

Progress = Callable[[int, int, str], None]


def checkout_dir(models: Path) -> Path:
    return models / "Spark-X2.6-1.7B"


def record_dir(models: Path) -> Path:
    return models / FOLDER


def list_files(host: str = _HOST, repo: str = REPO) -> list[tuple[str, int]]:
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


def quantize(src: Path, dest: Path) -> Path:
    """Pack ``src`` into ``dest``. A finished manifest is left in place."""
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
        manifest_path.unlink()
        (dest / "weights.bin").unlink(missing_ok=True)
    dest.mkdir(parents=True, exist_ok=True)
    index_path = src / "model.safetensors.index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        message = f"{index_path} has no weight map"
        raise RuntimeError(message)
    tensors, cursor = _pack(dest / "weights.bin", src, weight_map)
    if cursor != record_bytes():
        message = f"packed {cursor} bytes, expected {record_bytes()}"
        raise RuntimeError(message)
    for name in _SIDECARS:
        source = src / name
        if source.is_file():
            shutil.copy2(source, dest / name)
    if not (dest / "chat_template.jinja").is_file() or not (dest / "tokenizer.json").is_file():
        message = f"{src} is missing the chat template or the tokenizer"
        raise RuntimeError(message)
    manifest_path.write_text(
        json.dumps(
            {"complete": True, "weight": "fp4", "kv": "e4m3", "source": REPO, "tensors": tensors},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return dest


def _pack(path: Path, src: Path, weight_map: dict[str, Any]) -> tuple[dict[str, Any], int]:
    expected = _expected_names()
    unknown = sorted(set(weight_map) - expected - {"lm_head.weight"})
    if unknown:
        message = f"unexpected tensors: {', '.join(unknown)}"
        raise RuntimeError(message)
    missing = sorted(expected - set(weight_map))
    if missing:
        message = f"missing tensors: {', '.join(missing)}"
        raise RuntimeError(message)
    tensors: dict[str, Any] = {}
    cursor = 0
    shards: dict[str, tuple[dict[str, Any], int]] = {}
    with path.open("wb") as out:
        for raw, logical, kind, rows, inner in _order():
            array = _load(src, weight_map, shards, raw)
            if array.shape != (rows, inner) and not (kind == "f16" and array.shape == (inner,)):
                message = f"{raw} has shape {array.shape}"
                raise RuntimeError(message)
            if kind == "fp4":
                data, scale = quantize_fp4(array.reshape(rows, inner))
                spec = {
                    "kind": "fp4",
                    "rows": rows,
                    "inner": inner,
                    "data": cursor,
                    "data_bytes": len(data),
                    "scale": cursor + len(data),
                    "scale_bytes": len(scale),
                }
                out.write(data)
                out.write(scale)
                cursor += len(data) + len(scale)
            else:
                data = f16_bytes(array.reshape(rows, inner))
                spec = {
                    "kind": "f16",
                    "rows": rows,
                    "inner": inner,
                    "data": cursor,
                    "data_bytes": len(data),
                    "scale": 0,
                    "scale_bytes": 0,
                }
                out.write(data)
                cursor += len(data)
            tensors[logical] = spec
    return tensors, cursor


def _expected_names() -> set[str]:
    return {raw for raw, *_rest in _order()}


def _order() -> list[tuple[str, str, str, int, int]]:
    rows: list[tuple[str, str, str, int, int]] = [
        ("model.embedding.weight", "embedding", "fp4", VOCAB, HIDDEN),
    ]
    for index in range(LAYERS):
        stem = f"model.layers.{index}."
        rows.extend(
            (
                (stem + "self_attn.q_k_v_proj.weight", f"layer.{index}.qkv", "fp4", _QKV, HIDDEN),
                (stem + "self_attn.out_proj.weight", f"layer.{index}.out", "fp4", HIDDEN, HIDDEN),
                (stem + "mlp.gate_proj.weight", f"layer.{index}.gate", "fp4", INTERMEDIATE, HIDDEN),
                (stem + "mlp.up_proj.weight", f"layer.{index}.up", "fp4", INTERMEDIATE, HIDDEN),
                (stem + "mlp.down_proj.weight", f"layer.{index}.down", "fp4", HIDDEN, INTERMEDIATE),
                (stem + "self_attn.g_proj.weight", f"layer.{index}.g", "f16", HEADS, HIDDEN),
                (stem + "input_layernorm.weight", f"layer.{index}.in_norm", "f16", 1, HIDDEN),
                (stem + "post_attention_layernorm.weight", f"layer.{index}.post_norm", "f16", 1, HIDDEN),
            )
        )
    rows.append(("model.norm.weight", "final_norm", "f16", 1, HIDDEN))
    return rows


def _load(
    src: Path,
    weight_map: dict[str, Any],
    shards: dict[str, tuple[dict[str, Any], int]],
    name: str,
) -> np.ndarray:
    file_name = weight_map[name]
    if not isinstance(file_name, str):
        message = f"{name} has no shard"
        raise RuntimeError(message)
    header, data_start = shards.get(file_name, (None, 0))
    shard = src / file_name
    if header is None:
        with shard.open("rb") as handle:
            header_len = struct.unpack("<Q", handle.read(8))[0]
            header = json.loads(handle.read(header_len))
        data_start = 8 + header_len
        shards[file_name] = (header, data_start)
    info = header[name]
    start, end = info["data_offsets"]
    with shard.open("rb") as handle:
        handle.seek(data_start + int(start))
        raw = handle.read(int(end) - int(start))
    values = np.frombuffer(raw, dtype=np.uint16)
    return bf16_to_f32(values).reshape(tuple(info["shape"]))


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
