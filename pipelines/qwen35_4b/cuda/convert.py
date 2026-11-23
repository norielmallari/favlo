"""Download the official BF16 checkpoint and pack the CUDA integer plane."""

from __future__ import annotations

import http.client
import importlib.util
import json
import os
import shutil
import sys
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any
from urllib.parse import quote

import numpy as np

from favlo.quantize import half_bytes, parallel_map, quantize_int4, quantize_int8, to_f32, write_at

_HERE = Path(__file__).resolve().parent
_layout_spec = importlib.util.spec_from_file_location(
    "qwen35_cuda_layout_convert",
    _HERE / "layout.py",
)
if _layout_spec is None or _layout_spec.loader is None:
    message = f"cannot load {_HERE / 'layout.py'}"
    raise ImportError(message)
_layout = importlib.util.module_from_spec(_layout_spec)
sys.modules[_layout_spec.name] = _layout
_layout_spec.loader.exec_module(_layout)
CHECKOUT = _layout.CHECKOUT
FOLDER = _layout.FOLDER
REPO = _layout.REPO

_order_spec = importlib.util.spec_from_file_location(
    "qwen35_cuda_order_convert",
    _HERE / "order.py",
)
if _order_spec is None or _order_spec.loader is None:
    message = f"cannot load {_HERE / 'order.py'}"
    raise ImportError(message)
_order = importlib.util.module_from_spec(_order_spec)
sys.modules[_order_spec.name] = _order
_order_spec.loader.exec_module(_order)
tensors = _order.tensors

_HOST = "https://huggingface.co"
_CHUNK = 1024 * 1024
_SPAN = 64 * 1024 * 1024
_AGENT = "favlo"
Progress = Callable[[int, int, str], None]

SIDECARS = (
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "processor_config.json",
)


def checkout_dir(models: Path) -> Path:
    return models / CHECKOUT


def record_dir(models: Path, plane: str) -> Path:
    folder = FOLDER.get(plane)
    if folder is None:
        message = f"weight plane {plane} is not int8 or int4"
        raise ValueError(message)
    return models / folder


def quantize(
    src: Path,
    dest: Path,
    *,
    plane: str,
    vision: bool,
    context: int,
    on_progress: Progress | None = None,
) -> Path:
    """Pack ``src`` into ``dest/weights.bin``. An already complete record is kept."""
    src = Path(src)
    dest = Path(dest)
    wanted = tensors(vision=vision)
    manifest_path = dest / "manifest.json"
    if _complete(manifest_path, dest, wanted, plane):
        return dest
    manifest_path.unlink(missing_ok=True)
    (dest / "weights.bin").unlink(missing_ok=True)
    dest.mkdir(parents=True, exist_ok=True)
    sources = _sources(src)
    stored, cursor = _pack(dest / "weights.bin", sources, wanted, plane, on_progress)
    for name in SIDECARS:
        origin = src / name
        if origin.is_file():
            shutil.copy2(origin, dest / name)
    if not (dest / "tokenizer.json").is_file():
        message = f"{src} is missing the tokenizer"
        raise RuntimeError(message)
    manifest_path.write_text(
        json.dumps(
            {
                "complete": True,
                "weight": plane,
                "kv": "int8",
                "vision": vision,
                "context": context,
                "source": src.name,
                "official": REPO,
                "tensors": stored,
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
    tree = f"{host}/api/models/{repo}/tree/main"
    with _request(tree) as response:
        body = json.loads(response.read().decode())
    if not isinstance(body, list) or not body:
        message = f"{repo} published no files"
        raise RuntimeError(message)
    files: list[tuple[str, int]] = []
    for item in body:
        if not isinstance(item, dict) or item.get("type") != "file":
            continue
        name = item.get("path")
        if not isinstance(name, str) or not name or name.startswith(".."):
            continue
        size = item.get("size")
        files.append((name, size if isinstance(size, int) and size >= 0 else 0))
    if not files:
        message = f"{repo} published no files"
        raise RuntimeError(message)
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

        def _note(got: int, base: int = finished, label: str = name) -> None:
            on_progress(base + got, total or base + got, label)

        _fetch(url, path, size, _note)
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


def _complete(
    path: Path,
    dest: Path,
    wanted: list[tuple[str, str, str, tuple[int, ...]]],
    plane: str,
) -> bool:
    if not path.is_file() or not (dest / "weights.bin").is_file():
        return False
    try:
        current = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    stored = current.get("tensors")
    if current.get("complete") is not True or current.get("weight") != plane:
        return False
    if not isinstance(stored, dict):
        return False
    return {logical for _raw, logical, _kind, _shape in wanted} <= set(stored)


def _sources(src: Path) -> dict[str, tuple[dict[str, Any], int, Path]]:
    single = src / "model.safetensors"
    if single.is_file():
        header, data_start = _file_header(single)
        return {name: (header, data_start, single) for name in header if name != "__metadata__"}
    index_path = src / "model.safetensors.index.json"
    if not index_path.is_file():
        message = f"{src} has no model.safetensors"
        raise FileNotFoundError(message)
    index = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        message = f"{index_path} has no weight map"
        raise RuntimeError(message)
    headers: dict[str, tuple[dict[str, Any], int]] = {}
    found: dict[str, tuple[dict[str, Any], int, Path]] = {}
    for name, file_name in weight_map.items():
        if not isinstance(name, str) or not isinstance(file_name, str):
            continue
        path = src / file_name
        if file_name not in headers:
            headers[file_name] = _file_header(path)
        header, data_start = headers[file_name]
        found[name] = (header, data_start, path)
    return found


def _file_header(path: Path) -> tuple[dict[str, Any], int]:
    with path.open("rb") as handle:
        raw_size = handle.read(8)
        if len(raw_size) != 8:
            message = f"{path} is not a safetensors file"
            raise RuntimeError(message)
        size = int.from_bytes(raw_size, "little")
        header = json.loads(handle.read(size))
    if not isinstance(header, dict):
        message = f"{path} header is not an object"
        raise RuntimeError(message)
    return header, 8 + size


def _pack(
    path: Path,
    sources: dict[str, tuple[dict[str, Any], int, Path]],
    wanted: list[tuple[str, str, str, tuple[int, ...]]],
    plane: str,
    on_progress: Progress | None,
) -> tuple[dict[str, dict[str, Any]], int]:
    missing = [raw for raw, _logical, _kind, _shape in wanted if raw not in sources]
    if missing:
        message = f"missing tensors: {', '.join(missing[:8])}"
        raise RuntimeError(message)
    placed: list[tuple[str, str, str, tuple[int, ...], int, int]] = []
    cursor = 0
    for raw, logical, kind, shape in wanted:
        nbytes = _packed_bytes(kind, shape, plane)
        placed.append((raw, logical, kind, shape, cursor, nbytes))
        cursor += nbytes
    fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_TRUNC, 0o644)
    done = 0
    lock = threading.Lock()
    whole = len(wanted)

    def pack_one(task: tuple[Any, ...]) -> tuple[str, dict[str, Any]]:
        nonlocal done
        logical, body = _pack_tensor(task)
        if on_progress is not None:
            with lock:
                done += 1
                current = done
            on_progress(current, whole, logical)
        return logical, body

    try:
        os.ftruncate(fd, cursor)
        tasks = [
            (fd, sources[raw], raw, logical, kind, shape, offset, nbytes, plane)
            for raw, logical, kind, shape, offset, nbytes in placed
        ]
        packed = parallel_map(pack_one, tasks)
    finally:
        os.close(fd)
    return dict(packed), cursor


def _packed_bytes(kind: str, shape: tuple[int, ...], plane: str) -> int:
    if kind == "half":
        count = 1
        for dim in shape:
            count *= dim
        return count * 2
    rows, inner = shape
    scales = rows * 4
    if plane == "int8":
        return rows * inner + scales
    if plane == "int4":
        return rows * (inner // 2) + scales
    message = f"weight plane {plane} is not int8 or int4"
    raise ValueError(message)


def _pack_tensor(task: tuple[Any, ...]) -> tuple[str, dict[str, Any]]:
    fd, source, raw, logical, kind, shape, offset, nbytes, plane = task
    header, data_start, origin = source
    values = _load(origin, header, data_start, raw).reshape(shape)
    if kind == "int":
        codes, scale = quantize_int8(values) if plane == "int8" else quantize_int4(values)
        blob = np.ascontiguousarray(codes).tobytes() + np.ascontiguousarray(scale).tobytes()
    elif kind == "half":
        blob = half_bytes(values)
    else:
        message = f"{logical} kind {kind} is not int or half"
        raise RuntimeError(message)
    if len(blob) != nbytes:
        message = f"{raw} packed {len(blob)} bytes, expected {nbytes}"
        raise RuntimeError(message)
    write_at(fd, blob, offset)
    return logical, {
        "kind": plane if kind == "int" else "half",
        "shape": list(shape),
        "offset": offset,
        "bytes": nbytes,
    }


def _load(path: Path, header: dict[str, Any], data_start: int, name: str) -> np.ndarray:
    info = header[name]
    start, end = info["data_offsets"]
    with path.open("rb") as handle:
        handle.seek(data_start + int(start))
        raw = handle.read(int(end) - int(start))
    dtype = info.get("dtype")
    shape = tuple(int(dim) for dim in info["shape"])
    if dtype == "BF16":
        return to_f32(np.frombuffer(raw, dtype=np.uint16)).reshape(shape)
    if dtype == "F16":
        return np.frombuffer(raw, dtype=np.float16).astype(np.float32).reshape(shape)
    if dtype == "F32":
        return np.frombuffer(raw, dtype=np.float32).reshape(shape)
    message = f"{name} has unsupported dtype {dtype}"
    raise RuntimeError(message)


def _fetch(
    url: str,
    path: Path,
    size: int,
    on_bytes: Callable[[int], None] | None,
) -> None:
    """Download ``path``. A slow connection is reopened so the transfer keeps moving."""
    stalls = 0
    while True:
        have = path.stat().st_size if path.is_file() else 0
        if size and have == size:
            return
        if size and have > size:
            path.unlink()
            have = 0
        span = None if not size or stalls >= 2 else _SPAN
        try:
            _stream(url, path, have if size else 0, on_bytes, span)
        except (urllib.error.URLError, TimeoutError, OSError, http.client.HTTPException):
            stalls += 1
            if stalls > 8:
                message = f"download failed: {path.name}"
                raise RuntimeError(message) from None
            time.sleep(min(20, stalls * 2))
            continue
        if not size:
            return
        grown = path.stat().st_size - have
        if grown <= 0:
            stalls += 1
            if stalls > 8:
                message = f"download stalled: {path.name}"
                raise RuntimeError(message)
            time.sleep(min(20, stalls * 2))
            continue
        if grown < _CHUNK:
            time.sleep(2)
        stalls = 0


def _stream(
    url: str,
    path: Path,
    have: int,
    on_bytes: Callable[[int], None] | None = None,
    limit: int | None = None,
) -> None:
    headers = {"Range": f"bytes={have}-"} if have else None
    try:
        response = _request(url, headers)
    except urllib.error.HTTPError as exc:
        if exc.code in {429, 500, 502, 503, 504}:
            raise
        message = f"download failed: {exc.code} {url}"
        raise RuntimeError(message) from exc
    with response:
        status = getattr(response, "status", None) or response.getcode()
        mode = "ab" if status == 206 else "wb"
        written = have if mode == "ab" else 0
        taken = 0
        if on_bytes is not None:
            on_bytes(written)
        with path.open(mode) as handle:
            while True:
                started = time.perf_counter()
                chunk = response.read(_CHUNK)
                if not chunk:
                    break
                handle.write(chunk)
                written += len(chunk)
                taken += len(chunk)
                if on_bytes is not None:
                    on_bytes(written)
                if time.perf_counter() - started > 8 or (limit is not None and taken >= limit):
                    break
            handle.flush()
            os.fsync(handle.fileno())


def _request(url: str, headers: dict[str, str] | None = None) -> Any:
    merged = {"User-Agent": _AGENT}
    if headers:
        merged.update(headers)
    return urllib.request.urlopen(urllib.request.Request(url, headers=merged), timeout=120)
