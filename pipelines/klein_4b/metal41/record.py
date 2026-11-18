"""Pack the diffusers snapshot as symmetric int8."""

from __future__ import annotations

import json
import shutil
import struct
from pathlib import Path

import numpy as np

from .constants import FORMAT, REPO, TEXT_LAYERS, WEIGHTS
from .formats import pack_matrix, pad_group
from .source import tokenizer_file

_FLOATS = {"BF16", "F16", "F32"}


def quantize(origin: Path, dest: Path) -> None:
    """Write ``dest`` from the snapshot. A finished record of this format is left as it is."""
    root = Path(origin)
    out = Path(dest)
    if _finished(out):
        print(f"quantize already complete {out.name}", flush=True)
        return
    _require(root)
    out.mkdir(parents=True, exist_ok=True)
    matrices: dict[str, dict[str, object]] = {}
    vectors: dict[str, dict[str, object]] = {}
    stage = {"text": 0, "dit": 0, "vae": 0}
    files = (
        ("text/", root / "text_encoder" / "model-00001-of-00002.safetensors", True),
        ("text/", root / "text_encoder" / "model-00002-of-00002.safetensors", True),
        ("dit/", root / "transformer" / "diffusion_pytorch_model.safetensors", False),
        ("vae/", root / "vae" / "diffusion_pytorch_model.safetensors", False),
    )
    seen: set[str] = set()
    for prefix, path, text in files:
        _pack_file(path, out, prefix, text, matrices, vectors, stage, seen)
    _attach_bias(matrices, vectors)
    tok_dir = out / "tokenizer"
    tok_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(tokenizer_file(root), tok_dir / "tokenizer.json")
    (out / "index.json").write_text(
        json.dumps({"matrices": matrices, "vectors": vectors}, indent=1),
        encoding="utf-8",
    )
    nbytes = sum(path.stat().st_size for path in out.rglob("*") if path.is_file())
    manifest = {
        "format": FORMAT,
        "complete": True,
        "nbytes": nbytes,
        "repo": REPO,
        "stage_bytes": stage,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print(
        f"quantize {out.name} text {stage['text']} dit {stage['dit']} vae {stage['vae']}",
        flush=True,
    )


def _finished(path: Path) -> bool:
    manifest = path / "manifest.json"
    if not manifest.is_file():
        return False
    body = json.loads(manifest.read_text(encoding="utf-8"))
    return body.get("format") == FORMAT and body.get("complete") is True


def _require(root: Path) -> None:
    short: list[str] = []
    for name, size in WEIGHTS.items():
        path = root / name
        have = path.stat().st_size if path.is_file() else 0
        if have != size:
            short.append(f"{name} {have}/{size}")
    if short:
        message = "checkpoint is incomplete\n" + "\n".join(short)
        raise FileNotFoundError(message)


def _pack_file(
    path: Path,
    dest: Path,
    prefix: str,
    text: bool,
    matrices: dict[str, dict[str, object]],
    vectors: dict[str, dict[str, object]],
    stage: dict[str, int],
    seen: set[str],
) -> None:
    data_at, meta = _header(path)
    blob = np.memmap(path, dtype=np.uint8, mode="r")
    names = [key for key, spec in meta.items() if isinstance(spec, dict) and "shape" in spec]
    for index, key in enumerate(names):
        if text and _skip_text(key):
            continue
        full = prefix + key
        if full in seen:
            message = f"duplicate tensor {full}"
            raise RuntimeError(message)
        seen.add(full)
        spec = meta[key]
        dtype = str(spec.get("dtype", ""))
        shape = [int(dim) for dim in spec["shape"]]
        if dtype not in _FLOATS or not shape:
            continue
        start, end = (int(item) for item in spec["data_offsets"])
        if len(shape) == 1:
            _write_vector(blob, data_at, start, end, shape, dtype, dest, full, vectors, stage, prefix)
        elif len(shape) >= 2:
            _write_matrix(blob, data_at, start, shape, dtype, dest, full, matrices, stage, prefix)
        if index % 40 == 0:
            print(f"quantize {prefix}{index}/{len(names)}", flush=True)
    print(f"quantize {prefix} done", flush=True)


def _skip_text(key: str) -> bool:
    if key == "model.norm.weight" or key.startswith("lm_head"):
        return True
    if not key.startswith("model.layers."):
        return False
    return int(key.split(".")[2]) >= TEXT_LAYERS


def _write_vector(
    blob: np.ndarray,
    data_at: int,
    start: int,
    end: int,
    shape: list[int],
    dtype: str,
    dest: Path,
    key: str,
    vectors: dict[str, dict[str, object]],
    stage: dict[str, int],
    prefix: str,
) -> None:
    count = int(shape[0])
    values = _widen(blob[data_at + start : data_at + end], dtype, count)
    wide = "running_mean" in key or "running_var" in key
    stored = values.astype(np.float32 if wide else np.float16)
    rel = _rel(key, "f32" if wide else "f16")
    path = dest / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    stored.tofile(path)
    vectors[key] = {"file": rel, "dtype": "f32" if wide else "f16", "n": count}
    stage[prefix.strip("/")] += path.stat().st_size


def _write_matrix(
    blob: np.ndarray,
    data_at: int,
    start: int,
    shape: list[int],
    dtype: str,
    dest: Path,
    key: str,
    matrices: dict[str, dict[str, object]],
    stage: dict[str, int],
    prefix: str,
) -> None:
    rows = int(shape[0])
    cols = int(np.prod(shape[1:], dtype=np.int64))
    stored_n = pad_group(rows)
    stored_k = pad_group(cols)
    codes_rel = _rel(key, "i8")
    scales_rel = _rel(key, "f32")
    codes_path = dest / codes_rel
    scales_path = dest / scales_rel
    codes_path.parent.mkdir(parents=True, exist_ok=True)
    codes = np.memmap(codes_path, dtype=np.int8, mode="w+", shape=(stored_n, stored_k))
    scales = np.memmap(scales_path, dtype=np.float32, mode="w+", shape=(stored_n,))
    codes[:] = 0
    scales[:] = 0
    item = 2 if dtype == "BF16" or dtype == "F16" else 4
    for row0 in range(0, rows, 1024):
        row1 = min(rows, row0 + 1024)
        byte0 = data_at + start + row0 * cols * item
        byte1 = data_at + start + row1 * cols * item
        block = _widen(blob[byte0:byte1], dtype, (row1 - row0) * cols).reshape(row1 - row0, cols)
        packed, row_scales = pack_matrix(block)
        codes[row0:row1] = packed[: row1 - row0]
        scales[row0:row1] = row_scales[: row1 - row0]
    codes.flush()
    scales.flush()
    record: dict[str, object] = {
        "codes": codes_rel,
        "scales": scales_rel,
        "n": rows,
        "k": cols,
        "stored_n": stored_n,
        "stored_k": stored_k,
    }
    if len(shape) == 4:
        record["kh"] = int(shape[2])
        record["kw"] = int(shape[3])
    matrices[key] = record
    stage[prefix.strip("/")] += codes_path.stat().st_size + scales_path.stat().st_size


def _attach_bias(matrices: dict[str, dict[str, object]], vectors: dict[str, dict[str, object]]) -> None:
    for key, spec in matrices.items():
        if not key.endswith(".weight"):
            continue
        bias_key = key[: -len(".weight")] + ".bias"
        found = vectors.get(bias_key)
        if found is None:
            continue
        spec["bias"] = found["file"]


def _rel(key: str, suffix: str) -> str:
    return f"weights/{key}.{suffix}"


def _header(path: Path) -> tuple[int, dict[str, object]]:
    with path.open("rb") as handle:
        size = struct.unpack("<Q", handle.read(8))[0]
        payload = handle.read(size)
    if len(payload) != size:
        message = f"{path.name} header is short"
        raise RuntimeError(message)
    meta = json.loads(payload)
    if not isinstance(meta, dict):
        message = f"{path.name} header is not an object"
        raise RuntimeError(message)
    return 8 + size, meta


def _widen(raw: np.ndarray, dtype: str, count: int) -> np.ndarray:
    blob = np.ascontiguousarray(raw)
    if dtype == "BF16":
        units = np.frombuffer(blob, dtype=np.uint16, count=count).astype(np.uint32)
        return (units << 16).view(np.float32).copy()
    if dtype == "F16":
        return np.frombuffer(blob, dtype=np.float16, count=count).astype(np.float32)
    return np.frombuffer(blob, dtype=np.float32, count=count).copy()
