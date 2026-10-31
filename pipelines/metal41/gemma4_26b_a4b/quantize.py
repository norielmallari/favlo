"""Write the packed record. This module does not download a checkpoint."""

from __future__ import annotations

import json
import os
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from gemma4_26b_a4b.checkpoint import load_tensor, tensor_names, tensor_shape
from gemma4_26b_a4b.geometry import (
    CHECKPOINT_CONTEXT,
    CONTEXT,
    EXPERT_DOWN,
    EXPERT_UP,
    EXPERTS,
    FP16,
    LAYERS,
    RESIDENT,
    expected,
)
from gemma4_26b_a4b.planes import (
    Q40,
    align_k,
    dequant_nbytes,
    matrix_nbytes,
    pack_matrix,
    store_fp16,
)

RECORD_STEM = "gemma-4-26B-A4B-it"
SOURCE_DIR = "gemma-4-26B-A4B-it-qat-q4_0-unquantized"
RECORD_NAME = f"{RECORD_STEM}-q4_0"
ALIGN = 128
_SIDECARS = (
    "config.json",
    "generation_config.json",
    "chat_template.jinja",
    "tokenizer.json",
    "tokenizer_config.json",
    "processor_config.json",
)


def record_name() -> str:
    return RECORD_NAME


def estimate() -> dict[str, int]:
    """Byte totals from the checkpoint shapes."""
    resident = 0
    scratch = 0
    fp16 = 0
    per_up = per_down = 0
    for role, shape in expected().values():
        if role == FP16:
            fp16 += int(np.prod(shape)) * 2
        elif role == RESIDENT:
            resident += matrix_nbytes(int(shape[0]), int(shape[-1]), Q40)
            scratch = max(scratch, dequant_nbytes(int(shape[0]), int(shape[-1])))
        elif role == EXPERT_UP and per_up == 0:
            per_up = matrix_nbytes(int(shape[1]), int(shape[2]), Q40)
        elif role == EXPERT_DOWN and per_down == 0:
            per_down = matrix_nbytes(int(shape[1]), int(shape[2]), Q40)
    return {
        "resident": resident,
        "scratch": scratch,
        "fp16": fp16,
        "experts": LAYERS * EXPERTS * (per_up + per_down),
    }


def quantize(src: Path, dest: Path) -> None:
    manifest_path = dest / "manifest.json"
    if manifest_path.is_file():
        stored = json.loads(manifest_path.read_text(encoding="utf-8"))
        if stored.get("complete") is True:
            return
    names = tensor_names(src)
    catalog = expected()
    missing = [name for name in catalog if name not in names]
    if missing:
        message = f"checkpoint is missing {missing[0]}"
        raise FileNotFoundError(message)
    unknown = [name for name in names if name not in catalog]
    if unknown:
        message = f"checkpoint tensor {unknown[0]} is not in the Gemma 4 map"
        raise ValueError(message)
    for name, (_role, shape) in catalog.items():
        found = tensor_shape(src, names[name], name)
        if found != shape:
            message = f"{name} shape {found} != {shape}"
            raise ValueError(message)
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "experts").mkdir(exist_ok=True)
    _limit_blas()
    with ProcessPoolExecutor(max_workers=_worker_count(), initializer=_limit_blas) as pool:
        _write_experts(src, names, dest, pool)
        _write_resident(src, names, catalog, dest, pool)
    _copy_sidecars(src, dest)
    body = {
        "complete": True,
        "pipeline": "gemma4-26b-a4b",
        "source_name": src.name,
        "context_length": CONTEXT,
        "checkpoint_context": CHECKPOINT_CONTEXT,
        "planes": {
            "resident": Q40,
            "experts": Q40,
            "vectors": "fp16",
            "kv": "fp8",
        },
        "registered": False,
    }
    manifest_path.write_text(json.dumps(body, indent=2) + "\n", encoding="utf-8")
    progress = dest / "quantize_progress.json"
    if progress.is_file():
        progress.unlink()


def _worker_count() -> int:
    count = os.process_cpu_count() or os.cpu_count() or 2
    return max(1, count - 2)


def _limit_blas() -> None:
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"


def _pack_expert(task: tuple[str, str, str, str, str, int]) -> tuple[bytes, bytes]:
    src, gate_shard, down_shard, gate_name, down_name, expert = task
    root = Path(src)
    gate = load_tensor(root, gate_shard, gate_name, expert)
    down = load_tensor(root, down_shard, down_name, expert)
    return pack_matrix(gate, Q40), pack_matrix(down, Q40)


_ROW_CHUNK = 8192


def _pack_resident(
    src: Path,
    shard: str,
    name: str,
    rows: int,
    pool: ProcessPoolExecutor,
) -> bytes:
    if rows < _ROW_CHUNK:
        values = load_tensor(src, shard, name)
        return pack_matrix(np.ascontiguousarray(values), Q40)
    spans = []
    start = 0
    while start < rows:
        stop = min(rows, start + _ROW_CHUNK)
        spans.append((str(src), shard, name, start, stop))
        start = stop
    return b"".join(pool.map(_pack_rows, spans))


def _pack_rows(task: tuple[str, str, str, int, int]) -> bytes:
    src, shard, name, start, stop = task
    values = load_tensor(Path(src), shard, name, rows=(start, stop))
    return pack_matrix(np.ascontiguousarray(values), Q40)


def _write_experts(
    src: Path,
    names: dict[str, str],
    dest: Path,
    pool: ProcessPoolExecutor,
) -> None:
    progress = _read_progress(dest)
    done = {int(layer) for layer in progress.get("packed_layers", [])}
    offsets_path = dest / "experts" / "offsets.bin"
    width_count = LAYERS * EXPERTS
    if offsets_path.is_file() and offsets_path.stat().st_size == width_count * 8 and done:
        offsets = bytearray(offsets_path.read_bytes())
    else:
        offsets = bytearray(width_count * 8)
        done = set()
    for layer in range(LAYERS):
        if layer in done and (dest / "experts" / f"{layer}.bin").is_file():
            continue
        base = f"model.language_model.layers.{layer}"
        gate_name = f"{base}.experts.gate_up_proj"
        down_name = f"{base}.experts.down_proj"
        tasks = [
            (str(src), names[gate_name], names[down_name], gate_name, down_name, expert)
            for expert in range(EXPERTS)
        ]
        blob = bytearray()
        for expert, (gate, down) in enumerate(pool.map(_pack_expert, tasks)):
            start = len(blob)
            blob += gate
            blob += down
            slot = layer * EXPERTS + expert
            offsets[slot * 8 : (slot + 1) * 8] = start.to_bytes(8, "little")
        (dest / "experts" / f"{layer}.bin").write_bytes(blob)
        offsets_path.write_bytes(offsets)
        done.add(layer)
        progress["packed_layers"] = sorted(done)
        _write_progress(dest, progress)
        print(f"packed experts {layer}", flush=True)


def _write_resident(
    src: Path,
    names: dict[str, str],
    catalog: dict[str, tuple[str, tuple[int, ...]]],
    dest: Path,
    pool: ProcessPoolExecutor,
) -> None:
    index: list[dict[str, object]] = []
    with (dest / "resident.bin").open("wb") as handle:
        for name, (role, shape) in catalog.items():
            if role in {EXPERT_UP, EXPERT_DOWN}:
                continue
            _pad(handle)
            offset = handle.tell()
            if role == RESIDENT:
                if len(shape) != 2:
                    message = f"{name} is not a matrix"
                    raise ValueError(message)
                packed = _pack_resident(src, names[name], name, int(shape[0]), pool)
                stored = Q40
                padded = align_k(int(shape[1]), Q40)
            else:
                values = load_tensor(src, names[name], name)
                if tuple(values.shape) != shape:
                    message = f"{name} shape {tuple(values.shape)} != {shape}"
                    raise ValueError(message)
                packed = store_fp16(values)
                stored = "fp16"
                padded = None
            handle.write(packed)
            index.append(
                {
                    "name": name,
                    "plane": stored,
                    "shape": list(shape),
                    "padded_k": padded,
                    "offset": offset,
                    "nbytes": len(packed),
                }
            )
    (dest / "resident.json").write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")


def _copy_sidecars(src: Path, dest: Path) -> None:
    for name in _SIDECARS:
        path = src / name
        if path.is_file():
            shutil.copyfile(path, dest / name, follow_symlinks=True)


def _pad(handle: object) -> None:
    tell = handle.tell
    write = handle.write
    extra = (-int(tell())) % ALIGN
    if extra:
        write(b"\0" * extra)


def _read_progress(dest: Path) -> dict[str, object]:
    path = dest / "quantize_progress.json"
    if not path.is_file():
        return {"packed_layers": []}
    stored = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(stored, dict):
        return {"packed_layers": []}
    return stored


def _write_progress(dest: Path, body: dict[str, object]) -> None:
    path = dest / "quantize_progress.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(body) + "\n", encoding="utf-8")
    temporary.replace(path)
