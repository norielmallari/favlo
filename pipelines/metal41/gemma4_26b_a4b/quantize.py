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
    HALF,
    LAYERS,
    RESIDENT,
    expected,
)
from gemma4_26b_a4b.planes import (
    FP4,
    FP8,
    align_k,
    expert_bytes,
    fp8_parts,
    matrix_nbytes,
    pack_matrix,
    restoration_rms,
    store_half,
)
from gemma4_26b_a4b.promote import promotion_count, select_promoted

RECORD_STEM = "gemma-4-26B-A4B-it"
SOURCE_DIR = "gemma-4-26B-A4B-it"
RECORD_NAME = f"{RECORD_STEM}-fp8-e20"
ALIGN = 128
EXPERT_FP8_PERCENT = 20
EXPERT_FORMAT = "fp8-e20"
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
    half = 0
    for role, shape in expected().values():
        if role == HALF:
            half += int(np.prod(shape)) * 2
        elif role == RESIDENT:
            rows = _stored_rows(int(shape[0]))
            resident += matrix_nbytes(rows, int(shape[-1]), FP8)
    pool = LAYERS * EXPERTS
    promoted = promotion_count(pool, EXPERT_FP8_PERCENT)
    experts = promoted * expert_bytes(8) + (pool - promoted) * expert_bytes(4)
    return {
        "resident": resident,
        "scratch": scratch,
        "half": half,
        "experts": experts,
    }


def quantize(src: Path, dest: Path) -> None:
    manifest_path = dest / "manifest.json"
    if manifest_path.is_file():
        stored = json.loads(manifest_path.read_text(encoding="utf-8"))
        planes = stored.get("planes")
        if (
            stored.get("complete") is True
            and isinstance(planes, dict)
            and planes.get("resident") == FP8
            and planes.get("experts") == EXPERT_FORMAT
            and planes.get("expert_fp8_percent") == EXPERT_FP8_PERCENT
        ):
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
        if not (dest / "resident.bin").is_file() or not (dest / "resident.json").is_file():
            _write_resident(src, names, catalog, dest, pool)
        _write_experts(src, names, dest, pool)
    _copy_sidecars(src, dest)
    body = {
        "complete": True,
        "pipeline": "gemma4-26b-a4b",
        "source_name": src.name,
        "context_length": CONTEXT,
        "checkpoint_context": CHECKPOINT_CONTEXT,
        "planes": {
            "resident": FP8,
            "experts": EXPERT_FORMAT,
            "expert_fp8_percent": EXPERT_FP8_PERCENT,
            "vectors": "half",
            "kv": "fp8_e4m3",
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


def _score_expert(task: tuple[str, str, str, str, str, int, int]) -> tuple[int, int, float, float]:
    src, gate_shard, down_shard, gate_name, down_name, layer, expert = task
    root = Path(src)
    gate = load_tensor(root, gate_shard, gate_name, expert)
    down = load_tensor(root, down_shard, down_name, expert)
    return layer, expert, restoration_rms(gate, FP4), restoration_rms(down, FP4)


def _pack_expert(task: tuple[str, str, str, str, str, int, int]) -> bytes:
    src, gate_shard, down_shard, gate_name, down_name, expert, width = task
    root = Path(src)
    gate = load_tensor(root, gate_shard, gate_name, expert)
    down = load_tensor(root, down_shard, down_name, expert)
    plane = FP8 if width == 8 else FP4
    blob = pack_matrix(gate, plane) + pack_matrix(down, plane)
    if len(blob) != expert_bytes(width):
        message = "expert image does not match its width"
        raise RuntimeError(message)
    return blob


_ROW_CHUNK = 8192


def _stored_rows(rows: int) -> int:
    return (int(rows) + 31) & ~31


def _pack_resident(
    src: Path,
    shard: str,
    name: str,
    rows: int,
    k: int,
    pool: ProcessPoolExecutor,
) -> bytes:
    spans = []
    start = 0
    while start < rows:
        stop = min(rows, start + _ROW_CHUNK)
        spans.append((str(src), shard, name, start, stop, stop == rows))
        start = stop
    blobs = list(pool.map(_pack_rows, spans))
    data: list[bytes] = []
    scales: list[bytes] = []
    stored = _stored_rows(rows)
    for blob, task in zip(blobs, spans, strict=True):
        count = stored - task[3] if task[5] else task[4] - task[3]
        nbytes, scale = fp8_parts(count, k)
        if len(blob) != nbytes + scale:
            message = f"{name} fp8 row span is {len(blob)} bytes"
            raise RuntimeError(message)
        data.append(blob[:nbytes])
        scales.append(blob[nbytes:])
    return b"".join(data) + b"".join(scales)


def _pack_rows(task: tuple[str, str, str, int, int, bool]) -> bytes:
    src, shard, name, start, stop, last = task
    values = load_tensor(Path(src), shard, name, rows=(start, stop))
    if last:
        extra = _stored_rows(stop) - stop
        if extra:
            values = np.pad(values, ((0, extra), (0, 0)))
    return pack_matrix(np.ascontiguousarray(values), FP8)


def _write_experts(
    src: Path,
    names: dict[str, str],
    dest: Path,
    pool: ProcessPoolExecutor,
) -> None:
    chosen = _promotion(src, names, dest, pool)
    progress = _read_progress(dest)
    if progress.get("format") != EXPERT_FORMAT:
        for path in (dest / "experts").glob("*.bin"):
            path.unlink()
        progress = {"format": EXPERT_FORMAT, "packed_layers": []}
        _write_progress(dest, progress)
    done = {int(layer) for layer in progress.get("packed_layers", [])}
    widths = np.zeros(LAYERS * EXPERTS, dtype=np.uint8)
    for layer in range(LAYERS):
        for expert in range(EXPERTS):
            widths[layer * EXPERTS + expert] = 8 if (layer, expert) in chosen else 4
    (dest / "experts" / "widths.bin").write_bytes(widths.tobytes())
    for layer in range(LAYERS):
        expected = int(sum(expert_bytes(int(widths[layer * EXPERTS + expert])) for expert in range(EXPERTS)))
        target = dest / "experts" / f"{layer}.bin"
        if layer in done and target.is_file() and target.stat().st_size == expected:
            continue
        base = f"model.language_model.layers.{layer}"
        gate_name = f"{base}.experts.gate_up_proj"
        down_name = f"{base}.experts.down_proj"
        tasks = [
            (
                str(src),
                names[gate_name],
                names[down_name],
                gate_name,
                down_name,
                expert,
                int(widths[layer * EXPERTS + expert]),
            )
            for expert in range(EXPERTS)
        ]
        blob = b"".join(pool.map(_pack_expert, tasks))
        if len(blob) != expected:
            message = f"expert layer {layer} is not the fp8-e20 layout"
            raise RuntimeError(message)
        target.write_bytes(blob)
        done.add(layer)
        progress["packed_layers"] = sorted(done)
        progress["format"] = EXPERT_FORMAT
        _write_progress(dest, progress)
        print(f"packed experts {layer}", flush=True)


def _promotion(
    src: Path,
    names: dict[str, str],
    dest: Path,
    pool: ProcessPoolExecutor,
) -> set[tuple[int, int]]:
    path = dest / "expert_error.json"
    if path.is_file():
        body = json.loads(path.read_text(encoding="utf-8"))
        reused = _reuse_scores(body)
        if reused is not None:
            _replace_json(path, reused[0])
            return reused[1]
    tasks = []
    for layer in range(LAYERS):
        base = f"model.language_model.layers.{layer}"
        gate_name = f"{base}.experts.gate_up_proj"
        down_name = f"{base}.experts.down_proj"
        for expert in range(EXPERTS):
            tasks.append(
                (str(src), names[gate_name], names[down_name], gate_name, down_name, layer, expert)
            )
    parts: dict[tuple[int, int], tuple[float, float]] = {}
    rows: list[tuple[float, int, int]] = []
    scored = -1
    for layer, expert, gate_error, down_error in pool.map(_score_expert, tasks, chunksize=8):
        parts[(layer, expert)] = (gate_error, down_error)
        rows.append((max(gate_error, down_error), layer, expert))
        if layer != scored and expert == EXPERTS - 1:
            scored = layer
            print(f"scored experts {layer}", flush=True)
    chosen = select_promoted(rows, EXPERT_FP8_PERCENT)
    body = {
        "percent": EXPERT_FP8_PERCENT,
        "pool": LAYERS * EXPERTS,
        "count": len(chosen),
        "promoted": [[layer, expert] for layer, expert in chosen],
        "scores": {
            f"{layer}.{expert}": {
                "gate_up": parts[(layer, expert)][0],
                "down": parts[(layer, expert)][1],
                "expert": max(parts[(layer, expert)]),
            }
            for layer in range(LAYERS)
            for expert in range(EXPERTS)
        },
    }
    _replace_json(path, body)
    return set(chosen)


def _reuse_scores(body: object) -> tuple[dict[str, object], set[tuple[int, int]]] | None:
    """Keep a finished fp4 score file and apply the current percentage."""
    if not isinstance(body, dict):
        return None
    scores = body.get("scores")
    if not isinstance(scores, dict) or len(scores) != LAYERS * EXPERTS:
        return None
    rows: list[tuple[float, int, int]] = []
    for key, item in scores.items():
        if not isinstance(item, dict) or "expert" not in item or "." not in str(key):
            return None
        layer_text, expert_text = str(key).split(".", 1)
        rows.append((float(item["expert"]), int(layer_text), int(expert_text)))
    if len(rows) != LAYERS * EXPERTS:
        return None
    chosen = select_promoted(rows, EXPERT_FP8_PERCENT)
    stored = dict(body)
    stored["percent"] = EXPERT_FP8_PERCENT
    stored["pool"] = LAYERS * EXPERTS
    stored["count"] = len(chosen)
    stored["promoted"] = [[layer, expert] for layer, expert in chosen]
    return stored, set(chosen)


def _chosen(body: object) -> set[tuple[int, int]]:
    if not isinstance(body, dict):
        message = "expert_error.json is not a record"
        raise RuntimeError(message)
    if int(body.get("percent", -1)) != EXPERT_FP8_PERCENT or int(body.get("pool", -1)) != LAYERS * EXPERTS:
        message = "expert_error.json does not match the fp8-e20 percentage"
        raise RuntimeError(message)
    count = promotion_count(LAYERS * EXPERTS, EXPERT_FP8_PERCENT)
    if int(body.get("count", -1)) != count:
        message = "expert_error.json promote count does not match"
        raise RuntimeError(message)
    promoted = body.get("promoted")
    if not isinstance(promoted, list) or len(promoted) != count:
        message = "expert_error.json promote list does not match"
        raise RuntimeError(message)
    chosen: set[tuple[int, int]] = set()
    for pair in promoted:
        if not isinstance(pair, list) or len(pair) != 2:
            message = "expert_error.json promote list does not match"
            raise RuntimeError(message)
        layer, expert = int(pair[0]), int(pair[1])
        if layer < 0 or layer >= LAYERS or expert < 0 or expert >= EXPERTS:
            message = f"promoted expert {layer}.{expert} is outside the pool"
            raise RuntimeError(message)
        chosen.add((layer, expert))
    if len(chosen) != count:
        message = "expert_error.json promote list does not match"
        raise RuntimeError(message)
    return chosen


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
                packed = _pack_resident(
                    src, names[name], name, int(shape[0]), int(shape[1]), pool
                )
                stored = FP8
                padded = align_k(int(shape[1]), FP8)
                rows_pad = _stored_rows(int(shape[0]))
            else:
                values = load_tensor(src, names[name], name)
                if tuple(values.shape) != shape:
                    message = f"{name} shape {tuple(values.shape)} != {shape}"
                    raise ValueError(message)
                packed = store_half(values)
                stored = "half"
                padded = None
            handle.write(packed)
            index.append(
                {
                    "name": name,
                    "plane": stored,
                    "shape": list(shape),
                    "padded_k": padded,
                    "padded_rows": rows_pad if role == RESIDENT else None,
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
    _replace_json(dest / "quantize_progress.json", body)


def _replace_json(path: Path, body: object) -> None:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(body) + "\n", encoding="utf-8")
    temporary.replace(path)
