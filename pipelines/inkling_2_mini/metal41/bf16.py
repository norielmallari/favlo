"""Quantize a BF16 Inkling checkpoint to a mixed fp4 and fp8 expert record.

Each routed expert is scored by the relative RMS of an fp4 restoration of its
gate-up and down matrices. The highest scores, up to ``--expert-fp8`` percent
of the 3072 experts, are stored as MXFP8. The rest are Metal fp4. An equal
score keeps the lower index. ``--promote firing`` ranks a finished firing log
instead and does not rescore. Attention and the output projection stay float16.
The embedding and the head stay float16.

The source folder is left in place and recorded as the registry source.
The weight files are read there. The output is a sealed record in a new
folder, plus ``experts/widths.bin``.
"""

from __future__ import annotations

import json
import math
import shutil
from collections.abc import Callable
from pathlib import Path

import numpy as np

from inkling_2_mini.metal41.gptq import config_digest, dequant, e2m1, e8m0
from inkling_2_mini.metal41.layout import EXPERTS, HIDDEN, INTERMEDIATE, LAYERS, stored_k
from inkling_2_mini.metal41.weights import (
    Safetensors,
    _quantize_fp4,
    quantize_mxfp8,
    safetensor_files,
)

PACK_FORMAT = "favlo-inkling-expert-mix-1"
POOL = LAYERS * EXPERTS
_CHUNK = 4
_PIECE = 8 << 20
_PROJECTIONS = (("w13", INTERMEDIATE * 2, HIDDEN), ("w2", HIDDEN, INTERMEDIATE))

Log = Callable[[str], None] | None


def promote_count(percent: float, pool: int) -> int:
    """Largest whole count that does not exceed ``percent`` of ``pool``."""
    if pool < 1:
        message = f"expert pool {pool} is empty"
        raise ValueError(message)
    if not math.isfinite(percent) or percent < 0.0 or percent > 100.0:
        message = f"expert fp8 percentage {percent} is outside 0..100"
        raise ValueError(message)
    count = int(math.floor((percent * pool / 100.0) + 1e-9))
    if count > pool:
        return pool
    return count


def promoted_mask(scores: np.ndarray, count: int) -> np.ndarray:
    """Highest scores. An equal score keeps the lower index."""
    flat = np.asarray(scores, dtype=np.float64).reshape(-1)
    if count < 0 or count > flat.size:
        message = f"promote count {count} does not fit {flat.size} experts"
        raise ValueError(message)
    order = np.lexsort((np.arange(flat.size), -flat))
    chosen = np.zeros(flat.size, dtype=bool)
    chosen[order[:count]] = True
    return chosen


def fp4_relative(weight: np.ndarray) -> np.ndarray:
    """Relative RMS of an fp4 restoration. One score per leading expert."""
    matrix = np.ascontiguousarray(weight, dtype=np.float32)
    if matrix.ndim == 2:
        matrix = matrix.reshape(1, *matrix.shape)
    count, rows, cols = matrix.shape
    if cols % 32:
        message = f"inner dimension {cols} is not a multiple of 32"
        raise ValueError(message)
    num = np.zeros(count, dtype=np.float64)
    den = np.zeros(count, dtype=np.float64)
    blocks = cols // 32
    for start in range(0, rows, 256):
        stop = min(start + 256, rows)
        values = matrix[:, start:stop, :].reshape(count, stop - start, blocks, 32)
        _biased, scale = e8m0(np.max(np.abs(values), axis=-1))
        restored = dequant(e2m1(values / scale[..., None])) * scale[..., None]
        diff = values.astype(np.float64) - restored.astype(np.float64)
        num += np.sum(diff * diff, axis=(1, 2, 3))
        den += np.sum(values.astype(np.float64) ** 2, axis=(1, 2, 3))
    out = np.zeros(count, dtype=np.float64)
    live = den > 0.0
    out[live] = np.sqrt(num[live] / den[live])
    return out


def projection_image(width: int, rows: int, cols: int) -> tuple[int, int]:
    """Data bytes and scale bytes of one expert. Both lengths divide by 128."""
    if width == 8:
        data_n = rows * cols
        scale_n = rows * (cols // 32)
    elif width == 4:
        stored = stored_k(cols)
        data_n = rows * (stored // 2)
        scale_n = rows * (stored // 32)
    else:
        message = f"expert width {width} is not 4 or 8"
        raise ValueError(message)
    if data_n % 128 or scale_n % 128:
        message = f"expert image {(data_n, scale_n)} is not 128-byte aligned"
        raise ValueError(message)
    return data_n, scale_n


def quantize(
    src: Path,
    dest: Path,
    *,
    expert_fp8: float = 50.0,
    promote: str = "error",
    firing_db: Path | None = None,
    firing_since: int | None = None,
    log: Log = None,
) -> Path:
    """Seal ``dest`` from the BF16 checkpoint in ``src`` and register the folder."""
    source = Path(src).resolve()
    output = Path(dest).resolve()
    if source == output:
        message = "quantize writes a new folder"
        raise ValueError(message)
    if promote not in ("error", "firing"):
        message = f"promote {promote} is not error or firing"
        raise ValueError(message)
    safetensor_files(source)
    _import_sidecars(source, output)
    count = promote_count(expert_fp8, POOL)
    if promote == "firing":
        widths = _widths_from_firing(output, count, firing_db, firing_since, log)
    else:
        scores = _scores(output, source, expert_fp8, log)
        widths = np.full((LAYERS, EXPERTS), 4, dtype=np.uint8)
        widths.ravel()[promoted_mask(scores, count)] = 8
    if int(np.sum(widths == 8)) != count:
        message = "fp8 expert count does not match the requested percentage"
        raise RuntimeError(message)
    from inkling_2_mini.metal41.record import ready

    if ready(output):
        found = output / "experts" / "widths.bin"
        if found.is_file() and np.array_equal(np.fromfile(found, dtype=np.uint8), widths):
            if log is not None:
                log("record is current")
            _register(output, source)
            return output
        message = "destination record is already sealed"
        raise RuntimeError(message)
    _write_experts(output, source, widths, expert_fp8, log)
    from inkling_2_mini.metal41.vision import ensure_pack

    ensure_pack(output, log=log, language=source)
    from inkling_2_mini.metal41.record import seal

    seal(output, log=log, language=source)
    folder = output / "experts"
    folder.mkdir(parents=True, exist_ok=True)
    found = folder / "widths.bin"
    found.write_bytes(np.ascontiguousarray(widths).tobytes())
    _register(output, source)
    if log is not None:
        log(f"registered {output.name}")
    return output


def source_variants() -> dict[str, dict[str, float | int | str]]:
    """Sampler for a folder quantized from a BF16 source.

    Temperature 1, top-p 0.95, top-k 64, and presence penalty 0. Instruct
    uses minimal effort. Thinking keeps the row default.
    """
    card: dict[str, float | int | str] = {
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 64,
        "presence_penalty": 0.0,
    }
    instruct = dict(card)
    instruct["reasoning_effort"] = "minimal"
    return {"thinking": card, "instruct": instruct}


def _register(folder: Path, source: Path) -> None:
    from favlo.registry import load_registry, save_registry
    from inkling_2_mini.metal41.commands import register_folder

    register_folder(folder, alias=None, source=source)
    template = source / "chat_template.jinja"
    text = template.read_text() if template.is_file() else ""
    if "reason_mods" not in text:
        return
    data = load_registry()
    row = data["models"][folder.name]
    row["variants"] = source_variants()
    save_registry(data)


def _import_sidecars(src: Path, dest: Path) -> None:
    """Copy the tokenizer and config. Weight files stay in ``src``."""
    dest.mkdir(parents=True, exist_ok=True)
    for path in dest.glob("*.safetensors"):
        path.unlink()
    for path in src.rglob("*"):
        if path.name.endswith(".incomplete") or not path.is_file() or path.suffix == ".safetensors":
            continue
        rel = path.relative_to(src)
        target = dest / rel
        if target.exists():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)


def _scores(root: Path, language: Path, percent: float, log: Log) -> np.ndarray:
    path = root / "expert_error.json"
    scores, start_layer = _saved_scores(path, percent)
    if start_layer >= LAYERS:
        if log is not None:
            log("expert scores are current")
        return scores
    tensors = Safetensors(language)
    try:
        for layer in range(start_layer, LAYERS):
            slot = layer * EXPERTS
            scores[slot : slot + EXPERTS] = np.nan
            loaded = _read_projections(tensors, layer)
            for kind, _rows, _cols in _PROJECTIONS:
                weight = loaded[kind]
                for start in range(0, EXPERTS, _CHUNK):
                    stop = min(start + _CHUNK, EXPERTS)
                    piece = fp4_relative(weight[start:stop])
                    dest = scores[slot + start : slot + stop]
                    missing = np.isnan(dest)
                    dest[missing] = piece[missing]
                    dest[~missing] = np.maximum(dest[~missing], piece[~missing])
                if log is not None:
                    log(f"scored layer {layer} {kind}")
            _write_scores(path, percent, scores, layer + 1)
    finally:
        tensors.close()
    if not np.all(np.isfinite(scores)):
        message = "expert scores are incomplete"
        raise RuntimeError(message)
    return scores


def _saved_scores(path: Path, percent: float) -> tuple[np.ndarray, int]:
    scores = np.full(POOL, np.nan, dtype=np.float64)
    if not path.is_file():
        return scores, 0
    body = json.loads(path.read_text())
    saved = body.get("scores") if isinstance(body, dict) else None
    if not (
        isinstance(body, dict)
        and body.get("percent") == percent
        and isinstance(saved, list)
        and len(saved) == POOL
    ):
        return scores, 0
    for index, item in enumerate(saved):
        if item is not None:
            scores[index] = float(item)
    done = int(body.get("layers_done") or 0)
    if done < 0 or done > LAYERS or not np.all(np.isfinite(scores[: done * EXPERTS])):
        return np.full(POOL, np.nan, dtype=np.float64), 0
    return scores, done


def _write_scores(path: Path, percent: float, scores: np.ndarray, layers_done: int) -> None:
    values = [None if not np.isfinite(item) else float(item) for item in scores]
    path.write_text(
        json.dumps(
            {
                "percent": percent,
                "promote": "error",
                "layers_done": layers_done,
                "scores": values,
            }
        )
    )


def _widths_from_firing(
    root: Path,
    count: int,
    firing_db: Path | None,
    since: int | None,
    log: Log,
) -> np.ndarray:
    from inkling_2_mini.metal41.firing import flat_hit_scores, telemetry_dir

    path = Path(firing_db) if firing_db is not None else telemetry_dir(root.name) / "firing.sqlite"
    if not path.is_file():
        message = f"firing log is missing for {root.name}"
        raise FileNotFoundError(message)
    hits = flat_hit_scores(path, since=since)
    scores = np.array([hits[index] for index in range(POOL)], dtype=np.float64)
    widths = np.full((LAYERS, EXPERTS), 4, dtype=np.uint8)
    widths.ravel()[promoted_mask(scores, count)] = 8
    note = {
        "percent": 100.0 * count / POOL,
        "promote": "firing",
        "since": since,
        "promoted": [int(index) for index in np.flatnonzero(widths.ravel() == 8)],
    }
    (root / "expert_firing.json").write_text(json.dumps(note))
    if log is not None:
        log(f"ranked {count} experts from {path.name}")
    return widths


def _write_experts(
    root: Path, language: Path, widths: np.ndarray, percent: float, log: Log
) -> None:
    from inkling_2_mini.metal41.resident import EXPERT_PACK, Sink

    entries: list[tuple[str, tuple[int, ...]]] = [("widths", (LAYERS, EXPERTS))]
    sizes: dict[tuple[int, str], int] = {}
    for layer in range(LAYERS):
        for kind, rows, cols in _PROJECTIONS:
            total = 0
            for expert in range(EXPERTS):
                data_n, scale_n = projection_image(int(widths[layer, expert]), rows, cols)
                total += data_n + scale_n
            sizes[(layer, kind)] = total
            entries.append((f"layers.{layer}.{kind}", (total,)))
            entries.append((f"layers.{layer}.{kind}.data_off", (EXPERTS, 4)))
            entries.append((f"layers.{layer}.{kind}.scale_off", (EXPERTS, 4)))
    sink = Sink(
        root / EXPERT_PACK,
        entries,
        {
            "format": PACK_FORMAT,
            "config": config_digest(root),
            "expert_fp8": str(percent),
        },
    )
    tensors = Safetensors(language)
    try:
        sink.write_view("widths", _flat_view(widths))
        for layer in range(LAYERS):
            loaded = _read_projections(tensors, layer)
            for kind, _rows, _cols in _PROJECTIONS:
                blob, data_off, scale_off = _pack_projection(
                    loaded[kind], widths[layer], sizes[(layer, kind)]
                )
                sink.write_view(f"layers.{layer}.{kind}", _flat_view(blob))
                sink.write_view(f"layers.{layer}.{kind}.data_off", _flat_view(_u32_plane(data_off)))
                sink.write_view(
                    f"layers.{layer}.{kind}.scale_off", _flat_view(_u32_plane(scale_off))
                )
            if log is not None:
                log(f"packed layer {layer}")
        sink.close()
    except Exception:
        sink.abort()
        raise
    finally:
        tensors.close()


def _pack_projection(
    weight: np.ndarray,
    widths: np.ndarray,
    total: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    blob = np.zeros(total, dtype=np.uint8)
    data_off = np.zeros(EXPERTS, dtype=np.uint32)
    scale_off = np.zeros(EXPERTS, dtype=np.uint32)
    cursor = 0
    for expert in range(EXPERTS):
        data, scale = _quantize_expert(weight[expert], int(widths[expert]))
        data_off[expert] = cursor
        scale_off[expert] = cursor + data.nbytes
        blob[cursor : cursor + data.nbytes] = data
        cursor += data.nbytes
        blob[cursor : cursor + scale.nbytes] = scale
        cursor += scale.nbytes
    if cursor != total:
        message = f"expert projection packed {cursor} bytes, not {total}"
        raise RuntimeError(message)
    return blob, data_off, scale_off


def _quantize_expert(matrix: np.ndarray, width: int) -> tuple[np.ndarray, np.ndarray]:
    values = np.ascontiguousarray(matrix, dtype=np.float32)
    _rows, cols = values.shape
    if width == 8:
        codes, scales = quantize_mxfp8(values)
    else:
        stored = stored_k(cols)
        if stored != cols:
            padded = np.zeros((values.shape[0], stored), dtype=np.float32)
            padded[:, :cols] = values
            values = padded
        codes, scales = _quantize_fp4(values)
    return np.ascontiguousarray(codes).ravel(), np.ascontiguousarray(scales).ravel()


def _flat_view(values: np.ndarray) -> memoryview:
    """One-dimensional bytes. ``len`` of a matrix memoryview is its row count."""
    flat = np.ascontiguousarray(values).reshape(-1)
    return memoryview(flat)


def _u32_plane(values: np.ndarray) -> np.ndarray:
    raw = np.ascontiguousarray(values, dtype="<u4")
    return raw.view(np.uint8).reshape(EXPERTS, 4)


def _require_bf16(tensors: Safetensors, name: str, rows: int, cols: int) -> tuple[int, int]:
    info = tensors.header.get(name)
    if not isinstance(info, dict):
        message = f"checkpoint is missing {name}"
        raise KeyError(message)
    shape = tuple(info.get("shape", ()))
    expect = (EXPERTS, rows, cols)
    if info.get("dtype") != "BF16" or shape != expect:
        message = f"{name} is {info.get('dtype')} {shape}, not BF16 {expect}"
        raise ValueError(message)
    return tensors.locate(name)


def _grouped_windows(
    spans: list[tuple[str, int, int, int, int, int]],
) -> list[list[tuple[str, int, int, int, int, int]]]:
    """One group per sequential run of one file. A gap shorter than 8 MiB stays in the run."""
    if not spans:
        return []
    ordered = sorted(spans, key=lambda item: (item[3], item[4]))
    groups = [[ordered[0]]]
    end = ordered[0][4] + ordered[0][5]
    file_id = ordered[0][3]
    for item in ordered[1:]:
        fd, off, nbytes = item[3], item[4], item[5]
        if fd == file_id and off < end:
            message = f"expert span {off} overlaps a span that ends at {end}"
            raise ValueError(message)
        if fd == file_id and off - end < _PIECE:
            groups[-1].append(item)
            end = max(end, off + nbytes)
            continue
        groups.append([item])
        file_id = fd
        end = off + nbytes
    return groups


def _read_projections(tensors: Safetensors, layer: int) -> dict[str, np.ndarray]:
    """Both expert projections, each run read in 8 MiB pieces before scoring."""
    from inkling_2_mini.metal41.resident import pread_view

    spans: list[tuple[str, int, int, int, int, int]] = []
    for kind, rows, cols in _PROJECTIONS:
        name = f"model.llm.layers.{layer}.mlp.experts.{kind}_weight"
        fd, base = _require_bf16(tensors, name, rows, cols)
        spans.append((kind, rows, cols, fd, base, EXPERTS * rows * cols * 2))
    found: dict[str, np.ndarray] = {}
    for group in _grouped_windows(spans):
        start = group[0][4]
        stop = group[-1][4] + group[-1][5]
        window = np.empty(stop - start, dtype=np.uint8)
        pread_view(group[0][3], memoryview(window), start)
        for kind, rows, cols, _fd, off, nbytes in group:
            piece = window[off - start : off - start + nbytes]
            found[kind] = _bf16_matrix(piece, rows, cols)
    return found


def _bf16_matrix(raw: np.ndarray, rows: int, cols: int) -> np.ndarray:
    codes = np.ascontiguousarray(raw).view(np.uint16)
    wide = codes.astype(np.uint32) << 16
    return wide.view(np.float32).reshape(EXPERTS, rows, cols)
