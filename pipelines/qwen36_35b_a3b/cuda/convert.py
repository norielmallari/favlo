"""Pack a BF16 checkout into the CUDA record.

Dense language matrices are int8. Twenty percent of the routed experts, ranked
by int4 restoration error, are int8. The rest are int4. Norms, the router, and
the convolution stay fp16. The expert images are the host atlas.
"""

from __future__ import annotations

import json
import os
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from favlo.quantize import half_bytes, parallel_map, to_f32
from qwen36_35b_a3b.cuda.formats import pack_matrix, relative_error
from qwen36_35b_a3b.cuda.layout import (
    CHECKOUT,
    EXPERT_PERCENT,
    EXPERTS,
    INT4_IMAGE,
    INT8_IMAGE,
    LANGUAGE_EXPERTS,
    LAYERS,
    REPO,
    promote_count,
)

_HOST = "https://huggingface.co"
_AGENT = "favlo"
SIDECARS = (
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "preprocessor_config.json",
)


def checkout_dir(models: Path) -> Path:
    return models / CHECKOUT


def quantize(
    src: Path,
    output: Path,
    *,
    expert_int8: float = EXPERT_PERCENT,
    on_progress: Callable[[int, int, str], None] | None = None,
) -> Path:
    """Pack ``src``. A finished manifest with the same percentage stays."""
    percent = float(expert_int8)
    count = promote_count(percent, LANGUAGE_EXPERTS)
    src = src.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    if manifest_path.is_file():
        body = json.loads(manifest_path.read_text(encoding="utf-8"))
        if body.get("complete") is True and float(body.get("expert_int8_percent", -1)) == percent:
            return output
    index = json.loads((src / "model.safetensors.index.json").read_text(encoding="utf-8"))
    weight_map: dict[str, str] = index["weight_map"]
    promoted = _rank(src, weight_map, output, count, on_progress)
    tensors, language_bytes = _pack_dense(src, weight_map, output, on_progress)
    _pack_experts(src, weight_map, output, promoted, tensors, on_progress)
    manifest = {
        "complete": True,
        "pipeline": "qwen36_35b_a3b",
        "runtime": "cuda",
        "weight": "int8-e20" if percent == 20 else f"int8-e{percent:g}",
        "expert_int8_percent": percent,
        "expert_int8": count,
        "expert_int4": LANGUAGE_EXPERTS - count,
        "language_bytes": language_bytes,
        "layers": LAYERS,
        "experts": EXPERTS,
        "kv_cache": "int8",
        "tensors": tensors,
    }
    if percent <= 0:
        manifest["weight"] = "int4"
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest), encoding="utf-8")
    os.replace(temporary, manifest_path)
    for name in SIDECARS:
        source = src / name
        if source.is_file():
            target = output / name
            if not target.exists():
                target.write_bytes(source.read_bytes())
    return output


def list_files() -> list[tuple[str, int]]:
    """Files in the official repository, largest first."""
    url = f"{_HOST}/api/models/{REPO}/tree/main?recursive=1"
    request = urllib.request.Request(url, headers={"User-Agent": _AGENT})
    with urllib.request.urlopen(request, timeout=60) as response:
        listing = json.loads(response.read().decode("utf-8"))
    files = [
        (str(item["path"]), int(item.get("size") or 0))
        for item in listing
        if item.get("type") == "file"
    ]
    return sorted(files, key=lambda item: item[1], reverse=True)


def download(dest: Path, files: list[tuple[str, int]], on_progress: Callable[[int, int, str],
    None]) -> None:
    """Download ``files`` into ``dest``. A file that already has its size stays."""
    dest.mkdir(parents=True, exist_ok=True)
    done = 0
    total = sum(size for _name, size in files) or 1
    for name, size in files:
        path = dest / name
        path.parent.mkdir(parents=True, exist_ok=True)
        have = path.stat().st_size if path.is_file() else 0
        if size and have == size:
            done += size
            on_progress(done, total, name)
            continue
        url = f"{_HOST}/{REPO}/resolve/main/{name}"
        request = urllib.request.Request(url, headers={"User-Agent": _AGENT})
        with urllib.request.urlopen(request, timeout=120) as response, path.open("wb") as handle:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
                done += len(chunk)
                on_progress(min(done, total), total, name)


def _rank(
    src: Path,
    weight_map: dict[str, str],
    output: Path,
    count: int,
    on_progress: Callable[[int, int, str], None] | None,
) -> set[tuple[int, int]]:
    path = output / "expert_error.json"
    if path.is_file():
        body = json.loads(path.read_text(encoding="utf-8"))
        if int(body.get("count", -1)) == count and int(body.get("pool", 0)) == LANGUAGE_EXPERTS:
            return {(int(layer), int(expert)) for layer, expert in body["promoted"]}
    scores: dict[tuple[int, int], list[float]] = {}
    shards = _shards(weight_map, expert=True)
    seen = 0
    total = sum(len(names) for names in shards.values()) or 1
    for shard, names in shards.items():
        header, base = _header(src / shard)
        for name in names:
            array = _load(src / shard, header, base, name)
            layer = _layer(name)
            pairs = list(_iter_experts(name, array))
            errors = parallel_map(relative_error, [matrix for _expert, matrix in pairs])
            for (expert, _matrix), error in zip(pairs, errors, strict=True):
                scores.setdefault((layer, expert), []).append(float(error))
            del array
            seen += 1
            if on_progress is not None:
                on_progress(seen, total, name)
    combined = {key: max(values) for key, values in scores.items() if len(values) >= 2}
    if len(combined) != LANGUAGE_EXPERTS:
        message = f"scored {len(combined)} experts, not {LANGUAGE_EXPERTS}"
        raise RuntimeError(message)
    ranked = sorted(combined, key=lambda key: (-combined[key], key[0], key[1]))
    promoted = ranked[:count]
    body = {
        "pool": LANGUAGE_EXPERTS,
        "count": count,
        "promoted": [[layer, expert] for layer, expert in sorted(promoted)],
        "scores": {f"{layer}.{expert}": combined[(layer, expert)] for layer, expert in combined},
    }
    path.write_text(json.dumps(body), encoding="utf-8")
    return set(promoted)


def _pack_dense(
    src: Path,
    weight_map: dict[str, str],
    output: Path,
    on_progress: Callable[[int, int, str], None] | None,
) -> tuple[dict[str, dict[str, Any]], int]:
    resident = output / "resident.bin"
    vision = output / "vision.bin"
    if resident.exists():
        resident.unlink()
    if vision.exists():
        vision.unlink()
    tensors: dict[str, dict[str, Any]] = {}
    pending: dict[str, dict[str, np.ndarray]] = {}
    shards = _shards(weight_map, expert=False)
    seen = 0
    total = sum(len(names) for names in shards.values()) or 1
    for shard, names in shards.items():
        header, base = _header(src / shard)
        jobs: list[tuple[str, np.ndarray]] = []
        for name in names:
            if name.startswith("mtp."):
                seen += 1
                continue
            array = _load(src / shard, header, base, name)
            gate = name.endswith("shared_expert.gate_proj.weight")
            up = name.endswith("shared_expert.up_proj.weight")
            if gate or up:
                key = name.rsplit(".", 2)[0]
                slot = pending.setdefault(key, {})
                slot["gate" if name.endswith("gate_proj.weight") else "up"] = array
                if "gate" in slot and "up" in slot:
                    fused = np.concatenate([slot.pop("gate"), slot.pop("up")], axis=0)
                    jobs.append((key + ".gate_up", fused))
                    pending.pop(key, None)
                seen += 1
                continue
            jobs.append((name, array))
            seen += 1
        packed = parallel_map(_pack_named, jobs)
        for name, blob, body in packed:
            path = vision if _vision(name) else resident
            offset = _append(path, blob)
            tensors[name] = {**body, "file": path.name, "offset": offset}
        if on_progress is not None:
            on_progress(seen, total, shard)
    if pending:
        message = f"shared expert pair is incomplete: {sorted(pending)}"
        raise RuntimeError(message)
    return tensors, resident.stat().st_size if resident.is_file() else 0


def _pack_experts(
    src: Path,
    weight_map: dict[str, str],
    output: Path,
    promoted: set[tuple[int, int]],
    tensors: dict[str, dict[str, Any]],
    on_progress: Callable[[int, int, str], None] | None,
) -> None:
    folder = output / "experts"
    folder.mkdir(parents=True, exist_ok=True)
    staged: dict[int, dict[str, list[bytes]]] = {}
    shards = _shards(weight_map, expert=True)
    for shard, names in shards.items():
        header, base = _header(src / shard)
        for name in names:
            if name.startswith("mtp."):
                continue
            array = _load(src / shard, header, base, name)
            layer = _layer(name)
            kind = _expert_kind(name)
            pairs = list(_iter_experts(name, array))
            jobs = [
                (matrix, "int8" if (layer, expert) in promoted else "int4")
                for expert, matrix in pairs
            ]
            blobs = parallel_map(_pack_one, jobs)
            slot = staged.setdefault(layer, {}).setdefault(kind, [b""] * EXPERTS)
            for (expert, _matrix), blob in zip(pairs, blobs, strict=True):
                slot[expert] = blob
            tensors[name] = {"dtype": "expert", "file": f"experts/{layer}.bin", "kind": kind}
            del array
            if on_progress is not None:
                on_progress(1, 1, name)
    widths = np.full((LAYERS, EXPERTS), 4, dtype=np.uint8)
    offsets = np.zeros((LAYERS, EXPERTS), dtype=np.uint64)
    for layer in range(LAYERS):
        planes = staged.get(layer)
        missing = (
            planes is None
            or "gate_up" not in planes
            or "down" not in planes
            or b"" in planes["gate_up"]
            or b"" in planes["down"]
        )
        if missing:
            message = f"layer {layer} is missing an expert plane"
            raise RuntimeError(message)
        gate = planes["gate_up"]
        down = planes["down"]
        if len(gate) != EXPERTS or len(down) != EXPERTS:
            message = f"layer {layer} does not have {EXPERTS} experts"
            raise RuntimeError(message)
        # The planes were packed separately. Rebuild each image from the stored pair
        # by concatenating the already packed plane blobs.
        cursor = 0
        blob = bytearray()
        for expert in range(EXPERTS):
            piece = gate[expert] + down[expert]
            wide = (layer, expert) in promoted
            expect = INT8_IMAGE if wide else INT4_IMAGE
            if len(piece) != expect:
                message = f"layer {layer} expert {expert} is {len(piece)} bytes, not {expect}"
                raise RuntimeError(message)
            widths[layer, expert] = 8 if wide else 4
            offsets[layer, expert] = cursor
            blob.extend(piece)
            cursor += len(piece)
        target = folder / f"{layer}.bin"
        temporary = target.with_suffix(".bin.tmp")
        temporary.write_bytes(blob)
        os.replace(temporary, target)
    (folder / "widths.bin").write_bytes(widths.tobytes())
    (folder / "offsets.bin").write_bytes(np.ascontiguousarray(offsets, dtype="<u8").tobytes())


def _pack_one(job: tuple[np.ndarray, str]) -> bytes:
    matrix, plane = job
    return pack_matrix(matrix, plane)


def _iter_experts(name: str, array: np.ndarray) -> list[tuple[int, np.ndarray]]:
    """``(expert, matrix)`` for a stacked plane or one indexed expert."""
    index = _expert_index(name)
    if index is not None and array.ndim == 2:
        return [(index, array)]
    plane = array if array.ndim == 3 else array.reshape(1, *array.shape)
    return [(expert, plane[expert]) for expert in range(plane.shape[0])]


def _expert_index(name: str) -> int | None:
    tail = name.partition(".mlp.experts.")[2]
    head = tail.split(".", 1)[0]
    if head.isdigit():
        return int(head)
    return None


def _pack_named(job: tuple[str, np.ndarray]) -> tuple[str, bytes, dict[str, Any]]:
    name, array = job
    matrix = np.ascontiguousarray(array, dtype=np.float32)
    if _keep_f16(name):
        flat = matrix.reshape(matrix.shape[0], -1) if matrix.ndim > 2 else matrix
        if name.endswith("conv1d.weight"):
            flat = matrix.reshape(matrix.shape[0], matrix.shape[-1])
        blob = half_bytes(flat)
        body = {"dtype": "fp16", "shape": list(flat.shape), "data_nbytes": len(blob),
            "scale_nbytes": 0}
        return name, blob, body
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    elif matrix.ndim > 2:
        matrix = matrix.reshape(matrix.shape[0], -1)
    blob = pack_matrix(matrix, "int8")
    scale = matrix.shape[0] * 4
    body = {
        "dtype": "int8",
        "shape": [int(matrix.shape[0]), int(matrix.shape[1])],
        "data_nbytes": len(blob) - scale,
        "scale_nbytes": scale,
    }
    return name, blob, body


def _keep_f16(name: str) -> bool:
    if name.endswith("mlp.gate.weight") or name.endswith("shared_expert_gate.weight"):
        return True
    if name.endswith(".bias") or name.endswith("A_log") or name.endswith("dt_bias"):
        return True
    if "norm" in name or name.endswith("conv1d.weight"):
        return True
    return False


def _vision(name: str) -> bool:
    if "language_model" in name or name.startswith("lm_head"):
        return False
    return True


def _expert_kind(name: str) -> str:
    if name.endswith("gate_up_proj") or name.endswith("gate_up_proj.weight"):
        return "gate_up"
    if name.endswith("down_proj") or name.endswith("down_proj.weight"):
        return "down"
    message = f"{name} is not a routed expert plane"
    raise RuntimeError(message)


def _layer(name: str) -> int:
    marker = ".mlp.experts."
    head = name.partition(marker)[0]
    return int(head.split(".")[-1])


def _shards(weight_map: dict[str, str], *, expert: bool) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    for name, shard in weight_map.items():
        is_expert = ".mlp.experts." in name and not name.startswith("mtp.")
        if is_expert != expert:
            continue
        grouped.setdefault(shard, []).append(name)
    return dict(sorted(grouped.items()))


def _header(path: Path) -> tuple[dict[str, Any], int]:
    with path.open("rb") as handle:
        size = int.from_bytes(handle.read(8), "little")
        header = json.loads(handle.read(size))
    return header, 8 + size


def _load(path: Path, header: dict[str, Any], base: int, name: str) -> np.ndarray:
    info = header[name]
    start, stop = info["data_offsets"]
    with path.open("rb") as handle:
        handle.seek(base + int(start))
        raw = handle.read(int(stop) - int(start))
    dtype = info["dtype"]
    shape = tuple(int(dim) for dim in info["shape"])
    if dtype == "BF16":
        return to_f32(np.frombuffer(raw, dtype=np.uint16)).reshape(shape)
    if dtype == "F16":
        return np.frombuffer(raw, dtype=np.float16).astype(np.float32).reshape(shape)
    if dtype == "F32":
        return np.frombuffer(raw, dtype=np.float32).reshape(shape)
    message = f"{name} dtype {dtype} is not packed"
    raise RuntimeError(message)


def _append(path: Path, blob: bytes) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as handle:
        end = handle.seek(0, 2)
        pad = (128 - (end % 128)) % 128
        if pad:
            handle.write(b"\x00" * pad)
            end += pad
        handle.write(blob)
    return end
