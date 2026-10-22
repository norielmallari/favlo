"""Stream the BF16 checkout into the Metal record."""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from qwen36_35b_a3b.formats import (
    EXPERT_IMAGE,
    expert_bytes,
    expert_offsets,
    pack_fp4_matrix,
    pack_fp8_matrix,
    promote_count,
    promoted_indices,
    quantize_fp4,
    quantize_fp8,
    relative_error,
)

GATE_UP_BYTES = 1_114_112
DOWN_BYTES = 557_056
LANGUAGE_LAYERS = 40
LANGUAGE_EXPERTS = LANGUAGE_LAYERS * 256
MTP_EXPERTS = 256
SLOT_MAP_BYTES = LANGUAGE_EXPERTS
_POOL: ThreadPoolExecutor | None = None


def _worker_count() -> int:
    """One worker per core, capped so 256 expert images do not each allocate a full plane."""
    count = os.cpu_count() or 1
    return max(1, min(8, count))


def _pool() -> ThreadPoolExecutor:
    global _POOL
    if _POOL is None:
        _POOL = ThreadPoolExecutor(max_workers=_worker_count())
    return _POOL


def _pack_image(job: tuple[np.ndarray, int]) -> tuple[bytes, bytes]:
    matrix, width = job
    if width == 8:
        return quantize_fp8(matrix)
    return quantize_fp4(matrix)


def record_name(source: Path, percent: float = 0) -> str:
    """`<model>-fp4`, or `<model>-fp8-e<P>` when any plane is promoted.

    A hub snapshot uses the repository name, not the commit. A checkout
    directory ending in `-bf16` drops that suffix.
    """
    name = source.name
    for parent in (source, *source.parents):
        folder = parent.name
        if not folder.startswith("models--"):
            continue
        pieces = folder.removeprefix("models--").split("--")
        name = "--".join(pieces[1:]) if len(pieces) > 1 else pieces[0]
        break
    if name.endswith("-bf16"):
        name = name[: -len("-bf16")]
    if percent <= 0:
        return f"{name}-fp4"
    token = str(int(percent)) if float(percent) == int(percent) else f"{percent:g}"
    return f"{name}-fp8-e{token}"


def _align(value: int) -> int:
    return (value + 127) & ~127


def _bf16(raw: np.ndarray, shape: list[int]) -> np.ndarray:
    bits = np.frombuffer(raw, dtype=np.uint16).astype(np.uint32)
    return (bits << 16).view(np.float32).reshape(shape)


def _header(path: Path) -> tuple[dict, int]:
    with path.open("rb") as handle:
        size = int.from_bytes(handle.read(8), "little")
        header = json.loads(handle.read(size))
    return header, 8 + size


def _load(path: Path, header: dict, base: int, name: str) -> np.ndarray:
    info = header[name]
    start, stop = info["data_offsets"]
    with path.open("rb") as handle:
        handle.seek(base + start)
        raw = handle.read(stop - start)
    dtype = info["dtype"]
    shape = info["shape"]
    if dtype == "BF16":
        return _bf16(np.frombuffer(raw, dtype=np.uint16).copy(), shape)
    if dtype == "F32":
        return np.frombuffer(raw, dtype=np.float32).copy().reshape(shape)
    message = f"{name} dtype {dtype} is not packed"
    raise ValueError(message)


def _append(path: Path, blob: bytes) -> int:
    with path.open("ab") as handle:
        end = handle.seek(0, 2)
        pad = _align(end) - end
        if pad:
            handle.write(b"\x00" * pad)
            end += pad
        handle.write(blob)
    return end


def _f16_entry(array: np.ndarray) -> tuple[bytes, dict]:
    flat = np.ascontiguousarray(array, dtype=np.float16).reshape(-1)
    body = {
        "dtype": "f16",
        "shape": list(array.shape),
        "stored_shape": list(array.shape),
    }
    return flat.tobytes(), body


def _fp4_entry(
    array: np.ndarray, *, rows: int | None = None, inner: int | None = None
) -> tuple[bytes, dict]:
    matrix = np.ascontiguousarray(array, dtype=np.float32)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    data, scales, stored_rows, stored_inner = pack_fp4_matrix(matrix, rows=rows, inner=inner)
    pad = _align(len(data)) - len(data)
    blob = data + (b"\x00" * pad) + scales
    body = {
        "dtype": "fp4",
        "shape": list(array.shape),
        "stored_shape": [stored_rows, stored_inner],
        "data_nbytes": len(data),
        "scale_nbytes": len(scales),
        "scale_pad": pad,
    }
    return blob, body


def _fp8_entry(array: np.ndarray) -> tuple[bytes, dict]:
    matrix = np.ascontiguousarray(array, dtype=np.float32)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    elif matrix.ndim > 2:
        matrix = matrix.reshape(matrix.shape[0], -1)
    data, scales, stored_rows, stored_inner = pack_fp8_matrix(matrix)
    pad = _align(len(data)) - len(data)
    blob = data + (b"\x00" * pad) + scales
    body = {
        "dtype": "fp8",
        "shape": list(array.shape),
        "stored_shape": [stored_rows, stored_inner],
        "data_nbytes": len(data),
        "scale_nbytes": len(scales),
        "scale_pad": pad,
    }
    return blob, body


def _keep_f16(name: str) -> bool:
    if name.endswith("mlp.gate.weight") or name.endswith("shared_expert_gate.weight"):
        return True
    if name.endswith(".bias") or name.endswith("A_log") or name.endswith("dt_bias"):
        return True
    if "norm" in name or name.endswith("conv1d.weight"):
        return True
    return False


def _expert_file(root: Path, name: str) -> tuple[Path, int] | None:
    marker = ".mlp.experts."
    if marker not in name:
        return None
    if ".layers." not in name:
        return None
    head, _, _tail = name.partition(marker)
    layer = head.split(".")[-1]
    folder = "mtp" if name.startswith("mtp.") else layer
    return root / "experts" / f"{folder}.bin", int(layer) if folder != "mtp" else 0


def _routed_plane(name: str) -> tuple[str, int, str] | None:
    marker = ".mlp.experts."
    if marker not in name or ".layers." not in name:
        return None
    if name.endswith("gate_up_proj"):
        kind = "gate_up"
    elif name.endswith("down_proj"):
        kind = "down"
    else:
        return None
    pool = "mtp" if name.startswith("mtp.") else "language"
    layer = int(name.partition(marker)[0].split(".")[-1])
    return pool, layer, kind


def _replace_json(path: Path, body: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(body))
    os.replace(temporary, path)


def _same_budget(body: dict, percent: float) -> None:
    stored = body.get("expert_fp8_percent", 0)
    if float(stored) != float(percent):
        message = "record percentage does not match --expert-fp8"
        raise RuntimeError(message)


def _score_planes(
    src: Path, weight_map: dict[str, str]
) -> tuple[dict[tuple[int, int], dict[str, float]], dict[int, dict[str, float]]]:
    language: dict[tuple[int, int], dict[str, float]] = {}
    mtp: dict[int, dict[str, float]] = {}
    groups: dict[str, list[str]] = {}
    for name, shard in weight_map.items():
        groups.setdefault(shard, []).append(name)
    for shard, names in sorted(groups.items()):
        path = src / shard
        header, base = _header(path)
        for name in names:
            found = _routed_plane(name)
            if found is None:
                continue
            pool, layer, kind = found
            array = _load(path, header, base, name)
            if array.ndim < 2 or array.shape[0] != 256:
                message = f"{name} is not 256 routed experts"
                raise RuntimeError(message)
            print(f"  score {name}", flush=True)
            planes = [np.ascontiguousarray(array[expert]) for expert in range(256)]
            scored = _pool().map(relative_error, planes)
            for expert, score in enumerate(scored):
                if pool == "mtp":
                    mtp.setdefault(expert, {})[kind] = score
                else:
                    language.setdefault((layer, expert), {})[kind] = score
            del array
            del planes
    return language, mtp


def _flat_scores(parts: dict, width: int, *, language: bool) -> dict[int, float]:
    scores: dict[int, float] = {}
    for index in range(width):
        key: tuple[int, int] | int
        if language:
            layer, expert = divmod(index, 256)
            key = (layer, expert)
        else:
            key = index
        plane = parts.get(key, {})
        if set(plane) != {"gate_up", "down"}:
            message = f"expert {key} is missing a plane score"
            raise RuntimeError(message)
        scores[index] = max(plane.values())
    return scores


def _promoted_sets(
    src: Path,
    weight_map: dict[str, str],
    output: Path,
    percent: float,
) -> tuple[set[tuple[int, int]], set[int]]:
    language_count = promote_count(percent, LANGUAGE_EXPERTS)
    mtp_count = promote_count(percent, MTP_EXPERTS)
    if language_count == 0 and mtp_count == 0:
        return set(), set()
    path = output / "expert_error.json"
    progress = output / "quantize_progress.json"
    if path.is_file():
        body = json.loads(path.read_text())
        if float(body.get("percent", -1)) != float(percent):
            message = "expert_error.json does not match --expert-fp8"
            raise RuntimeError(message)
        return _sets_from(body, language_count, mtp_count)
    if progress.is_file():
        message = "expert_error.json is missing from an unfinished record"
        raise RuntimeError(message)
    language_parts, mtp_parts = _score_planes(src, weight_map)
    language_scores = _flat_scores(language_parts, LANGUAGE_EXPERTS, language=True)
    mtp_scores = _flat_scores(mtp_parts, MTP_EXPERTS, language=False)
    language_ids = promoted_indices(language_scores, language_count)
    mtp_ids = promoted_indices(mtp_scores, mtp_count)
    body = {
        "percent": percent,
        "language": {
            "pool": LANGUAGE_EXPERTS,
            "count": language_count,
            "promoted": [[index // 256, index % 256] for index in language_ids],
            "scores": {
                f"{index // 256}.{index % 256}": {
                    "gate_up": language_parts[(index // 256, index % 256)]["gate_up"],
                    "down": language_parts[(index // 256, index % 256)]["down"],
                    "expert": language_scores[index],
                }
                for index in range(LANGUAGE_EXPERTS)
            },
        },
        "mtp": {
            "pool": MTP_EXPERTS,
            "count": mtp_count,
            "promoted": mtp_ids,
            "scores": {
                str(index): {
                    "gate_up": mtp_parts[index]["gate_up"],
                    "down": mtp_parts[index]["down"],
                    "expert": mtp_scores[index],
                }
                for index in range(MTP_EXPERTS)
            },
        },
    }
    _replace_json(path, body)
    return _sets_from(body, language_count, mtp_count)


def _sets_from(
    body: dict, language_count: int, mtp_count: int
) -> tuple[set[tuple[int, int]], set[int]]:
    language = body["language"]
    mtp = body["mtp"]
    if int(language["count"]) != language_count or int(mtp["count"]) != mtp_count:
        message = "expert_error.json promote count does not match"
        raise RuntimeError(message)
    chosen = {(int(layer), int(expert)) for layer, expert in language["promoted"]}
    mtp_ids = {int(expert) for expert in mtp["promoted"]}
    if len(chosen) != language_count or len(mtp_ids) != mtp_count:
        message = "expert_error.json promote list does not match"
        raise RuntimeError(message)
    for layer, expert in chosen:
        if layer < 0 or layer >= LANGUAGE_LAYERS or expert < 0 or expert >= 256:
            message = f"promoted expert {layer}.{expert} is outside the language pool"
            raise RuntimeError(message)
    if any(expert < 0 or expert >= MTP_EXPERTS for expert in mtp_ids):
        message = "a promoted MTP expert is outside the pool"
        raise RuntimeError(message)
    return chosen, mtp_ids


def _expert_width(
    path: Path,
    expert: int,
    language: set[tuple[int, int]],
    mtp: set[int],
) -> int:
    if path.stem == "mtp":
        return 8 if expert in mtp else 4
    return 8 if (int(path.stem), expert) in language else 4


def _pack_plane(
    array: np.ndarray,
    kind: str,
    path: Path,
    language: set[tuple[int, int]],
    mtp: set[int],
) -> list[tuple[bytes, bytes]]:
    jobs = [
        (
            np.ascontiguousarray(array[expert], dtype=np.float32),
            _expert_width(path, expert, language, mtp),
        )
        for expert in range(array.shape[0])
    ]
    packed = list(_pool().map(_pack_image, jobs))
    if kind not in {"gate_up", "down"}:
        message = f"{kind} is not a routed expert plane"
        raise RuntimeError(message)
    return packed


def _write_layer(
    path: Path,
    gate: list[tuple[bytes, bytes]],
    down: list[tuple[bytes, bytes]],
    language: set[tuple[int, int]],
    mtp: set[int],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        for expert in range(len(gate)):
            blob = gate[expert][0] + gate[expert][1] + down[expert][0] + down[expert][1]
            expect = expert_bytes(_expert_width(path, expert, language, mtp))
            if len(blob) != expect:
                message = f"{path.name} expert {expert} is {len(blob)} bytes"
                raise RuntimeError(message)
            handle.write(blob)
    os.replace(temporary, path)


def _stage_expert(
    staged: dict[Path, dict[str, tuple[str, list[tuple[bytes, bytes]]]]],
    path: Path,
    name: str,
    array: np.ndarray,
    kind: str,
    output: Path,
    language: set[tuple[int, int]],
    mtp: set[int],
) -> dict[str, dict[str, str]] | None:
    """Hold one plane until its pair arrives, then write the layer in router order."""
    plane = _pack_plane(array, kind, path, language, mtp)
    entry = staged.setdefault(path, {})
    entry[kind] = (name, plane)
    if "gate_up" not in entry or "down" not in entry:
        return None
    gate_name, gate = entry.pop("gate_up")
    down_name, down = entry.pop("down")
    staged.pop(path, None)
    _write_layer(path, gate, down, language, mtp)
    relative = str(path.relative_to(output))
    widths = [_expert_width(path, expert, language, mtp) for expert in range(len(gate))]
    if any(width == 8 for width in widths) and any(width == 4 for width in widths):
        dtype = "mixed-expert"
    elif widths and widths[0] == 8:
        dtype = "fp8-expert"
    else:
        dtype = "fp4-expert"
    return {
        gate_name: {"dtype": dtype, "file": relative, "kind": "gate_up"},
        down_name: {"dtype": dtype, "file": relative, "kind": "down"},
    }


def _write_widths(
    output: Path, language: set[tuple[int, int]], mtp: set[int]
) -> None:
    widths = np.full((LANGUAGE_LAYERS, 256), 4, dtype=np.uint8)
    for layer, expert in language:
        widths[layer, expert] = 8
    offsets = expert_offsets(widths)
    folder = output / "experts"
    folder.mkdir(parents=True, exist_ok=True)
    _replace_bytes(folder / "widths.bin", widths.tobytes())
    _replace_bytes(folder / "offsets.bin", np.ascontiguousarray(offsets, dtype="<u8").tobytes())
    mtp_widths = np.full((1, MTP_EXPERTS), 4, dtype=np.uint8)
    for expert in mtp:
        mtp_widths[0, expert] = 8
    mtp_offsets = expert_offsets(mtp_widths)
    _replace_bytes(folder / "mtp.widths", mtp_widths.tobytes())
    _replace_bytes(folder / "mtp.offsets", np.ascontiguousarray(mtp_offsets, dtype="<u8").tobytes())
    for layer in range(LANGUAGE_LAYERS):
        path = folder / f"{layer}.bin"
        expect = int(offsets[layer, -1]) + expert_bytes(int(widths[layer, -1]))
        got = path.stat().st_size
        if got != expect:
            message = f"{path.name} is {got} bytes"
            raise RuntimeError(message)
    mtp_path = folder / "mtp.bin"
    expect = int(mtp_offsets[0, -1]) + expert_bytes(int(mtp_widths[0, -1]))
    if mtp_path.stat().st_size != expect:
        message = f"mtp.bin is {mtp_path.stat().st_size} bytes"
        raise RuntimeError(message)


def _replace_bytes(path: Path, blob: bytes) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(blob)
    os.replace(temporary, path)


def quantize(
    src: Path,
    output: Path,
    on_bytes: Callable[[int, str], None] | None = None,
    *,
    expert_fp8: float = 20.0,
) -> Path:
    """Pack ``src``. A finished manifest with the same percentage stays."""

    def _notify(detail: str) -> None:
        if on_bytes is None:
            return
        total = 0
        for path in output.rglob("*"):
            if path.is_file():
                total += path.stat().st_size
        on_bytes(total, detail)

    percent = float(expert_fp8)
    language_count = promote_count(percent, LANGUAGE_EXPERTS)
    mtp_count = promote_count(percent, MTP_EXPERTS)
    src = src.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    if manifest_path.is_file():
        body = json.loads(manifest_path.read_text())
        if body.get("complete"):
            _same_budget(body, percent)
            return output
    index = json.loads((src / "model.safetensors.index.json").read_text())
    weight_map: dict[str, str] = index["weight_map"]
    language, mtp = _promoted_sets(src, weight_map, output, percent)
    dense_fp8 = percent > 0
    progress_path = output / "quantize_progress.json"
    done: dict[str, dict] = {}
    if progress_path.is_file():
        done = json.loads(progress_path.read_text()).get("tensors", {})
    resident = output / "resident.bin"
    shards: dict[str, list[str]] = {}
    for name, shard in weight_map.items():
        shards.setdefault(shard, []).append(name)
    pending: dict[str, dict[str, np.ndarray]] = {}
    staged: dict[Path, dict[str, tuple[str, list[tuple[bytes, bytes]]]]] = {}
    config = json.loads((src / "config.json").read_text())
    _notify(output.name)
    for shard in sorted(shards):
        path = src / shard
        header, base = _header(path)
        if on_bytes is None:
            print(f"shard {shard}", flush=True)
        for name in shards[shard]:
            if name in done:
                continue
            array = _load(path, header, base, name)
            expert = _expert_file(output, name)
            if expert is not None:
                kind = "gate_up" if name.endswith("gate_up_proj") else "down"
                if on_bytes is None:
                    print(f"  {name}", flush=True)
                committed = _stage_expert(
                    staged, expert[0], name, array, kind, output, language, mtp
                )
                if committed is None:
                    del array
                    continue
                done.update(committed)
            elif name.endswith("shared_expert.gate_proj.weight") or name.endswith(
                "shared_expert.up_proj.weight"
            ):
                key = name.rsplit(".", 2)[0]
                slot = pending.setdefault(key, {})
                slot["gate" if name.endswith("gate_proj.weight") else "up"] = array
                if "gate" in slot and "up" in slot:
                    fused = np.concatenate([slot.pop("gate"), slot.pop("up")], axis=0)
                    blob, body = _fp8_entry(fused) if dense_fp8 else _fp4_entry(fused)
                    offset = _append(resident, blob)
                    done[key + ".gate_up"] = {**body, "file": "resident.bin", "offset": offset}
                    pending.pop(key, None)
                else:
                    continue
            elif _keep_f16(name):
                flat = array
                if name.endswith("conv1d.weight"):
                    flat = array.reshape(array.shape[0], array.shape[-1])
                blob, body = _f16_entry(flat)
                offset = _append(resident, blob)
                done[name] = {**body, "file": "resident.bin", "offset": offset}
            else:
                matrix = array
                if matrix.ndim > 2:
                    matrix = matrix.reshape(matrix.shape[0], -1)
                blob, body = _fp8_entry(matrix) if dense_fp8 else _fp4_entry(matrix)
                offset = _append(resident, blob)
                done[name] = {
                    **body,
                    "file": "resident.bin",
                    "offset": offset,
                    "logical_shape": list(array.shape),
                }
            del array
            tmp = progress_path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps({"tensors": done}))
            os.replace(tmp, progress_path)
            _notify(name)
    if pending:
        message = f"shared expert pair is incomplete: {sorted(pending)}"
        raise RuntimeError(message)
    if staged:
        message = f"routed expert pair is incomplete: {sorted(path.name for path in staged)}"
        raise RuntimeError(message)
    if language or mtp:
        _write_widths(output, language, mtp)
    manifest = {
        "complete": True,
        "pipeline": "qwen36-35b-a3b",
        "source": src.name,
        "expert_order": "router",
        "expert_fp8_percent": percent,
        "tensors": done,
        "expert_image": EXPERT_IMAGE,
        "gate_up_bytes": GATE_UP_BYTES,
        "down_bytes": DOWN_BYTES,
        "layers": 40,
        "experts": 256,
        "vocab": int((config.get("text_config") or config)["vocab_size"]),
        "hidden": 2048,
    }
    if language or mtp:
        manifest["expert_fp8"] = language_count
        manifest["expert_fp4"] = LANGUAGE_EXPERTS - language_count
        manifest["mtp_expert_fp8"] = mtp_count
        manifest["mtp_expert_fp4"] = MTP_EXPERTS - mtp_count
        manifest["expert_widths"] = "experts/widths.bin"
        manifest["expert_offsets"] = "experts/offsets.bin"
    tmp = manifest_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(manifest))
    os.replace(tmp, manifest_path)
    for extra in (
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "config.json",
        "generation_config.json",
    ):
        src_file = src / extra
        if src_file.is_file():
            target = output / extra
            if not target.exists():
                target.write_bytes(src_file.read_bytes())
    return output
