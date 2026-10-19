"""Stream the BF16 checkout into the fp4 record."""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from pathlib import Path

import numpy as np

from qwen36_35b_a3b.formats import EXPERT_IMAGE, pack_fp4_matrix, quantize_fp4

GATE_UP_BYTES = 1_114_112
DOWN_BYTES = 557_056
LANGUAGE_LAYERS = 40
SLOT_MAP_BYTES = LANGUAGE_LAYERS * 256


def record_name(source: Path) -> str:
    """`<model>-fp4`. A hub snapshot uses the repository name, not the commit."""
    name = source.name
    for parent in (source, *source.parents):
        folder = parent.name
        if not folder.startswith("models--"):
            continue
        pieces = folder.removeprefix("models--").split("--")
        name = "--".join(pieces[1:]) if len(pieces) > 1 else pieces[0]
        break
    return f"{name}-fp4"


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


def expert_slot_map(gate_scales: list[bytes], down_scales: list[bytes]) -> np.ndarray:
    """Router row to file slot. The sort key is the median E8M0 byte of the image.

    Ties keep the lower router row in the earlier slot. The nibbles and scale
    bytes move with the image and are not recomputed.
    """
    count = len(gate_scales)
    if count != len(down_scales) or count < 1 or count > 256:
        message = f"expert planes {count} and {len(down_scales)} are not one layer"
        raise ValueError(message)
    ranked: list[tuple[float, int]] = []
    for index in range(count):
        scales = np.frombuffer(gate_scales[index] + down_scales[index], dtype=np.uint8)
        ranked.append((float(np.median(scales)), index))
    order = sorted(range(count), key=lambda index: ranked[index])
    mapping = np.empty(count, dtype=np.uint8)
    for slot, expert in enumerate(order):
        mapping[expert] = np.uint8(slot)
    return mapping


def _pack_plane(array: np.ndarray, kind: str) -> list[tuple[bytes, bytes]]:
    expect = GATE_UP_BYTES if kind == "gate_up" else DOWN_BYTES
    packed: list[tuple[bytes, bytes]] = []
    for expert in range(array.shape[0]):
        matrix = np.ascontiguousarray(array[expert], dtype=np.float32)
        data, scales = quantize_fp4(matrix)
        if len(data) + len(scales) != expect:
            message = f"{kind} image {len(data) + len(scales)} != {expect}"
            raise RuntimeError(message)
        packed.append((data, scales))
    return packed


def _write_ordered_layer(
    path: Path,
    gate: list[tuple[bytes, bytes]],
    down: list[tuple[bytes, bytes]],
    mapping: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    order = sorted(range(mapping.size), key=lambda expert: int(mapping[expert]))
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as handle:
        for expert in order:
            handle.write(gate[expert][0])
            handle.write(gate[expert][1])
            handle.write(down[expert][0])
            handle.write(down[expert][1])
    os.replace(tmp, path)


def _store_slot_map(output: Path, path: Path, mapping: np.ndarray) -> None:
    if path.stem == "mtp":
        target = path.with_suffix(".order")
        tmp = target.with_suffix(".order.tmp")
        tmp.write_bytes(mapping.tobytes())
        os.replace(tmp, target)
        return
    layer = int(path.stem)
    if layer < 0 or layer >= LANGUAGE_LAYERS or mapping.size != 256:
        message = f"{path.name} is not a language expert file"
        raise RuntimeError(message)
    order_path = output / "experts" / "order.bin"
    if not order_path.exists() or order_path.stat().st_size != SLOT_MAP_BYTES:
        order_path.write_bytes(b"\x00" * SLOT_MAP_BYTES)
    blob = bytearray(order_path.read_bytes())
    start = layer * 256
    blob[start : start + 256] = mapping.tobytes()
    tmp = order_path.with_suffix(".bin.tmp")
    tmp.write_bytes(blob)
    os.replace(tmp, order_path)


def _stage_expert(
    staged: dict[Path, dict[str, tuple[str, list[tuple[bytes, bytes]]]]],
    path: Path,
    name: str,
    array: np.ndarray,
    kind: str,
    output: Path,
) -> dict[str, dict[str, str]] | None:
    """Hold one plane until its pair arrives, then write the ordered layer."""
    plane = _pack_plane(array, kind)
    entry = staged.setdefault(path, {})
    entry[kind] = (name, plane)
    if "gate_up" not in entry or "down" not in entry:
        return None
    gate_name, gate = entry.pop("gate_up")
    down_name, down = entry.pop("down")
    staged.pop(path, None)
    mapping = expert_slot_map(
        [scales for _data, scales in gate],
        [scales for _data, scales in down],
    )
    if sorted(int(slot) for slot in mapping) != list(range(mapping.size)):
        message = f"{path.name} expert map is not a permutation"
        raise RuntimeError(message)
    _write_ordered_layer(path, gate, down, mapping)
    _store_slot_map(output, path, mapping)
    relative = str(path.relative_to(output))
    return {
        gate_name: {"dtype": "fp4-expert", "file": relative, "kind": "gate_up"},
        down_name: {"dtype": "fp4-expert", "file": relative, "kind": "down"},
    }


def quantize(
    src: Path,
    output: Path,
    on_bytes: Callable[[int, str], None] | None = None,
) -> Path:
    """Pack ``src`` into ``output``. A finished manifest is left in place."""

    def _notify(detail: str) -> None:
        if on_bytes is None:
            return
        total = 0
        for path in output.rglob("*"):
            if path.is_file():
                total += path.stat().st_size
        on_bytes(total, detail)

    src = src.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    if manifest_path.is_file():
        body = json.loads(manifest_path.read_text())
        if body.get("complete"):
            return output
    progress_path = output / "quantize_progress.json"
    done: dict[str, dict] = {}
    if progress_path.is_file():
        done = json.loads(progress_path.read_text()).get("tensors", {})
    resident = output / "resident.bin"
    index = json.loads((src / "model.safetensors.index.json").read_text())
    weight_map: dict[str, str] = index["weight_map"]
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
                committed = _stage_expert(staged, expert[0], name, array, kind, output)
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
                    blob, body = _fp4_entry(fused)
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
                blob, body = _fp4_entry(matrix)
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
    order_path = output / "experts" / "order.bin"
    order = np.frombuffer(order_path.read_bytes(), dtype=np.uint8)
    if order.size != SLOT_MAP_BYTES:
        message = f"expert slot map is {order.size} bytes"
        raise RuntimeError(message)
    for layer in range(LANGUAGE_LAYERS):
        mapping = order[layer * 256 : (layer + 1) * 256]
        if sorted(int(slot) for slot in mapping) != list(range(256)):
            message = f"layer {layer} expert map is not a permutation"
            raise RuntimeError(message)
    manifest = {
        "complete": True,
        "pipeline": "qwen36-35b-a3b",
        "source": src.name,
        "expert_order": "median-e8m0",
        "expert_slots": "experts/order.bin",
        "tensors": done,
        "expert_image": EXPERT_IMAGE,
        "gate_up_bytes": GATE_UP_BYTES,
        "down_bytes": DOWN_BYTES,
        "layers": 40,
        "experts": 256,
        "vocab": int((config.get("text_config") or config)["vocab_size"]),
        "hidden": 2048,
    }
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
