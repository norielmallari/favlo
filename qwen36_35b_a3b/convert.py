"""Stream the BF16 checkout into the fp4 record."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from qwen36_35b_a3b.formats import EXPERT_IMAGE, pack_fp4_matrix, quantize_fp4

GATE_UP_BYTES = 1_114_112
DOWN_BYTES = 557_056


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


def _write_experts(path: Path, array: np.ndarray, kind: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() or path.stat().st_size != 256 * EXPERT_IMAGE:
        with path.open("wb") as handle:
            handle.truncate(256 * EXPERT_IMAGE)
    experts = array.shape[0]
    with path.open("r+b") as handle:
        for expert in range(experts):
            matrix = np.ascontiguousarray(array[expert], dtype=np.float32)
            data, scales = quantize_fp4(matrix)
            packed = len(data) + len(scales)
            expect = GATE_UP_BYTES if kind == "gate_up" else DOWN_BYTES
            if packed != expect:
                message = f"{kind} image {packed} != {expect}"
                raise RuntimeError(message)
            base = expert * EXPERT_IMAGE + (0 if kind == "gate_up" else GATE_UP_BYTES)
            handle.seek(base)
            handle.write(data)
            handle.write(scales)


def quantize(src: Path, output: Path) -> Path:
    """Pack ``src`` into ``output``. A finished manifest is left in place."""
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
    config = json.loads((src / "config.json").read_text())
    for shard in sorted(shards):
        path = src / shard
        header, base = _header(path)
        print(f"shard {shard}", flush=True)
        for name in shards[shard]:
            if name in done:
                continue
            array = _load(path, header, base, name)
            expert = _expert_file(output, name)
            if expert is not None:
                kind = "gate_up" if name.endswith("gate_up_proj") else "down"
                print(f"  {name}", flush=True)
                _write_experts(expert[0], array, kind)
                done[name] = {
                    "dtype": "fp4-expert",
                    "file": str(expert[0].relative_to(output)),
                    "kind": kind,
                }
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
    if pending:
        message = f"shared expert pair is incomplete: {sorted(pending)}"
        raise RuntimeError(message)
    manifest = {
        "complete": True,
        "pipeline": "qwen36-35b-a3b",
        "source": src.name,
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
