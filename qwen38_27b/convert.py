"""Stream a BF16 checkout into one uniform-fp4 record.

Every matrix is fp4 E2M1 with an E8M0 scale every 32 along K, including the
feed-forward, the embedding, and the output head. One-dimensional parameters
and the depthwise convolution stay float16. The feed-forward is not scored.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import numpy as np

from qwen38_27b.formats import pack_fp4_matrix, stored_rows

RECORD_NAME = "Qwen3.8-27B-madmetal-fp8-fp4"


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
    shape = [int(value) for value in info["shape"]]
    if dtype == "BF16":
        return _bf16(np.frombuffer(raw, dtype=np.uint16).copy(), shape)
    if dtype == "F16":
        return np.frombuffer(raw, dtype=np.float16).copy().reshape(shape).astype(np.float32)
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
    flat = np.ascontiguousarray(array, dtype=np.float16)
    body = {"dtype": "f16", "shape": list(array.shape), "stored_shape": list(flat.shape)}
    return flat.tobytes(), body


def _fp4_entry(array: np.ndarray) -> tuple[bytes, dict]:
    matrix = np.ascontiguousarray(array, dtype=np.float32)
    logical = list(matrix.shape)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    elif matrix.ndim > 2:
        matrix = matrix.reshape(matrix.shape[0], -1)
    rows = stored_rows(matrix.shape[0])
    data, scales, stored_r, stored_c = pack_fp4_matrix(matrix, rows=rows)
    pad = _align(len(data)) - len(data)
    blob = data + (b"\x00" * pad) + scales
    body = {
        "dtype": "fp4",
        "shape": logical,
        "stored_shape": [stored_r, stored_c],
        "data_nbytes": len(data),
        "scale_nbytes": len(scales),
        "scale_pad": pad,
    }
    return blob, body


def _keep_f16(name: str, array: np.ndarray) -> bool:
    if name.endswith("conv1d.weight") or name.endswith("A_log") or name.endswith("dt_bias"):
        return True
    if name.endswith(".bias") or array.ndim == 1:
        return True
    return False


def _weight_map(src: Path) -> dict[str, str]:
    index = src / "model.safetensors.index.json"
    if index.is_file():
        body = json.loads(index.read_text())
        mapping = body.get("weight_map")
        if isinstance(mapping, dict):
            return {str(name): str(shard) for name, shard in mapping.items()}
    singles = sorted(src.glob("*.safetensors"))
    if len(singles) == 1:
        header, _base = _header(singles[0])
        return {name: singles[0].name for name in header if name != "__metadata__"}
    message = f"{src.name} has no safetensors index"
    raise FileNotFoundError(message)


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
    weight_map = _weight_map(src)
    shards: dict[str, list[str]] = {}
    for name, shard in weight_map.items():
        shards.setdefault(shard, []).append(name)
    config = json.loads((src / "config.json").read_text())
    text = config.get("text_config") or config
    for shard in sorted(shards):
        path = src / shard
        header, base = _header(path)
        print(f"shard {shard}", flush=True)
        for name in shards[shard]:
            if name in done:
                continue
            array = _load(path, header, base, name)
            print(f"  {name}", flush=True)
            if _keep_f16(name, array):
                flat = array
                if name.endswith("conv1d.weight") and array.ndim >= 2:
                    flat = array.reshape(array.shape[0], -1)
                blob, body = _f16_entry(flat)
            else:
                blob, body = _fp4_entry(array)
            offset = _append(resident, blob)
            done[name] = {**body, "file": "resident.bin", "offset": offset}
            del array
            tmp = progress_path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps({"tensors": done}))
            os.replace(tmp, progress_path)
    manifest = {
        "complete": True,
        "pipeline": "qwen38-27b",
        "source": src.name,
        "tensors": done,
        "kv_cache": "e4m3",
        "weight": "fp4",
        "feed_forward": "fp4",
        "layers": int(text["num_hidden_layers"]),
        "hidden": int(text["hidden_size"]),
        "vocab": int(text["vocab_size"]),
        "layer_types": list(text["layer_types"]),
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
                shutil.copyfile(src_file, target)
    return output
