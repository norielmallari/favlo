"""Pack ``google/gemma-4-26B-A4B-it``. Dense matrices are int8. Experts are int4."""

from __future__ import annotations

import json
import os
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from gemma4_26b_a4b.cuda.formats import pack_expert, pack_matrix, store_f16
from gemma4_26b_a4b.cuda.layout import (
    EXPERTS,
    FOLDER,
    GLOBAL_LAYERS,
    HIDDEN,
    LAYERS,
    REPO,
    expert_image_bytes,
)

_SIDECARS = (
    "config.json",
    "generation_config.json",
    "chat_template.jinja",
    "tokenizer.json",
    "tokenizer_config.json",
    "processor_config.json",
)
_HALF_SUFFIX = (
    "input_layernorm.weight",
    "post_attention_layernorm.weight",
    "pre_feedforward_layernorm.weight",
    "pre_feedforward_layernorm_2.weight",
    "post_feedforward_layernorm.weight",
    "post_feedforward_layernorm_1.weight",
    "post_feedforward_layernorm_2.weight",
    "self_attn.q_norm.weight",
    "self_attn.k_norm.weight",
    "router.proj.weight",
    "router.scale",
    "router.per_expert_scale",
)


def record_name() -> str:
    return FOLDER


def _silence() -> None:
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"


def _header(path: Path) -> tuple[dict[str, Any], int]:
    with path.open("rb") as handle:
        size = int.from_bytes(handle.read(8), "little")
        header = json.loads(handle.read(size))
    return header, 8 + size


def _bf16(raw: bytes, shape: tuple[int, ...]) -> np.ndarray:
    bits = np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << np.uint32(16)
    return bits.view(np.float32).reshape(shape).copy()


def _index(root: Path) -> dict[str, str]:
    path = root / "model.safetensors.index.json"
    if path.is_file():
        body = json.loads(path.read_text(encoding="utf-8"))
        return dict(body["weight_map"])
    names: dict[str, str] = {}
    for shard in sorted(root.glob("*.safetensors")):
        header, _base = _header(shard)
        for name in header:
            if name != "__metadata__":
                names[name] = shard.name
    return names


def _load(root: Path, names: dict[str, str], name: str) -> np.ndarray:
    shard = root / names[name]
    header, base = _header(shard)
    info = header[name]
    start, stop = info["data_offsets"]
    shape = tuple(int(dim) for dim in info["shape"])
    with shard.open("rb") as handle:
        handle.seek(base + int(start))
        raw = handle.read(int(stop) - int(start))
    dtype = info["dtype"]
    if dtype == "BF16":
        return _bf16(raw, shape)
    if dtype == "F16":
        return np.frombuffer(raw, dtype=np.float16).astype(np.float32).reshape(shape).copy()
    if dtype == "F32":
        return np.frombuffer(raw, dtype=np.float32).reshape(shape).copy()
    message = f"{name} dtype {dtype} is not packed"
    raise RuntimeError(message)


def _pack_layer(task: tuple[str, str, str, int, bytes]) -> bytes:
    _silence()
    root, gate_file, down_file, layer, widths = task
    source = Path(root)
    gate_name = f"model.language_model.layers.{layer}.experts.gate_up_proj"
    down_name = f"model.language_model.layers.{layer}.experts.down_proj"
    mapping = {gate_name: gate_file, down_name: down_file}
    gate = _load(source, mapping, gate_name)
    down = _load(source, mapping, down_name)
    marks = np.frombuffer(widths, dtype=np.uint8)
    pieces: list[bytes] = []
    for expert in range(EXPERTS):
        plane = "int8" if int(marks[layer * EXPERTS + expert]) == 8 else "int4"
        blob = pack_expert(gate[expert], down[expert], plane)
        expect = expert_image_bytes(plane)
        if len(blob) != expect:
            message = f"layer {layer} expert {expert} packed {len(blob)} bytes"
            raise RuntimeError(message)
        pieces.append(blob)
    return b"".join(pieces)


def _workers() -> int:
    """One process per CPU, unless a layer's float32 copy would make the host swap."""
    cpus = os.process_cpu_count() or os.cpu_count() or 1
    # Gate-up is about 2.0 GiB and down about 1.0 GiB after the bf16 expand.
    layer_bytes = 3_400_000_000
    try:
        from favlo.host import available_bytes

        free = available_bytes()
    except OSError:
        return cpus
    fit = max(1, (free - (2 << 30)) // layer_bytes)
    return max(1, min(int(cpus), int(fit)))


def quantize(src: Path, dest: Path) -> None:
    """Write the int8 dense record and the int4 expert atlas under ``dest``."""
    names = _index(src)
    dest.mkdir(parents=True, exist_ok=True)
    for sidecar in _SIDECARS:
        found = src / sidecar
        if found.is_file():
            shutil.copyfile(found, dest / sidecar)
    workers = _workers()
    print(f"quantize workers {workers}", flush=True)
    gate_key = "model.language_model.layers.0.experts.gate_up_proj"
    if gate_key not in names:
        message = f"{src} has no Gemma 4 expert tensor"
        raise FileNotFoundError(message)
    promoted = _choose(src, names, dest, workers)
    _write_experts(src, names, dest, promoted, workers)
    _write_resident(src, names, dest)
    manifest = {
        "model": REPO,
        "runtime": "cuda",
        "weights": "int8",
        "experts": "int4",
        "expert_int8_percent": 0,
        "hot_path": "fp16",
        "down_k": 768,
    }
    (dest / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def _choose(src: Path, names: dict[str, str], dest: Path, workers: int) -> set[tuple[int, int]]:
    del src, names, workers
    path = dest / "expert_error.json"
    body = {
        "percent": 0,
        "pool": LAYERS * EXPERTS,
        "count": 0,
        "promoted": [],
    }
    path.write_text(json.dumps(body) + "\n", encoding="utf-8")
    return set()


def _write_experts(
    src: Path,
    names: dict[str, str],
    dest: Path,
    chosen: set[tuple[int, int]],
    workers: int,
) -> None:
    folder = dest / "experts"
    folder.mkdir(exist_ok=True)
    widths = np.zeros(LAYERS * EXPERTS, dtype=np.uint8)
    for layer in range(LAYERS):
        for expert in range(EXPERTS):
            widths[layer * EXPERTS + expert] = 8 if (layer, expert) in chosen else 4
    (folder / "widths.bin").write_bytes(widths.tobytes())
    tasks = []
    for layer in range(LAYERS):
        path = folder / f"{layer}.bin"
        expect = 0
        for expert in range(EXPERTS):
            plane = "int8" if int(widths[layer * EXPERTS + expert]) == 8 else "int4"
            expect += expert_image_bytes(plane)
        if path.is_file() and path.stat().st_size == expect:
            print(f"packed experts {layer}", flush=True)
            continue
        gate = f"model.language_model.layers.{layer}.experts.gate_up_proj"
        down = f"model.language_model.layers.{layer}.experts.down_proj"
        tasks.append((str(src), names[gate], names[down], layer, widths.tobytes()))
    if not tasks:
        return
    with ProcessPoolExecutor(max_workers=workers, initializer=_silence) as pool:
        for task, blob in zip(tasks, pool.map(_pack_layer, tasks), strict=True):
            layer = int(task[3])
            (folder / f"{layer}.bin").write_bytes(blob)
            print(f"packed experts {layer}", flush=True)


def _write_resident(src: Path, names: dict[str, str], dest: Path) -> None:
    resident = dest / "resident.bin"
    index_path = dest / "resident.json"
    if resident.is_file() and index_path.is_file():
        print(f"packed resident {resident.stat().st_size}", flush=True)
        return
    pieces: list[bytes] = []
    index: dict[str, dict[str, Any]] = {}
    cursor = 0

    def add(name: str, blob: bytes, shape: list[int], kind: str) -> None:
        nonlocal cursor
        pad = (16 - (cursor & 15)) & 15
        if pad:
            pieces.append(b"\x00" * pad)
            cursor += pad
        index[name] = {"offset": cursor, "nbytes": len(blob), "shape": shape, "kind": kind}
        pieces.append(blob)
        cursor += len(blob)

    embed = _load(src, names, "model.language_model.embed_tokens.weight")
    add("embed", pack_matrix(embed, "int8"), [int(embed.shape[0]), HIDDEN], "int8")
    final = _load(src, names, "model.language_model.norm.weight")
    add("final_norm", store_f16(final), [HIDDEN], "fp16")
    for layer in range(LAYERS):
        base = f"model.language_model.layers.{layer}"
        for suffix in _HALF_SUFFIX:
            tensor = f"{base}.{suffix}"
            if tensor not in names:
                continue
            values = _load(src, names, tensor).reshape(-1)
            add(tensor, store_f16(values), [int(values.shape[0])], "fp16")
        global_layer = layer in GLOBAL_LAYERS
        projections = ["self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.o_proj.weight"]
        if not global_layer:
            projections.insert(2, "self_attn.v_proj.weight")
        projections.extend(["mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight"])
        for suffix in projections:
            tensor = f"{base}.{suffix}"
            matrix = _load(src, names, tensor)
            add(tensor, pack_matrix(matrix, "int8"), [int(matrix.shape[0]), int(matrix.shape[1])], "int8")
        scalar = _load(src, names, f"{base}.layer_scalar").reshape(-1)
        add(f"{base}.layer_scalar", store_f16(scalar), [1], "fp16")
    blob = b"".join(pieces)
    (dest / "resident.bin").write_bytes(blob)
    (dest / "resident.json").write_text(json.dumps(index) + "\n", encoding="utf-8")
    print(f"packed resident {len(blob)}", flush=True)
