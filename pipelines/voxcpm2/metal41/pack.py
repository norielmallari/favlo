"""Write one kernel-order blob from the published checkout."""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np

from voxcpm2.metal41.metal import pad_linear
from voxcpm2.metal41.store import (
    ALIGN,
    EMBED_NAME,
    PACK_NAME,
    PACK_VERSION,
    WEIGHTS_NAME,
    _bf16_f32,
)
from voxcpm2.metal41.vae import load_decoder

_ORDER = {"linear": 0, "gain": 1, "bias": 2, "embed": 3, "vector": 4}


def write_pack(src: Path, dest: Path | None = None) -> list[str]:
    """Write ``weights.bin`` and ``pack.json``. The safetensors file and the VAE zip stay."""
    target = src if dest is None else dest
    weights = src / "model.safetensors"
    vae = src / "audiovae.pth"
    if not weights.is_file():
        message = f"{src.name} has no model.safetensors"
        raise FileNotFoundError(message)
    if not vae.is_file():
        message = f"{src.name} has no audiovae.pth"
        raise FileNotFoundError(message)
    target.mkdir(parents=True, exist_ok=True)
    tmp_bin = target / f".{WEIGHTS_NAME}.new"
    tmp_meta = target / f".{PACK_NAME}.new"
    try:
        state = load_decoder(vae)
        records = _write_blob(weights, state, tmp_bin)
        body = {
            "version": PACK_VERSION,
            "align": ALIGN,
            "file": WEIGHTS_NAME,
            "tensors": records,
        }
        tmp_meta.write_text(json.dumps(body, indent=2) + "\n", encoding="utf-8")
        tmp_bin.replace(target / WEIGHTS_NAME)
        tmp_meta.replace(target / PACK_NAME)
    except Exception:
        tmp_bin.unlink(missing_ok=True)
        tmp_meta.unlink(missing_ok=True)
        raise
    return [WEIGHTS_NAME, PACK_NAME]


def _write_blob(
    weights: Path,
    state: dict[str, np.ndarray],
    dest: Path,
) -> dict[str, dict[str, object]]:
    with weights.open("rb") as src, dest.open("wb") as out:
        count = struct.unpack("<Q", _exact(src, 8))[0]
        header = json.loads(_exact(src, count))
        header.pop("__metadata__", None)
        origin = 8 + count
        ranked: list[tuple[str, str, dict[str, object], tuple[int, ...]]] = []
        for name, meta in header.items():
            shape = tuple(int(item) for item in meta["shape"])
            ranked.append((_kind(name, shape), name, meta, shape))
        ranked.sort(key=lambda item: (_ORDER[item[0]], item[1]))
        writer = _Blob(out)
        for kind, name, meta, shape in ranked:
            if meta.get("dtype") != "BF16":
                message = f"{name} is {meta.get('dtype')}"
                raise ValueError(message)
            start, end = (int(item) for item in meta["data_offsets"])
            src.seek(origin + start)
            values = _bf16_f32(_exact(src, end - start)).reshape(shape)
            writer.add(name, kind, values)
        for name in sorted(state):
            writer.add_vae(name, state[name])
        return writer.tensors


def _kind(name: str, shape: tuple[int, ...]) -> str:
    rank = len(shape)
    if name == EMBED_NAME:
        if rank != 2:
            message = f"{name} rank is {rank}"
            raise ValueError(message)
        return "embed"
    if name.endswith(".bias"):
        if rank != 1:
            message = f"{name} rank is {rank}"
            raise ValueError(message)
        return "bias"
    if name.endswith(".weight") and rank == 1:
        return "gain"
    if name.endswith(".weight") and rank == 2:
        return "linear"
    if name.endswith(".weight"):
        message = f"{name} rank {rank} is not a linear or a norm"
        raise ValueError(message)
    return "vector"


class _Blob:
    def __init__(self, handle: object) -> None:
        self._handle = handle
        self._cursor = 0
        self.tensors: dict[str, dict[str, object]] = {}

    def add(self, name: str, kind: str, values: np.ndarray) -> None:
        if kind == "linear":
            padded, _out, _inner, out_pad, inner_pad = pad_linear(values)
            payload = np.ascontiguousarray(padded).tobytes()
            self._write(
                name,
                payload,
                kind=kind,
                dtype="F16",
                shape=_ints(values.shape),
                pad=[out_pad, inner_pad],
            )
            return
        if kind == "gain":
            flat = np.ascontiguousarray(values.reshape(-1), dtype=np.float16)
            self._write(name, flat.tobytes(), kind=kind, dtype="F16", shape=_ints(flat.shape))
            return
        array = np.ascontiguousarray(values, dtype=np.float32)
        self._write(name, array.tobytes(), kind=kind, dtype="F32", shape=_ints(array.shape))

    def add_vae(self, name: str, values: np.ndarray) -> None:
        array = np.ascontiguousarray(values, dtype=np.float32)
        self._write(name, array.tobytes(), kind="vae", dtype="F32", shape=_ints(array.shape))

    def _write(self, name: str, payload: bytes, **fields: object) -> None:
        aligned = (self._cursor + ALIGN - 1) // ALIGN * ALIGN
        gap = aligned - self._cursor
        if gap:
            self._handle.write(b"\0" * gap)
        self._handle.write(payload)
        self.tensors[name] = {"offset": aligned, "nbytes": len(payload), **fields}
        self._cursor = aligned + len(payload)


def _ints(shape: object) -> list[int]:
    return [int(item) for item in shape]


def _exact(handle: object, count: int) -> bytes:
    raw = handle.read(count)
    if len(raw) != count:
        message = "the safetensors file ended early"
        raise ValueError(message)
    return raw
