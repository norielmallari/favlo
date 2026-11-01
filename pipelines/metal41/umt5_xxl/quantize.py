"""Pack the BF16 umT5 encoder as int8 and copy the tokenizer into that pack."""

from __future__ import annotations

import io
import json
import pickle
import shutil
import zipfile
from collections import OrderedDict
from pathlib import Path

import numpy as np

from umt5_xxl.formats import pack_int8, relative_error, restore_int8
from umt5_xxl.source import (
    ENCODER_FILE,
    TOKENIZER_EXTRAS,
    TOKENIZER_FILE,
    encoder_path,
    pack_dir,
)

_LAYERS = 24
_PREFIX = "umt5-xxl-enc-bf16/data"
_PICKLE = "umt5-xxl-enc-bf16/data.pkl"
_MATRICES = (
    "attn.q.weight",
    "attn.k.weight",
    "attn.v.weight",
    "attn.o.weight",
    "ffn.gate.0.weight",
    "ffn.fc1.weight",
    "ffn.fc2.weight",
)


def quantize(
    src: Path | None = None,
    dest: Path | None = None,
    *,
    tokenizer_src: Path | None = None,
    drop_source: bool = True,
) -> float:
    """Pack the bf16 encoder as int8. Returns the worst relative RMS.

    The tokenizer files beside the encoder are copied into ``dest``. When
    ``drop_source`` is true, the BF16 checkout is removed after a successful pack.
    """
    weights = Path(src) if src is not None else encoder_path()
    if not weights.is_file():
        message = f"umT5 encoder is missing: {weights}"
        raise FileNotFoundError(message)
    out = Path(dest) if dest is not None else pack_dir()
    out.mkdir(parents=True, exist_ok=True)
    source = tokenizer_src if tokenizer_src is not None else weights.parent
    archive = zipfile.ZipFile(weights)
    try:
        meta = _metadata(archive)
        _write_half(archive, meta, "token_embedding.weight", out / "embed.npy")
        _write_half(archive, meta, "norm.weight", out / "norm.npy")
        for layer in range(_LAYERS):
            prefix = f"blocks.{layer}"
            _write_half(archive, meta, f"{prefix}.norm1.weight", out / f"{prefix}.norm1.npy")
            _write_half(archive, meta, f"{prefix}.norm2.weight", out / f"{prefix}.norm2.npy")
            _write_half(
                archive,
                meta,
                f"{prefix}.pos_embedding.embedding.weight",
                out / f"{prefix}.pos.npy",
            )
        worst = 0.0
        worst_name = ""
        for layer in range(_LAYERS):
            for suffix in _MATRICES:
                name = f"blocks.{layer}.{suffix}"
                path = out / f"{name}.npz"
                if _stored_kind(path) == "int8":
                    continue
                weight = _matrix(archive, meta, name)
                pack = pack_int8(weight)
                error = relative_error(restore_int8(pack), weight)
                _save(path, pack)
                del weight, pack
                print(f"int8 {name} {error:.6f}", flush=True)
                if error >= worst:
                    worst = error
                    worst_name = name
    finally:
        archive.close()
    _copy_tokenizer(Path(source), out)
    (out / "manifest.json").write_text(
        json.dumps({"weight": "int8", "worst": worst, "worst_name": worst_name}),
        encoding="utf-8",
    )
    print(f"umt5 int8 worst {worst_name} {worst:.6f}", flush=True)
    if drop_source:
        _drop_source(weights.parent, out)
    return worst


def _copy_tokenizer(source: Path, dest: Path) -> None:
    required = source / TOKENIZER_FILE
    if not required.is_file():
        message = f"tokenizer is missing: {required}"
        raise FileNotFoundError(message)
    shutil.copy2(required, dest / TOKENIZER_FILE)
    print(f"tokenizer {TOKENIZER_FILE}", flush=True)
    for name in TOKENIZER_EXTRAS:
        path = source / name
        if path.is_file():
            shutil.copy2(path, dest / name)
            print(f"tokenizer {name}", flush=True)


def _drop_source(source: Path, pack: Path) -> None:
    if not source.is_dir():
        return
    if source.resolve() == pack.resolve():
        return
    encoder = source / ENCODER_FILE
    if not encoder.is_file():
        return
    shutil.rmtree(source)
    print(f"removed {source.name}", flush=True)


def _metadata(archive: zipfile.ZipFile) -> dict[str, tuple]:
    class _Unpickler(pickle.Unpickler):
        def find_class(self, module: str, name: str) -> object:
            if name == "OrderedDict":
                return OrderedDict
            if name == "_rebuild_tensor_v2":
                return _rebuild
            if name.endswith("Storage"):
                return type(name, (), {})
            message = f"unexpected pickle type {module}.{name}"
            raise RuntimeError(message)

        def persistent_load(self, pid: tuple) -> tuple:
            return (str(pid[2]), int(pid[4]))

    def _rebuild(
        storage: tuple,
        offset: int,
        size: tuple[int, ...],
        stride: tuple[int, ...],
        requires_grad: bool,
        hooks: object,
        metadata: object = None,
    ) -> tuple:
        del requires_grad, hooks, metadata
        return (storage[0], int(offset), tuple(int(v) for v in size), tuple(int(v) for v in stride))

    loaded = _Unpickler(io.BytesIO(archive.read(_PICKLE))).load()
    if not isinstance(loaded, dict):
        message = "umT5 checkpoint is not a tensor map"
        raise RuntimeError(message)
    return {str(name): value for name, value in loaded.items()}


def _float(
    storage: np.ndarray, offset: int, shape: tuple[int, ...], stride: tuple[int, ...]
) -> np.ndarray:
    view = np.lib.stride_tricks.as_strided(
        storage[offset:],
        shape=shape,
        strides=tuple(step * 2 for step in stride),
    )
    copied = np.ascontiguousarray(view)
    bits = copied.astype(np.uint32) << np.uint32(16)
    return bits.view(np.float32)


def _tensor(archive: zipfile.ZipFile, meta: dict[str, tuple], name: str) -> np.ndarray:
    key, offset, shape, stride = meta[name]
    raw = archive.read(f"{_PREFIX}/{key}")
    storage = np.frombuffer(raw, dtype="<u2")
    return np.ascontiguousarray(_float(storage, offset, shape, stride), dtype=np.float32)


def _matrix(archive: zipfile.ZipFile, meta: dict[str, tuple], name: str) -> np.ndarray:
    array = _tensor(archive, meta, name)
    if array.ndim != 2:
        message = f"{name} is not a matrix"
        raise RuntimeError(message)
    return array


def _write_half(
    archive: zipfile.ZipFile, meta: dict[str, tuple], name: str, dest: Path
) -> None:
    if dest.is_file():
        return
    if name == "token_embedding.weight":
        _write_embed(archive, meta, dest)
        return
    np.save(dest, np.ascontiguousarray(_tensor(archive, meta, name), dtype=np.float16))


def _write_embed(archive: zipfile.ZipFile, meta: dict[str, tuple], dest: Path) -> None:
    key, offset, shape, stride = meta["token_embedding.weight"]
    if tuple(stride) != (shape[1], 1):
        message = "token embedding is not row major"
        raise RuntimeError(message)
    rows, width = int(shape[0]), int(shape[1])
    out = np.lib.format.open_memmap(dest, mode="w+", dtype=np.float16, shape=(rows, width))
    member = archive.open(f"{_PREFIX}/{key}")
    try:
        step = 512
        for start in range(0, rows, step):
            count = min(step, rows - start)
            member.seek((int(offset) + start * width) * 2)
            raw = np.frombuffer(member.read(count * width * 2), dtype="<u2").copy()
            bits = raw.astype(np.uint32) << np.uint32(16)
            out[start : start + count] = bits.view(np.float32).reshape(count, width)
    finally:
        member.close()
        out.flush()


def _save(path: Path, pack) -> None:
    np.savez(
        path,
        kind=np.array(pack.kind),
        packed=pack.packed,
        scales=pack.scales,
        bias=pack.bias,
        perm=pack.perm,
        logical_n=np.int32(pack.logical_n),
        logical_k=np.int32(pack.logical_k),
        padded_n=np.int32(pack.padded_n),
        stored_k=np.int32(pack.stored_k),
    )


def _stored_kind(path: Path) -> str | None:
    if not path.is_file():
        return None
    try:
        return str(np.load(path)["kind"].item())
    except (OSError, ValueError, EOFError, zipfile.BadZipFile):
        path.unlink(missing_ok=True)
        return None
