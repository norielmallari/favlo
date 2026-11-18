"""Pack the BF16 umT5 encoder as int8 and copy the tokenizer into that pack."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import zipfile
from pathlib import Path

import numpy as np

from umt5_xxl.metal41.formats import pack_int8, relative_error, restore_int8
from umt5_xxl.metal41.source import (
    TOKENIZER_EXTRAS,
    TOKENIZER_FILE,
    encoder_path,
    pack_dir,
)
from umt5_xxl.metal41.weights import FORMATS, open_weights

_LAYERS = 24
_MATRICES = (
    "attn.q.weight",
    "attn.k.weight",
    "attn.v.weight",
    "attn.o.weight",
    "ffn.gate.0.weight",
    "ffn.fc1.weight",
    "ffn.fc2.weight",
)
_PLANE_ALIGN = 128
_PLANE_FIELDS = ("packed", "scales", "bias")
_PLANES_BEFORE_POS = ("attn.q", "attn.k", "attn.v", "attn.o")
_PLANES_AFTER_NORM2 = ("ffn.gate.0", "ffn.fc1", "ffn.fc2")


def quantize(
    src: Path | None = None,
    dest: Path | None = None,
    *,
    encoder_format: str,
    tokenizer_src: Path | None = None,
    drop_source: bool = True,
) -> float:
    """Pack the bf16 encoder as int8. Returns the worst relative RMS.

    ``encoder_format`` is ``wan`` (zip ``.pth``) or ``hf`` (``.safetensors``).
    The tokenizer files beside the encoder, or under ``tokenizer_src``, are
    copied into ``dest``. When ``drop_source`` is true, the BF16 checkout is
    removed after a successful pack.
    """
    if encoder_format not in FORMATS:
        message = f"encoder format {encoder_format} is not wan or hf"
        raise ValueError(message)
    weights = Path(src) if src is not None else encoder_path()
    if not weights.is_file():
        message = f"umT5 encoder is missing: {weights}"
        raise FileNotFoundError(message)
    out = Path(dest) if dest is not None else pack_dir()
    out.mkdir(parents=True, exist_ok=True)
    source = tokenizer_src if tokenizer_src is not None else weights.parent
    reader = open_weights(encoder_format, weights)
    try:
        _write_embed(reader, out / "embed.npy")
        _write_half(reader, "norm.weight", out / "norm.npy")
        for layer in range(_LAYERS):
            prefix = f"blocks.{layer}"
            _write_half(reader, f"{prefix}.norm1.weight", out / f"{prefix}.norm1.npy")
            _write_half(reader, f"{prefix}.norm2.weight", out / f"{prefix}.norm2.npy")
            _write_half(
                reader,
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
                weight = reader.tensor(name)
                if weight.ndim != 2:
                    message = f"{name} is not a matrix"
                    raise RuntimeError(message)
                pack = pack_int8(weight)
                error = relative_error(restore_int8(pack), weight)
                _save(path, pack)
                del weight, pack
                print(f"int8 {name} {error:.6f}", flush=True)
                if error >= worst:
                    worst = error
                    worst_name = name
    finally:
        reader.close()
    _write_planes(out)
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
    from umt5_xxl.metal41.source import ENCODER_FILE

    if not source.is_dir():
        return
    if source.resolve() == pack.resolve():
        return
    encoder = source / ENCODER_FILE
    if encoder.is_file():
        shutil.rmtree(source)
        print(f"removed {source.name}", flush=True)
        return
    # Hugging Face checkouts keep a .safetensors file, not the Wan zip name.
    if any(path.suffix == ".safetensors" for path in source.iterdir() if path.is_file()):
        shutil.rmtree(source)
        print(f"removed {source.name}", flush=True)


def _write_half(reader, name: str, dest: Path) -> None:
    if dest.is_file():
        return
    np.save(dest, np.ascontiguousarray(reader.tensor(name), dtype=np.float16))


def _write_embed(reader, dest: Path) -> None:
    if dest.is_file():
        return
    rows, width = reader.embed_shape()
    out = np.lib.format.open_memmap(dest, mode="w+", dtype=np.float16, shape=(rows, width))
    step = 512
    try:
        for start in range(0, rows, step):
            count = min(step, rows - start)
            out[start : start + count] = reader.read_embed_rows(start, count)
    finally:
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


def _write_planes(dest: Path) -> None:
    """Write ``weights.bin`` and ``planes.json`` from the arrays already stored.

    The ``.npy`` and ``.npz`` files stay. Codes are copied. ``perm`` is not.
    """
    blob_fd, blob_name = tempfile.mkstemp(prefix=".weights.", suffix=".tmp", dir=dest)
    blob_path: Path | None = Path(blob_name)
    json_path: Path | None = None
    try:
        with os.fdopen(blob_fd, "wb") as handle:
            planes = _fill_planes(dest, handle)
            handle.flush()
            os.fsync(handle.fileno())
        json_fd, json_name = tempfile.mkstemp(prefix=".planes.", suffix=".tmp", dir=dest)
        json_path = Path(json_name)
        with os.fdopen(json_fd, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(planes, indent=2))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(blob_path, dest / "weights.bin")
        blob_path = None
        os.replace(json_path, dest / "planes.json")
        json_path = None
    finally:
        if blob_path is not None:
            blob_path.unlink(missing_ok=True)
        if json_path is not None:
            json_path.unlink(missing_ok=True)
    print(f"weights.bin {len(planes)}", flush=True)


def _fill_planes(dest: Path, handle) -> list[dict[str, object]]:
    planes: list[dict[str, object]] = []
    offset = _append_npy(handle, planes, 0, "embed", dest / "embed.npy")
    for layer in range(_LAYERS):
        prefix = f"blocks.{layer}"
        offset = _append_npy(
            handle, planes, offset, f"{prefix}.norm1", dest / f"{prefix}.norm1.npy"
        )
        for stem in _PLANES_BEFORE_POS:
            offset = _append_linear(handle, planes, offset, dest, f"{prefix}.{stem}")
        offset = _append_npy(handle, planes, offset, f"{prefix}.pos", dest / f"{prefix}.pos.npy")
        offset = _append_npy(
            handle, planes, offset, f"{prefix}.norm2", dest / f"{prefix}.norm2.npy"
        )
        for stem in _PLANES_AFTER_NORM2:
            offset = _append_linear(handle, planes, offset, dest, f"{prefix}.{stem}")
    _append_npy(handle, planes, offset, "norm", dest / "norm.npy")
    return planes


def _append_npy(handle, planes: list[dict[str, object]], offset: int, name: str, path: Path) -> int:
    if not path.is_file():
        message = f"umT5 plane is missing: {path.name}"
        raise FileNotFoundError(message)
    array = np.load(path, mmap_mode="r")
    try:
        return _append_array(handle, planes, offset, name, array)
    finally:
        del array


def _append_linear(
    handle, planes: list[dict[str, object]], offset: int, dest: Path, stem: str
) -> int:
    path = dest / f"{stem}.weight.npz"
    if not path.is_file():
        message = f"umT5 plane is missing: {path.name}"
        raise FileNotFoundError(message)
    with np.load(path) as pack:
        for field in _PLANE_FIELDS:
            if field not in pack.files:
                message = f"{path.name} is missing {field}"
                raise KeyError(message)
            array = np.ascontiguousarray(pack[field])
            offset = _append_array(handle, planes, offset, f"{stem}.weight.{field}", array)
            del array
    return offset


def _append_array(handle, planes: list[dict[str, object]], offset: int, name: str, array) -> int:
    if not array.flags.c_contiguous:
        array = np.ascontiguousarray(array)
    start = _aligned(offset)
    gap = start - offset
    if gap:
        handle.write(b"\x00" * gap)
    nbytes = int(array.nbytes)
    before = handle.tell()
    array.tofile(handle)
    written = handle.tell() - before
    if written != nbytes:
        message = f"{name} wrote {written} bytes, expected {nbytes}"
        raise RuntimeError(message)
    planes.append(
        {
            "name": name,
            "dtype": array.dtype.name,
            "offset": start,
            "nbytes": nbytes,
            "shape": [int(dim) for dim in array.shape],
        }
    )
    return start + nbytes


def _aligned(offset: int) -> int:
    mask = _PLANE_ALIGN - 1
    return (offset + mask) & ~mask


def _stored_kind(path: Path) -> str | None:
    if not path.is_file():
        return None
    try:
        return str(np.load(path)["kind"].item())
    except (OSError, ValueError, EOFError, zipfile.BadZipFile):
        path.unlink(missing_ok=True)
        return None
