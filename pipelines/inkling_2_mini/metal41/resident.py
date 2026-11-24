"""Expert planes baked for a sequential SSD read.

Layers 0–19 are NVFP4 in the checkpoint. ``bake`` nearest-rounds them to the
Metal fp4 the kernels read when the pack is missing. ``calibrate`` writes
that pack with GPTQ-FOEM instead. Layers 20–23 are copied as stored MXFP8.
``publish`` copies that payload into ``experts.bin``. The open preads
``experts.bin`` and the attention planes in ``resident.bin`` in 8 MiB pieces.
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import math
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

from inkling_2_mini.metal41.gptq import config_digest
from inkling_2_mini.metal41.layout import (
    EXPERTS,
    HIDDEN,
    INTERMEDIATE,
    LAYERS,
    NVFP4_LAYERS,
    OUTER_SCALE_DENOMINATOR,
    stored_k,
)

EXPERT_PACK = "experts-fp4.safetensors"
EXPERT_FORMAT = "favlo-fp4-expert-1"
_PIECE = 8 << 20
_LIBC = ctypes.CDLL("/usr/lib/libSystem.dylib", use_errno=True)
_LIBC.pread.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int64]
_LIBC.pread.restype = ctypes.c_ssize_t

Note = Callable[[int], None]


class Spec:
    """One uint8 tensor in the expert pack."""

    def __init__(
        self,
        name: str,
        shape: tuple[int, ...],
        *,
        layer: int,
        proj: str,
        kind: str,
        nvfp4: bool,
    ) -> None:
        self.name = name
        self.shape = shape
        self.layer = layer
        self.proj = proj
        self.kind = kind
        self.nvfp4 = nvfp4
        self.nbytes = math.prod(shape)


def catalog() -> tuple[Spec, ...]:
    """Metal fp4 for layers 0–19, MXFP8 as stored for layers 20–23."""
    specs: list[Spec] = []
    for index in range(LAYERS):
        nvfp4 = index < NVFP4_LAYERS
        for proj, rows, cols in (("w13", INTERMEDIATE * 2, HIDDEN), ("w2", HIDDEN, INTERMEDIATE)):
            if nvfp4:
                wide = stored_k(cols)
                data = (EXPERTS, rows, wide // 2)
                scale = (EXPERTS, rows, wide // 32)
            else:
                data = (EXPERTS, rows, cols)
                scale = (EXPERTS, rows, cols // 32)
            for kind, shape in (("data", data), ("scale", scale)):
                specs.append(
                    Spec(
                        f"layers.{index}.{proj}.{kind}",
                        shape,
                        layer=index,
                        proj=proj,
                        kind=kind,
                        nvfp4=nvfp4,
                    )
                )
    return tuple(specs)


def expert_pack_bytes() -> int:
    """Bytes of the expert pack, including a small header slack."""
    return sum(spec.nbytes for spec in catalog()) + (1 << 20)


def baked_shapes(layer: int, proj: str) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """``(data shape, scale shape)`` for one projection in the pack."""
    found = [spec for spec in catalog() if spec.layer == layer and spec.proj == proj]
    data = next(spec.shape for spec in found if spec.kind == "data")
    scale = next(spec.shape for spec in found if spec.kind == "scale")
    return data, scale


def pread_view(fd: int, view: memoryview, file_off: int, note: Note | None = None) -> None:
    """Fill ``view`` from ``file_off``. A long view is split into 8 MiB pieces."""
    offset = 0
    total = len(view)
    while offset < total:
        step = min(_PIECE, total - offset)
        chunk = view[offset : offset + step]
        holder = ctypes.c_char.from_buffer(chunk)
        try:
            address = ctypes.addressof(holder)
            got = 0
            while got < step:
                rc = _LIBC.pread(
                    ctypes.c_int(fd),
                    ctypes.c_void_p(address + got),
                    ctypes.c_size_t(step - got),
                    ctypes.c_int64(file_off + offset + got),
                )
                if rc < 0:
                    err = ctypes.get_errno()
                    raise OSError(err, os.strerror(err))
                if rc == 0:
                    message = f"read stopped at {file_off + offset + got}"
                    raise OSError(message)
                got += int(rc)
        finally:
            del holder
        offset += step
        if note is not None:
            note(step)


def pread_handle(
    fd: int,
    handle: Any,
    file_off: int,
    nbytes: int,
    *,
    dest_off: int = 0,
    note: Note | None = None,
) -> None:
    """Read ``nbytes`` from ``file_off`` into a shared device span."""
    from favlo.memory import memory

    view = memory.map(handle, int(dest_off), int(nbytes))
    try:
        pread_view(fd, view, int(file_off), note)
    finally:
        memory.unmap(handle)


class ExpertFile:
    """The baked expert pack. The header is read. The payload stays on disk."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._fd = os.open(self.path, os.O_RDONLY)
        try:
            size = int.from_bytes(os.pread(self._fd, 8, 0), "little")
            raw = os.pread(self._fd, size, 8)
            if len(raw) != size:
                message = f"{self.path.name} header is short"
                raise ValueError(message)
            header = json.loads(raw)
            if not isinstance(header, dict):
                message = f"{self.path.name} header is not an object"
                raise ValueError(message)
            self.metadata = header.pop("__metadata__", None) or {}
            self.header = header
            self.header_digest = hashlib.sha256(raw).hexdigest()
            self._data = 8 + size
        except Exception:
            os.close(self._fd)
            raise

    def close(self) -> None:
        if self._fd >= 0:
            os.close(self._fd)
            self._fd = -1

    def span(self, name: str, shape: tuple[int, ...]) -> tuple[int, int]:
        info = self.header.get(name)
        if not isinstance(info, dict):
            message = f"expert pack is missing {name}"
            raise KeyError(message)
        found = tuple(info.get("shape", ()))
        if found != shape:
            message = f"{name} shape {found} is not {shape}"
            raise ValueError(message)
        offsets = info.get("data_offsets")
        if not isinstance(offsets, list) or len(offsets) != 2:
            message = f"{name} has no data range"
            raise ValueError(message)
        start, end = int(offsets[0]), int(offsets[1])
        return self._data + start, end - start

    def pread(
        self,
        handle: Any,
        file_off: int,
        nbytes: int,
        *,
        dest_off: int = 0,
        note: Note | None = None,
    ) -> None:
        pread_handle(self._fd, handle, file_off, nbytes, dest_off=dest_off, note=note)

    def fileno(self) -> int:
        return self._fd


def open_expert_pack(root: Path, config_root: Path | None = None) -> ExpertFile | None:
    """The baked experts, when the file matches this checkpoint."""
    path = Path(root) / EXPERT_PACK
    if not path.is_file():
        return None
    try:
        record = ExpertFile(path)
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    meta = record.metadata
    names = {spec.name for spec in catalog()}
    check = Path(config_root or root)
    if (
        meta.get("format") != EXPERT_FORMAT
        or meta.get("config") != config_digest(check)
        or names - set(record.header)
    ):
        record.close()
        return None
    return record


class Sink:
    """A safetensors file written in catalog order, beside the final name."""

    def __init__(
        self,
        path: Path,
        entries: list[tuple[str, tuple[int, ...]]],
        metadata: dict[str, str],
    ) -> None:
        self.path = Path(path)
        self.partial = self.path.with_suffix(".partial")
        header: dict[str, object] = {"__metadata__": metadata}
        self.places: dict[str, int] = {}
        self.sizes: dict[str, int] = {}
        offset = 0
        for name, shape in entries:
            nbytes = math.prod(shape)
            header[name] = {
                "dtype": "U8",
                "shape": list(shape),
                "data_offsets": [offset, offset + nbytes],
            }
            self.places[name] = offset
            self.sizes[name] = nbytes
            offset += nbytes
        raw = json.dumps(header, separators=(",", ":")).encode()
        raw += b" " * (-len(raw) % 8)
        self._data = 8 + len(raw)
        self._total = offset
        self._cursor = 0
        self._file = self.partial.open("wb")
        self._file.write(len(raw).to_bytes(8, "little"))
        self._file.write(raw)

    def write_view(self, name: str, view: memoryview) -> None:
        at = self._place(name, len(view))
        self._file.seek(self._data + at)
        offset = 0
        while offset < len(view):
            step = min(_PIECE, len(view) - offset)
            self._file.write(view[offset : offset + step])
            offset += step
        self._cursor = at + len(view)

    def copy_fd(self, name: str, fd: int, file_off: int, nbytes: int) -> None:
        at = self._place(name, nbytes)
        self._file.seek(self._data + at)
        left = nbytes
        pos = file_off
        buf = bytearray(_PIECE)
        window = memoryview(buf)
        while left:
            step = min(_PIECE, left)
            got = 0
            while got < step:
                piece = os.pread(fd, step - got, pos + got)
                if not piece:
                    message = f"read stopped at {pos + got}"
                    raise OSError(message)
                window[got : got + len(piece)] = piece
                got += len(piece)
            self._file.write(window[:step])
            left -= step
            pos += step
        self._cursor = at + nbytes

    def close(self) -> None:
        if self._file is None:
            return
        if self._cursor != self._total:
            self.abort()
            message = "expert pack ended early"
            raise RuntimeError(message)
        self._file.close()
        self._file = None
        self.partial.replace(self.path)

    def abort(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None
        self.partial.unlink(missing_ok=True)

    def _place(self, name: str, nbytes: int) -> int:
        at = self.places.get(name)
        expect = self.sizes.get(name)
        if at is None or expect is None or at != self._cursor:
            message = f"{name} is out of pack order"
            raise RuntimeError(message)
        if nbytes != expect:
            message = f"{name} is {nbytes} bytes, not {expect}"
            raise ValueError(message)
        return at


def bake(
    root: Path, dest: Path | None = None, log: Callable[[str], None] | None = None
) -> None:
    """Requant NVFP4 experts once and write the pack the open reads."""
    from favlo.memory import memory
    from inkling_2_mini.metal41.record import ready as record_ready
    from inkling_2_mini.metal41.session import Session
    from inkling_2_mini.metal41.weights import Tensors, checkpoint_file

    base = Path(root)
    pack = Path(dest or root)
    if record_ready(pack):
        if log is not None:
            log("record is current")
        return
    pack.mkdir(parents=True, exist_ok=True)
    ready = open_expert_pack(pack, base)
    if ready is not None:
        ready.close()
        if log is not None:
            log("expert fp4 pack is current")
        return
    owner = "inkling_2_mini:bake"
    session = Session(owner)
    source = Tensors(checkpoint_file(base))
    sink = Sink(
        pack / EXPERT_PACK,
        [(spec.name, spec.shape) for spec in catalog()],
        {"format": EXPERT_FORMAT, "config": config_digest(base)},
    )
    try:
        _write_experts(session, source, sink, owner, log)
        sink.close()
    except Exception:
        sink.abort()
        raise
    finally:
        source.close()
        session.close()
        memory.sweep(owner)
    if log is not None:
        log(f"wrote {EXPERT_PACK}")


def _write_experts(session: Any, source: Any, sink: Sink, owner: str, log: Any) -> None:
    specs = catalog()
    index = 0
    while index < len(specs):
        data = specs[index]
        scale = specs[index + 1]
        if data.nvfp4:
            _write_nvfp4(session, source, sink, owner, data, scale)
        else:
            _write_stored(source, sink, data)
            _write_stored(source, sink, scale)
        if log is not None:
            log(f"layer {data.layer} {data.proj}")
        index += 2


def _write_stored(source: Any, sink: Sink, spec: Spec) -> None:
    file_off, nbytes = source.span(_source_name(spec), spec.shape)
    sink.copy_fd(spec.name, source.fileno(), file_off, nbytes)


def _write_nvfp4(
    session: Any, source: Any, sink: Sink, owner: str, data: Spec, scale: Spec
) -> None:
    from favlo.memory import memory
    from inkling_2_mini.metal41.session import u32

    src_k = HIDDEN if data.proj == "w13" else INTERMEDIATE
    rows = data.shape[1]
    wide = stored_k(src_k)
    raw_data = _fill_source(session, source, owner, data, (EXPERTS, rows, src_k // 2))
    raw_scale = _fill_source(session, source, owner, scale, (EXPERTS, rows, src_k // 16))
    stem = f"model.llm.layers.{data.layer}.mlp.experts.{data.proj}"
    amax = source.vector(f"{stem}_weight_amax", (EXPERTS,))
    denom = source.vector(f"{stem}_weight_outer_scale_denominator", (1,))
    if float(denom[0]) != OUTER_SCALE_DENOMINATOR:
        message = f"{stem} outer scale denominator is not {OUTER_SCALE_DENOMINATOR}"
        raise ValueError(message)
    gs = _allocate(session, owner, EXPERTS * 4)
    memory.store(gs, 0, (amax / float(denom[0])).astype("float32").tobytes())
    dest_data = _allocate(session, owner, data.nbytes)
    dest_scale = _allocate(session, owner, scale.nbytes)
    blocks = EXPERTS * rows * (wide // 32)
    session.launch(
        "pack_nvfp4_fp4",
        [raw_data, raw_scale, gs, dest_data, dest_scale],
        [u32(EXPERTS), u32(rows), u32(src_k), u32(wide)],
        blocks,
    )
    _dump(sink, data.name, dest_data, data.nbytes)
    _dump(sink, scale.name, dest_scale, scale.nbytes)
    for handle in (raw_data, raw_scale, gs, dest_data, dest_scale):
        _drop(session, handle)


def _fill_source(session: Any, source: Any, owner: str, spec: Spec, shape: tuple[int, ...]) -> Any:
    file_off, nbytes = source.span(_source_name(spec), shape)
    handle = _allocate(session, owner, nbytes)
    pread_handle(source.fileno(), handle, file_off, nbytes)
    return handle


def _source_name(spec: Spec) -> str:
    stem = f"model.llm.layers.{spec.layer}.mlp.experts.{spec.proj}"
    if spec.kind == "data":
        return f"{stem}_weight"
    return f"{stem}_weight_scale"


def _allocate(session: Any, owner: str, nbytes: int) -> Any:
    from favlo.memory import memory

    return session._keep(memory.allocate(int(nbytes), owner=owner, heap="resident"))


def _dump(sink: Sink, name: str, handle: Any, nbytes: int) -> None:
    from favlo.memory import memory

    view = memory.map(handle, 0, nbytes)
    try:
        sink.write_view(name, view)
    finally:
        memory.unmap(handle)


def _drop(session: Any, handle: Any) -> None:
    from favlo.memory import memory

    memory.release(handle)
    try:
        session._owned.remove(handle)
    except ValueError:
        return


__all__ = [
    "EXPERT_FORMAT",
    "EXPERT_PACK",
    "ExpertFile",
    "Sink",
    "Spec",
    "bake",
    "baked_shapes",
    "catalog",
    "expert_pack_bytes",
    "open_expert_pack",
    "pread_handle",
    "pread_view",
]
