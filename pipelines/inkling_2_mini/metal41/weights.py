"""Load the safetensors checkpoint. Experts stay packed in the file.

The fused attention plane and the output projection are float16, from the
checkpoint bf16. The head stays half.
The router, the sink, the dynamic convolution taps, and the message gates
stay half. The head keeps the unpadded vocabulary rows.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from inkling_2_mini.metal41.codec import e4m3_to_float
from inkling_2_mini.metal41.gptq import (
    PACK_FORMAT,
    PACK_NAME,
    config_digest,
    e2m1,
    e8m0,
    pack_nibbles,
)
from inkling_2_mini.metal41.layout import (
    CORE_END,
    CORE_START,
    EXPERTS,
    FULL_LAYERS,
    HIDDEN,
    INTERMEDIATE,
    KV_DIM,
    LAYERS,
    NVFP4_LAYERS,
    OUTER_SCALE_DENOMINATOR,
    Q_DIM,
    REL_DIM,
    UNPADDED_VOCAB,
    VOCAB,
    stored_k,
)
from inkling_2_mini.metal41.session import Session

# Row offsets inside the fused attention plane.
ROW_Q = 0
ROW_REL = Q_DIM
ROW_GATE = Q_DIM + REL_DIM
ROW_K = 2 * Q_DIM + REL_DIM
ROW_V = ROW_K + KV_DIM
ATTN_ROWS = ROW_V + KV_DIM
Q_ROWS = ROW_K
HEAD_ROWS = (UNPADDED_VOCAB + 63) // 64 * 64


class Plane:
    """A matrix stored [padded, inner]. ``scale`` is set for a quantized plane."""

    def __init__(
        self,
        handle: Any,
        cols: int,
        padded: int,
        inner: int,
        scale: Any = None,
        *,
        fp8: bool = False,
    ) -> None:
        self.handle = handle
        self.cols = int(cols)
        self.padded = int(padded)
        self.inner = int(inner)
        self.scale = scale
        self.fp8 = bool(fp8)
        self.fp4 = scale is not None and not fp8

    def rows_at(self, row: int) -> tuple[tuple[Any, int], tuple[Any, int]]:
        """(data, offset) and (scale, offset) bindings that start at ``row``."""
        return (self.handle, row * (self.inner // 2)), (self.scale, row * (self.inner // 32))


class Experts:
    """One fused expert tensor. The atlas fills ``data`` and ``scale``."""

    def __init__(
        self,
        rows: int,
        cols: int,
        scale_cols: int,
        data_stride: int,
        scale_stride: int,
        global_scale: np.ndarray | None,
        file_data: tuple[int, int],
        file_scale: tuple[int, int],
        nvfp4: bool,
    ) -> None:
        self.data = None
        self.scale = None
        self.rows = int(rows)
        self.cols = int(cols)
        self.scale_cols = int(scale_cols)
        self.data_stride = int(data_stride)
        self.scale_stride = int(scale_stride)
        self.global_scale = global_scale
        self.file_data = file_data
        self.file_scale = file_scale
        self.nvfp4 = bool(nvfp4)
        self.fp4 = False
        self.baked = False
        self.inner = int(cols)
        self.data_offsets = None
        self.scale_offsets = None
        self.widths = None
        self.host_data_off = None
        self.host_scale_off = None
        self.host_data_bytes = None
        self.host_scale_bytes = None


class Layer:
    def __init__(self) -> None:
        self.index = 0
        self.full = False
        self.nvfp4 = True
        self.core = False
        self.attn = None
        self.wo = None
        self.sink = None
        self.k_proj = None
        self.v_proj = None
        self.router = None
        self.router_bias = None
        self.q_scale = None
        self.rel_scale = None
        self.rel_table = None
        self.norms = None
        self.mlp_norm = None
        self.attn_conv = None
        self.mlp_conv = None
        self.k_conv = None
        self.v_conv = None
        self.ctl_down = None
        self.ctl_fg = None
        self.w13 = None
        self.w2 = None
        self.attn_query = None
        self.attn_queries = 1
        self.mlp_query = None
        self.mixed = False
        self.uniform = 0
        self.widths = None


def _bf16_f32(raw: bytes, shape: tuple[int, ...]) -> np.ndarray:
    wide = np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16
    return wide.view(np.float32).reshape(shape).copy()


_FP8_MAX = np.float64(448.0)
_E4M3_TABLE = np.array([e4m3_to_float(i) for i in range(256)], dtype=np.float64)
_E4M3_EDGE = (_E4M3_TABLE[:126] + _E4M3_TABLE[1:127]) * 0.5


def _e8m0_fp8(amax: np.ndarray) -> np.ndarray:
    """E8M0 whose E4M3 peak of 448 covers ``amax``. The power of two rounds up."""
    ratio = np.maximum(amax.astype(np.float64) / _FP8_MAX, np.finfo(np.float64).tiny)
    mantissa, exponent = np.frexp(ratio)
    power = exponent.astype(np.int32) - (mantissa == 0.5).astype(np.int32)
    power = np.where(amax <= 0.0, np.int32(-127), power)
    return np.clip(power + 127, 0, 254).astype(np.uint8)


def quantize_mxfp8(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """E4M3 codes and one E8M0 scale per 32 along K. Inner is a multiple of 128."""
    values = np.asarray(matrix)
    rows, inner = values.shape
    if inner % 128:
        message = f"inner dimension {inner} is not a multiple of 128"
        raise ValueError(message)
    codes = np.empty((rows, inner), dtype=np.uint8)
    scales = np.empty((rows, inner // 32), dtype=np.uint8)
    groups = inner // 32
    for start in range(0, rows, 1024):
        stop = min(start + 1024, rows)
        tile = np.asarray(values[start:stop], dtype=np.float64)
        blocks = tile.reshape(stop - start, groups, 32)
        biased = _e8m0_fp8(np.max(np.abs(blocks), axis=-1))
        factor = np.ldexp(np.ones(biased.shape, dtype=np.float64), biased.astype(np.int32) - 127)
        scaled = blocks / factor[..., None]
        choice = np.searchsorted(_E4M3_EDGE, np.abs(scaled), side="left").astype(np.uint8)
        choice = np.minimum(choice, np.uint8(126))
        signed = np.where(np.signbit(scaled), choice | np.uint8(0x80), choice)
        codes[start:stop] = signed.reshape(stop - start, inner)
        scales[start:stop] = biased
    return codes, scales


def dequant_mxfp8(codes: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """Rebuild a matrix from ``quantize_mxfp8``."""
    values = _E4M3_TABLE[codes].astype(np.float32)
    factor = np.ldexp(np.ones(scales.shape, dtype=np.float32), scales.astype(np.int32) - 127)
    return values * np.repeat(factor, 32, axis=1)


def _quantize_fp4(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pack [rows, inner] as E2M1 with one E8M0 scale per 32. Inner is a multiple of 32."""
    rows, inner = matrix.shape
    if inner % 32:
        message = f"inner dimension {inner} is not a multiple of 32"
        raise ValueError(message)
    packed = np.empty((rows, inner // 2), dtype=np.uint8)
    scales = np.empty((rows, inner // 32), dtype=np.uint8)
    for start in range(0, rows, 2048):
        stop = min(start + 2048, rows)
        tile = np.asarray(matrix[start:stop], dtype=np.float32)
        values = tile.reshape(stop - start, inner // 32, 32)
        biased, scale = e8m0(np.max(np.abs(values), axis=-1))
        nibbles = e2m1(values / scale[..., None]).reshape(stop - start, inner)
        packed[start:stop] = pack_nibbles(nibbles)
        scales[start:stop] = biased
    return packed, scales


class Tensors:
    """One safetensors file. The header is read. Payloads are pread in 8 MiB pieces."""

    def __init__(self, path: Path, on_read: Any = None) -> None:
        self.path = Path(path)
        self._file = self.path.open("rb")
        self._on_read = on_read
        fd = self._file.fileno()
        header_bytes = int.from_bytes(os.pread(fd, 8, 0), "little")
        raw = os.pread(fd, header_bytes, 8)
        if len(raw) != header_bytes:
            message = f"{self.path.name} header is short"
            raise OSError(message)
        header = json.loads(raw)
        self.metadata = header.pop("__metadata__", None) or {}
        self.header = header
        self.header_digest = hashlib.sha256(raw).hexdigest()
        self._data = 8 + header_bytes

    def close(self) -> None:
        self._file.close()

    def fileno(self) -> int:
        return self._file.fileno()

    def into(
        self,
        handle: Any,
        file_off: int,
        nbytes: int,
        *,
        dest_off: int = 0,
        note: Any = None,
    ) -> None:
        """Read a span straight into a shared device buffer, in 8 MiB pieces."""
        from inkling_2_mini.metal41.resident import pread_handle

        pread_handle(
            self.fileno(), handle, file_off, nbytes, dest_off=dest_off, note=note or self._on_read
        )

    def raw(self, name: str) -> tuple[bytes, dict[str, Any]]:
        info = self.header.get(name)
        if info is None:
            message = f"checkpoint is missing {name}"
            raise KeyError(message)
        start, end = info["data_offsets"]
        nbytes = int(end) - int(start)
        buf = bytearray(nbytes)
        from inkling_2_mini.metal41.resident import pread_view

        pread_view(self.fileno(), memoryview(buf), self._data + int(start), self._on_read)
        return bytes(buf), info

    def vector(self, name: str, shape: tuple[int, ...]) -> np.ndarray:
        raw, info = self.raw(name)
        found = tuple(info["shape"])
        if found != shape:
            message = f"{name} shape {found} is not {shape}"
            raise ValueError(message)
        if info["dtype"] == "F32":
            return np.frombuffer(raw, dtype=np.float32).reshape(shape).copy()
        if info["dtype"] == "F16":
            return np.frombuffer(raw, dtype=np.float16).reshape(shape).astype(np.float32)
        if info["dtype"] == "BF16":
            return _bf16_f32(raw, shape)
        message = f"{name} dtype {info['dtype']} is not a float tensor"
        raise ValueError(message)

    def bytes_of(self, name: str, shape: tuple[int, ...]) -> np.ndarray:
        raw, info = self.raw(name)
        if info["dtype"] != "U8" or tuple(info["shape"]) != shape:
            message = f"{name} is not U8 {shape}"
            raise ValueError(message)
        return np.frombuffer(raw, dtype=np.uint8).reshape(shape)

    def span(self, name: str, shape: tuple[int, ...]) -> tuple[int, int]:
        info = self.header.get(name)
        if info is None:
            message = f"checkpoint is missing {name}"
            raise KeyError(message)
        found = tuple(info["shape"])
        if found != shape:
            message = f"{name} shape {found} is not {shape}"
            raise ValueError(message)
        start, end = info["data_offsets"]
        return self._data + int(start), int(end) - int(start)

    def blob(self, offset: int, nbytes: int) -> bytes:
        buf = bytearray(int(nbytes))
        from inkling_2_mini.metal41.resident import pread_view

        pread_view(self.fileno(), memoryview(buf), int(offset), self._on_read)
        return bytes(buf)


def checkpoint_file(root: Path) -> Path:
    return Path(root) / "model-00001-of-00001.safetensors"


def safetensor_files(path: Path) -> list[Path]:
    """The safetensors files of a checkpoint. Payloads stay where they are.

    A file is that file. A directory with an index opens the files the index
    names. A directory without an index opens every safetensors file there.
    """
    path = Path(path)
    if path.is_file():
        if path.suffix != ".safetensors":
            message = f"{path.name} is not a safetensors file"
            raise ValueError(message)
        return [path]
    index = path / "model.safetensors.index.json"
    if index.is_file():
        body = json.loads(index.read_text())
        mapping = body.get("weight_map") if isinstance(body, dict) else None
        if not isinstance(mapping, dict) or not mapping:
            message = f"{index.name} has no weight map"
            raise ValueError(message)
        names = sorted({str(item) for item in mapping.values()})
        files = [path / name for name in names]
        missing = [item.name for item in files if not item.is_file()]
        if missing:
            message = f"checkpoint is missing {', '.join(missing)}"
            raise FileNotFoundError(message)
        return files
    files = sorted(
        item for item in path.iterdir() if item.is_file() and item.suffix == ".safetensors"
    )
    if not files:
        message = f"{path.name} has no safetensors file"
        raise FileNotFoundError(message)
    return files


class Safetensors:
    """One checkpoint, looked up by tensor name. Payloads stay in place."""

    def __init__(self, path: Path, on_read: Any = None) -> None:
        self.files = safetensor_files(path)
        self._open = [Tensors(item, on_read) for item in self.files]
        self._by_name: dict[str, Tensors] = {}
        self.header: dict[str, Any] = {}
        for tensors in self._open:
            for name, info in tensors.header.items():
                if name in self._by_name:
                    message = f"{name} is in two shards"
                    raise ValueError(message)
                self._by_name[name] = tensors
                self.header[name] = info

    def close(self) -> None:
        for tensors in self._open:
            tensors.close()

    def vector(self, name: str, shape: tuple[int, ...]) -> np.ndarray:
        tensors = self._by_name.get(name)
        if tensors is None:
            message = f"checkpoint is missing {name}"
            raise KeyError(message)
        return tensors.vector(name, shape)

    def locate(self, name: str) -> tuple[int, int]:
        """File descriptor and absolute payload offset of one tensor."""
        tensors = self._by_name.get(name)
        if tensors is None:
            message = f"checkpoint is missing {name}"
            raise KeyError(message)
        info = tensors.header[name]
        start, _end = info["data_offsets"]
        return tensors.fileno(), tensors._data + int(start)


DENSE_PACK = "dense-fp8.safetensors"
DENSE_FORMAT = "favlo-dense-fp8-nearest-1"


def dense_catalog() -> list[tuple[str, tuple[int, int]]]:
    """Attention and output-projection planes, one pair per layer."""
    entries: list[tuple[str, tuple[int, int]]] = []
    for index in range(LAYERS):
        entries.append((f"layers.{index}.attn.data", (ATTN_ROWS, HIDDEN)))
        entries.append((f"layers.{index}.attn.scale", (ATTN_ROWS, HIDDEN // 32)))
        entries.append((f"layers.{index}.wo.data", (HIDDEN, Q_DIM)))
        entries.append((f"layers.{index}.wo.scale", (HIDDEN, Q_DIM // 32)))
    return entries


def open_dense_pack(root: Path) -> Any:
    """The fp8 attention pack, when the file matches this layout."""
    from inkling_2_mini.metal41.resident import ExpertFile

    path = Path(root) / DENSE_PACK
    if not path.is_file():
        return None
    try:
        record = ExpertFile(path)
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    names = {name for name, _shape in dense_catalog()}
    if record.metadata.get("format") != DENSE_FORMAT or names - set(record.header):
        record.close()
        return None
    return record


def ensure_dense(root: Path, log: Any = None, *, language: Path | None = None) -> None:
    """Quantize attention and the output projection once. The open preads the file."""
    ready = open_dense_pack(root)
    if ready is not None:
        ready.close()
        if log is not None:
            log("dense fp8 pack is current")
        return
    source = Safetensors(Path(language) if language is not None else checkpoint_file(root))
    try:
        _write_dense_nearest(root, source, log)
    finally:
        source.close()


def _write_pair(sink: Any, name: str, codes: np.ndarray, scales: np.ndarray) -> None:
    sink.write_view(f"{name}.data", memoryview(np.ascontiguousarray(codes).ravel()))
    sink.write_view(f"{name}.scale", memoryview(np.ascontiguousarray(scales).ravel()))


def _write_dense_nearest(root: Path, source: Safetensors, log: Any) -> None:
    from inkling_2_mini.metal41.resident import Sink

    sink = Sink(Path(root) / DENSE_PACK, dense_catalog(), {"format": DENSE_FORMAT})
    try:
        for index in range(LAYERS):
            if log is not None:
                log(f"quantizing layer {index} attention and output projection")
            codes, scales = _fused_fp8(source, index)
            _write_pair(sink, f"layers.{index}.attn", codes, scales)
            weight = source.vector(f"model.llm.layers.{index}.attn.wo_ud.weight", (HIDDEN, Q_DIM))
            _write_pair(sink, f"layers.{index}.wo", *quantize_mxfp8(weight))
        sink.close()
    except Exception:
        sink.abort()
        raise


def _fused_fp8(tensors: Safetensors, index: int) -> tuple[np.ndarray, np.ndarray]:
    """E4M3 rows of ``[wq; wr; gate; wk; wv]`` and their E8M0 scales."""
    attn = f"model.llm.layers.{index}.attn"
    parts = (
        (f"{attn}.wq_du.weight", Q_DIM),
        (f"{attn}.wr_du.weight", REL_DIM),
        (f"{attn}.gate_attn.weight", Q_DIM),
        (f"{attn}.wk_dv.weight", KV_DIM),
        (f"{attn}.wv_dv.weight", KV_DIM),
    )
    codes = []
    scales = []
    for name, rows in parts:
        packed, scale = quantize_mxfp8(tensors.vector(name, (rows, HIDDEN)))
        codes.append(packed)
        scales.append(scale)
    return np.concatenate(codes), np.concatenate(scales)


def attn_matrix(tensors: Tensors, index: int) -> np.ndarray:
    """The fused ``[wq; wr; gate; wk; wv]`` rows of one layer."""
    attn = f"model.llm.layers.{index}.attn"
    return np.concatenate(
        [
            tensors.vector(f"{attn}.wq_du.weight", (Q_DIM, HIDDEN)),
            tensors.vector(f"{attn}.wr_du.weight", (REL_DIM, HIDDEN)),
            tensors.vector(f"{attn}.gate_attn.weight", (Q_DIM, HIDDEN)),
            tensors.vector(f"{attn}.wk_dv.weight", (KV_DIM, HIDDEN)),
            tensors.vector(f"{attn}.wv_dv.weight", (KV_DIM, HIDDEN)),
        ],
        axis=0,
    )


def wo_matrix(tensors: Tensors, index: int) -> np.ndarray:
    return tensors.vector(f"model.llm.layers.{index}.attn.wo_ud.weight", (HIDDEN, Q_DIM))


def head_matrix(tensors: Tensors) -> np.ndarray:
    return tensors.vector("model.llm.unembed.weight", (VOCAB, HIDDEN))[:UNPADDED_VOCAB]


def open_pack(root: Path) -> Tensors | None:
    """The calibrated dense planes, when the file matches this checkpoint."""
    path = Path(root) / PACK_NAME
    if not path.is_file():
        return None
    pack = Tensors(path)
    meta = pack.metadata
    if meta.get("format") != PACK_FORMAT or meta.get("config") != config_digest(root):
        pack.close()
        return None
    return pack


def _pad_rows(matrix: np.ndarray, padded: int) -> np.ndarray:
    rows, inner = matrix.shape
    if rows == padded:
        return np.ascontiguousarray(matrix)
    out = np.zeros((padded, inner), dtype=matrix.dtype)
    out[:rows] = matrix
    return out


class Checkpoint:
    """The language tensors. Vision and the drafter stay on disk."""

    def __init__(
        self,
        root: Path,
        session: Session,
        progress: Any = None,
        *,
        calibrated: bool = True,
        pack_root: Path | None = None,
    ) -> None:
        from inkling_2_mini.metal41.record import open_language, ready

        self.root = root
        self.pack_root = Path(pack_root or root)
        self.session = session
        self._progress = progress
        self._done = 0
        self._noted = 0
        del calibrated
        self._pack = None
        self.dense_tag = "fp8-nearest"
        self._shared_dense = False
        if ready(root):
            self._tensors, self.experts = open_language(root, on_read=self._read)
            self._dense = self._tensors
            self._shared_dense = True
        else:
            path = checkpoint_file(root)
            if not path.is_file():
                message = f"checkpoint is missing {path.name}"
                raise FileNotFoundError(message)
            self._tensors = Tensors(path, on_read=self._read)
            from inkling_2_mini.metal41.resident import open_expert_pack

            self.experts = open_expert_pack(self.pack_root, self.root)
            self._dense = open_dense_pack(self.pack_root)
        try:
            self._load()
        except Exception:
            self.close()
            raise
        finally:
            if self._pack is not None:
                self._pack.close()
                self._pack = None
            dense = getattr(self, "_dense", None)
            if dense is not None and not self._shared_dense:
                dense.close()
                self._dense = None

    def _load(self) -> None:
        session = self.session
        record = getattr(self, "experts", None)
        self._mixed = record is not None and "widths" in record.header
        self.embed = self._f16_rows("model.llm.embed.weight", VOCAB, HIDDEN, VOCAB).handle
        self.embed_norm = self._f32("model.llm.embed_norm.weight", (HIDDEN,))
        self.final_norm = self._f32("model.llm.norm.weight", (HIDDEN,))
        self.unembed = self._f16_rows("model.llm.unembed.weight", UNPADDED_VOCAB, HIDDEN, HEAD_ROWS)
        self.output_query = self._f32("model.llm.attn_res.output_query", (HIDDEN,))
        cross_q = self._vector_f32("model.llm.middle_cycle_cross_pass_queries", (24, HIDDEN))
        cross_kv = self._vector_f32("model.llm.middle_cycle_cross_pass_kv_queries", (24, HIDDEN))
        # Rows 0..23 are the query and MLP messages, rows 24..47 the key and value messages.
        self.cross = session.upload(
            np.ascontiguousarray(np.concatenate([cross_q, cross_kv], axis=0))
        )
        self.layers = [self._layer(index) for index in range(LAYERS)]
        self.dense_tag = "fp16" if not self.layers[0].attn.fp8 else "fp8-nearest"
        self.expert_bytes = 0
        for layer in self.layers:
            for experts in (layer.w13, layer.w2):
                self.expert_bytes += experts.file_data[1] + experts.file_scale[1]

    def close(self) -> None:
        experts = getattr(self, "experts", None)
        if experts is not None:
            experts.close()
            self.experts = None
        tensors = getattr(self, "_tensors", None)
        if tensors is not None:
            tensors.close()
            self._tensors = None

    def _read(self, nbytes: int) -> None:
        self._done += nbytes
        progress = self._progress
        if progress is not None and self._done - self._noted > (32 << 20):
            self._noted = self._done
            note = getattr(progress, "tick", None)
            if callable(note):
                note(self._done)

    def blob(self, offset: int, nbytes: int) -> bytes:
        """Bytes from the mapped checkpoint. ``offset`` is into the map."""
        return self._tensors.blob(offset, nbytes)

    def _vector_f32(self, name: str, shape: tuple[int, ...]) -> np.ndarray:
        return self._tensors.vector(name, shape)

    def _matrix(self, name: str, shape: tuple[int, int]) -> np.ndarray:
        return self._tensors.vector(name, shape)

    def _f32(self, name: str, shape: tuple[int, ...]) -> Any:
        return self.session.upload(np.ascontiguousarray(self._vector_f32(name, shape)))

    def _dense(self, name: str, cols: int, rows: int, inner: int, build: Any) -> Plane:
        """A Metal fp4 plane: the calibrated pack when present, else round to nearest."""
        pack = self._pack
        if pack is not None:
            data_off, data_n = pack.span(f"{name}.data", (rows, inner // 2))
            scale_off, scale_n = pack.span(f"{name}.scale", (rows, inner // 32))
            data = self._device(data_n)
            scale = self._device(scale_n)
            pack.into(data, data_off, data_n, note=self._read)
            pack.into(scale, scale_off, scale_n, note=self._read)
            return Plane(data, cols, rows, inner, scale=scale)
        packed, scales = _quantize_fp4(build())
        return Plane(
            self.session.upload(packed), cols, rows, inner, scale=self.session.upload(scales)
        )

    def _device(self, nbytes: int) -> Any:
        from favlo.memory import memory

        handle = memory.allocate(int(nbytes), owner=self.session._owner, heap="resident")
        return self.session._keep(handle)

    def _direct_f16(self, name: str, rows: int, inner: int) -> bool:
        info = self._tensors.header.get(name)
        if not isinstance(info, dict) or info.get("dtype") != "F16":
            return False
        shape = tuple(info.get("shape") or ())
        return len(shape) == 2 and int(shape[0]) >= rows and int(shape[1]) == inner

    def _f16_rows(self, name: str, rows: int, inner: int, padded: int) -> Plane:
        """Pread a float16 matrix. A BF16 tensor is cast once, for an unrepacked file."""
        if not self._direct_f16(name, rows, inner):
            info = self._tensors.header[name]
            matrix = self._matrix(name, tuple(info["shape"]))
            return self._half(np.asarray(matrix)[:rows], rows, padded, inner)
        info = self._tensors.header[name]
        start = int(info["data_offsets"][0])
        total = padded * inner * 2
        handle = self._device(total) if padded == rows else self.session.zeros(total)
        self._tensors.into(
            handle,
            self._tensors._data + start,
            rows * inner * 2,
            note=self._read,
        )
        return Plane(handle, rows, padded, inner)

    def _half_attention(self, index: int) -> bool:
        """True when this layer's attention weights are float16 or bf16."""
        name = f"model.llm.layers.{index}.attn.wq_du.weight"
        info = self._tensors.header.get(name)
        return isinstance(info, dict) and info.get("dtype") in {"F16", "BF16"}

    def _fp8_plane(self, name: str, rows: int, inner: int) -> Plane:
        """Pread one baked E4M3 plane and its E8M0 scales."""
        pack = self._dense
        if pack is None:
            message = f"dense fp8 pack is missing {name}"
            raise FileNotFoundError(message)
        data_off, data_n = pack.span(f"{name}.data", (rows, inner))
        scale_off, scale_n = pack.span(f"{name}.scale", (rows, inner // 32))
        data = self._device(data_n)
        scale = self._device(scale_n)
        pack.pread(data, data_off, data_n, note=self._read)
        pack.pread(scale, scale_off, scale_n, note=self._read)
        return Plane(data, rows, rows, inner, scale=scale, fp8=True)

    def _attn(self, index: int) -> Plane:
        """The fused plane. Each float16 member is pread at its row offset."""
        attn = f"model.llm.layers.{index}.attn"
        parts = (
            (f"{attn}.wq_du.weight", Q_DIM, ROW_Q),
            (f"{attn}.wr_du.weight", REL_DIM, ROW_REL),
            (f"{attn}.gate_attn.weight", Q_DIM, ROW_GATE),
            (f"{attn}.wk_dv.weight", KV_DIM, ROW_K),
            (f"{attn}.wv_dv.weight", KV_DIM, ROW_V),
        )
        if not all(self._direct_f16(name, rows, HIDDEN) for name, rows, _row in parts):
            return self._half(attn_matrix(self._tensors, index), ATTN_ROWS, ATTN_ROWS, HIDDEN)
        handle = self._device(ATTN_ROWS * HIDDEN * 2)
        for name, rows, row in parts:
            info = self._tensors.header[name]
            start = int(info["data_offsets"][0])
            self._tensors.into(
                handle,
                self._tensors._data + start,
                rows * HIDDEN * 2,
                dest_off=row * HIDDEN * 2,
                note=self._read,
            )
        return Plane(handle, ATTN_ROWS, ATTN_ROWS, HIDDEN)

    def _half(self, matrix: np.ndarray, cols: int, padded: int, inner: int) -> Plane:
        array = np.ascontiguousarray(_pad_rows(matrix, padded).astype(np.float16))
        return Plane(self.session.upload(array), cols, padded, inner)

    def _span(self, name: str, shape: tuple[int, ...]) -> tuple[int, int]:
        return self._tensors.span(name, shape)

    def _conv(self, name: str, channels: int) -> Any:
        array = self._vector_f32(name, (channels, 1, 4)).reshape(channels, 4)
        return self.session.upload(np.ascontiguousarray(array))

    def _mixed_widths(self) -> np.ndarray:
        cached = getattr(self, "_widths", None)
        if cached is not None:
            return cached
        raw, info = self.experts.raw("widths")
        shape = tuple(info.get("shape", ()))
        if info.get("dtype") != "U8" or shape != (LAYERS, EXPERTS):
            message = f"widths shape {shape} is not {(LAYERS, EXPERTS)}"
            raise ValueError(message)
        self._widths = np.frombuffer(raw, dtype=np.uint8).reshape(LAYERS, EXPERTS).copy()
        return self._widths

    def _u32_row(self, name: str) -> np.ndarray:
        raw, info = self.experts.raw(name)
        if tuple(info.get("shape", ())) != (EXPERTS, 4):
            message = f"{name} shape {info.get('shape')} is not {(EXPERTS, 4)}"
            raise ValueError(message)
        return np.frombuffer(raw, dtype="<u4").reshape(EXPERTS).copy()

    def _mixed_experts(self, layer: int, kind: str, rows: int, cols: int) -> Experts:
        record = self.experts
        widths = self._mixed_widths()[layer]
        uniform = 4 if np.all(widths == 4) else 8 if np.all(widths == 8) else 0
        data_off = self._u32_row(f"layers.{layer}.{kind}.data_off")
        scale_off = self._u32_row(f"layers.{layer}.{kind}.scale_off")
        info = record.header[f"layers.{layer}.{kind}"]
        shape = tuple(info["shape"])
        blob_off, blob_n = record.span(f"layers.{layer}.{kind}", shape)
        ends = np.empty(EXPERTS, dtype=np.uint32)
        ends[:-1] = data_off[1:]
        ends[-1] = np.uint32(blob_n)
        data_bytes = (scale_off - data_off).astype(np.uint32)
        scale_bytes = (ends - scale_off).astype(np.uint32)
        if np.any(data_bytes <= 0) or np.any(scale_bytes <= 0):
            message = f"layers.{layer}.{kind} offsets are not packed"
            raise ValueError(message)
        experts = Experts(
            rows,
            cols,
            cols // 32,
            0,
            0,
            None,
            (blob_off, blob_n),
            (blob_off, 0),
            False,
        )
        experts.baked = True
        experts.host_data_off = data_off
        experts.host_scale_off = scale_off
        experts.host_data_bytes = data_bytes
        experts.host_scale_bytes = scale_bytes
        if uniform == 0:
            experts.inner = stored_k(cols) if kind == "w2" else int(cols)
            return experts
        if not np.all(data_bytes == data_bytes[0]) or not np.all(scale_bytes == scale_bytes[0]):
            message = f"layers.{layer}.{kind} widths are uniform but the images are not"
            raise ValueError(message)
        experts.fp4 = uniform == 4
        experts.inner = stored_k(cols) if uniform == 4 else int(cols)
        experts.data_stride = int(data_bytes[0])
        experts.scale_stride = int(scale_bytes[0])
        return experts

    def _experts(self, layer: int, kind: str, rows: int, cols: int, nvfp4: bool) -> Experts:
        record = getattr(self, "experts", None)
        if record is not None and "widths" in record.header:
            return self._mixed_experts(layer, kind, rows, cols)
        if record is not None:
            from inkling_2_mini.metal41.resident import baked_shapes

            data_shape, scale_shape = baked_shapes(layer, kind)
            experts = Experts(
                rows,
                cols,
                int(scale_shape[-1]),
                int(np.prod(data_shape[1:])),
                int(np.prod(scale_shape[1:])),
                None,
                record.span(f"layers.{layer}.{kind}.data", data_shape),
                record.span(f"layers.{layer}.{kind}.scale", scale_shape),
                nvfp4,
            )
            experts.baked = True
            return experts
        stem = f"model.llm.layers.{layer}.mlp.experts.{kind}"
        scale_cols = cols // 16 if nvfp4 else cols // 32
        data_shape = (EXPERTS, rows, cols // 2 if nvfp4 else cols)
        scale_shape = (EXPERTS, rows, scale_cols)
        file_data = self._span(f"{stem}_weight", data_shape)
        file_scale = self._span(f"{stem}_weight_scale", scale_shape)
        global_scale = None
        if nvfp4:
            amax = self._vector_f32(f"{stem}_weight_amax", (EXPERTS,))
            denom = self._vector_f32(f"{stem}_weight_outer_scale_denominator", (1,))
            if float(denom[0]) != OUTER_SCALE_DENOMINATOR:
                message = f"{stem} outer scale denominator is not {OUTER_SCALE_DENOMINATOR}"
                raise ValueError(message)
            global_scale = (amax / denom[0]).astype(np.float32)
        return Experts(
            rows,
            cols,
            scale_cols,
            int(np.prod(data_shape[1:])),
            int(np.prod(scale_shape[1:])),
            global_scale,
            file_data,
            file_scale,
            nvfp4,
        )

    def _controllers(self, layer: int) -> tuple[Any, Any]:
        downs = []
        fgs = []
        for name in ("attn", "k", "v", "mlp"):
            stem = f"model.llm.layers.{layer}.controllers.{name}"
            downs.append(self._matrix(f"{stem}.down_weight", (64, HIDDEN * 2)))
            forget = self._matrix(f"{stem}.forget_weight", (HIDDEN, 64))
            gain = self._matrix(f"{stem}.gain_weight", (HIDDEN, 64))
            fgs.append(np.concatenate([forget, gain], axis=0))
        down = np.ascontiguousarray(np.stack(downs).astype(np.float16))
        fg = np.ascontiguousarray(np.stack(fgs).astype(np.float16))
        return self.session.upload(down), self.session.upload(fg)

    def _query(self, name: str) -> tuple[Any, int]:
        info = self._tensors.header[name]
        shape = tuple(info["shape"])
        if shape == (HIDDEN,):
            count = 1
        elif len(shape) == 2 and shape[1] == HIDDEN:
            count = int(shape[0])
        else:
            message = f"{name} shape {shape} is not a hidden query"
            raise ValueError(message)
        array = self._vector_f32(name, shape).reshape(count, HIDDEN)
        return self.session.upload(np.ascontiguousarray(array)), count

    def _layer(self, index: int) -> Layer:
        layer = Layer()
        layer.index = index
        layer.full = index in FULL_LAYERS
        layer.nvfp4 = index < NVFP4_LAYERS and not self._mixed
        layer.mixed = self._mixed
        if self._mixed:
            layer.widths = self._mixed_widths()[index]
            row = layer.widths
            layer.uniform = 4 if np.all(row == 4) else 8 if np.all(row == 8) else 0
        layer.core = CORE_START <= index < CORE_END
        stem = f"model.llm.layers.{index}"
        attn = f"{stem}.attn"
        if self._half_attention(index):
            layer.attn = self._attn(index)
            layer.wo = self._f16_rows(f"{attn}.wo_ud.weight", HIDDEN, Q_DIM, HIDDEN)
        else:
            layer.attn = self._fp8_plane(f"layers.{index}.attn", ATTN_ROWS, HIDDEN)
            layer.wo = self._fp8_plane(f"layers.{index}.wo", HIDDEN, Q_DIM)
        layer.sink = self._f16_rows(f"{attn}.attn_sink_proj.weight", 24, HIDDEN, 64)
        layer.q_scale = self._f32(f"{attn}.q_l2_clip.scale", (24, 128))
        layer.rel_scale = self._f32(f"{attn}.rel_l2_clip.scale", (24, 16))
        layer.rel_table = self._f32(f"{attn}.rel_logits_proj.proj", (16, 1024))
        layer.attn_conv = self._conv(f"{stem}.attn_sconv.weight", HIDDEN)
        layer.mlp_conv = self._conv(f"{stem}.mlp_sconv.weight", HIDDEN)
        layer.mlp_norm = self._f32(f"{stem}.mlp_norm.weight", (HIDDEN,))
        layer.router = self._f16_rows(f"{stem}.mlp.gate.weight", 128, HIDDEN, 128)
        layer.router_bias = self._f32(f"{stem}.mlp.gate.bias", (128,))
        if layer.full:
            norms = np.stack(
                [
                    self._vector_f32(f"{stem}.q_input_norm.weight", (HIDDEN,)),
                    self._vector_f32(f"{stem}.k_input_norm.weight", (HIDDEN,)),
                    self._vector_f32(f"{stem}.v_input_norm.weight", (HIDDEN,)),
                ]
            )
            layer.norms = self.session.upload(np.ascontiguousarray(norms))
            layer.k_conv = self._f32(f"{attn}.k_sconv.static_w", (4, KV_DIM))
            layer.v_conv = self._f32(f"{attn}.v_sconv.static_w", (4, KV_DIM))
            layer.k_proj = self._f16_rows(f"{attn}.k_sconv.weight_proj", 32, HIDDEN, 64)
            layer.v_proj = self._f16_rows(f"{attn}.v_sconv.weight_proj", 32, HIDDEN, 64)
        else:
            layer.norms = self._f32(f"{stem}.attn_norm.weight", (HIDDEN,))
            layer.k_conv = self._conv(f"{attn}.k_sconv.weight", KV_DIM)
            layer.v_conv = self._conv(f"{attn}.v_sconv.weight", KV_DIM)
        layer.w13 = self._experts(index, "w13", INTERMEDIATE * 2, HIDDEN, layer.nvfp4)
        layer.w2 = self._experts(index, "w2", HIDDEN, INTERMEDIATE, layer.nvfp4)
        if layer.core:
            layer.ctl_down, layer.ctl_fg = self._controllers(index)
        if index == 0:
            layer.attn_query, layer.attn_queries = None, 0
        else:
            layer.attn_query, layer.attn_queries = self._query(
                f"model.llm.attn_res.attn_queries.{index - 1}"
            )
        layer.mlp_query, _count = self._query(f"model.llm.attn_res.mlp_queries.{index}")
        return layer
