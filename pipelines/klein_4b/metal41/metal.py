"""Metal device. Int8 products and float32 attention sync after each dispatch."""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import Metal
import numpy as np

from favlo.memory import device_buffer, memory
from klein_4b.portable.constants import BN_EPS, GROUP, NORM_EPS, TILE_M, TILE_N, VAE_EPS, VAE_GROUPS
from klein_4b.portable.formats import pad_rows

_SHADER = Path(__file__).resolve().parent / "shaders" / "kernels.metal"
_LOAD_CHUNK = 8 << 20
_NAMES = (
    "quantize_f16_i8",
    "quantize_f32_i8",
    "gather_quantize_f16",
    "gather_quantize_f32",
    "matmul_i8",
    "attend_f32",
    "attend_flash",
    "store_f16",
    "store_f32",
    "rmsnorm_row",
    "modulate_row",
    "silu_f16",
    "silu_mul_f16",
    "swiglu_f16",
    "gate_f16",
    "rope_text",
    "rope_dit",
    "repeat_heads",
    "copy_f16",
    "copy_f32",
    "cast_f16_f32",
    "cast_f32_f16",
    "slice_cols_f16",
    "place_cols_f16",
    "pack_attn_mlp",
    "groupnorm_f16",
    "groupnorm_f32",
    "im2col_f16",
    "chw_from_rows",
    "chw_from_rows_f32",
    "seq_from_chw",
    "seq_from_chw_f32",
    "nearest2_f16",
    "nearest2_f32",
    "add_f16",
    "add_f32",
    "patch_pack",
    "unpack_latent",
    "euler_f16",
)


@dataclass
class View:
    """One resident tensor. ``owner`` returns the buffer to the pool on ``drop``."""

    buf: Any
    shape: tuple[int, ...]
    kind: str
    offset: int = 0
    capacity: int = 0
    owner: bool = False

    @property
    def count(self) -> int:
        total = 1
        for dim in self.shape:
            total *= int(dim)
        return total


@dataclass
class Bank:
    codes: Any
    scales: Any
    bias: Any | None
    n: int
    k: int
    stored_n: int
    stored_k: int
    kh: int | None
    kw: int | None


class Gpu:
    """One device. ``close`` sweeps the owner."""

    def __init__(self, owner: str | None = None) -> None:
        self._owner = owner or f"klein_4b:{id(self)}"
        self.device = memory.device
        self.queue = memory.make_queue()
        source = _SHADER.read_text(encoding="utf-8")
        options = Metal.MTLCompileOptions.alloc().init()
        options.setLanguageVersion_((4 << 16) + 1)
        library, error = self.device.newLibraryWithSource_options_error_(source, options, None)
        if library is None:
            message = f"shader compile failed: {error}"
            raise RuntimeError(message)
        self.pipes: dict[str, Any] = {}
        for name in _NAMES:
            fn = library.newFunctionWithName_(name)
            if fn is None:
                message = f"shader is missing {name}"
                raise RuntimeError(message)
            pipe, pipe_error = self.device.newComputePipelineStateWithFunction_error_(fn, None)
            if pipe is None:
                message = f"pipeline {name} failed: {pipe_error}"
                raise RuntimeError(message)
            self.pipes[name] = pipe
        width = int(self.pipes["matmul_i8"].threadExecutionWidth())
        self._threads = (width * 4, 1, 1)
        self._cmd: Any = None
        self._enc: Any = None
        self._held: list[bytes] = []
        self._owned: list[Any] = []
        self._keep: list[Any] = []
        self._bank: dict[str, Bank] = {}
        self._vectors: dict[str, View] = {}
        self._scratch: dict[str, tuple[int, Any]] = {}
        self._retired: list[Any] = []
        self._free: list[tuple[int, Any]] = []
        self._store: Any = None

    def bind_store(self, store: Any) -> None:
        self._store = store

    def upload(self, array: np.ndarray, *, keep: bool = False) -> Any:
        buf = memory.upload(np.ascontiguousarray(array), owner=self._owner)
        (self._keep if keep else self._owned).append(buf)
        return buf

    def scratch(self, name: str, nbytes: int) -> Any:
        """A reusable workspace. A larger request keeps the old buffer until the next wait.

        The buffer is not cleared. The next kernel writes every element it reads.
        Clearing it here walked the whole span in 8 MiB stores, and a convolution
        that released this workspace repeated that walk.
        """
        have, buf = self._scratch.get(name, (0, None))
        if buf is not None and have >= nbytes:
            return buf
        fresh = device_buffer(self._owner, int(nbytes), zero=False)
        self._keep.append(fresh)
        if buf is not None:
            if self._enc is None:
                self._release_handle(buf)
            else:
                self._retired.append(buf)
        self._scratch[name] = (int(nbytes), fresh)
        return fresh

    def spec(self, key: str) -> Bank:
        found = self._bank.get(key)
        if found is not None:
            return found
        if self._store is None:
            message = f"{key} is not loaded"
            raise KeyError(message)
        spec = self._store.index["matrices"][key]
        root = self._store.path
        bias_file = spec.get("bias")
        bias = self._read_file(root / bias_file) if isinstance(bias_file, str) else None
        kh = spec.get("kh")
        kw = spec.get("kw")
        bank = Bank(
            codes=self._read_file(root / spec["codes"]),
            scales=self._read_file(root / spec["scales"]),
            bias=bias,
            n=int(spec["n"]),
            k=int(spec["k"]),
            stored_n=int(spec["stored_n"]),
            stored_k=int(spec["stored_k"]),
            kh=int(kh) if isinstance(kh, int) else None,
            kw=int(kw) if isinstance(kw, int) else None,
        )
        self._bank[key] = bank
        return bank

    def _read_file(self, path: Path) -> Any:
        """Read one existing plane into the span the kernel binds. No second host array."""
        nbytes = path.stat().st_size
        buf = device_buffer(self._owner, max(nbytes, 16), zero=False)
        self._keep.append(buf)
        if nbytes < 1:
            return buf
        view = memory.map(buf, 0, nbytes)
        with path.open("rb", buffering=0) as source:
            offset = 0
            while offset < nbytes:
                step = min(_LOAD_CHUNK, nbytes - offset)
                pending = view[offset : offset + step]
                filled = 0
                while filled < step:
                    wrote = source.readinto(pending[filled:])
                    if not wrote:
                        message = f"{path.name} ended before {nbytes} bytes"
                        raise RuntimeError(message)
                    filled += int(wrote)
                offset += step
        return buf

    def linear(self, activation: np.ndarray, key: str) -> np.ndarray:
        """Float16 ``[rows, K]`` times one packed weight. The result is float16 ``[rows, N]``."""
        bank = self.spec(key)
        values = np.ascontiguousarray(activation, dtype=np.float16)
        if values.ndim != 2 or values.shape[1] != bank.k:
            message = f"{key} wants K {bank.k}, got {values.shape}"
            raise ValueError(message)
        return self._product(values, bank)[:, : bank.n]

    def product_arrays(
        self,
        activation: np.ndarray,
        codes: np.ndarray,
        scales: np.ndarray,
        logical_n: int,
        logical_k: int,
    ) -> np.ndarray:
        """One int8 product. The result keeps the padded columns."""
        bank = Bank(
            codes=self.upload(np.ascontiguousarray(codes)),
            scales=self.upload(np.ascontiguousarray(scales, dtype=np.float32)),
            bias=None,
            n=int(logical_n),
            k=int(logical_k),
            stored_n=int(codes.shape[0]),
            stored_k=int(codes.shape[1]),
            kh=None,
            kw=None,
        )
        values = np.ascontiguousarray(activation, dtype=np.float16)
        return self._product(values, bank)

    def attend(
        self,
        query: np.ndarray,
        key: np.ndarray,
        value: np.ndarray,
        *,
        causal: bool,
        valid: int,
    ) -> np.ndarray:
        """Float32 attention. ``query`` is ``[seq, heads, dim]``."""
        if query.shape != key.shape or query.shape != value.shape or query.ndim != 3:
            message = "attention wants matching [seq, heads, dim] arrays"
            raise ValueError(message)
        seq, heads, dim = (int(query.shape[0]), int(query.shape[1]), int(query.shape[2]))
        if dim < 1 or dim > 1024 or dim % 32:
            message = f"attention head {dim} is outside 32..1024 in steps of 32"
            raise ValueError(message)
        out = self.scratch("attn", seq * heads * dim * 4)
        self._attend(
            self.upload(np.ascontiguousarray(query, dtype=np.float32)),
            self.upload(np.ascontiguousarray(key, dtype=np.float32)),
            self.upload(np.ascontiguousarray(value, dtype=np.float32)),
            out,
            seq,
            heads,
            heads,
            dim,
            int(valid),
            causal,
            None,
        )
        raw = self.read_f32(out, seq * heads * dim)
        return raw.reshape(seq, heads, dim)

    def sync(self) -> None:
        """Commit the open command buffer and wait."""
        self._sync()

    def copy_in(self, array: np.ndarray) -> View:
        """Upload one contiguous float tensor. The caller drops it."""
        data = np.ascontiguousarray(array)
        if data.dtype == np.float16:
            kind = "f16"
        elif data.dtype == np.float32:
            kind = "f32"
        else:
            data = np.ascontiguousarray(data, dtype=np.float32)
            kind = "f32"
        buf = memory.upload(data, owner=self._owner)
        self._keep.append(buf)
        return View(buf, tuple(int(dim) for dim in data.shape), kind, 0, int(data.nbytes), True)

    def copy_out(self, view: View) -> np.ndarray:
        """Read a tensor back. This waits for the open command buffer."""
        self.sync()
        item = 2 if view.kind == "f16" else 4
        mapped = memory.map(view.buf, view.offset, view.count * item)
        dtype = np.float16 if view.kind == "f16" else np.float32
        copied = np.frombuffer(mapped, dtype=dtype, count=view.count).copy()
        return copied.reshape(view.shape)

    def vector(self, key: str) -> View:
        """A resident norm or BatchNorm vector. It stays until ``drop_prefix``."""
        found = self._vectors.get(key)
        if found is not None:
            return found
        if self._store is None:
            message = f"{key} is not loaded"
            raise KeyError(message)
        spec = self._store.index["vectors"][key]
        kind = "f32" if spec["dtype"] == "f32" else "f16"
        count = int(spec["n"])
        item = 4 if kind == "f32" else 2
        view = View(
            self._read_file(self._store.path / spec["file"]),
            (count,),
            kind,
            0,
            max(count * item, 16),
            False,
        )
        self._vectors[key] = view
        return view

    def drop(self, *views: View | None) -> None:
        """Return owning tensors to the pool. A slice is left in place."""
        for view in views:
            if view is None or not view.owner:
                continue
            self._free.append((view.capacity, view.buf))
            view.owner = False

    def reshape(self, view: View, shape: tuple[int, ...]) -> View:
        count = 1
        for dim in shape:
            count *= int(dim)
        if count != view.count:
            message = f"reshape {view.shape} to {shape} changes the count"
            raise ValueError(message)
        view.shape = tuple(int(dim) for dim in shape)
        return view

    def slice_rows(self, view: View, start: int, count: int) -> View:
        """A contiguous row span. It does not own the buffer."""
        if count < 1 or start < 0 or start + count > int(view.shape[0]):
            message = f"rows {start}:{start + count} are outside {view.shape}"
            raise ValueError(message)
        width = view.count // int(view.shape[0])
        item = 2 if view.kind == "f16" else 4
        offset = int(view.offset) + int(start) * width * item
        return View(view.buf, (int(count), *view.shape[1:]), view.kind, offset, 0, False)

    def span(self, view: View, start: int, count: int) -> View:
        """A contiguous element span of a vector."""
        item = 2 if view.kind == "f16" else 4
        offset = int(view.offset) + int(start) * item
        return View(view.buf, (int(count),), view.kind, offset, 0, False)

    def prefix_channels(self, view: View, channels: int) -> View:
        taken = int(channels)
        if taken < 1 or taken > int(view.shape[0]):
            message = f"channels {taken} are outside {view.shape}"
            raise ValueError(message)
        return View(view.buf, (taken, *view.shape[1:]), view.kind, view.offset, 0, False)

    def clone(self, src: View) -> View:
        dst = self._alloc(src.shape, src.kind)
        self._copy_into(src, dst, 0)
        return dst

    def concat_rows(self, parts: list[View]) -> View:
        kind = parts[0].kind
        rest = parts[0].shape[1:]
        rows = 0
        for part in parts:
            if part.kind != kind or part.shape[1:] != rest:
                message = "concat rows want the same tail"
                raise ValueError(message)
            rows += int(part.shape[0])
        dst = self._alloc((rows, *rest), kind)
        cursor = 0
        for part in parts:
            width = part.count // int(part.shape[0])
            self._copy_into(part, dst, cursor * width)
            cursor += int(part.shape[0])
        return dst

    def concat_cols(self, parts: list[View]) -> View:
        rows = int(parts[0].shape[0])
        kind = parts[0].kind
        cols = 0
        for part in parts:
            if part.kind != "f16" or int(part.shape[0]) != rows:
                message = "concat columns want matching float16 rows"
                raise ValueError(message)
            cols += int(part.shape[1])
        dst = self._alloc((rows, cols), kind)
        col = 0
        for part in parts:
            width = int(part.shape[1])
            self._dispatch(
                "place_cols_f16",
                [part.buf, dst.buf],
                [_u32(rows), _u32(width), _u32(cols), _u32(col)],
                *self._launch(rows * width),
                [part.offset, dst.offset],
            )
            col += width
        return dst

    def linear_view(self, src: View, key: str, *, store: str = "f16") -> View:
        """``[rows, K]`` times one packed weight. ``store`` is ``f16`` or ``f32``."""
        bank = self.spec(key)
        if src.kind not in ("f16", "f32") or len(src.shape) != 2 or int(src.shape[1]) != bank.k:
            message = f"{key} wants float [rows, {bank.k}], got {src.kind} {src.shape}"
            raise ValueError(message)
        if store not in ("f16", "f32"):
            message = f"store {store} is not f16 or f32"
            raise ValueError(message)
        rows = int(src.shape[0])
        padded = pad_rows(max(rows, 1))
        groups = bank.stored_k // GROUP
        codes = self.scratch("codes", padded * bank.stored_k)
        scales = self.scratch("scales", padded * groups * 4)
        dest = self.scratch("dest", padded * bank.stored_n * 4)
        quant = "quantize_f32_i8" if src.kind == "f32" else "quantize_f16_i8"
        self._dispatch(
            quant,
            [src.buf, codes, scales],
            [_u32(rows), _u32(bank.k), _u32(bank.stored_k), _u32(bank.k)],
            (groups, max(rows, 1), 1),
            (1, 1, 1),
            [src.offset, 0, 0],
        )
        return self._finish_linear(rows, bank, codes, scales, dest, store)

    def _finish_linear(
        self,
        rows: int,
        bank: Bank,
        codes: Any,
        scales: Any,
        dest: Any,
        store: str,
    ) -> View:
        padded = pad_rows(max(rows, 1))
        bias = bank.bias if bank.bias is not None else self.scratch("nobias", 16)
        self._dispatch(
            "matmul_i8",
            [codes, scales, bank.codes, bank.scales, bias, dest],
            [
                _u32(padded),
                _u32(bank.stored_n),
                _u32(bank.stored_k),
                _u32(bank.n),
                _u32(bank.stored_n),
                _u32(0 if bank.bias is None else 1),
            ],
            (bank.stored_n // TILE_N, padded // TILE_M, 1),
            self._threads,
        )
        out = self._alloc((rows, bank.n), store)
        name = "store_f32" if store == "f32" else "store_f16"
        self._dispatch(
            name,
            [dest, out.buf],
            [_u32(rows), _u32(bank.stored_n), _u32(bank.n)],
            *self._launch(rows * bank.n),
            [0, out.offset],
        )
        return out

    def rmsnorm(self, src: View, weight: View, *, store: str) -> View:
        dim = int(src.shape[-1])
        rows = src.count // dim
        kind = "f32" if store == "f32" else "f16"
        dst = self._alloc(src.shape, kind)
        dummy = self.scratch("dummy", 256)
        half_buf, half_off = (dst.buf, dst.offset) if kind == "f16" else (dummy, 0)
        float_buf, float_off = (dst.buf, dst.offset) if kind == "f32" else (dummy, 0)
        self._dispatch(
            "rmsnorm_row",
            [src.buf, weight.buf, half_buf, float_buf],
            [_u32(rows), _u32(dim), _u32(0 if kind == "f16" else 1), _f32(NORM_EPS)],
            (max(rows, 1), 1, 1),
            (256, 1, 1),
            [src.offset, weight.offset, half_off, float_off],
        )
        return dst

    def modulate(self, src: View, shift: View, scale: View) -> View:
        dim = int(src.shape[-1])
        rows = src.count // dim
        dst = self._alloc(src.shape, "f16")
        self._dispatch(
            "modulate_row",
            [src.buf, shift.buf, scale.buf, dst.buf],
            [_u32(rows), _u32(dim), _f32(NORM_EPS)],
            (max(rows, 1), 1, 1),
            (256, 1, 1),
            [src.offset, shift.offset, scale.offset, dst.offset],
        )
        return dst

    def silu(self, src: View) -> View:
        dst = self._alloc(src.shape, "f16")
        self._dispatch(
            "silu_f16",
            [src.buf, dst.buf],
            [_u32(src.count)],
            *self._launch(src.count),
            [src.offset, dst.offset],
        )
        return dst

    def silu_mul(self, gate: View, up: View) -> View:
        dst = self._alloc(gate.shape, "f16")
        self._dispatch(
            "silu_mul_f16",
            [gate.buf, up.buf, dst.buf],
            [_u32(gate.count)],
            *self._launch(gate.count),
            [gate.offset, up.offset, dst.offset],
        )
        return dst

    def swiglu(self, src: View) -> View:
        rows, cols = int(src.shape[0]), int(src.shape[1])
        dst = self._alloc((rows, cols // 2), "f16")
        self._dispatch(
            "swiglu_f16",
            [src.buf, dst.buf],
            [_u32(rows), _u32(cols)],
            *self._launch(dst.count),
            [src.offset, dst.offset],
        )
        return dst

    def gate(self, residual: View, gate: View | None, branch: View) -> View:
        cols = int(residual.shape[-1])
        dst = self._alloc(residual.shape, "f16")
        gate_buf = self.scratch("dummy", 256) if gate is None else gate.buf
        gate_off = 0 if gate is None else gate.offset
        self._dispatch(
            "gate_f16",
            [residual.buf, gate_buf, branch.buf, dst.buf],
            [_u32(residual.count), _u32(cols), _u32(0 if gate is None else 1)],
            *self._launch(residual.count),
            [residual.offset, gate_off, branch.offset, dst.offset],
        )
        return dst

    def rope_text(self, src: View, cos: View, sin: View) -> View:
        return self._rope("rope_text", src, cos, sin)

    def rope_dit(self, src: View, cos: View, sin: View) -> View:
        return self._rope("rope_dit", src, cos, sin)

    def repeat_heads(self, src: View, times: int) -> View:
        seq, kv_heads, dim = (int(src.shape[0]), int(src.shape[1]), int(src.shape[2]))
        dst = self._alloc((seq, kv_heads * int(times), dim), "f32")
        self._dispatch(
            "repeat_heads",
            [src.buf, dst.buf],
            [_u32(seq), _u32(kv_heads), _u32(int(times)), _u32(dim)],
            *self._launch(dst.count),
            [src.offset, dst.offset],
        )
        return dst

    def attend_view(self, query: View, key: View, value: View, *, causal: bool, valid: int) -> View:
        seq, heads, dim = (int(query.shape[0]), int(query.shape[1]), int(query.shape[2]))
        kv_heads = int(key.shape[1])
        dst = self._alloc(query.shape, "f32")
        self._attend(
            query.buf,
            key.buf,
            value.buf,
            dst.buf,
            seq,
            heads,
            kv_heads,
            dim,
            int(valid),
            causal,
            [query.offset, key.offset, value.offset, dst.offset],
        )
        return dst

    def _attend(
        self,
        query: Any,
        key: Any,
        value: Any,
        dest: Any,
        seq: int,
        q_heads: int,
        kv_heads: int,
        dim: int,
        valid: int,
        causal: bool,
        offsets: list[int] | None,
    ) -> None:
        flash = dim == 128
        if flash:
            rows = (seq + 31) // 32
            self._dispatch(
                "attend_flash",
                [query, key, value, dest],
                [
                    _u32(seq),
                    _u32(valid),
                    _u32(q_heads),
                    _u32(kv_heads),
                    _u32(dim),
                    _u32(1 if causal else 0),
                ],
                (q_heads, rows, 1),
                (128, 1, 1),
                offsets,
            )
            return
        self._dispatch(
            "attend_f32",
            [query, key, value, dest],
            [_u32(seq), _u32(q_heads), _u32(dim), _u32(valid), _u32(1 if causal else 0)],
            (q_heads, seq, 1),
            (dim, 1, 1),
            offsets,
        )

    def cast(self, src: View, kind: str) -> View:
        name = "cast_f16_f32" if kind == "f32" else "cast_f32_f16"
        dst = self._alloc(src.shape, kind)
        self._dispatch(
            name,
            [src.buf, dst.buf],
            [_u32(src.count)],
            *self._launch(src.count),
            [src.offset, dst.offset],
        )
        return dst

    def slice_cols(self, src: View, start: int, count: int) -> View:
        rows, stride = int(src.shape[0]), int(src.shape[1])
        dst = self._alloc((rows, int(count)), "f16")
        self._dispatch(
            "slice_cols_f16",
            [src.buf, dst.buf],
            [_u32(rows), _u32(stride), _u32(int(start)), _u32(int(count))],
            *self._launch(dst.count),
            [src.offset, dst.offset],
        )
        return dst

    def pack_attn_mlp(self, attn: View, mlp: View) -> View:
        rows = int(attn.shape[0])
        attn_cols = attn.count // rows
        mlp_cols = int(mlp.shape[1])
        dst = self._alloc((rows, attn_cols + mlp_cols), "f16")
        self._dispatch(
            "pack_attn_mlp",
            [attn.buf, mlp.buf, dst.buf],
            [_u32(rows), _u32(attn_cols), _u32(mlp_cols)],
            *self._launch(dst.count),
            [attn.offset, mlp.offset, dst.offset],
        )
        return dst

    def groupnorm(self, src: View, weight: View, bias: View, *, silu: bool) -> View:
        channels, height, width = (int(src.shape[0]), int(src.shape[1]), int(src.shape[2]))
        if channels % VAE_GROUPS:
            message = f"group norm wants a multiple of {VAE_GROUPS} channels, got {channels}"
            raise ValueError(message)
        kind = "f32" if src.kind == "f32" else "f16"
        name = "groupnorm_f32" if kind == "f32" else "groupnorm_f16"
        dst = self._alloc(src.shape, kind)
        self._dispatch(
            name,
            [src.buf, weight.buf, bias.buf, dst.buf],
            [
                _u32(channels),
                _u32(height),
                _u32(width),
                _u32(VAE_GROUPS),
                _u32(1 if silu else 0),
                _f32(VAE_EPS),
            ],
            (VAE_GROUPS, 1, 1),
            (256, 1, 1),
            [src.offset, weight.offset, bias.offset, dst.offset],
        )
        return dst

    def conv(self, volume: View, key: str, *, stride: int, pads: tuple[int, int, int, int]) -> View:
        """Convolution. ``pads`` is top, bottom, left, right. The input stays."""
        bank = self.spec(key)
        if bank.kh is None or bank.kw is None:
            message = f"{key} is not a convolution"
            raise ValueError(message)
        channels, height, width = (int(volume.shape[0]), int(volume.shape[1]), int(volume.shape[2]))
        top, bottom, left, right = pads
        out_h = (height + top + bottom - int(bank.kh)) // stride + 1
        out_w = (width + left + right - int(bank.kw)) // stride + 1
        k_total = channels * int(bank.kh) * int(bank.kw)
        if k_total != bank.k or out_h < 1 or out_w < 1:
            message = f"{key} cannot map {volume.shape} with stride {stride}"
            raise ValueError(message)
        return self._conv_spans(
            volume, bank, channels, height, width, k_total, stride, top, left, out_h, out_w
        )

    def _conv_spans(
        self,
        volume: View,
        bank: Bank,
        channels: int,
        height: int,
        width: int,
        k_total: int,
        stride: int,
        top: int,
        left: int,
        out_h: int,
        out_w: int,
    ) -> View:
        """Quantize the volume into the int8 product. Each span fits in a 32-bit tensor index."""
        rows = out_h * out_w
        kind = "f32" if volume.kind == "f32" else "f16"
        span = _span_rows(bank.stored_k)
        padded = pad_rows(max(min(rows, span), 1))
        groups = bank.stored_k // GROUP
        codes = self.scratch("codes", padded * bank.stored_k)
        scales = self.scratch("scales", padded * groups * 4)
        dest = self.scratch("dest", padded * bank.stored_n * 4)
        gather = "gather_quantize_f32" if kind == "f32" else "gather_quantize_f16"
        chw_name = "chw_from_rows_f32" if kind == "f32" else "chw_from_rows"
        chw = self._alloc((bank.n, out_h, out_w), kind)
        row0 = 0
        while row0 < rows:
            take = min(span, rows - row0)
            self._dispatch(
                gather,
                [volume.buf, codes, scales],
                [
                    _u32(channels),
                    _u32(height),
                    _u32(width),
                    _u32(int(bank.kh)),
                    _u32(int(bank.kw)),
                    _u32(stride),
                    _u32(top),
                    _u32(left),
                    _u32(out_w),
                    _u32(k_total),
                    _u32(bank.stored_k),
                    _u32(row0),
                    _u32(take),
                ],
                (groups, max(take, 1), 1),
                (1, 1, 1),
                [volume.offset, 0, 0],
            )
            flat = self._finish_linear(take, bank, codes, scales, dest, kind)
            self._dispatch(
                chw_name,
                [flat.buf, chw.buf],
                [_u32(bank.n), _u32(out_h), _u32(out_w), _u32(row0), _u32(take)],
                *self._launch(take * bank.n),
                [flat.offset, chw.offset],
            )
            self.drop(flat)
            row0 += take
        return chw

    def nearest2(self, src: View) -> View:
        channels, height, width = (int(src.shape[0]), int(src.shape[1]), int(src.shape[2]))
        kind = "f32" if src.kind == "f32" else "f16"
        name = "nearest2_f32" if kind == "f32" else "nearest2_f16"
        dst = self._alloc((channels, height * 2, width * 2), kind)
        self._dispatch(
            name,
            [src.buf, dst.buf],
            [_u32(channels), _u32(height), _u32(width)],
            *self._launch(dst.count),
            [src.offset, dst.offset],
        )
        return dst

    def add(self, left: View, right: View) -> View:
        kind = "f32" if left.kind == "f32" else "f16"
        name = "add_f32" if kind == "f32" else "add_f16"
        dst = self._alloc(left.shape, kind)
        self._dispatch(
            name,
            [left.buf, right.buf, dst.buf],
            [_u32(left.count)],
            *self._launch(left.count),
            [left.offset, right.offset, dst.offset],
        )
        return dst

    def seq_from_chw(self, src: View) -> View:
        channels, height, width = (int(src.shape[0]), int(src.shape[1]), int(src.shape[2]))
        kind = "f32" if src.kind == "f32" else "f16"
        name = "seq_from_chw_f32" if kind == "f32" else "seq_from_chw"
        dst = self._alloc((height * width, channels), kind)
        self._dispatch(
            name,
            [src.buf, dst.buf],
            [_u32(channels), _u32(height), _u32(width)],
            *self._launch(src.count),
            [src.offset, dst.offset],
        )
        return dst

    def chw_from_seq(self, src: View, height: int, width: int) -> View:
        channels = int(src.shape[1])
        kind = "f32" if src.kind == "f32" else "f16"
        name = "chw_from_rows_f32" if kind == "f32" else "chw_from_rows"
        dst = self._alloc((channels, int(height), int(width)), kind)
        self._dispatch(
            name,
            [src.buf, dst.buf],
            [_u32(channels), _u32(int(height)), _u32(int(width)), _u32(0), _u32(int(height) * int(width))],
            *self._launch(dst.count),
            [src.offset, dst.offset],
        )
        return dst

    def patch_pack(self, src: View, mean: View, var: View) -> View:
        _channels, height, width = (int(src.shape[0]), int(src.shape[1]), int(src.shape[2]))
        grid_w = width // 2
        dst = self._alloc(((height // 2) * grid_w, 128), "f16")
        self._dispatch(
            "patch_pack",
            [src.buf, mean.buf, var.buf, dst.buf],
            [_u32(height), _u32(width), _u32(grid_w), _f32(BN_EPS)],
            *self._launch2(128, dst.shape[0]),
            [src.offset, mean.offset, var.offset, dst.offset],
        )
        return dst

    def unpack_latent(self, src: View, mean: View, var: View, grid_h: int, grid_w: int) -> View:
        height, width = int(grid_h) * 2, int(grid_w) * 2
        dst = self._alloc((32, height, width), "f32")
        self._dispatch(
            "unpack_latent",
            [src.buf, mean.buf, var.buf, dst.buf],
            [_u32(height), _u32(width), _u32(int(grid_w)), _f32(BN_EPS)],
            *self._launch2(128, int(grid_h) * int(grid_w)),
            [src.offset, mean.offset, var.offset, dst.offset],
        )
        return dst

    def euler(self, state: View, velocity: View, delta: float) -> View:
        dst = self._alloc(state.shape, "f16")
        self._dispatch(
            "euler_f16",
            [state.buf, velocity.buf, dst.buf],
            [_u32(state.count), _f32(float(delta))],
            *self._launch(state.count),
            [state.offset, velocity.offset, dst.offset],
        )
        return dst

    def im2col(self, volume: View, kh: int, kw: int, stride: int, pads: tuple[int, int, int, int]) -> View:
        """Gather for the kernels check. ``pads`` is top, bottom, left, right."""
        channels, height, width = (int(volume.shape[0]), int(volume.shape[1]), int(volume.shape[2]))
        top, bottom, left, right = pads
        out_h = (height + top + bottom - kh) // stride + 1
        out_w = (width + left + right - kw) // stride + 1
        k_total = channels * kh * kw
        dst = self._alloc((out_h * out_w, k_total), "f16")
        self._dispatch(
            "im2col_f16",
            [volume.buf, dst.buf],
            [
                _u32(channels),
                _u32(height),
                _u32(width),
                _u32(kh),
                _u32(kw),
                _u32(stride),
                _u32(top),
                _u32(left),
                _u32(out_h),
                _u32(out_w),
            ],
            *self._launch2(k_total, out_h * out_w),
            [volume.offset, dst.offset],
        )
        return dst

    def release_pool(self) -> None:
        """Free pooled tensors after the open command buffer finishes. Scratch stays."""
        self.sync()
        pooled = self._free
        self._free = []
        for _capacity, buf in pooled:
            self._release_handle(buf)

    def release_workspace(self) -> None:
        """Free scratch and the pool. Live views stay. Call after the stage has synced."""
        self.sync()
        for _size, buf in self._scratch.values():
            self._release_handle(buf)
        self._scratch.clear()
        for _capacity, buf in self._free:
            self._release_handle(buf)
        self._free.clear()
        self._release_retired()

    def drop_prefix(self, prefix: str) -> None:
        """Release device copies whose key starts with ``prefix``."""
        for key in list(self._bank):
            if not key.startswith(prefix):
                continue
            bank = self._bank.pop(key)
            self._release_handle(bank.codes)
            self._release_handle(bank.scales)
            if bank.bias is not None:
                self._release_handle(bank.bias)
        for key in list(self._vectors):
            if key.startswith(prefix):
                self._release_handle(self._vectors.pop(key).buf)

    def close(self) -> None:
        if self.queue is not None:
            self.sync()
        self._cmd = None
        self._enc = None
        self._held.clear()
        for buf in self._owned + self._keep:
            release = getattr(buf, "release", None)
            if callable(release):
                release()
        self._owned.clear()
        self._keep.clear()
        self._bank.clear()
        self._vectors.clear()
        self._scratch.clear()
        self._free.clear()
        self.pipes.clear()
        self.queue = None
        self.device = None
        memory.sweep(self._owner)

    def _product(self, values: np.ndarray, bank: Bank) -> np.ndarray:
        rows = int(values.shape[0])
        padded = pad_rows(max(rows, 1))
        groups = bank.stored_k // GROUP
        if bank.stored_k % GROUP or bank.stored_n % TILE_N:
            message = f"tile mismatch n={bank.stored_n} k={bank.stored_k}"
            raise ValueError(message)
        codes = self.scratch("codes", padded * bank.stored_k)
        scales = self.scratch("scales", padded * groups * 4)
        dest = self.scratch("dest", padded * bank.stored_n * 4)
        memory.zero(codes, 0, padded * bank.stored_k)
        memory.zero(scales, 0, padded * groups * 4)
        self._dispatch(
            "quantize_f16_i8",
            [self.upload(values), codes, scales],
            [_u32(rows), _u32(bank.k), _u32(bank.stored_k), _u32(bank.k)],
            (groups, max(rows, 1), 1),
            (1, 1, 1),
        )
        bias = bank.bias if bank.bias is not None else self.scratch("nobias", 16)
        self._dispatch(
            "matmul_i8",
            [codes, scales, bank.codes, bank.scales, bias, dest],
            [
                _u32(padded),
                _u32(bank.stored_n),
                _u32(bank.stored_k),
                _u32(bank.n),
                _u32(bank.stored_n),
                _u32(0 if bank.bias is None else 1),
            ],
            (bank.stored_n // TILE_N, padded // TILE_M, 1),
            self._threads,
        )
        raw = self.read_f32(dest, rows * bank.stored_n)
        return raw.reshape(rows, bank.stored_n).astype(np.float16)

    def read_f32(self, buf: Any, count: int) -> np.ndarray:
        self._sync()
        if count < 1:
            return np.zeros(0, dtype=np.float32)
        view = buf.contents().as_buffer(int(count) * 4)
        copied = np.frombuffer(view, dtype=np.float32, count=int(count)).copy()
        self._drop_owned()
        return copied

    def _release_retired(self) -> None:
        retired = self._retired
        self._retired = []
        for buf in retired:
            self._release_handle(buf)

    def _sync(self) -> None:
        enc = self._enc
        cmd = self._cmd
        if enc is None or cmd is None:
            self._release_retired()
            return
        self._enc = None
        self._cmd = None
        memory.handoff(enc, cmd)
        enc.endEncoding()
        cmd.commit()
        status, error = memory.retire(cmd)
        self._held.clear()
        self._release_retired()
        if status == int(Metal.MTLCommandBufferStatusError):
            message = f"metal command failed: {error}"
            raise RuntimeError(message)

    def _drop_owned(self) -> None:
        for buf in self._owned:
            release = getattr(buf, "release", None)
            if callable(release):
                release()
        self._owned.clear()

    def _alloc(self, shape: tuple[int, ...], kind: str) -> View:
        item = 2 if kind == "f16" else 4
        count = 1
        for dim in shape:
            count *= int(dim)
        need = max(count * item, 16)
        best_at = -1
        best = 0
        for index, (capacity, _buf) in enumerate(self._free):
            if capacity >= need and (best_at < 0 or capacity < best):
                best = capacity
                best_at = index
        if best_at >= 0 and best <= need * 2:
            capacity, buf = self._free.pop(best_at)
            return View(buf, tuple(int(dim) for dim in shape), kind, 0, capacity, True)
        capacity = 16
        while capacity < need:
            capacity *= 2
        buf = device_buffer(self._owner, capacity, zero=False)
        self._keep.append(buf)
        return View(buf, tuple(int(dim) for dim in shape), kind, 0, capacity, True)

    def _launch(self, count: int) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
        groups = max(1, (int(count) + 255) // 256)
        return (groups, 1, 1), (256, 1, 1)

    def _launch2(self, width: int, height: int) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
        return (
            (max(1, (int(width) + 15) // 16), max(1, (int(height) + 15) // 16), 1),
            (16, 16, 1),
        )

    def _copy_into(self, src: View, dst: View, dst_element: int) -> None:
        item = 2 if src.kind == "f16" else 4
        name = "copy_f16" if src.kind == "f16" else "copy_f32"
        self._dispatch(
            name,
            [src.buf, dst.buf],
            [_u32(src.count)],
            *self._launch(src.count),
            [src.offset, dst.offset + int(dst_element) * item],
        )

    def _rope(self, name: str, src: View, cos: View, sin: View) -> View:
        seq, heads, dim = (int(src.shape[0]), int(src.shape[1]), int(src.shape[2]))
        dst = self._alloc(src.shape, "f32")
        self._dispatch(
            name,
            [src.buf, cos.buf, sin.buf, dst.buf],
            [_u32(seq), _u32(heads), _u32(dim)],
            *self._launch(src.count),
            [src.offset, cos.offset, sin.offset, dst.offset],
        )
        return dst

    def _release_handle(self, buf: Any) -> None:
        identity = getattr(buf, "id", None)
        self._keep = [item for item in self._keep if getattr(item, "id", None) != identity]
        self._free = [item for item in self._free if getattr(item[1], "id", None) != identity]
        buf.release()

    def _dispatch(
        self,
        name: str,
        buffers: list[Any],
        constants: list[bytes],
        grid: tuple[int, int, int],
        threads: tuple[int, int, int],
        offsets: list[int] | None = None,
    ) -> None:
        if self._enc is None:
            self._cmd = memory.command_buffer(self.queue)
            self._enc = self._cmd.computeCommandEncoder()
        enc = self._enc
        enc.setComputePipelineState_(self.pipes[name])
        if offsets is None:
            offsets = [0] * len(buffers)
        for index, buf in enumerate(buffers):
            offset = int(offsets[index])
            if offset % 16:
                message = f"{name} buffer {index} offset {offset} is not 16-byte aligned"
                raise ValueError(message)
            memory.bind(enc, index, buf, offset)
        for index, blob in enumerate(constants):
            self._held.append(blob)
            enc.setBytes_length_atIndex_(blob, len(blob), len(buffers) + index)
        enc.dispatchThreadgroups_threadsPerThreadgroup_(
            Metal.MTLSizeMake(int(grid[0]), int(grid[1]), int(grid[2])),
            Metal.MTLSizeMake(int(threads[0]), int(threads[1]), int(threads[2])),
        )


def _span_rows(stored_k: int) -> int:
    """Rows whose ``stored_k`` product stays under 2^31 elements."""
    room = 2_000_000_000 // max(int(stored_k), 1)
    return max(TILE_M, (room // TILE_M) * TILE_M)


def _u32(value: int) -> bytes:
    return struct.pack("I", int(value))


def _f32(value: float) -> bytes:
    return struct.pack("f", float(value))
