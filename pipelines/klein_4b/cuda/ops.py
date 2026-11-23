"""Device planes. Weights stay in the store. Activations are freed by the caller."""

from __future__ import annotations

import ctypes
from dataclasses import dataclass

import numpy as np

from klein_4b.cuda import engine
from klein_4b.cuda.store import Tensor


@dataclass
class Plane:
    ptr: int
    rows: int
    cols: int


@dataclass
class Volume:
    ptr: int
    channels: int
    height: int
    width: int
    fp32: bool = False


class Ops:
    def __init__(self) -> None:
        self.live: list[int] = []

    def close(self) -> None:
        engine.sync()
        for ptr in self.live:
            engine.free(ptr)
        self.live.clear()

    def free(self, *items: Plane | Volume | int | None) -> None:
        ptrs = [self._ptr(item) for item in items if item is not None]
        if not ptrs:
            return
        engine.sync()
        for ptr in ptrs:
            if ptr in self.live:
                self.live.remove(ptr)
                engine.free(ptr)

    def plane(self, rows: int, cols: int) -> Plane:
        ptr = engine.alloc(rows * cols * 2)
        self.live.append(ptr)
        return Plane(ptr, rows, cols)

    def volume(self, channels: int, height: int, width: int, *, fp32: bool) -> Volume:
        ptr = engine.alloc(channels * height * width * (4 if fp32 else 2))
        self.live.append(ptr)
        return Volume(ptr, channels, height, width, fp32)

    def upload_f16(self, values: np.ndarray, rows: int, cols: int) -> Plane:
        array = np.ascontiguousarray(values, dtype=np.float16)
        ptr = self._copy_host(array.tobytes())
        return Plane(ptr, rows, cols)

    def upload_f32(self, values: np.ndarray) -> int:
        array = np.ascontiguousarray(values, dtype=np.float32)
        return self._copy_host(array.tobytes())

    def upload_i32(self, values: np.ndarray) -> int:
        array = np.ascontiguousarray(values, dtype=np.int32)
        return self._copy_host(array.tobytes())

    def upload_volume(self, values: np.ndarray, *, fp32: bool) -> Volume:
        array = np.ascontiguousarray(values, dtype=np.float32 if fp32 else np.float16)
        channels, height, width = array.shape
        ptr = self._copy_host(array.tobytes())
        return Volume(ptr, channels, height, width, fp32)

    def download_f16(self, plane: Plane) -> np.ndarray:
        count = plane.rows * plane.cols
        host = np.empty(count, dtype=np.float16)
        self._d2h(host, plane.ptr)
        return host.reshape(plane.rows, plane.cols)

    def download_volume(self, volume: Volume) -> np.ndarray:
        count = volume.channels * volume.height * volume.width
        host = np.empty(count, dtype=np.float32 if volume.fp32 else np.float16)
        self._d2h(host, volume.ptr)
        return host.reshape(volume.channels, volume.height, volume.width)

    def linear(self, act: Plane, weight: Tensor) -> Plane:
        src = act
        if act.cols != weight.stored_k:
            src = self.pad_k(act, weight.stored_k)
        out = self.plane(act.rows, weight.rows)
        engine.call(
            "klein_gemm",
            engine.pointer(src.ptr),
            engine.pointer(weight.codes),
            engine.pointer(weight.scales),
            engine.pointer(weight.bias),
            engine.pointer(out.ptr),
            act.rows,
            weight.rows,
            weight.stored_k,
        )
        if src is not act:
            self.free(src)
        return out

    def embed(self, ids: np.ndarray, weight: Tensor) -> Plane:
        uploaded = self.upload_i32(ids)
        out = self.plane(ids.shape[0], weight.stored_k)
        engine.call(
            "klein_embed",
            engine.pointer(uploaded),
            engine.pointer(weight.codes),
            engine.pointer(weight.scales),
            engine.pointer(out.ptr),
            int(ids.shape[0]),
            weight.stored_k,
        )
        self.free(uploaded)
        if weight.inner != weight.stored_k:
            sliced = self.slice_cols(out, 0, weight.inner)
            self.free(out)
            return sliced
        return out

    def rmsnorm(self, act: Plane, weight: Tensor | None, eps: float) -> Plane:
        out = self.plane(act.rows, act.cols)
        engine.call(
            "klein_rmsnorm",
            engine.pointer(act.ptr),
            engine.pointer(None if weight is None else weight.ptr),
            engine.pointer(out.ptr),
            act.rows,
            act.cols,
            ctypes.c_float(eps),
        )
        return out

    def rmsnorm_heads(self, act: Plane, weight: Tensor, heads: int, dim: int, eps: float) -> Plane:
        rows = act.rows * heads
        out = self.plane(act.rows, act.cols)
        engine.call(
            "klein_rmsnorm",
            engine.pointer(act.ptr),
            engine.pointer(weight.ptr),
            engine.pointer(out.ptr),
            rows,
            dim,
            ctypes.c_float(eps),
        )
        return out

    def layernorm(self, act: Plane, eps: float) -> Plane:
        out = self.plane(act.rows, act.cols)
        engine.call(
            "klein_layernorm",
            engine.pointer(act.ptr),
            engine.pointer(out.ptr),
            act.rows,
            act.cols,
            ctypes.c_float(eps),
        )
        return out

    def silu(self, act: Plane) -> Plane:
        out = self.plane(act.rows, act.cols)
        engine.call("klein_silu", engine.pointer(act.ptr), engine.pointer(out.ptr), act.rows * act.cols)
        return out

    def silu_mul(self, gate: Plane, up: Plane) -> Plane:
        out = self.plane(gate.rows, gate.cols)
        engine.call(
            "klein_silu_mul",
            engine.pointer(gate.ptr),
            engine.pointer(up.ptr),
            engine.pointer(out.ptr),
            gate.rows * gate.cols,
        )
        return out

    def swiglu(self, act: Plane) -> Plane:
        out = self.plane(act.rows, act.cols // 2)
        engine.call("klein_swiglu", engine.pointer(act.ptr), engine.pointer(out.ptr), act.rows, act.cols)
        return out

    def modulate(self, act: Plane, shift: Plane, scale: Plane) -> Plane:
        out = self.plane(act.rows, act.cols)
        engine.call(
            "klein_modulate",
            engine.pointer(act.ptr),
            engine.pointer(shift.ptr),
            engine.pointer(scale.ptr),
            engine.pointer(out.ptr),
            act.rows,
            act.cols,
        )
        return out

    def gated(self, residual: Plane, gate: Plane | None, branch: Plane) -> Plane:
        out = self.plane(residual.rows, residual.cols)
        engine.call(
            "klein_gate",
            engine.pointer(residual.ptr),
            engine.pointer(None if gate is None else gate.ptr),
            engine.pointer(branch.ptr),
            engine.pointer(out.ptr),
            residual.rows,
            residual.cols,
        )
        return out

    def rope(self, act: Plane, seq: int, heads: int, dim: int, cos: int, sin: int, *, text: bool) -> Plane:
        out = self.plane(act.rows, act.cols)
        name = "klein_rope_text" if text else "klein_rope_dit"
        engine.call(
            name,
            engine.pointer(act.ptr),
            engine.pointer(cos),
            engine.pointer(sin),
            engine.pointer(out.ptr),
            seq,
            heads,
            dim,
        )
        return out

    def attend(
        self,
        query: Plane,
        key: Plane,
        value: Plane,
        seq: int,
        heads: int,
        kv_heads: int,
        dim: int,
        causal: int,
        valid: int,
    ) -> Plane:
        out = self.plane(seq, heads * dim)
        scale = float(dim) ** -0.5
        engine.call(
            "klein_attend",
            engine.pointer(query.ptr),
            engine.pointer(key.ptr),
            engine.pointer(value.ptr),
            engine.pointer(out.ptr),
            seq,
            heads,
            kv_heads,
            dim,
            causal,
            valid,
            ctypes.c_float(scale),
        )
        return out

    def slice_cols(self, act: Plane, start: int, width: int) -> Plane:
        out = self.plane(act.rows, width)
        engine.call(
            "klein_slice_cols",
            engine.pointer(act.ptr),
            engine.pointer(out.ptr),
            act.rows,
            act.cols,
            start,
            width,
        )
        return out

    def slice_rows(self, act: Plane, start: int, count: int) -> Plane:
        out = self.plane(count, act.cols)
        src = act.ptr + start * act.cols * 2
        engine.call(
            "klein_copy_d2d",
            engine.pointer(out.ptr),
            engine.pointer(src),
            ctypes.c_uint64(count * act.cols * 2),
        )
        return out

    def concat_rows(self, parts: list[Plane]) -> Plane:
        cols = parts[0].cols
        rows = sum(part.rows for part in parts)
        out = self.plane(rows, cols)
        offset = 0
        for part in parts:
            nbytes = part.rows * cols * 2
            engine.call(
                "klein_copy_d2d",
                engine.pointer(out.ptr + offset),
                engine.pointer(part.ptr),
                ctypes.c_uint64(nbytes),
            )
            offset += nbytes
        return out

    def concat_cols(self, left: Plane, right: Plane) -> Plane:
        out = self.plane(left.rows, left.cols + right.cols)
        engine.call(
            "klein_concat_cols",
            engine.pointer(left.ptr),
            engine.pointer(right.ptr),
            engine.pointer(out.ptr),
            left.rows,
            left.cols,
            right.cols,
        )
        return out

    def pad_k(self, act: Plane, stored: int) -> Plane:
        out = self.plane(act.rows, stored)
        engine.call(
            "klein_pad_k",
            engine.pointer(act.ptr),
            engine.pointer(out.ptr),
            act.rows,
            act.cols,
            stored,
        )
        return out

    def euler(self, latent: Plane, velocity: Plane, delta: float) -> Plane:
        out = self.plane(latent.rows, latent.cols)
        engine.call(
            "klein_euler",
            engine.pointer(latent.ptr),
            engine.pointer(velocity.ptr),
            engine.pointer(out.ptr),
            latent.rows * latent.cols,
            ctypes.c_float(delta),
        )
        return out

    def conv(self, volume: Volume, weight: Tensor, *, stride: int, pad: int) -> Volume:
        out_c, cin, kernel = _conv_shape(weight, volume.channels)
        out_h = (volume.height + 2 * pad - kernel) // stride + 1
        out_w = (volume.width + 2 * pad - kernel) // stride + 1
        out = self.volume(out_c, out_h, out_w, fp32=volume.fp32)
        logical = cin * kernel * kernel
        engine.call(
            "klein_conv",
            engine.pointer(volume.ptr),
            engine.pointer(weight.codes),
            engine.pointer(weight.scales),
            engine.pointer(weight.bias),
            engine.pointer(out.ptr),
            volume.channels,
            volume.height,
            volume.width,
            kernel,
            stride,
            pad,
            out_h,
            out_w,
            out_c,
            logical,
            weight.stored_k,
            1 if volume.fp32 else 0,
        )
        return out

    def groupnorm(self, volume: Volume, weight: Tensor, *, groups: int, eps: float, silu: bool) -> Volume:
        out = self.volume(volume.channels, volume.height, volume.width, fp32=volume.fp32)
        scratch = engine.alloc(groups * 2 * 4)
        self.live.append(scratch)
        spatial = volume.height * volume.width
        engine.call(
            "klein_groupnorm",
            engine.pointer(volume.ptr),
            engine.pointer(weight.ptr),
            engine.pointer(weight.bias),
            engine.pointer(out.ptr),
            engine.pointer(scratch),
            volume.channels,
            spatial,
            groups,
            ctypes.c_float(eps),
            1 if silu else 0,
            1 if volume.fp32 else 0,
        )
        self.free(scratch)
        return out

    def pad_edge(self, volume: Volume) -> Volume:
        out = self.volume(volume.channels, volume.height + 1, volume.width + 1, fp32=volume.fp32)
        engine.call(
            "klein_pad_edge",
            engine.pointer(volume.ptr),
            engine.pointer(out.ptr),
            volume.channels,
            volume.height,
            volume.width,
            1,
            1,
            1 if volume.fp32 else 0,
        )
        return out

    def nearest2(self, volume: Volume) -> Volume:
        out = self.volume(volume.channels, volume.height * 2, volume.width * 2, fp32=volume.fp32)
        engine.call(
            "klein_nearest2",
            engine.pointer(volume.ptr),
            engine.pointer(out.ptr),
            volume.channels,
            volume.height,
            volume.width,
            1 if volume.fp32 else 0,
        )
        return out

    def prefix_channels(self, volume: Volume, keep: int) -> Volume:
        out = self.volume(keep, volume.height, volume.width, fp32=volume.fp32)
        engine.call(
            "klein_prefix_channels",
            engine.pointer(volume.ptr),
            engine.pointer(out.ptr),
            keep,
            volume.channels,
            volume.height * volume.width,
        )
        return out

    def to_tokens(self, volume: Volume) -> Plane:
        spatial = volume.height * volume.width
        out = self.plane(spatial, volume.channels)
        engine.call(
            "klein_transpose",
            engine.pointer(volume.ptr),
            engine.pointer(out.ptr),
            volume.channels,
            spatial,
            1,
            1 if volume.fp32 else 0,
        )
        return out

    def to_volume(self, tokens: Plane, height: int, width: int, *, fp32: bool) -> Volume:
        out = self.volume(tokens.cols, height, width, fp32=fp32)
        engine.call(
            "klein_transpose",
            engine.pointer(tokens.ptr),
            engine.pointer(out.ptr),
            tokens.cols,
            height * width,
            0,
            1 if fp32 else 0,
        )
        return out

    def add_volume(self, left: Volume, right: Volume) -> Volume:
        out = self.volume(left.channels, left.height, left.width, fp32=left.fp32)
        count = left.channels * left.height * left.width
        name = "klein_add_f32" if left.fp32 else "klein_add"
        engine.call(name, engine.pointer(left.ptr), engine.pointer(right.ptr), engine.pointer(out.ptr), count)
        return out

    def pack_latent(self, mean: Volume, running_mean: Tensor, running_var: Tensor, grid_h: int, grid_w: int, eps: float) -> Plane:
        out = self.plane(grid_h * grid_w, 128)
        engine.call(
            "klein_pack_latent",
            engine.pointer(mean.ptr),
            engine.pointer(running_mean.ptr),
            engine.pointer(running_var.ptr),
            engine.pointer(out.ptr),
            grid_h,
            grid_w,
            ctypes.c_float(eps),
        )
        return out

    def unpack_latent(self, tokens: Plane, running_mean: Tensor, running_var: Tensor, grid_h: int, grid_w: int, eps: float) -> Volume:
        height = grid_h * 2
        width = grid_w * 2
        out = self.volume(32, height, width, fp32=True)
        engine.call(
            "klein_unpack_latent",
            engine.pointer(tokens.ptr),
            engine.pointer(running_mean.ptr),
            engine.pointer(running_var.ptr),
            engine.pointer(out.ptr),
            grid_h,
            grid_w,
            ctypes.c_float(eps),
        )
        return out

    def _copy_host(self, payload: bytes) -> int:
        ptr = engine.alloc(len(payload))
        self.live.append(ptr)
        buffer = ctypes.create_string_buffer(payload)
        engine.call("klein_copy_h2d", engine.pointer(ptr), ctypes.cast(buffer, ctypes.c_void_p), ctypes.c_uint64(len(payload)))
        return ptr

    def _d2h(self, host: np.ndarray, ptr: int) -> None:
        engine.call(
            "klein_copy_d2h",
            host.ctypes.data_as(ctypes.c_void_p),
            engine.pointer(ptr),
            ctypes.c_uint64(host.nbytes),
        )

    def _ptr(self, item: Plane | Volume | int) -> int:
        if isinstance(item, int):
            return item
        return item.ptr


def _conv_shape(weight: Tensor, channels: int) -> tuple[int, int, int]:
    shape = weight.shape
    if len(shape) == 4:
        out_c, cin, kh, kw = shape
        if kh != kw:
            message = "conv kernel is not square"
            raise ValueError(message)
        return out_c, cin, kh
    if len(shape) == 2:
        return shape[0], channels, 1
    message = f"bad conv shape {shape}"
    raise ValueError(message)
