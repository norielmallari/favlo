"""Metal device and the dispatches this pipeline's kernels command accepts."""

from __future__ import annotations

import gc
import struct
from collections.abc import Callable
from pathlib import Path
from typing import Any

import Metal
import numpy as np
import objc

from favlo.memory import memory
from wan22_i2v.formats import INT8_TILE_N, K_ALIGN, TILE_M, TILE_N, LinearPack, pad_to


class BufferView:
    """A byte window of a span. Dispatches bind ``offset`` inside ``base``."""

    def __init__(self, base: Any, offset: int) -> None:
        self.base = base
        self.offset = int(offset)

_SHADER = Path(__file__).resolve().parent / "shaders" / "kernels.metal"
_NAMES = (
    "matmul_fp4",
    "matmul_fp8",
    "quantize_i8",
    "matmul_i8",
    "scale_i8",
    "zero_bytes",
    "gather_half",
    "scale_gather",
    "rescale_rows",
    "add_bias",
    "gelu_tanh",
    "silu",
    "modulate",
    "gate_residual",
    "rmsnorm",
    "layernorm",
    "layernorm_affine",
    "rope3d",
    "attend",
    "attend_q4",
    "patch_embed",
    "conv3d",
    "conv_pack",
    "conv_im2col",
    "conv_scatter",
    "matmul_f16",
    "cast_half",
    "add_f32",
    "mul_f32",
    "add_half",
    "add_embed_half",
    "copy_f32",
    "axpy",
    "sinusoid",
    "unpatchify",
    "condition36",
    "affine_mu",
    "affine_inv",
    "channel_rms",
    "silu_clip",
    "spatial_attend",
    "spatial_attend_lane",
    "end_pad",
    "nearest2",
    "concat_time",
    "stack_pairs",
    "slice_time",
    "attend_bias",
    "pixels_u8",
)


class Gpu:
    """One device and one command encoder. ``end`` waits for the batch."""

    def __init__(
        self, on_pipeline: Callable[[], None] | None = None, *, owner: str | None = None
    ) -> None:
        self._owner = owner or f"wan22-i2v:{id(self)}"
        self.device = memory.device
        self.queue = memory.make_queue()
        self.pipes: dict[str, Any] = {}
        with objc.autorelease_pool():
            source = _SHADER.read_text(encoding="utf-8")
            options = Metal.MTLCompileOptions.alloc().init()
            options.setLanguageVersion_((4 << 16) + 1)
            library, error = self.device.newLibraryWithSource_options_error_(source, options, None)
            if library is None:
                message = f"shader compile failed: {error}"
                raise RuntimeError(message)
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
                if on_pipeline is not None:
                    on_pipeline()
        self._cmd: Any = None
        self._enc: Any = None
        self._inflight: list[Any] = []
        self._held: list[bytes] = []
        self._owned: list[Any] = []
        self._kept: set[int] = set()
        self._factors: Any = None
        self._factor_rows = 0
        self._conv_pack: dict[tuple, Any] = {}
        self._conv_col: tuple[Any, int] | None = None
        self._conv_tile: tuple[Any, int] | None = None
        width = int(self.pipes["matmul_fp4"].threadExecutionWidth())
        self._matmul_threads = (width * 4, 1, 1)

    def commit(self) -> None:
        """Commit the open encoder and leave the previous buffer running."""
        if self._enc is None:
            return
        memory.handoff(self._enc, self._cmd)
        self._enc.endEncoding()
        self._cmd.commit()
        self._inflight.append(self._cmd)
        self._enc = None
        self._cmd = None
        self._held.clear()
        if len(self._inflight) > 1:
            self._finish(self._inflight.pop(0))

    def end(self) -> None:
        self.commit()
        pending = self._inflight
        self._inflight = []
        for cmd in pending:
            self._finish(cmd)

    def gpu_ms(self) -> float:
        """Finish earlier buffers, then return this encoder's GPU milliseconds.

        The sample is ``GPUEndTime − GPUStartTime`` for the one buffer that
        holds the dispatches since the previous finish.
        """
        pending = self._inflight
        self._inflight = []
        for cmd in pending:
            self._finish(cmd)
        if self._enc is None:
            return 0.0
        memory.handoff(self._enc, self._cmd)
        self._enc.endEncoding()
        cmd = self._cmd
        self._enc = None
        self._cmd = None
        self._held.clear()
        cmd.commit()
        cmd.waitUntilCompleted()
        start = float(cmd.GPUStartTime())
        end = float(cmd.GPUEndTime())
        status, error = memory.retire(cmd)
        if status == int(Metal.MTLCommandBufferStatusError):
            message = f"metal command failed: {error}"
            raise RuntimeError(message)
        return max(0.0, end - start) * 1000.0

    def _finish(self, cmd: Any) -> None:
        status, error = memory.retire(cmd)
        if status == int(Metal.MTLCommandBufferStatusError):
            message = f"metal command failed: {error}"
            raise RuntimeError(message)

    def keep(self, buf: Any) -> Any:
        self._kept.add(id(buf))
        return buf

    def reclaim(self) -> None:
        """Drop buffers that are not marked with ``keep``."""
        try:
            self.end()
        except RuntimeError:
            self._enc = None
            self._cmd = None
            self._held.clear()
        with objc.autorelease_pool():
            kept: list[Any] = []
            temps: list[Any] = []
            for buf in self._owned:
                if id(buf) in self._kept:
                    kept.append(buf)
                else:
                    temps.append(buf)
            self._owned = kept
            for buf in temps:
                buf.release()
            temps.clear()

    def close(self) -> None:
        """Finish device work, release every buffer, and drain the autorelease pool."""
        try:
            self.end()
        except RuntimeError:
            self._enc = None
            self._cmd = None
            self._held.clear()
        with objc.autorelease_pool():
            owned = self._owned
            self._owned = []
            self._kept.clear()
            self._conv_pack.clear()
            self._conv_col = None
            self._conv_tile = None
            for buf in owned:
                buf.release()
            owned.clear()
            self.pipes.clear()
            self.queue = None
            self.device = None
            memory.sweep(self._owner)
            gc.collect()

    def upload(self, array: np.ndarray) -> Any:
        buf = memory.upload(array, owner=self._owner)
        self._owned.append(buf)
        return buf

    def empty(self, nbytes: int) -> Any:
        buf = memory.allocate(int(nbytes), owner=self._owner, zero=True)
        self._owned.append(buf)
        return buf

    def read_f32(self, buf: Any, count: int) -> np.ndarray:
        self.end()
        view = buf.contents().as_buffer(count * 4)
        return np.frombuffer(view, dtype=np.float32, count=count).copy()

    def read_u8(self, buf: Any, count: int) -> np.ndarray:
        self.end()
        view = buf.contents().as_buffer(count)
        return np.frombuffer(view, dtype=np.uint8, count=count).copy()

    def store_f32(self, buf: Any, array: np.ndarray) -> None:
        self.end()
        raw = np.ascontiguousarray(array, dtype=np.float32).ravel()
        view = np.frombuffer(buf.contents().as_buffer(raw.nbytes), dtype=np.float32)
        view[:] = raw

    def store_f16(self, buf: Any, array: np.ndarray) -> None:
        self.end()
        raw = np.ascontiguousarray(array, dtype=np.float16).ravel()
        view = np.frombuffer(buf.contents().as_buffer(raw.nbytes), dtype=np.float16)
        view[:] = raw

    def unkeep(self, buf: Any) -> None:
        self._kept.discard(id(buf))

    def drop(self, buf: Any) -> None:
        """Wait, then release one buffer this device allocated."""
        self.end()
        self.release(buf)

    def release(self, buf: Any) -> None:
        """Release a buffer the open command buffer still retains."""
        self.unkeep(buf)
        self._drop([buf])

    def zero(self, buf: Any, nbytes: int) -> None:
        threads = 256
        groups = (int(nbytes) + threads - 1) // threads
        self._dispatch(
            "zero_bytes",
            [buf],
            [struct.pack("I", int(nbytes))],
            (groups, 1, 1),
            (threads, 1, 1),
        )

    def matmul(
        self,
        kind: str,
        act: Any,
        weight: Any,
        scale: Any,
        dest: Any,
        rows: int,
        cols: int,
        inner: int,
    ) -> None:
        if rows % TILE_M or cols % TILE_N or inner % 256:
            message = f"matmul tile mismatch rows={rows} cols={cols} inner={inner}"
            raise ValueError(message)
        name = "matmul_fp4" if kind == "fp4" else "matmul_fp8"
        self._dispatch(
            name,
            [act, weight, scale, dest],
            [struct.pack("I", rows), struct.pack("I", cols), struct.pack("I", inner)],
            (cols // TILE_N, rows // TILE_M, 1),
            self._matmul_threads,
        )

    def gather(
        self,
        src: Any,
        perm: Any,
        dst: Any,
        rows: int,
        logical_k: int,
        stored: int,
        src_cols: int,
    ) -> None:
        count = rows * stored
        threads = 256
        self._dispatch(
            "gather_half",
            [src, perm, dst],
            [
                struct.pack("I", rows),
                struct.pack("I", logical_k),
                struct.pack("I", stored),
                struct.pack("I", src_cols),
            ],
            ((count + threads - 1) // threads, 1, 1),
            (threads, 1, 1),
        )

    def project(
        self,
        src: Any,
        src_cols: int,
        rows: int,
        pack: LinearPack,
        dest: Any,
        uploaded: dict[str, Any],
        half: Any,
    ) -> None:
        """Gather, ``matmul2d``, and bias. ``dest`` is tight ``[rows, padded_n]``."""
        if pack.kind == "int8":
            self._project_i8(src, src_cols, rows, pack, dest, uploaded, half)
            return
        factors = self._factor_buffer(rows)
        self._dispatch(
            "scale_gather",
            [src, uploaded["perm"], half, factors],
            [
                struct.pack("I", rows),
                struct.pack("I", pack.logical_k),
                struct.pack("I", pack.stored_k),
                struct.pack("I", src_cols),
            ],
            (rows, 1, 1),
            (128, 1, 1),
        )
        self.matmul(
            pack.kind,
            half,
            uploaded["packed"],
            uploaded["scales"],
            dest,
            rows,
            pack.padded_n,
            pack.stored_k,
        )
        self._span(
            "rescale_rows",
            [dest, factors],
            rows,
            pack.padded_n,
            [],
        )
        self._span(
            "add_bias",
            [dest, uploaded["bias"]],
            rows,
            pack.padded_n,
            [struct.pack("I", pack.logical_n)],
        )

    def _project_i8(
        self,
        src: Any,
        src_cols: int,
        rows: int,
        pack: LinearPack,
        dest: Any,
        uploaded: dict[str, Any],
        codes: Any,
    ) -> None:
        """Per-row int8 codes, int8 ``matmul2d``, then the scale and the bias."""
        if rows % TILE_M or pack.padded_n % INT8_TILE_N or pack.stored_k % 128:
            message = (
                f"int8 tile mismatch rows={rows} cols={pack.padded_n} inner={pack.stored_k}"
            )
            raise ValueError(message)
        factors = self._factor_buffer(rows)
        self._dispatch(
            "quantize_i8",
            [src, codes, factors],
            [
                struct.pack("I", rows),
                struct.pack("I", pack.logical_k),
                struct.pack("I", pack.stored_k),
                struct.pack("I", src_cols),
            ],
            (rows, 1, 1),
            (128, 1, 1),
        )
        self._dispatch(
            "matmul_i8",
            [codes, uploaded["packed"], dest],
            [
                struct.pack("I", rows),
                struct.pack("I", pack.padded_n),
                struct.pack("I", pack.stored_k),
            ],
            (pack.padded_n // INT8_TILE_N, rows // TILE_M, 1),
            self._matmul_threads,
        )
        count = rows * pack.padded_n
        threads = 256
        self._dispatch(
            "scale_i8",
            [dest, factors, uploaded["scales"], uploaded["bias"]],
            [
                struct.pack("I", rows),
                struct.pack("I", pack.padded_n),
                struct.pack("I", pack.logical_n),
            ],
            ((count + threads - 1) // threads, 1, 1),
            (threads, 1, 1),
        )

    def _factor_buffer(self, rows: int) -> Any:
        if self._factors is not None and self._factor_rows >= rows:
            return self._factors
        if self._factors is not None:
            self.unkeep(self._factors)
        self._factors = self.keep(self.empty(rows * 4))
        self._factor_rows = rows
        return self._factors

    def pointwise(self, name: str, buf: Any, rows: int, cols: int, logical: int) -> None:
        self._span(name, [buf], rows, cols, [struct.pack("I", logical)])

    def modulate(
        self, data: Any, shift: Any, scale: Any, rows: int, cols: int, logical: int
    ) -> None:
        self._span("modulate", [data, shift, scale], rows, cols, [struct.pack("I", logical)])

    def gate(self, left: Any, right: Any, gate: Any, rows: int, cols: int, logical: int) -> None:
        self._span("gate_residual", [left, right, gate], rows, cols, [struct.pack("I", logical)])

    def rmsnorm(
        self, source: Any, weight: Any, dest: Any, rows: int, width: int, eps: float
    ) -> None:
        self._dispatch(
            "rmsnorm",
            [source, weight, dest],
            [struct.pack("I", rows), struct.pack("I", width), struct.pack("f", eps)],
            (rows, 1, 1),
            (128, 1, 1),
        )

    def layernorm(self, source: Any, dest: Any, rows: int, width: int, eps: float) -> None:
        self._dispatch(
            "layernorm",
            [source, dest],
            [struct.pack("I", rows), struct.pack("I", width), struct.pack("f", eps)],
            (rows, 1, 1),
            (128, 1, 1),
        )

    def layernorm_affine(
        self,
        source: Any,
        weight: Any,
        bias: Any,
        dest: Any,
        rows: int,
        width: int,
        eps: float,
    ) -> None:
        self._dispatch(
            "layernorm_affine",
            [source, weight, bias, dest],
            [struct.pack("I", rows), struct.pack("I", width), struct.pack("f", eps)],
            (rows, 1, 1),
            (128, 1, 1),
        )

    def rope(
        self,
        values: Any,
        seq: int,
        heads: int,
        dim: int,
        grid: tuple[int, int, int],
        axes: tuple[int, int, int],
    ) -> None:
        self._dispatch(
            "rope3d",
            [values],
            [
                struct.pack("I", seq),
                struct.pack("I", heads),
                struct.pack("I", dim),
                struct.pack("I", grid[0]),
                struct.pack("I", grid[1]),
                struct.pack("I", grid[2]),
                struct.pack("I", axes[0]),
                struct.pack("I", axes[1]),
                struct.pack("I", axes[2]),
                struct.pack("f", 10000.0),
            ],
            (seq, 1, 1),
            (128, 1, 1),
        )

    def attend(
        self,
        query: Any,
        key: Any,
        value: Any,
        dest: Any,
        q_len: int,
        kv_len: int,
        heads: int,
        dim: int,
    ) -> None:
        scale = float(dim) ** -0.5
        if dim == 128:
            self._dispatch(
                "attend_q4",
                [query, key, value, dest],
                [
                    struct.pack("IIII", q_len, kv_len, heads, 0),
                    struct.pack("f", scale),
                ],
                (heads, (q_len + 3) // 4, 1),
                (32, 1, 1),
            )
            return
        self._dispatch(
            "attend",
            [query, key, value, dest],
            [
                struct.pack("I", q_len),
                struct.pack("I", kv_len),
                struct.pack("I", heads),
                struct.pack("I", dim),
                struct.pack("f", scale),
            ],
            (heads, q_len, 1),
            (32, 1, 1),
        )

    def conv3d(
        self,
        source: np.ndarray,
        weight: np.ndarray,
        bias: np.ndarray,
        stride: tuple[int, int, int],
        pad: tuple[int, int, int],
    ) -> np.ndarray:
        """Causal-style 3D convolution. ``pad`` is the front, top, and left padding.

        The matching back, bottom, and right padding is the same amount, except
        time, which is not padded on the future side. ``source`` is
        ``[C, T, H, W]`` and ``weight`` is ``[O, C, Kt, Kh, Kw]``.
        """
        cin, tin, hin, win = (int(v) for v in source.shape)
        cout, cin_w, kt, kh, kw = (int(v) for v in weight.shape)
        if cin_w != cin or bias.shape != (cout,):
            message = f"conv weight {weight.shape} bias {bias.shape} source {source.shape}"
            raise ValueError(message)
        stride_t, stride_h, stride_w = stride
        pad_t, pad_h, pad_w = pad
        tout = (tin + pad_t - kt) // stride_t + 1
        hout = (hin + pad_h + pad_h - kh) // stride_h + 1
        wout = (win + pad_w + pad_w - kw) // stride_w + 1
        if tout < 1 or hout < 1 or wout < 1:
            message = f"conv output is empty for source {source.shape} weight {weight.shape}"
            raise ValueError(message)
        count = cout * tout * hout * wout
        with objc.autorelease_pool():
            result = self._conv3d(
                source,
                weight,
                bias,
                count,
                (
                    cin,
                    cout,
                    tin,
                    hin,
                    win,
                    tout,
                    hout,
                    wout,
                    kt,
                    kh,
                    kw,
                    stride_t,
                    stride_h,
                    stride_w,
                ),
                (pad_t, pad_h, pad_w),
            )
        return result.reshape(cout, tout, hout, wout)

    def _conv3d(
        self,
        source: np.ndarray,
        weight: np.ndarray,
        bias: np.ndarray,
        count: int,
        shape: tuple[int, ...],
        pad: tuple[int, int, int],
    ) -> np.ndarray:
        buffers = [
            self.upload(source),
            self.upload(weight),
            self.upload(bias),
            self._alloc(count * 4),
        ]
        constants = [struct.pack("I", int(value)) for value in shape]
        constants.extend(struct.pack("i", int(value)) for value in pad)
        threads = 256
        self._dispatch(
            "conv3d",
            buffers,
            constants,
            ((count + threads - 1) // threads, 1, 1),
            (threads, 1, 1),
        )
        result = self.read_f32(buffers[3], count).copy()
        self._drop(buffers)
        buffers.clear()
        return result

    def _alloc(self, nbytes: int) -> Any:
        buf = memory.allocate(int(nbytes), owner=self._owner)
        self._owned.append(buf)
        return buf

    def _drop(self, buffers: list[Any]) -> None:
        for buf in buffers:
            if buf in self._owned:
                self._owned.remove(buf)
            if id(buf) not in self._kept:
                buf.release()

    def alloc_f32(self, count: int) -> Any:
        return self._alloc(int(count) * 4)

    def drop_many(self, buffers: list[Any]) -> None:
        self.end()
        for buf in buffers:
            self.unkeep(buf)
        self._drop(buffers)

    def _u(self, *values: int) -> list[bytes]:
        return [struct.pack("I", int(value)) for value in values]

    def _count(self, name: str, buffers: list[Any], constants: list[bytes], count: int) -> None:
        threads = 256
        groups = (int(count) + threads - 1) // threads
        self._dispatch(name, buffers, constants, (groups, 1, 1), (threads, 1, 1))

    def add_bias(self, data: Any, bias: Any, rows: int, cols: int, logical: int) -> None:
        self._span("add_bias", [data, bias], rows, cols, [struct.pack("I", logical)])

    def matmul_f16(
        self, act: Any, weight: Any, dest: Any, rows: int, cols: int, inner: int
    ) -> None:
        if rows % TILE_M or cols % TILE_N or inner % 256:
            message = f"matmul tile mismatch rows={rows} cols={cols} inner={inner}"
            raise ValueError(message)
        self._dispatch(
            "matmul_f16",
            [act, weight, dest],
            self._u(rows, cols, inner),
            (cols // TILE_N, rows // TILE_M, 1),
            self._matmul_threads,
        )

    def cast_half(self, source: Any, dest: Any, count: int) -> None:
        self._count("cast_half", [source, dest], self._u(count), count)

    def add_f32(self, left: Any, right: Any, count: int) -> None:
        self._count("add_f32", [left, right], self._u(count), count)

    def mul_f32(self, left: Any, right: Any, count: int) -> None:
        self._count("mul_f32", [left, right], self._u(count), count)

    def add_half(self, left: Any, right: Any, dest: Any, count: int) -> None:
        self._count("add_half", [left, right, dest], self._u(count), count)

    def add_embed_half(
        self, modulation: Any, embed: Any, dest: Any, slices: int, dim: int
    ) -> None:
        self._count(
            "add_embed_half",
            [modulation, embed, dest],
            self._u(slices, dim),
            slices * dim,
        )

    def copy_f32(self, source: Any, offset: int, dest: Any, count: int) -> None:
        self._count("copy_f32", [source, dest], self._u(offset, count), count)

    def axpy(self, left: Any, right: Any, scale: float, count: int) -> None:
        threads = 256
        groups = (int(count) + threads - 1) // threads
        self._dispatch(
            "axpy",
            [left, right],
            [struct.pack("f", float(scale)), struct.pack("I", int(count))],
            (groups, 1, 1),
            (threads, 1, 1),
        )

    def sinusoid(self, dest: Any, freq: int, position: float) -> None:
        threads = 256
        groups = (int(freq) + threads - 1) // threads
        self._dispatch(
            "sinusoid",
            [dest],
            [struct.pack("I", int(freq)), struct.pack("f", float(position))],
            (groups, 1, 1),
            (threads, 1, 1),
        )

    def unpatchify(
        self, tokens: Any, latent: Any, time: int, height: int, width: int
    ) -> None:
        count = 16 * time * height * 2 * width * 2
        self._count("unpatchify", [tokens, latent], self._u(time, height, width), count)

    def condition(
        self,
        noise: Any,
        mask: Any,
        latent: Any,
        dest: Any,
        time: int,
        height: int,
        width: int,
    ) -> None:
        count = 36 * time * height * width
        self._count(
            "condition36",
            [noise, mask, latent, dest],
            self._u(time, height, width),
            count,
        )

    def affine_mu(self, source: Any, mean: Any, std: Any, dest: Any, plane: int) -> None:
        self._count("affine_mu", [source, mean, std, dest], self._u(plane), 16 * plane)

    def affine_inv(self, source: Any, mean: Any, std: Any, dest: Any, plane: int) -> None:
        self._count("affine_inv", [source, mean, std, dest], self._u(plane), 16 * plane)

    def channel_rms(
        self, source: Any, gamma: Any, dest: Any, channels: int, plane: int, eps: float
    ) -> None:
        threads = 256
        groups = (int(plane) + threads - 1) // threads
        constants = [
            struct.pack("I", int(channels)),
            struct.pack("I", int(plane)),
            struct.pack("f", float(eps)),
        ]
        self._dispatch(
            "channel_rms",
            [source, gamma, dest],
            constants,
            (groups, 1, 1),
            (threads, 1, 1),
        )

    def silu_clip(self, source: Any, dest: Any, count: int) -> None:
        self._count("silu_clip", [source, dest], self._u(count), count)

    def spatial_attend(
        self, qkv: Any, dest: Any, channels: int, time: int, height: int, width: int
    ) -> None:
        spatial = height * width
        if channels % 32 == 0 and 0 < channels // 32 <= 12 and spatial > 0:
            self._dispatch(
                "spatial_attend_lane",
                [qkv, dest],
                self._u(channels, time, height, width),
                (time, (spatial + 3) // 4, 1),
                (32, 1, 1),
            )
            return
        self._count(
            "spatial_attend",
            [qkv, dest],
            self._u(channels, time, height, width),
            time * spatial,
        )

    def end_pad(self, source: Any, dest: Any, shape: tuple[int, int, int, int]) -> None:
        channels, time, height, width = shape
        count = channels * time * (height + 1) * (width + 1)
        self._count("end_pad", [source, dest], self._u(channels, time, height, width), count)

    def nearest2(self, source: Any, dest: Any, shape: tuple[int, int, int, int]) -> None:
        channels, time, height, width = shape
        count = channels * time * height * 2 * width * 2
        self._count("nearest2", [source, dest], self._u(channels, time, height, width), count)

    def concat_time(
        self,
        left: Any,
        right: Any,
        dest: Any,
        channels: int,
        left_time: int,
        right_time: int,
        height: int,
        width: int,
    ) -> None:
        count = channels * (left_time + right_time) * height * width
        self._count(
            "concat_time",
            [left, right, dest],
            self._u(channels, left_time, right_time, height, width),
            count,
        )

    def stack_pairs(self, source: Any, dest: Any, shape: tuple[int, int, int, int]) -> None:
        channels, time, height, width = shape
        count = channels * time * 2 * height * width
        self._count("stack_pairs", [source, dest], self._u(channels, time, height, width), count)

    def slice_time(
        self,
        source: Any,
        dest: Any,
        shape: tuple[int, int, int, int],
        start: int,
        count_time: int,
    ) -> None:
        channels, time, height, width = shape
        count = channels * count_time * height * width
        self._count(
            "slice_time",
            [source, dest],
            self._u(channels, time, height, width, start, count_time),
            count,
        )

    def attend_bias(
        self,
        query: Any,
        key: Any,
        value: Any,
        dest: Any,
        buckets: Any,
        table: Any,
        q_len: int,
        kv_len: int,
        heads: int,
        dim: int,
    ) -> None:
        self._dispatch(
            "attend_bias",
            [query, key, value, dest, buckets, table],
            self._u(q_len, kv_len, heads, dim),
            (heads, q_len, 1),
            (32, 1, 1),
        )

    def pixels_u8(self, source: Any, dest: Any, frames: int, height: int, width: int) -> None:
        count = frames * height * width
        self._count("pixels_u8", [source, dest], self._u(frames, height, width), count)

    def conv_into(
        self,
        source: Any,
        source_shape: tuple[int, int, int, int],
        weight: Any,
        weight_shape: tuple[int, int, int, int, int],
        bias: Any,
        stride: tuple[int, int, int],
        pad: tuple[int, int, int],
    ) -> tuple[Any, tuple[int, int, int, int]]:
        """Dispatch ``conv3d`` into a new buffer. The caller waits before reading it."""
        cin, tin, hin, win = source_shape
        cout, cin_w, kt, kh, kw = weight_shape
        if cin_w != cin:
            message = f"conv channels {cin_w} do not match {cin}"
            raise ValueError(message)
        stride_t, stride_h, stride_w = stride
        pad_t, pad_h, pad_w = pad
        tout = (tin + pad_t - kt) // stride_t + 1
        hout = (hin + pad_h + pad_h - kh) // stride_h + 1
        wout = (win + pad_w + pad_w - kw) // stride_w + 1
        if tout < 1 or hout < 1 or wout < 1:
            message = f"conv output is empty for {source_shape} kernel {weight_shape}"
            raise ValueError(message)
        spatial = tout * hout * wout
        k_raw = cin * kt * kh * kw
        if cout >= 32 and spatial >= 1024 and k_raw >= 64:
            dest = self._alloc(cout * spatial * 4)
            self._conv_gemm(
                source,
                source_shape,
                weight,
                bias,
                dest,
                (cout, tout, hout, wout),
                (kt, kh, kw),
                stride,
                pad,
                k_raw,
            )
            return dest, (cout, tout, hout, wout)
        dest = self._alloc(cout * tout * hout * wout * 4)
        shape = (
            cin,
            cout,
            tin,
            hin,
            win,
            tout,
            hout,
            wout,
            kt,
            kh,
            kw,
            stride_t,
            stride_h,
            stride_w,
        )
        constants = [struct.pack("I", int(value)) for value in shape]
        constants.extend(struct.pack("i", int(value)) for value in pad)
        threads = 256
        count = cout * tout * hout * wout
        self._dispatch(
            "conv3d",
            [source, weight, bias, dest],
            constants,
            ((count + threads - 1) // threads, 1, 1),
            (threads, 1, 1),
        )
        return dest, (cout, tout, hout, wout)

    def _conv_gemm(
        self,
        source: Any,
        source_shape: tuple[int, int, int, int],
        weight: Any,
        bias: Any,
        dest: Any,
        out_shape: tuple[int, int, int, int],
        kernel: tuple[int, int, int],
        stride: tuple[int, int, int],
        pad: tuple[int, int, int],
        k_raw: int,
    ) -> None:
        """im2col tiles plus half ``matmul2d``. ``conv3d`` covers the smaller shapes."""
        cin, tin, hin, win = source_shape
        cout, tout, hout, wout = out_shape
        kt, kh, kw = kernel
        k_pad = pad_to(k_raw, K_ALIGN)
        n_pad = pad_to(cout, TILE_N)
        packed = self._packed_conv(weight, cout, cin, kt, kh, kw, k_raw, k_pad, n_pad)
        chunk = 4096
        columns = self._fit_conv("_conv_col", chunk * k_pad * 2)
        tile = self._fit_conv("_conv_tile", chunk * n_pad * 4)
        spatial = tout * hout * wout
        start = 0
        while start < spatial:
            valid = min(chunk, spatial - start)
            rows = pad_to(valid, TILE_M)
            self._dispatch(
                "conv_im2col",
                [source, columns],
                [
                    *self._u(tin, hin, win, kt, kh, kw, *stride),
                    *self._i(*pad),
                    *self._u(hout, wout, k_raw, k_pad, start, valid),
                ],
                ((k_pad + 255) // 256, rows, 1),
                (256, 1, 1),
            )
            self.matmul_f16(columns, packed, tile, rows, n_pad, k_pad)
            self._dispatch(
                "conv_scatter",
                [tile, bias, dest],
                self._u(cout, tout, hout, wout, n_pad, valid, start),
                ((cout + 31) // 32, valid, 1),
                (32, 1, 1),
            )
            start += valid

    def _packed_conv(
        self,
        weight: Any,
        cout: int,
        cin: int,
        kt: int,
        kh: int,
        kw: int,
        k_raw: int,
        k_pad: int,
        n_pad: int,
    ) -> Any:
        key = (id(weight), cout, cin, kt, kh, kw, k_raw, k_pad, n_pad)
        found = self._conv_pack.get(key)
        if found is not None:
            return found[0]
        packed = self.keep(self._alloc(n_pad * k_pad * 2))
        self._dispatch(
            "conv_pack",
            [weight, packed],
            self._u(cout, cin, kt, kh, kw, k_raw, k_pad, n_pad),
            ((k_pad + 255) // 256, n_pad, 1),
            (256, 1, 1),
        )
        # Hold the weight object so a later buffer cannot reuse this id.
        self._conv_pack[key] = (packed, weight)
        return packed

    def _fit_conv(self, name: str, nbytes: int) -> Any:
        pair = getattr(self, name)
        if pair is not None and pair[1] >= nbytes:
            return pair[0]
        buf = self.keep(self._alloc(nbytes))
        setattr(self, name, (buf, nbytes))
        return buf

    def _i(self, *values: int) -> list[bytes]:
        return [struct.pack("i", int(value)) for value in values]

    def patch(
        self,
        volume: Any,
        weight: Any,
        bias: Any,
        dest: Any,
        channels: int,
        time: int,
        height: int,
        width: int,
        dim: int,
    ) -> None:
        tokens = time * (height // 2) * (width // 2)
        threads = 64
        self._dispatch(
            "patch_embed",
            [volume, weight, bias, dest],
            [
                struct.pack("I", channels),
                struct.pack("I", time),
                struct.pack("I", height),
                struct.pack("I", width),
                struct.pack("I", dim),
            ],
            ((dim + threads - 1) // threads, tokens, 1),
            (threads, 1, 1),
        )

    def _span(
        self, name: str, buffers: list[Any], rows: int, cols: int, extra: list[bytes]
    ) -> None:
        count = rows * cols
        threads = 256
        groups = (count + threads - 1) // threads
        constants = [struct.pack("I", rows), struct.pack("I", cols), *extra]
        self._dispatch(name, buffers, constants, (groups, 1, 1), (threads, 1, 1))

    def _dispatch(
        self,
        name: str,
        buffers: list[Any],
        constants: list[bytes],
        grid: tuple[int, int, int],
        threads: tuple[int, int, int],
    ) -> None:
        if self._enc is None:
            self._cmd = memory.command_buffer(self.queue)
            self._enc = self._cmd.computeCommandEncoder()
        enc = self._enc
        enc.setComputePipelineState_(self.pipes[name])
        for index, buf in enumerate(buffers):
            if isinstance(buf, BufferView):
                memory.bind(enc, index, buf.base, buf.offset)
            else:
                memory.bind(enc, index, buf, 0)
        for index, blob in enumerate(constants):
            self._held.append(blob)
            enc.setBytes_length_atIndex_(blob, len(blob), len(buffers) + index)
        enc.dispatchThreadgroups_threadsPerThreadgroup_(
            Metal.MTLSizeMake(int(grid[0]), int(grid[1]), int(grid[2])),
            Metal.MTLSizeMake(int(threads[0]), int(threads[1]), int(threads[2])),
        )
