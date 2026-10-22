"""Metal device and the dispatches this pipeline's kernels command accepts."""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Any

import numpy as np

import Metal
from wan22_i2v.formats import TILE_M, TILE_N, LinearPack

_SHADER = Path(__file__).resolve().parent / "shaders" / "kernels.metal"
_NAMES = (
    "matmul_fp4",
    "matmul_fp8",
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
    "patch_embed",
    "conv3d",
)


class Gpu:
    """One device and one command encoder. ``end`` waits for the batch."""

    def __init__(self) -> None:
        device = Metal.MTLCreateSystemDefaultDevice()
        if device is None:
            message = "Metal device is missing"
            raise RuntimeError(message)
        self.device = device
        self.queue = device.newCommandQueue()
        source = _SHADER.read_text(encoding="utf-8")
        options = Metal.MTLCompileOptions.alloc().init()
        options.setLanguageVersion_((4 << 16) + 1)
        library, error = device.newLibraryWithSource_options_error_(source, options, None)
        if library is None:
            message = f"shader compile failed: {error}"
            raise RuntimeError(message)
        self.pipes: dict[str, Any] = {}
        for name in _NAMES:
            fn = library.newFunctionWithName_(name)
            if fn is None:
                message = f"shader is missing {name}"
                raise RuntimeError(message)
            pipe, pipe_error = device.newComputePipelineStateWithFunction_error_(fn, None)
            if pipe is None:
                message = f"pipeline {name} failed: {pipe_error}"
                raise RuntimeError(message)
            self.pipes[name] = pipe
        self._cmd: Any = None
        self._enc: Any = None
        self._held: list[bytes] = []
        self._owned: list[Any] = []
        self._kept: set[int] = set()
        self._factors: Any = None
        self._factor_rows = 0
        width = int(self.pipes["matmul_fp4"].threadExecutionWidth())
        self._matmul_threads = (width * 4, 1, 1)

    def end(self) -> None:
        if self._enc is None:
            return
        self._enc.endEncoding()
        self._cmd.commit()
        self._cmd.waitUntilCompleted()
        status = int(self._cmd.status())
        error = self._cmd.error()
        self._enc = None
        self._cmd = None
        self._held.clear()
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
        kept: list[Any] = []
        for buf in self._owned:
            if id(buf) in self._kept:
                kept.append(buf)
            else:
                buf.release()
        self._owned = kept

    def close(self) -> None:
        try:
            self.end()
        except RuntimeError:
            self._enc = None
            self._cmd = None
            self._held.clear()
        for buf in self._owned:
            buf.release()
        self._owned.clear()
        self._kept.clear()
        self.pipes.clear()
        self.queue = None
        self.device = None

    def upload(self, array: np.ndarray) -> Any:
        data = np.ascontiguousarray(array)
        raw = data.tobytes()
        buf = self.device.newBufferWithBytes_length_options_(
            raw, len(raw), Metal.MTLResourceStorageModeShared
        )
        if buf is None:
            message = "Metal buffer allocation failed"
            raise RuntimeError(message)
        self._owned.append(buf)
        return buf

    def empty(self, nbytes: int) -> Any:
        buf = self.device.newBufferWithLength_options_(
            int(nbytes), Metal.MTLResourceStorageModeShared
        )
        if buf is None:
            message = f"Metal buffer allocation failed for {nbytes} bytes"
            raise RuntimeError(message)
        self._owned.append(buf)
        self.zero(buf, int(nbytes))
        return buf

    def read_f32(self, buf: Any, count: int) -> np.ndarray:
        self.end()
        view = buf.contents().as_buffer(count * 4)
        return np.frombuffer(view, dtype=np.float32, count=count).copy()

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
        buffers = [
            self.upload(source),
            self.upload(weight),
            self.upload(bias),
            self._alloc(count * 4),
        ]
        values = (
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
        constants = [struct.pack("I", int(v)) for v in values]
        constants.extend(struct.pack("i", int(v)) for v in (pad_t, pad_h, pad_w))
        threads = 256
        self._dispatch(
            "conv3d",
            buffers,
            constants,
            ((count + threads - 1) // threads, 1, 1),
            (threads, 1, 1),
        )
        result = self.read_f32(buffers[3], count).reshape(cout, tout, hout, wout)
        self._drop(buffers)
        return result

    def _alloc(self, nbytes: int) -> Any:
        buf = self.device.newBufferWithLength_options_(
            int(nbytes), Metal.MTLResourceStorageModeShared
        )
        if buf is None:
            message = f"Metal buffer allocation failed for {nbytes} bytes"
            raise RuntimeError(message)
        self._owned.append(buf)
        return buf

    def _drop(self, buffers: list[Any]) -> None:
        for buf in buffers:
            if buf in self._owned:
                self._owned.remove(buf)
            if id(buf) not in self._kept:
                buf.release()

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
            ((tokens + threads - 1) // threads, 1, 1),
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
            self._cmd = self.queue.commandBuffer()
            self._enc = self._cmd.computeCommandEncoder()
        enc = self._enc
        enc.setComputePipelineState_(self.pipes[name])
        for index, buf in enumerate(buffers):
            enc.setBuffer_offset_atIndex_(buf, 0, index)
        for index, blob in enumerate(constants):
            self._held.append(blob)
            enc.setBytes_length_atIndex_(blob, len(blob), len(buffers) + index)
        enc.dispatchThreadgroups_threadsPerThreadgroup_(
            Metal.MTLSizeMake(int(grid[0]), int(grid[1]), int(grid[2])),
            Metal.MTLSizeMake(int(threads[0]), int(threads[1]), int(threads[2])),
        )
