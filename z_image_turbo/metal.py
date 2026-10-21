"""Metal device, the fp4 matmul dispatch, and the elementwise kernels."""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Any

import numpy as np

import Metal
from z_image_turbo.formats import TILE_M, TILE_N, pad_to

_SHADER = Path(__file__).resolve().parent / "shaders" / "kernels.metal"
_NAMES = (
    "matmul_fp4",
    "zero_bytes",
    "rmsnorm",
    "layernorm",
    "rope_axes",
    "attend",
    "scale_to_half",
    "bias_cast",
    "channel_scale",
    "channel_gate",
    "add_rows",
    "silu_mul",
    "gather_fp4",
    "group_norm",
    "im2col",
    "scatter_bias",
    "nearest_up",
    "add_nchw",
    "silu_f32",
    "nchw_to_f32",
    "rows_add_nchw",
    "euler",
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
        self._half: Any = None
        self._half_bytes = 0
        self._inv: Any = None
        self._inv_bytes = 0
        width = int(self.pipes["matmul_fp4"].threadExecutionWidth())
        self._matmul_threads = (width * 4, 1, 1)

    def _workspace(self, name: str, nbytes: int) -> Any:
        current = getattr(self, name)
        held = int(getattr(self, f"{name}_bytes"))
        if current is None or held < nbytes:
            if current is not None:
                self._kept.discard(id(current))
            current = self.keep(self.empty(nbytes))
            setattr(self, name, current)
            setattr(self, f"{name}_bytes", nbytes)
        return current

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
        """Keep this buffer until ``close``. Other buffers are temporary."""
        self._kept.add(id(buf))
        return buf

    def reclaim(self) -> None:
        """Finish the open command buffer and release temporary buffers.

        A new Metal buffer has one extra retain. ``release`` balances it, and
        dropping the Python proxy frees the allocation.
        """
        try:
            self.end()
        except RuntimeError:
            self._enc = None
            self._cmd = None
            self._held.clear()
        self._release_owned(keep=True)

    def close(self) -> None:
        """Finish the open buffer, release every allocation, then drop the device."""
        try:
            self.end()
        except RuntimeError:
            self._enc = None
            self._cmd = None
            self._held.clear()
        self._release_owned(keep=False)
        self._half = None
        self._inv = None
        self._half_bytes = 0
        self._inv_bytes = 0
        self.pipes.clear()
        self.queue = None
        self.device = None

    def _release_owned(self, *, keep: bool) -> None:
        kept: list[Any] = []
        temps: list[Any] = []
        for buf in self._owned:
            if keep and id(buf) in self._kept:
                kept.append(buf)
            else:
                temps.append(buf)
        self._owned = kept
        if not keep:
            self._kept.clear()
        for buf in temps:
            buf.release()
        temps.clear()

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

    def read_f16(self, buf: Any, count: int) -> np.ndarray:
        self.end()
        view = buf.contents().as_buffer(count * 2)
        return np.frombuffer(view, dtype=np.float16, count=count).copy()

    def store_f16(self, buf: Any, array: np.ndarray) -> None:
        self.end()
        raw = np.ascontiguousarray(array, dtype=np.float16).ravel()
        view = np.frombuffer(buf.contents().as_buffer(raw.nbytes), dtype=np.float16)
        view[:] = raw

    def store_f32(self, buf: Any, array: np.ndarray) -> None:
        self.end()
        raw = np.ascontiguousarray(array, dtype=np.float32).ravel()
        view = np.frombuffer(buf.contents().as_buffer(raw.nbytes), dtype=np.float32)
        view[:] = raw

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
        self._dispatch(
            "matmul_fp4",
            [act, weight, scale, dest],
            [struct.pack("I", rows), struct.pack("I", cols), struct.pack("I", inner)],
            (cols // TILE_N, rows // TILE_M, 1),
            self._matmul_threads,
        )

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

    def rope(
        self,
        values: Any,
        pos: Any,
        seq: int,
        heads: int,
        dim: int,
        theta: float,
        axes: tuple[int, int, int],
        adjacent: bool = False,
    ) -> None:
        self._dispatch(
            "rope_axes",
            [values, pos],
            [
                struct.pack("I", seq),
                struct.pack("I", heads),
                struct.pack("I", dim),
                struct.pack("f", theta),
                struct.pack("I", axes[0]),
                struct.pack("I", axes[1]),
                struct.pack("I", axes[2]),
                struct.pack("I", 1 if adjacent else 0),
            ],
            (seq * heads, 1, 1),
            (1, 1, 1),
        )

    def attend(
        self,
        query: Any,
        key: Any,
        value: Any,
        dest: Any,
        seq: int,
        limit: int,
        q_heads: int,
        kv_heads: int,
        head_dim: int,
        causal: bool,
    ) -> None:
        if head_dim % 32 or head_dim > 512:
            message = f"head dim {head_dim} is not a 32-wide span up to 512"
            raise ValueError(message)
        self._dispatch(
            "attend",
            [query, key, value, dest],
            [
                struct.pack("I", seq),
                struct.pack("I", limit),
                struct.pack("I", q_heads),
                struct.pack("I", kv_heads),
                struct.pack("I", head_dim),
                struct.pack("I", 1 if causal else 0),
            ],
            (q_heads, seq, 1),
            (32, 1, 1),
        )

    def scale_to_half(self, source: Any, dest: Any, inv: Any, rows: int, inner: int) -> None:
        self._dispatch(
            "scale_to_half",
            [source, dest, inv],
            [struct.pack("I", rows), struct.pack("I", inner)],
            (rows, 1, 1),
            (128, 1, 1),
        )

    def bias_cast(
        self,
        source: Any,
        bias: Any | None,
        dest: Any,
        inv: Any,
        rows: int,
        cols: int,
        cols_padded: int,
    ) -> None:
        dummy = bias if bias is not None else dest
        self._dispatch(
            "bias_cast",
            [source, dummy, dest, inv],
            [
                struct.pack("I", rows),
                struct.pack("I", cols),
                struct.pack("I", cols_padded),
                struct.pack("I", 0 if bias is None else 1),
            ],
            _groups(cols, rows, 32, 8),
            (32, 8, 1),
        )

    def channel_scale(
        self, source: Any, scale: Any, dest: Any, rows: int, width: int, scale_offset: int = 0
    ) -> None:
        self._dispatch(
            "channel_scale",
            [source, scale, dest],
            [struct.pack("I", rows), struct.pack("I", width)],
            _groups(width, rows, 32, 8),
            (32, 8, 1),
            [0, scale_offset, 0],
        )

    def channel_gate(
        self,
        residual: Any,
        branch: Any,
        gate: Any,
        dest: Any,
        rows: int,
        width: int,
        gate_offset: int = 0,
    ) -> None:
        self._dispatch(
            "channel_gate",
            [residual, branch, gate, dest],
            [struct.pack("I", rows), struct.pack("I", width)],
            _groups(width, rows, 32, 8),
            (32, 8, 1),
            [0, 0, gate_offset, 0],
        )

    def add_rows(self, left: Any, right: Any, dest: Any, count: int) -> None:
        self._span("add_rows", [left, right, dest], count)

    def silu_mul(self, gate: Any, up: Any, dest: Any, count: int) -> None:
        self._span("silu_mul", [gate, up, dest], count)

    def gather_fp4(
        self,
        ids: Any,
        table: Any,
        scales: Any,
        dest: Any,
        seq: int,
        limit: int,
        hidden: int,
        stored: int,
    ) -> None:
        self._dispatch(
            "gather_fp4",
            [ids, table, scales, dest],
            [
                struct.pack("I", seq),
                struct.pack("I", limit),
                struct.pack("I", hidden),
                struct.pack("I", stored),
            ],
            ((hidden + 31) // 32, seq, 1),
            (32, 1, 1),
        )

    def group_norm(
        self,
        source: Any,
        weight: Any,
        bias: Any,
        dest: Any,
        channels: int,
        height: int,
        width: int,
        groups: int,
        eps: float,
    ) -> None:
        self._dispatch(
            "group_norm",
            [source, weight, bias, dest],
            [
                struct.pack("I", channels),
                struct.pack("I", height),
                struct.pack("I", width),
                struct.pack("I", groups),
                struct.pack("f", eps),
            ],
            (groups, 1, 1),
            (128, 1, 1),
        )

    def im2col(
        self,
        source: Any,
        dest: Any,
        channels: int,
        height: int,
        width: int,
        kernel: int,
        pad: int,
        stored: int,
        pixel0: int,
        pixels: int,
    ) -> None:
        self._dispatch(
            "im2col",
            [source, dest],
            [
                struct.pack("I", channels),
                struct.pack("I", height),
                struct.pack("I", width),
                struct.pack("I", kernel),
                struct.pack("I", pad),
                struct.pack("I", stored),
                struct.pack("I", pixel0),
                struct.pack("I", pixels),
            ],
            (pixels, 1, 1),
            (1, 1, 1),
        )

    def scatter_bias(
        self,
        source: Any,
        bias: Any,
        dest: Any,
        pixels: int,
        cols: int,
        cols_padded: int,
        height: int,
        width: int,
        pixel0: int,
    ) -> None:
        self._dispatch(
            "scatter_bias",
            [source, bias, dest],
            [
                struct.pack("I", pixels),
                struct.pack("I", cols),
                struct.pack("I", cols_padded),
                struct.pack("I", height),
                struct.pack("I", width),
                struct.pack("I", pixel0),
            ],
            ((cols + 31) // 32, pixels, 1),
            (32, 1, 1),
        )

    def nearest_up(
        self, source: Any, dest: Any, channels: int, height: int, width: int
    ) -> None:
        count = channels * (height * 2) * (width * 2)
        self._span("nearest_up", [source, dest], count, extra=[
            struct.pack("I", channels),
            struct.pack("I", height),
            struct.pack("I", width),
        ])

    def add_nchw(self, left: Any, right: Any, dest: Any, count: int) -> None:
        self._span("add_nchw", [left, right, dest], count)

    def silu_f32(self, source: Any, dest: Any, count: int) -> None:
        self._span("silu_f32", [source, dest], count)

    def nchw_to_f32(
        self,
        source: Any,
        dest: Any,
        channels: int,
        height: int,
        width: int,
        rows: int,
    ) -> None:
        self._dispatch(
            "nchw_to_f32",
            [source, dest],
            [
                struct.pack("I", channels),
                struct.pack("I", height),
                struct.pack("I", width),
                struct.pack("I", rows),
            ],
            ((channels + 31) // 32, rows, 1),
            (32, 1, 1),
        )

    def rows_add_nchw(
        self,
        rows_buf: Any,
        residual: Any,
        dest: Any,
        channels: int,
        height: int,
        width: int,
    ) -> None:
        pixels = height * width
        self._dispatch(
            "rows_add_nchw",
            [rows_buf, residual, dest],
            [
                struct.pack("I", channels),
                struct.pack("I", height),
                struct.pack("I", width),
            ],
            ((channels + 31) // 32, pixels, 1),
            (32, 1, 1),
        )

    def euler(self, latent: Any, velocity: Any, delta: float, count: int) -> None:
        threads = 256
        groups = (count + threads - 1) // threads
        self._dispatch(
            "euler",
            [latent, velocity],
            [struct.pack("f", float(delta)), struct.pack("I", count)],
            (groups, 1, 1),
            (threads, 1, 1),
        )

    def _span(
        self,
        name: str,
        buffers: list[Any],
        count: int,
        extra: list[bytes] | None = None,
    ) -> None:
        threads = 256
        groups = (count + threads - 1) // threads
        constants = [struct.pack("I", count)]
        if extra:
            constants = extra + constants
        self._dispatch(name, buffers, constants, (groups, 1, 1), (threads, 1, 1))

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
            self._cmd = self.queue.commandBuffer()
            self._enc = self._cmd.computeCommandEncoder()
        enc = self._enc
        enc.setComputePipelineState_(self.pipes[name])
        for index, buf in enumerate(buffers):
            shift = 0 if offsets is None else offsets[index]
            enc.setBuffer_offset_atIndex_(buf, shift, index)
        for index, blob in enumerate(constants):
            self._held.append(blob)
            enc.setBytes_length_atIndex_(blob, len(blob), len(buffers) + index)
        enc.dispatchThreadgroups_threadsPerThreadgroup_(
            Metal.MTLSizeMake(int(grid[0]), int(grid[1]), int(grid[2])),
            Metal.MTLSizeMake(int(threads[0]), int(threads[1]), int(threads[2])),
        )


def _groups(width: int, rows: int, tx: int, ty: int) -> tuple[int, int, int]:
    return ((width + tx - 1) // tx, (rows + ty - 1) // ty, 1)


def linear(
    gpu: Gpu,
    act: Any,
    weight: Any,
    scale: Any,
    dest_f32: Any,
    dest_f16: Any,
    rows: int,
    cols: int,
    inner: int,
    bias: Any | None = None,
    rows_padded: int | None = None,
    cols_padded: int | None = None,
) -> None:
    """fp4 matmul plus bias, stored as float32 of the logical column count.

    The tensor op reads float16 activations. A row whose peak exceeds 60000 is
    scaled into that range, and the product is scaled back.
    """
    row_n = rows_padded if rows_padded is not None else pad_to(rows, TILE_M)
    col_n = cols_padded if cols_padded is not None else pad_to(cols, TILE_N)
    half_act = gpu._workspace("_half", row_n * inner * 2)
    inv = gpu._workspace("_inv", row_n * 4)
    gpu.scale_to_half(act, half_act, inv, row_n, inner)
    gpu.matmul(half_act, weight, scale, dest_f32, row_n, col_n, inner)
    gpu.bias_cast(dest_f32, bias, dest_f16, inv, row_n, cols, col_n)
