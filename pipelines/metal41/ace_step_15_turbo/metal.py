"""Metal device for this pipeline. Every neural op dispatches a shader."""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Any

import Metal
import numpy as np

from ace_step_15_turbo.pack import TILE_M, TILE_N
from favlo.memory import memory

_SHADER = Path(__file__).resolve().parent / "shaders" / "kernels.metal"
_NAMES = (
    "matmul_fp4",
    "matmul_fp8",
    "cast_f16",
    "add_bias",
    "add_vec",
    "rmsnorm",
    "rope_half",
    "silu",
    "silu_mul",
    "adaln",
    "channel_residual",
    "residual",
    "quant_fp8",
    "attend",
    "sinusoid_time",
    "rope_tables",
    "euler",
    "snake",
    "conv1d",
    "conv_transpose1d",
    "gather_fp8",
    "copy_bytes",
    "fill_f32",
    "copy_rows",
    "copy_span",
    "transpose2d",
    "pack_latent",
    "block_amax",
    "reduce_amax",
    "scale_peak",
)


def _size(width: int, height: int = 1, depth: int = 1) -> Any:
    return Metal.MTLSizeMake(int(width), int(height), int(depth))


class Gpu:
    """One device. ``wait`` ends the command buffer and returns its GPU milliseconds."""

    def __init__(self) -> None:
        self._owner = f"ace:{id(self)}"
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
            function = library.newFunctionWithName_(name)
            if function is None:
                message = f"shader is missing {name}"
                raise RuntimeError(message)
            pipe, pipe_error = self.device.newComputePipelineStateWithFunction_error_(
                function, None
            )
            if pipe is None:
                message = f"pipeline {name} failed: {pipe_error}"
                raise RuntimeError(message)
            self.pipes[name] = pipe
        self._cmd: Any = None
        self._enc: Any = None
        self._held: list[Any] = []
        self._resident: list[Any] = []
        self._clip: list[Any] = []
        self.seconds = 0.0
        width = int(self.pipes["matmul_fp4"].threadExecutionWidth())
        self._matmul_threads = _size(width * 4, 1, 1)

    def resident(self, buf: Any) -> Any:
        """Keep a weight until ``close``."""
        self._resident.append(buf)
        return buf

    def clip(self, buf: Any) -> Any:
        """Keep an activation until the clip finishes."""
        self._clip.append(buf)
        return buf

    def reclaim(self, *alive: Any) -> None:
        """Wait, then drop activations that are not weights or clip state."""
        self.wait()
        self._held = [*self._resident, *self._clip, *alive]

    def end_clip(self) -> None:
        self._clip.clear()
        self.reclaim()

    def buffer(self, nbytes: int) -> Any:
        size = max(128, (int(nbytes) + 127) // 128 * 128)
        buf = memory.allocate(size, owner=self._owner)
        self._held.append(buf)
        return buf

    def write(self, buf: Any, array: np.ndarray) -> None:
        payload = np.ascontiguousarray(array)
        buf.contents().as_buffer(payload.nbytes)[: payload.nbytes] = payload.tobytes()

    def read(self, buf: Any, count: int, dtype: np.dtype) -> np.ndarray:
        view = np.frombuffer(buf.contents().as_buffer(count * dtype.itemsize), dtype=dtype)
        return np.array(view[:count], copy=True)

    def zeros(self, count: int) -> Any:
        return self.filled(0.0, count)

    def filled(self, value: float, count: int) -> Any:
        buf = self.buffer(count * 4)
        self._encode("fill_f32", (max(1, count), 1, 1), (256, 1, 1), [buf], [value, count], ["f", "I"])
        return buf

    def f32(self, array: np.ndarray) -> Any:
        payload = np.ascontiguousarray(array, dtype=np.float32).reshape(-1)
        buf = self.buffer(payload.nbytes)
        self.write(buf, payload)
        return buf

    def upload(self, array: np.ndarray) -> Any:
        payload = np.ascontiguousarray(array)
        buf = self.buffer(max(payload.nbytes, 128))
        self.write(buf, payload)
        return buf

    def wait(self) -> float:
        if self._enc is None:
            return 0.0
        self._enc.endEncoding()
        self._cmd.commit()
        self._cmd.waitUntilCompleted()
        status = int(self._cmd.status())
        error = self._cmd.error()
        start = float(self._cmd.GPUStartTime())
        end = float(self._cmd.GPUEndTime())
        memory.retire(self._cmd)
        self._enc = None
        self._cmd = None
        elapsed = max(0.0, end - start)
        self.seconds += elapsed
        if status == int(Metal.MTLCommandBufferStatusError):
            message = f"metal command failed: {error}"
            raise RuntimeError(message)
        return elapsed * 1000.0

    def _begin(self) -> None:
        if self._enc is not None:
            return
        self._cmd = memory.command_buffer(self.queue)
        self._enc = self._cmd.computeCommandEncoder()

    def _encode(
        self,
        name: str,
        grid: tuple[int, int, int],
        threads: tuple[int, int, int],
        buffers: list[Any],
        constants: list[Any],
        kinds: list[str],
    ) -> None:
        self._begin()
        encoder = self._enc
        pipe = self.pipes[name]
        encoder.setComputePipelineState_(pipe)
        for index, buf in enumerate(buffers):
            memory.bind(encoder, index, buf, 0)
        base = len(buffers)
        for offset, (value, kind) in enumerate(zip(constants, kinds, strict=True)):
            blob = struct.pack("<" + kind, value)
            self._held.append(blob)
            encoder.setBytes_length_atIndex_(blob, len(blob), base + offset)
        groups = (
            max(1, (grid[0] + threads[0] - 1) // threads[0]),
            max(1, (grid[1] + threads[1] - 1) // threads[1]),
            max(1, (grid[2] + threads[2] - 1) // threads[2]),
        )
        encoder.dispatchThreadgroups_threadsPerThreadgroup_(
            _size(*groups),
            _size(*threads),
        )

    def groups(
        self,
        name: str,
        groups: tuple[int, int, int],
        threads: tuple[int, int, int],
        buffers: list[Any],
        constants: list[Any],
        kinds: list[str],
    ) -> None:
        """One threadgroup per tile. ``groups`` is not divided by ``threads``."""
        self._begin()
        encoder = self._enc
        encoder.setComputePipelineState_(self.pipes[name])
        for index, buf in enumerate(buffers):
            memory.bind(encoder, index, buf, 0)
        base = len(buffers)
        for offset, (value, kind) in enumerate(zip(constants, kinds, strict=True)):
            blob = struct.pack("<" + kind, value)
            self._held.append(blob)
            encoder.setBytes_length_atIndex_(blob, len(blob), base + offset)
        encoder.dispatchThreadgroups_threadsPerThreadgroup_(_size(*groups), _size(*threads))

    def matmul(
        self,
        activation: Any,
        weight: Any,
        scale: Any,
        rows: int,
        cols: int,
        inner: int,
        kind: str = "fp4",
    ) -> Any:
        half = self.buffer(rows * inner * 2)
        self._encode("cast_f16", (rows * inner, 1, 1), (256, 1, 1), [activation, half], [rows * inner], ["I"])
        dest = self.buffer(rows * cols * 4)
        name = "matmul_fp8" if kind == "fp8" else "matmul_fp4"
        self.groups(
            name,
            (cols // TILE_N, rows // TILE_M, 1),
            (int(self._matmul_threads.width), 1, 1),
            [half, weight, scale, dest],
            [rows, cols, inner],
            ["I", "I", "I"],
        )
        return dest

    def linear(
        self,
        activation: Any,
        rows: int,
        cols_in: int,
        weight: Any,
        scale: Any,
        cols_out: int,
        inner: int,
        bias: Any | None = None,
        kind: str = "fp4",
    ) -> Any:
        """``[rows, cols_in]`` times a packed ``[cols_out, inner]`` weight."""
        padded = ((rows + TILE_M - 1) // TILE_M) * TILE_M
        staged = self.zeros(padded * inner)
        self._encode(
            "copy_rows",
            (rows * cols_in, 1, 1),
            (256, 1, 1),
            [activation, staged],
            [rows, cols_in, cols_in, inner],
            ["I", "I", "I", "I"],
        )
        dest = self.matmul(staged, weight, scale, padded, cols_out, inner, kind)
        if bias is not None:
            self._encode(
                "add_bias",
                (rows * cols_out, 1, 1),
                (256, 1, 1),
                [dest, bias],
                [rows, cols_out, cols_out],
                ["I", "I", "I"],
            )
        return dest

    def norm(self, values: Any, weight: Any, rows: int, width: int, eps: float = 1e-6) -> Any:
        dest = self.buffer(rows * width * 4)
        self.groups(
            "rmsnorm",
            (max(1, rows), 1, 1),
            (128, 1, 1),
            [values, weight, dest],
            [rows, width, eps],
            ["I", "I", "f"],
        )
        return dest

    def rope(self, values: Any, cos: Any, sin: Any, seq: int, heads: int, dim: int) -> Any:
        dest = self.buffer(seq * heads * dim * 4)
        self._encode(
            "rope_half",
            (seq * heads * dim, 1, 1),
            (256, 1, 1),
            [values, dest, cos, sin],
            [seq, heads, dim],
            ["I", "I", "I"],
        )
        return dest

    def rope_table(self, seq: int, dim: int, theta: float) -> tuple[Any, Any]:
        cos = self.buffer(seq * dim * 4)
        sin = self.buffer(seq * dim * 4)
        self._encode(
            "rope_tables",
            (seq * dim, 1, 1),
            (256, 1, 1),
            [cos, sin],
            [seq, dim, theta],
            ["I", "I", "f"],
        )
        return cos, sin

    def silu(self, values: Any, count: int) -> Any:
        dest = self.buffer(count * 4)
        self._encode(
            "silu",
            (count, 1, 1),
            (256, 1, 1),
            [values, dest],
            [count],
            ["I"],
        )
        return dest

    def silu_mul(self, gate: Any, up: Any, count: int) -> Any:
        dest = self.buffer(count * 4)
        self._encode(
            "silu_mul",
            (count, 1, 1),
            (256, 1, 1),
            [gate, up, dest],
            [count],
            ["I"],
        )
        return dest

    def adaln(self, values: Any, scale: Any, shift: Any, rows: int, width: int) -> Any:
        dest = self.buffer(rows * width * 4)
        self._encode(
            "adaln",
            (rows * width, 1, 1),
            (256, 1, 1),
            [values, scale, shift, dest],
            [rows, width],
            ["I", "I"],
        )
        return dest

    def add(self, base: Any, extra: Any, count: int) -> None:
        self._encode("residual", (count, 1, 1), (256, 1, 1), [base, extra], [count], ["I"])

    def add_gated(self, base: Any, extra: Any, gate: Any, rows: int, width: int) -> None:
        self._encode(
            "channel_residual",
            (rows * width, 1, 1),
            (256, 1, 1),
            [base, extra, gate],
            [rows, width],
            ["I", "I"],
        )

    def add_vectors(self, base: Any, extra: Any, count: int) -> None:
        self._encode("add_vec", (count, 1, 1), (256, 1, 1), [base, extra], [count], ["I"])

    def quantize(self, values: Any, count: int) -> tuple[Any, Any]:
        if count % 32 != 0:
            message = "fp8 length must be a multiple of 32"
            raise ValueError(message)
        coded = self.buffer(count)
        scales = self.buffer(count // 32)
        self.groups(
            "quant_fp8",
            (count // 32, 1, 1),
            (32, 1, 1),
            [values, coded, scales],
            [count // 32],
            ["I"],
        )
        return coded, scales

    def attend(
        self,
        query: Any,
        key: Any,
        key_scale: Any,
        value: Any,
        value_scale: Any,
        seq: int,
        kv_seq: int,
        heads: int,
        kv_heads: int,
        dim: int,
        window: int,
    ) -> Any:
        if dim != 128:
            message = "attention head dim must be 128"
            raise ValueError(message)
        dest = self.buffer(seq * heads * dim * 4)
        self.groups(
            "attend",
            (seq, heads, 1),
            (dim, 1, 1),
            [query, key, key_scale, value, value_scale, dest],
            [seq, kv_seq, heads, kv_heads, dim, window],
            ["I", "I", "I", "I", "I", "i"],
        )
        return dest

    def sinusoid(self, tick: float) -> Any:
        dest = self.buffer(256 * 4)
        self._encode("sinusoid_time", (256, 1, 1), (256, 1, 1), [dest], [tick], ["f"])
        return dest

    def euler(self, latent: Any, velocity: Any, dt: float, count: int) -> None:
        self._encode(
            "euler",
            (count, 1, 1),
            (256, 1, 1),
            [latent, velocity],
            [dt, count],
            ["f", "I"],
        )

    def snake(self, values: Any, alpha: Any, beta: Any, channels: int, length: int) -> Any:
        dest = self.buffer(channels * length * 4)
        self._encode(
            "snake",
            (channels * length, 1, 1),
            (256, 1, 1),
            [values, alpha, beta, dest],
            [channels, length],
            ["I", "I"],
        )
        return dest

    def conv(
        self,
        values: Any,
        weight: Any,
        scale: Any,
        bias: Any | None,
        cin: int,
        cout: int,
        kernel: int,
        inner: int,
        length: int,
        pad: int,
        stride: int,
        dilation: int = 1,
    ) -> tuple[Any, int]:
        out_len = (length + 2 * pad - dilation * (kernel - 1) - 1) // stride + 1
        dest = self.buffer(cout * out_len * 4)
        present = bias if bias is not None else weight
        self._encode(
            "conv1d",
            (cout, max(1, out_len), 1),
            (16, 16, 1),
            [values, weight, scale, present, dest],
            [cin, cout, kernel, inner, length, pad, stride, dilation, 0 if bias is None else 1],
            ["I", "I", "I", "I", "I", "I", "I", "I", "I"],
        )
        return dest, out_len

    def conv_transpose(
        self,
        values: Any,
        weight: Any,
        scale: Any,
        bias: Any | None,
        cin: int,
        cout: int,
        kernel: int,
        inner: int,
        length: int,
        pad: int,
        stride: int,
    ) -> tuple[Any, int]:
        out_len = length * stride
        dest = self.buffer(max(1, cout * out_len) * 4)
        present = bias if bias is not None else weight
        self._encode(
            "conv_transpose1d",
            (max(1, cout), max(1, out_len), 1),
            (16, 16, 1),
            [values, weight, scale, present, dest],
            [cin, cout, kernel, inner, length, pad, stride, 0 if bias is None else 1],
            ["I", "I", "I", "I", "I", "I", "I", "I"],
        )
        return dest, out_len

    def gather(self, table: Any, scale: Any, ids: Any, count: int, width: int, inner: int) -> Any:
        dest = self.buffer(count * width * 4)
        self._encode(
            "gather_fp8",
            (count * width, 1, 1),
            (256, 1, 1),
            [table, scale, ids, dest],
            [width, inner, count],
            ["I", "I", "I"],
        )
        return dest

    def transpose(self, values: Any, rows: int, cols: int) -> Any:
        dest = self.buffer(rows * cols * 4)
        self._encode(
            "transpose2d",
            (rows * cols, 1, 1),
            (256, 1, 1),
            [values, dest],
            [rows, cols],
            ["I", "I"],
        )
        return dest

    def pack_latent(self, source: Any, mask: Any, noise: Any, channels: int, length: int) -> Any:
        dest = self.buffer(3 * channels * length * 4)
        self._encode(
            "pack_latent",
            (3 * channels * length, 1, 1),
            (256, 1, 1),
            [source, mask, noise, dest],
            [channels, length],
            ["I", "I"],
        )
        return dest

    def span(self, src: Any, dst: Any, count: int, src_offset: int, dst_offset: int) -> None:
        self._encode(
            "copy_span",
            (count, 1, 1),
            (256, 1, 1),
            [src, dst],
            [count, src_offset, dst_offset],
            ["I", "I", "I"],
        )

    def normalize_peak(self, values: Any, count: int, target: float) -> None:
        stride = 4096
        groups = max(1, (count + stride - 1) // stride)
        partial = self.buffer(groups * 4)
        self.groups(
            "block_amax",
            (groups, 1, 1),
            (256, 1, 1),
            [values, partial],
            [count, stride],
            ["I", "I"],
        )
        peak = self.buffer(4)
        self.groups(
            "reduce_amax",
            (1, 1, 1),
            (256, 1, 1),
            [partial, peak],
            [groups],
            ["I"],
        )
        self._encode(
            "scale_peak",
            (count, 1, 1),
            (256, 1, 1),
            [values, peak],
            [target, count],
            ["f", "I"],
        )

    def close(self) -> None:
        """Finish device work and drop weights and activations."""
        self.wait()
        self._held.clear()
        self._clip.clear()
        self._resident.clear()
        memory.sweep(self._owner)
