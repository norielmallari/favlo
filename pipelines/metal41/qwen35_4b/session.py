"""Compile the Qwen3.5-4B shader library and encode its dispatches."""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Any

from favlo.memory import device_buffer, memory

_SHADER = Path(__file__).resolve().parent / "shaders" / "kernels.metal"

KERNELS = (
    "zero_f32",
    "zero_u32",
    "embed_row",
    "embed_rows",
    "cast_f32_f16",
    "copy_row_f32",
    "matmul_fp8",
    "matmul_fp8_n32",
    "rmsnorm",
    "rmsnorm_rows",
    "head_rmsnorm",
    "head_rmsnorm_rows",
    "gated_rmsnorm",
    "gated_rmsnorm_rows",
    "silu_mul",
    "silu_mul_rows",
    "residual_add",
    "residual_add_rows",
    "gemv_fp8",
    "gemv_fp8_add",
    "conv_silu",
    "conv_silu_chunk",
    "gdn_step",
    "gdn_chunk",
    "rope_heads",
    "rope_heads_rows",
    "store_kv",
    "store_kv_rows",
    "attend_decode",
    "attend_prefill",
    "penalize",
    "bump_count",
    "argmax_partial",
    "argmax_final",
)

PREFILL_TILE = 512

# topk kernels stay in the shader source for a later sampler; greedy + presence is enough now.

GEMV_ROWS = 8


def _metal() -> Any:
    import Metal

    return Metal


def _objc_pair(result: Any) -> tuple[Any, str | None]:
    if isinstance(result, tuple) and len(result) == 2:
        value, error = result
        if error is not None:
            return None, str(error)
        return value, None
    return result, None


def u32(value: int) -> bytes:
    return struct.pack("<I", int(value))


def f32(value: float) -> bytes:
    return struct.pack("<f", float(value))


class Session:
    """One shader library, one queue, and one open command buffer at a time."""

    def __init__(self, owner: str) -> None:
        metal = _metal()
        self.owner = owner
        device = memory.device
        options = metal.MTLCompileOptions.alloc().init()
        options.setLanguageVersion_((4 << 16) + 1)
        source = _SHADER.read_text(encoding="utf-8")
        library, error = _objc_pair(
            device.newLibraryWithSource_options_error_(source, options, None)
        )
        if library is None:
            message = f"shader library failed to compile: {error}"
            raise RuntimeError(message)
        self.metal = metal
        self.device = device
        self.queue = memory.make_queue()
        self.pipes: dict[str, Any] = {}
        for name in KERNELS:
            function = library.newFunctionWithName_(name)
            if function is None:
                message = f"{name} is not in the shader library"
                raise RuntimeError(message)
            pipeline, pipe_error = _objc_pair(
                device.newComputePipelineStateWithFunction_error_(function, None)
            )
            if pipeline is None:
                message = f"{name} pipeline failed: {pipe_error}"
                raise RuntimeError(message)
            self.pipes[name] = pipeline
        self._size = metal.MTLSize
        self._held: list[Any] = []
        self._inflight: list[tuple[Any, list[Any]]] = []
        self._command: Any = None
        self._encoder: Any = None
        self.gpu_seconds = 0.0

    def allocate(self, nbytes: int, *, zero: bool = False) -> Any:
        return device_buffer(self.owner, nbytes, zero=zero)

    def upload(self, data: bytes) -> Any:
        return memory.upload(data, owner=self.owner)

    @property
    def active(self) -> bool:
        return self._encoder is not None

    def begin(self) -> None:
        if self._encoder is not None:
            return
        self._command = memory.command_buffer(self.queue)
        self._encoder = self._command.computeCommandEncoder()

    def end(self, *, wait: bool = True) -> None:
        encoder = self._encoder
        command = self._command
        if encoder is None or command is None:
            if wait:
                self.wait()
            return
        self._encoder = None
        self._command = None
        encoder.endEncoding()
        command.commit()
        held = list(self._held)
        self._held.clear()
        self._inflight.append((command, held))
        if wait:
            self.wait()

    def wait(self) -> None:
        """Drain committed command buffers. Safe before overwriting a bound span."""
        pending = self._inflight
        self._inflight = []
        for command, _held in pending:
            command.waitUntilCompleted()
            started = getattr(command, "GPUStartTime", None)
            stopped = getattr(command, "GPUEndTime", None)
            if callable(started) and callable(stopped):
                span = float(stopped()) - float(started())
                if span > 0.0:
                    self.gpu_seconds += span
            status, error = memory.retire(command)
            if status != 4:
                message = f"metal command failed ({status}): {error}"
                raise RuntimeError(message)

    def flush(self) -> None:
        if self._encoder is not None:
            self.end(wait=True)
        else:
            self.wait()

    def launch(
        self,
        name: str,
        handles: list[Any],
        constants: list[bytes],
        groups: tuple[int, int, int],
        threads: tuple[int, int, int],
        offsets: list[int] | None = None,
    ) -> None:
        own = self._encoder is None
        if own:
            self.begin()
        encoder = self._encoder
        self._held.extend(handles)
        encoder.setComputePipelineState_(self.pipes[name])
        for index, handle in enumerate(handles):
            shift = 0 if offsets is None else int(offsets[index])
            memory.bind(encoder, index, handle, shift)
        base = len(handles)
        for index, blob in enumerate(constants):
            encoder.setBytes_length_atIndex_(blob, len(blob), base + index)
        encoder.dispatchThreadgroups_threadsPerThreadgroup_(
            self._size(*groups),
            self._size(*threads),
        )
        if own:
            self.end()

    def close(self) -> None:
        try:
            self.flush()
        finally:
            memory.sweep(self.owner)
