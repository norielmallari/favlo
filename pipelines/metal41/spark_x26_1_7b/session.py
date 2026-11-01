"""Compile this pipeline's shader library and dispatch it."""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Any

from favlo.memory import memory

_SHADER = Path(__file__).resolve().parent / "shaders" / "kernels.metal"

KERNELS = (
    "gemv_fp8",
    "gemv_fp8_vec",
    "gemv_fp8_row",
    "gemv_fp8_row_add",
    "gemv_fp8_vec_rms",
    "gemv_fp8_vec_add",
    "gemv_fp8_vec_add_cached",
    "gemv_fp8_vec_add_wide",
    "gemv_group",
    "gemv_group_rms",
    "gemv_group_add",
    "gemv_group_add_wide",
    "gemv_f16",
    "embed_row",
    "rmsnorm",
    "residual",
    "cast_f16",
    "geglu",
    "rope_qk",
    "store_full",
    "store_slide",
    "store_full_span",
    "store_slide_span",
    "store_full_keys",
    "store_slide_keys",
    "attend_span",
    "attend_part",
    "attend_chunk_merge",
    "attend_tile",
    "attend_merge",
    "gemv_fp8_rms",
    "gemv_fp8_add",
    "rebuild_full",
    "rebuild_slide",
    "score_keys",
    "mix_values",
    "gate_rows",
    "matmul_fp8_64",
    "matmul_fp8_32",
    "gemv_fp8_split",
    "gemv_split_sum",
    "mm_m8_n128",
    "mm_m8_n32",
    "mm_m16_n128",
    "zero_f32",
    "zero_u32",
    "partial_max",
    "final_max",
    "partial_exp",
    "final_sum",
    "write_probs",
    "penalize",
    "bump_count",
    "argmax_partial",
    "argmax_final",
    "fill_pairs",
    "bitonic",
    "export_probs",
    "block_scan",
    "scan_blocks",
    "add_block_prefix",
    "pick_nucleus",
    "partial_count",
    "final_count",
    "partial_min",
    "final_min",
    "mask_below",
    "copy_token_major",
    "copy_channel_major",
)


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
    """One Metal library. Persistent buffers use the resident heap."""

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
        self._held: list[Any] = []
        self._command: Any = None
        self._encoder: Any = None
        self.gpu_seconds = 0.0

    def allocate(self, nbytes: int, *, zero: bool = False) -> Any:
        return memory.allocate(nbytes, owner=self.owner, heap="resident", zero=zero)

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

    def end(self) -> None:
        encoder = self._encoder
        command = self._command
        if encoder is None or command is None:
            return
        self._encoder = None
        self._command = None
        encoder.endEncoding()
        command.commit()
        command.waitUntilCompleted()
        started = getattr(command, "GPUStartTime", None)
        stopped = getattr(command, "GPUEndTime", None)
        if callable(started) and callable(stopped):
            span = float(stopped()) - float(started())
            if span > 0.0:
                self.gpu_seconds += span
        status, error = memory.retire(command)
        self._held.clear()
        if status != 4:
            message = f"metal command failed ({status}): {error}"
            raise RuntimeError(message)

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
        self._held.extend(constants)
        encoder.setComputePipelineState_(self.pipes[name])
        for index, handle in enumerate(handles):
            shift = 0 if offsets is None else int(offsets[index])
            memory.bind(encoder, index, handle, shift)
        base = len(handles)
        for index, blob in enumerate(constants):
            encoder.setBytes_length_atIndex_(blob, len(blob), base + index)
        encoder.dispatchThreadgroups_threadsPerThreadgroup_(
            self.metal.MTLSize(*groups),
            self.metal.MTLSize(*threads),
        )
        if own:
            self.end()

    def close(self) -> None:
        if self._encoder is not None:
            self.end()
        memory.sweep(self.owner)
