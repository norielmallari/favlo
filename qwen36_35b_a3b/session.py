"""Device buffers and the shader library for this pipeline."""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Any

import numpy as np

KERNELS = (
    "prefill_fp4_e8m0",
    "prefill_fp8_e8m0",
    "prefill_fp8_n64",
    "prefill_fp8_n64_stride",
    "prefill_fp8_n64_sweep",
    "prefill_fp8_m8",
    "gemv_fp4",
    "gemv_f16",
    "rmsnorm_f32",
    "layernorm_f32",
    "silu_f32",
    "sigmoid_f32",
    "softplus_f32",
    "gelu_tanh_f32",
    "l2_heads_f32",
    "residual_add_f32",
    "swiglu_f32",
    "rope_partial",
    "gdn_conv",
    "gdn_state",
    "attend_gqa",
    "attend_head72",
    "topk8",
    "gather_row",
    "dequant_e4m3",
    "cast_f32_f16",
    "shift_history",
    "store_row",
    "split_q_gate",
    "axpy_f32",
    "zero_f32",
    "mul_f32",
    "softmax256",
    "gather_fp4_row",
    "gemv_fp4_batch",
    "gemv_fp4_x4",
    "gemv_fp4_vec",
    "gemv_fp4_vec_slots",
    "gemv_fp4_vec_acc",
    "gemv_f16_f32",
    "gdn_in_proj",
    "attn_qkv",
    "router_coeff",
    "gdn_conv_shift",
    "qk_norm_rope",
    "store_kv",
    "attend_gqa_heads",
    "kv_plane_k",
    "kv_tail_store",
    "kv_plane_v",
    "kv_cast_q",
    "kv_row_init",
    "kv_softmax_tile",
    "kv_softmax_span",
    "kv_softmax_max",
    "kv_softmax_prob",
    "kv_softmax_close",
    "kv_softmax_max_rows",
    "kv_softmax_prob_rows",
    "kv_pv_split",
    "kv_pv_split8",
    "kv_part_sum",
    "kv_acc_add",
    "kv_attend_write",
    "moe_gate_silu",
    "moe_down_mix",
    "rmsnorm_rows",
    "embed_fp4_rows",
    "residual_rows",
    "gdn_conv_rows",
    "gdn_state_rows",
    "split_q_gate_rows",
    "qk_norm_rope_rows",
    "store_kv_rows",
    "attend_prefill",
    "attend_prefill2",
    "gemv_f16_rows",
    "gemv_f16_grid",
    "coeff_rows",
    "router8_rows",
    "silu_mul_rows",
    "gather_half_rows",
    "gather_half_slots",
    "silu_wave",
    "prefill_fp4_expert_tile",
    "prefill_f16",
    "prefill_fp4_n64",
    "prefill_fp4_n64_stride",
    "scatter_add_rows",
    "silu_mul_slots",
    "router8",
    "mix_experts",
)

_SHADER = Path(__file__).resolve().parent / "shaders" / "kernels.metal"
_ZERO = b"\x00" * (8 << 20)


def _clear(buffer: Any, nbytes: int) -> None:
    view = memoryview(buffer.contents().as_buffer(nbytes))
    offset = 0
    while offset < nbytes:
        end = min(offset + len(_ZERO), nbytes)
        view[offset:end] = _ZERO[: end - offset]
        offset = end


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


class Session:
    """One Metal device. Compiling the library fails if a kernel has no pipeline."""

    def __init__(self) -> None:
        metal = _metal()
        device = metal.MTLCreateSystemDefaultDevice()
        if device is None:
            message = "this Mac has no Metal device"
            raise RuntimeError(message)
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
        self.queue = device.newCommandQueue()
        self.shared = metal.MTLResourceStorageModeShared
        self.pipes: dict[str, Any] = {}
        self._recording = False
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

    @property
    def device_name(self) -> str:
        name = self.device.name
        return str(name() if callable(name) else name)

    def upload(self, blob: bytes | np.ndarray) -> Any:
        if isinstance(blob, np.ndarray):
            raw = np.ascontiguousarray(blob).tobytes()
        else:
            raw = bytes(blob)
        buffer = self.device.newBufferWithLength_options_(max(len(raw), 16), self.shared)
        if raw:
            memoryview(buffer.contents().as_buffer(len(raw)))[:] = raw
        return buffer

    def allocate(self, nbytes: int) -> Any | None:
        """A shared buffer, or None when the device refuses the length."""
        return self.device.newBufferWithLength_options_(max(nbytes, 16), self.shared)

    def zeros(self, nbytes: int) -> Any:
        buffer = self.allocate(nbytes)
        if buffer is None:
            message = f"device refused a {nbytes} byte buffer"
            raise RuntimeError(message)
        _clear(buffer, nbytes)
        return buffer

    def download(self, buffer: Any, nbytes: int) -> bytes:
        return bytes(memoryview(buffer.contents().as_buffer(nbytes)))

    def download_f32(self, buffer: Any, count: int) -> np.ndarray:
        return np.frombuffer(self.download(buffer, count * 4), dtype=np.float32).copy()

    def dispatch(
        self,
        name: str,
        buffers: list[Any],
        constants: list[bytes],
        groups: tuple[int, int, int],
        threads: tuple[int, int, int],
    ) -> None:
        self._held.extend(buffers)
        self._held.extend(constants)
        command = self.queue.commandBuffer()
        encoder = command.computeCommandEncoder()
        encoder.setComputePipelineState_(self.pipes[name])
        for index, buffer in enumerate(buffers):
            encoder.setBuffer_offset_atIndex_(buffer, 0, index)
        base = len(buffers)
        for index, blob in enumerate(constants):
            encoder.setBytes_length_atIndex_(blob, len(blob), base + index)
        encoder.dispatchThreadgroups_threadsPerThreadgroup_(
            self.metal.MTLSize(*groups),
            self.metal.MTLSize(*threads),
        )
        encoder.endEncoding()
        command.commit()
        command.waitUntilCompleted()
        self._held.clear()
        if int(command.status()) != 4:
            message = f"{name} command status {command.status()} {command.error()}"
            raise RuntimeError(message)

    def begin(self) -> None:
        self._recording = True
        self._pass_held: list[Any] = []
        self._pass_command = self.queue.commandBuffer()
        self._pass_encoder = self._pass_command.computeCommandEncoder()

    def encode(
        self,
        name: str,
        bindings: list[tuple[Any, int]],
        constants: list[bytes],
        groups: tuple[int, int, int],
        threads: tuple[int, int, int],
    ) -> None:
        encoder = self._pass_encoder
        encoder.setComputePipelineState_(self.pipes[name])
        for index, (buffer, offset) in enumerate(bindings):
            self._pass_held.append(buffer)
            encoder.setBuffer_offset_atIndex_(buffer, int(offset), index)
        base = len(bindings)
        for index, blob in enumerate(constants):
            self._pass_held.append(blob)
            encoder.setBytes_length_atIndex_(blob, len(blob), base + index)
        encoder.dispatchThreadgroups_threadsPerThreadgroup_(
            self.metal.MTLSize(*groups),
            self.metal.MTLSize(*threads),
        )

    def end(self) -> None:
        self._recording = False
        self._pass_encoder.endEncoding()
        command = self._pass_command
        command.commit()
        command.waitUntilCompleted()
        self._pass_held.clear()
        self._pass_encoder = None
        self._pass_command = None
        if int(command.status()) != 4:
            message = f"command status {command.status()} {command.error()}"
            raise RuntimeError(message)

    def prefill(
        self,
        name: str,
        activation: np.ndarray,
        data: bytes,
        scales: bytes,
        rows: int,
        cols: int,
        inner: int,
    ) -> np.ndarray:
        if rows % 64 or cols % 32 or inner % 32:
            message = "prefill tile requires rows multiple of 64 and cols, inner multiple of 32"
            raise ValueError(message)
        destination = self.zeros(rows * cols * 4)
        width = int(self.pipes[name].threadExecutionWidth())
        self.dispatch(
            name,
            [self.upload(activation), self.upload(data), self.upload(scales), destination],
            [struct.pack("<I", rows), struct.pack("<I", cols), struct.pack("<I", inner)],
            (cols // 32, rows // 64, 1),
            (width * 4, 1, 1),
        )
        return self.download_f32(destination, rows * cols).reshape(rows, cols)

    def gemv_fp4(
        self, activation: np.ndarray, data: bytes, scales: bytes, cols: int, inner: int
    ) -> np.ndarray:
        destination = self.zeros(cols * 4)
        self.dispatch(
            "gemv_fp4",
            [self.upload(activation), self.upload(data), self.upload(scales), destination],
            [struct.pack("<I", cols), struct.pack("<I", inner)],
            (cols, 1, 1),
            (32, 1, 1),
        )
        return self.download_f32(destination, cols)

    def gemv_f16(self, activation: np.ndarray, weights: np.ndarray) -> np.ndarray:
        cols, inner = weights.shape
        destination = self.zeros(cols * 4)
        self.dispatch(
            "gemv_f16",
            [
                self.upload(activation),
                self.upload(np.ascontiguousarray(weights, dtype=np.float16)),
                destination,
            ],
            [struct.pack("<I", cols), struct.pack("<I", inner)],
            (cols, 1, 1),
            (32, 1, 1),
        )
        return self.download_f32(destination, cols)
