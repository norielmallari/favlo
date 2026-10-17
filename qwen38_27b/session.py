"""Device buffers and the shader library for this pipeline."""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Any

import numpy as np

KERNELS = (
    "prefill_fp4_n64",
    "prefill_fp4_e8m0",
    "prefill_fp8_n64",
    "prefill_fp8_n64_stride",
    "prefill_fp8_m8",
    "gemv_fp4",
    "gemv_fp4_vec",
    "gemv_fp4_x4",
    "gemv_fp4_span2",
    "copy_f32",
    "rmsnorm_f32",
    "rmsnorm_rows",
    "swiglu_f32",
    "residual_rows",
    "concat_pair",
    "cast_f32_f16",
    "zero_f32",
    "embed_fp4_rows",
    "gather_fp4_row",
    "gdn_in_proj",
    "gdn_in_proj_span2",
    "attn_qkv",
    "attn_qkv_span2",
    "ffn_gate_up",
    "ffn_gate_up_span2",
    "gdn_conv_rows",
    "gdn_conv_shift",
    "gdn_state_rows",
    "split_q_gate_rows",
    "qk_norm_rope",
    "qk_norm_rope_rows",
    "kv_plane_k",
    "kv_tail_store",
    "kv_plane_v",
    "kv_cast_q",
    "kv_row_init",
    "kv_softmax_tile",
    "kv_softmax_span",
    "kv_softmax_max",
    "kv_softmax_prob",
    "kv_softmax_max_rows",
    "kv_softmax_prob_rows",
    "kv_softmax_close",
    "kv_pv_split8",
    "kv_part_sum",
    "kv_acc_add",
    "kv_attend_write",
    "dequant_e4m3",
    "argmax_partial",
)

_SHADER = Path(__file__).resolve().parent / "shaders" / "kernels.metal"
_ZERO = b"\x00" * (8 << 20)


def _clear(buffer: Any, nbytes: int, note: Any = None) -> None:
    view = memoryview(buffer.contents().as_buffer(nbytes))
    offset = 0
    while offset < nbytes:
        end = min(offset + len(_ZERO), nbytes)
        view[offset:end] = _ZERO[: end - offset]
        step = end - offset
        offset = end
        if note is not None:
            note(step)


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
        self._load_note = None

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
        _clear(buffer, nbytes, self._load_note)
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
        destination = self.zeros(rows * cols * 4)
        width = int(self.pipes[name].threadExecutionWidth())
        tile = 64 if name.endswith("n64") else 32
        self.dispatch(
            name,
            [self.upload(activation), self.upload(data), self.upload(scales), destination],
            [struct.pack("<I", rows), struct.pack("<I", cols), struct.pack("<I", inner)],
            (cols // tile, rows // 64, 1),
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
