"""Device buffers, the int4 prefill tile, and the request-path shaders."""

from __future__ import annotations

import json
import math
import struct
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from qwen38_flash_next.formats import BLOCK, IO_CHUNK, NGRAM_SLOT, NGRAM_WIDTH, int4_plane_sizes
from qwen38_flash_next.shader_text import shader_text

VECTOR_SHADER = shader_text("vector.metal")
ACT_SHADER = shader_text("act.metal")
# Prefill is mpp::tensor_ops::matmul2d. The E8M0 plane requires transpose_right.
# Extents are (inner, outer) with the inner dimension contiguous, which is
# row-major [rows, inner] for the activation and [cols, inner] for the weight.
# WWDC26 session 330. The scale block is 32 along the inner dimension.
MATMUL_SHADER = shader_text("matmul.metal")


def _metal():
    import Metal

    return Metal


def _buffer_view(buffer: Any, count: int) -> memoryview:
    return memoryview(buffer.contents().as_buffer(count))


def _fill_buffer(buffer: Any, count: int, value: int) -> None:
    _buffer_view(buffer, count)[:] = bytes([value]) * count


def _write_u32(buffer: Any, value: int) -> None:
    _buffer_view(buffer, 4)[:] = value.to_bytes(4, "little")


def _buffer_bytes(buffer: Any, count: int) -> bytes:
    return bytes(_buffer_view(buffer, count))


def _buffer_slice(buffer: Any, offset: int, nbytes: int) -> bytes:
    view = _buffer_view(buffer, offset + nbytes)
    return bytes(view[offset : offset + nbytes])


def _objc_pair(result: Any) -> tuple[Any, str | None]:
    if isinstance(result, tuple) and len(result) == 2:
        value, error = result
        if error is not None:
            return None, str(error)
        return value, None
    return result, None


def _device_name(device: Any) -> str:
    name = device.name
    return str(name() if callable(name) else name)


def _shared_buffer(device: Any, blob: bytes) -> Any:
    metal = _metal()
    buffer = device.newBufferWithLength_options_(
        max(len(blob), 16), metal.MTLResourceStorageModeShared
    )
    if blob:
        _buffer_view(buffer, len(blob))[:] = blob
    return buffer


class Int4Gemm:
    """Prefill matmul2d and decode shader dot for int4 weights with E8M0 scales."""

    def __init__(self) -> None:
        metal = _metal()
        device = metal.MTLCreateSystemDefaultDevice()
        if device is None:
            message = "this Mac has no Metal device"
            raise RuntimeError(message)
        options = metal.MTLCompileOptions.alloc().init()
        options.setLanguageVersion_((4 << 16) + 1)
        library, error = _objc_pair(
            device.newLibraryWithSource_options_error_(MATMUL_SHADER, options, None)
        )
        if library is None:
            message = f"matmul shader failed to compile: {error}"
            raise RuntimeError(message)
        self.metal = metal
        self.device = device
        self.prefill = self._pipeline(library, "prefill_int4_e8m0")
        self.decode = self._pipeline(library, "decode_int4_e8m0")

    def _pipeline(self, library: Any, name: str) -> Any:
        function = library.newFunctionWithName_(name)
        pipeline, error = _objc_pair(
            self.device.newComputePipelineStateWithFunction_error_(function, None)
        )
        if pipeline is None:
            message = f"{name} pipeline failed: {error}"
            raise RuntimeError(message)
        return pipeline

    def prefill_int4(
        self, activation: bytes, weights: bytes, scales: bytes, rows: int, cols: int, inner: int
    ) -> bytes:
        """rows×inner half activation times cols×inner int4. Returns rows×cols float32."""
        if rows % 64 or cols % 32 or inner % 32:
            message = "prefill tile requires rows multiple of 64 and cols, inner multiple of 32"
            raise ValueError(message)
        destination = _shared_buffer(self.device, b"\x00" * (rows * cols * 4))
        self._dispatch_prefill(activation, weights, scales, destination, rows, cols, inner)
        return _buffer_bytes(destination, rows * cols * 4)

    def decode_int4(
        self, activation: bytes, weights: bytes, scales: bytes, cols: int, inner: int
    ) -> bytes:
        """One half vector of length inner times cols int4 rows. Returns cols float32."""
        if cols < 1 or inner % 32:
            message = "decode requires a positive column count and inner multiple of 32"
            raise ValueError(message)
        destination = _shared_buffer(self.device, b"\x00" * (cols * 4))
        self._dispatch_decode(activation, weights, scales, destination, cols, inner)
        return _buffer_bytes(destination, cols * 4)

    def _dispatch_prefill(
        self,
        activation: bytes,
        weights: bytes,
        scales: bytes,
        destination: Any,
        rows: int,
        cols: int,
        inner: int,
    ) -> None:
        buffers = (
            _shared_buffer(self.device, activation),
            _shared_buffer(self.device, weights),
            _shared_buffer(self.device, scales),
            destination,
            _shared_buffer(self.device, rows.to_bytes(4, "little")),
            _shared_buffer(self.device, cols.to_bytes(4, "little")),
            _shared_buffer(self.device, inner.to_bytes(4, "little")),
        )
        width = self.prefill.threadExecutionWidth()
        self._encode(
            self.prefill,
            buffers,
            self.metal.MTLSize(cols // 32, rows // 64, 1),
            self.metal.MTLSize(width * 4, 1, 1),
        )

    def _dispatch_decode(
        self,
        activation: bytes,
        weights: bytes,
        scales: bytes,
        destination: Any,
        cols: int,
        inner: int,
    ) -> None:
        buffers = (
            _shared_buffer(self.device, activation),
            _shared_buffer(self.device, weights),
            _shared_buffer(self.device, scales),
            destination,
            _shared_buffer(self.device, cols.to_bytes(4, "little")),
            _shared_buffer(self.device, inner.to_bytes(4, "little")),
        )
        self._encode(
            self.decode,
            buffers,
            self.metal.MTLSize(cols, 1, 1),
            self.metal.MTLSize(self.decode.threadExecutionWidth(), 1, 1),
        )

    def _encode(self, pipeline: Any, buffers: tuple[Any, ...], groups: Any, threads: Any) -> None:
        command = self.device.newCommandQueue().commandBuffer()
        encoder = command.computeCommandEncoder()
        encoder.setComputePipelineState_(pipeline)
        for index, buffer in enumerate(buffers):
            encoder.setBuffer_offset_atIndex_(buffer, 0, index)
        encoder.dispatchThreadgroups_threadsPerThreadgroup_(groups, threads)
        encoder.endEncoding()
        command.commit()
        command.waitUntilCompleted()


def _log(message: str) -> None:
    sys.stderr.write(message + "\n")
    sys.stderr.flush()


def _tok_s(label: str, tokens: int, seconds: float) -> str:
    if tokens <= 0 or seconds <= 0:
        return f"{label} 0 tok"
    return f"{label} {tokens / seconds:.2f} tok/s ({tokens} tok in {seconds:.2f}s)"


def _append_transcript(path: Path | None, prompt: str, output: str) -> None:
    """Append one prompt and its completion. Token rates stay on the console."""
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    record = f"prompt:\n{prompt.rstrip()}\n\noutput:\n{output.rstrip()}\n"
    existing = path.is_file() and path.stat().st_size > 0
    with path.open("a", encoding="utf-8") as handle:
        if existing:
            handle.write("\n")
        handle.write(record)


def _prompt_text(messages: Any) -> str:
    if isinstance(messages, str):
        return messages
    if not isinstance(messages, list):
        return json.dumps(messages, ensure_ascii=False)
    lines: list[str] = []
    for message in messages:
        if not isinstance(message, dict):
            lines.append(str(message))
            continue
        content = message.get("content", "")
        if not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False)
        lines.append(f"{message.get('role', '')}: {content}")
    return "\n".join(lines)


def _check_f32_gemv(gpu: MetalSession) -> None:
    weight = np.arange(8 * 4, dtype=np.float32).reshape(8, 4)
    vector = np.arange(4, dtype=np.float32) + 1.0
    got = gpu.f32(gpu.load_f32(weight), vector)
    expect = weight @ vector
    if not np.allclose(got, expect):
        message = "f32 shader dot diverged"
        raise RuntimeError(message)


def bf16_to_f32(raw: bytes) -> np.ndarray:
    units = np.frombuffer(raw, dtype=np.uint16).astype(np.uint32)
    return (units << 16).view(np.float32)


class Int4Matrix:
    def __init__(
        self,
        rows: int,
        inner: int,
        packed_rows: int,
        weights: Any,
        scales: Any,
        stored_inner: int | None = None,
        weight_off: int = 0,
        scale_off: int = 0,
    ) -> None:
        self.rows = rows
        self.inner = inner
        self.stored_inner = inner if stored_inner is None else stored_inner
        self.packed_rows = packed_rows
        self.weights = weights
        self.scales = scales
        self.weight_off = weight_off
        self.scale_off = scale_off


class F16Matrix:
    def __init__(self, rows: int, inner: int, weights: Any, weight_off: int = 0) -> None:
        self.rows = rows
        self.inner = inner
        self.weights = weights
        self.weight_off = weight_off


class F32Matrix:
    """One BF16 matrix, expanded to float32 and uploaded for the shader dot."""

    def __init__(self, rows: int, inner: int, weights: Any) -> None:
        self.rows = rows
        self.inner = inner
        self.weights = weights


class Dev:
    """A float buffer that stays on the device until the host has to read it."""

    def __init__(self, buf: Any, n: int, offset: int = 0) -> None:
        self.buf = buf
        self.n = n
        self.offset = offset

    def view(self, shift: int, n: int) -> Dev:
        if shift < 0 or n < 0 or shift + n > self.n:
            message = f"view {shift}+{n} outside {self.n}"
            raise ValueError(message)
        return Dev(self.buf, n, self.offset + shift)


def _dev_bind(src: Dev) -> tuple[Any, int]:
    return src.buf, src.offset * 4


class MetalSession:
    """Decode dots, prefill matmul2d, the GDN delta, and grouped-query attention."""

    def __init__(self) -> None:
        self.gemm = Int4Gemm()
        self.device = self.gemm.device
        self.metal = self.gemm.metal
        width = self.gemm.decode.threadExecutionWidth()
        if int(width) != 32:
            message = f"shader dot assumes simd width 32, device reported {width}"
            raise RuntimeError(message)
        self.queue = self.device.newCommandQueue()
        self.launch_count = 0
        self.command_count = 0
        self.gpu_seconds = 0.0
        self.encode_seconds = 0.0
        self.io_seconds = 0.0
        self.io_loads = 0
        self._encoder = None
        self._command = None
        self._batch_launches = 0
        self._retain: list[Any] = []
        self._held_constants: list[bytes] = []
        self._io = None
        self._event = None
        self._event_value = 0
        self._io_pending: list[Any] = []
        options = self.metal.MTLCompileOptions.alloc().init()
        options.setLanguageVersion_((4 << 16) + 1)
        library, error = _objc_pair(
            self.device.newLibraryWithSource_options_error_(
                VECTOR_SHADER + ACT_SHADER, options, None
            )
        )
        if library is None:
            message = f"vector shader failed to compile: {error}"
            raise RuntimeError(message)
        self.f16_pipe = self._pipeline(library, "gemv_f16")
        self.f32_pipe = self._pipeline(library, "gemv_f32")
        self.gdn_pipe = self._pipeline(library, "gdn_delta")
        self.attend_pipe = self._pipeline(library, "attend_gqa")
        act_names = (
            "silu_f32",
            "sigmoid_f32",
            "softplus_f32",
            "scale_f32",
            "mul_f32",
            "add_f32",
            "add_into_f32",
            "copy_f32",
            "axpy_slot",
            "scale_rows_f32",
            "rms_rows",
            "l2_rows",
            "rope_rows",
            "causal_conv",
            "dilated_conv",
            "repeat_rows",
            "residual_write",
            "branch_mean",
            "gdn_alpha",
            "ple_gate",
            "gate_broadcast",
            "split_q",
            "sigmoid_mul",
            "topk_softmax",
            "argmax_partial",
            "argmax_final",
            "cast_f16_pad",
            "cast_prefill",
            "compact_rows",
            "embed_gather",
            "dequant_ngram",
            "broadcast_rows",
            "sample_f32",
        )
        self.act = {name: self._pipeline(library, name) for name in act_names}
        self._argmax_peaks = self.alloc(1024 * 4)
        self._argmax_index = self.alloc(1024 * 4)
        self._argmax_out = self.alloc(4)
        self._topk_probs = self.alloc(512 * 4)
        self._sample_scratch = None
        self.activation = self.alloc(10240 * 2)
        self.f32_activation = self.alloc(10240 * 4)
        self.destination = self.alloc(248320 * 4)
        self.prefill_act = self.alloc(64 * 10240 * 2)
        self.prefill_dest = self.alloc(64 * 12288 * 4)
        self.scores = self.alloc(24 * 64 * 4)
        self.attend_q = self.alloc(24 * 256 * 4)
        self.attend_k = self.alloc(64 * 2 * 256 * 4)
        self.attend_v = self.alloc(64 * 2 * 256 * 4)
        self.attend_out = self.alloc(24 * 256 * 4)
        self.gdn_q = self.alloc(48 * 128 * 4)
        self.gdn_k = self.alloc(48 * 128 * 4)
        self.gdn_v = self.alloc(48 * 128 * 4)
        self.gdn_a = self.alloc(48 * 4)
        self.gdn_b = self.alloc(48 * 4)
        self.gdn_y = self.alloc(48 * 128 * 4)

    def _pipeline(self, library: Any, name: str) -> Any:
        function = library.newFunctionWithName_(name)
        pipeline, error = _objc_pair(
            self.device.newComputePipelineStateWithFunction_error_(function, None)
        )
        if pipeline is None:
            message = f"{name} pipeline failed: {error}"
            raise RuntimeError(message)
        return pipeline

    def alloc(self, nbytes: int) -> Any:
        return self.device.newBufferWithLength_options_(
            max(nbytes, 16), self.metal.MTLResourceStorageModeShared
        )

    def upload(self, blob: bytes) -> Any:
        buffer = self.alloc(len(blob))
        if blob:
            _buffer_view(buffer, len(blob))[:] = blob
        return buffer

    def zeros(self, nbytes: int) -> Any:
        buffer = self.alloc(nbytes)
        _fill_buffer(buffer, nbytes, 0)
        return buffer

    def load_int4(
        self, image: bytes | memoryview, rows: int, logical_inner: int, stored_inner: int
    ) -> Int4Matrix:
        """Upload a record image. Zero rows pad the output width to a multiple of 32."""
        data_n, scale_n = int4_plane_sizes(rows, stored_inner)
        if len(image) != data_n + scale_n:
            message = f"int4 image is {len(image)} bytes, expected {data_n + scale_n}"
            raise ValueError(message)
        packed_rows = ((rows + 31) // 32) * 32
        weights = memoryview(image)[:data_n]
        scales = memoryview(image)[data_n:]
        if packed_rows != rows:
            row_b = stored_inner // 2
            scale_b = stored_inner // BLOCK
            w = bytearray(weights)
            w.extend(b"\x00" * ((packed_rows - rows) * row_b))
            s = bytearray(scales)
            s.extend(b"\x00" * ((packed_rows - rows) * scale_b))
            weights = memoryview(w)
            scales = memoryview(s)
        return Int4Matrix(
            rows,
            logical_inner,
            packed_rows,
            self.upload(weights),
            self.upload(scales),
            stored_inner,
        )

    def load_f32(self, matrix: np.ndarray) -> F32Matrix:
        array = np.ascontiguousarray(matrix, dtype=np.float32)
        rows, inner = int(array.shape[0]), int(array.shape[1])
        return F32Matrix(rows, inner, self.upload(array.tobytes()))

    def f32(self, matrix: F32Matrix, vector: np.ndarray) -> np.ndarray:
        if vector.shape != (matrix.inner,):
            message = f"f32 vector {vector.shape} does not match inner {matrix.inner}"
            raise ValueError(message)
        if matrix.inner > 10240:
            message = f"f32 inner {matrix.inner} exceeds the activation buffer"
            raise ValueError(message)
        self._write_f32(self.f32_activation, vector)
        self._encode(
            self.f32_pipe,
            (self.f32_activation, matrix.weights, self.destination),
            (struct.pack("<I", matrix.rows), struct.pack("<I", matrix.inner)),
            self.metal.MTLSize(matrix.rows, 1, 1),
            self.metal.MTLSize(32, 1, 1),
        )
        raw = _buffer_bytes(self.destination, matrix.rows * 4)
        return np.frombuffer(raw, dtype=np.float32).copy()

    def int4(self, matrix: Int4Matrix, vector: np.ndarray) -> np.ndarray:
        if isinstance(matrix, F32Matrix):
            return self.f32(matrix, vector)
        if vector.shape != (matrix.inner,):
            message = f"int4 vector {vector.shape} does not match inner {matrix.inner}"
            raise ValueError(message)
        activation = np.zeros(matrix.stored_inner, dtype=np.float32)
        activation[: matrix.inner] = vector
        half = np.clip(activation, -65504.0, 65504.0).astype(np.float16)
        _buffer_view(self.activation, half.nbytes)[:] = half.tobytes()
        self._encode(
            self.gemm.decode,
            (self.activation, matrix.weights, matrix.scales, self.destination),
            (struct.pack("<I", matrix.rows), struct.pack("<I", matrix.stored_inner)),
            self.metal.MTLSize(matrix.rows, 1, 1),
            self.metal.MTLSize(32, 1, 1),
            (0, matrix.weight_off, matrix.scale_off, 0),
        )
        raw = _buffer_bytes(self.destination, matrix.rows * 4)
        return np.frombuffer(raw, dtype=np.float32).copy()

    def int4_rows(self, matrix: Int4Matrix, rows: np.ndarray) -> np.ndarray:
        if isinstance(matrix, F32Matrix):
            return np.stack([self.f32(matrix, row) for row in rows], axis=0)
        count = rows.shape[0]
        if count == 1:
            return self.int4(matrix, rows[0])[None, :]
        padded = ((count + 63) // 64) * 64
        cols = matrix.packed_rows
        act = np.zeros((padded, matrix.stored_inner), dtype=np.float16)
        act[:count, : matrix.inner] = np.clip(rows, -65504.0, 65504.0).astype(np.float16)
        self.prefill_act = self._grow(self.prefill_act, act.nbytes)
        self.prefill_dest = self._grow(self.prefill_dest, padded * cols * 4)
        _buffer_view(self.prefill_act, act.nbytes)[:] = act.tobytes()
        _fill_buffer(self.prefill_dest, padded * cols * 4, 0)
        self._encode(
            self.gemm.prefill,
            (self.prefill_act, matrix.weights, matrix.scales, self.prefill_dest),
            (
                struct.pack("<I", padded),
                struct.pack("<I", cols),
                struct.pack("<I", matrix.stored_inner),
            ),
            self.metal.MTLSize(cols // 32, padded // 64, 1),
            self.metal.MTLSize(32 * 4, 1, 1),
        )
        flat = np.frombuffer(
            _buffer_bytes(self.prefill_dest, padded * cols * 4), dtype=np.float32
        ).reshape(padded, cols)
        return flat[:count, : matrix.rows].copy()

    def f16(self, matrix: F16Matrix, vector: np.ndarray) -> np.ndarray:
        if isinstance(matrix, F32Matrix):
            return self.f32(matrix, vector)
        half = np.clip(vector, -65504.0, 65504.0).astype(np.float16)
        _buffer_view(self.activation, half.nbytes)[:] = half.tobytes()
        self._encode(
            self.f16_pipe,
            (self.activation, matrix.weights, self.destination),
            (struct.pack("<I", matrix.rows), struct.pack("<I", matrix.inner)),
            self.metal.MTLSize(matrix.rows, 1, 1),
            self.metal.MTLSize(32, 1, 1),
            (0, matrix.weight_off, 0),
        )
        raw = _buffer_bytes(self.destination, matrix.rows * 4)
        return np.frombuffer(raw, dtype=np.float32).copy()

    def f16_rows(self, matrix: F16Matrix, rows: np.ndarray) -> np.ndarray:
        return np.stack([self.f16(matrix, row) for row in rows], axis=0)

    def gdn(
        self,
        state: Any,
        query: np.ndarray,
        key: np.ndarray,
        value: np.ndarray,
        alpha: np.ndarray,
        beta: np.ndarray,
    ) -> np.ndarray:
        self._write_f32(self.gdn_q, query)
        self._write_f32(self.gdn_k, key)
        self._write_f32(self.gdn_v, value)
        self._write_f32(self.gdn_a, alpha)
        self._write_f32(self.gdn_b, beta)
        self._encode(
            self.gdn_pipe,
            (state, self.gdn_q, self.gdn_k, self.gdn_v, self.gdn_a, self.gdn_b, self.gdn_y),
            (),
            self.metal.MTLSize(48, 1, 1),
            self.metal.MTLSize(128, 1, 1),
        )
        return np.frombuffer(_buffer_bytes(self.gdn_y, 48 * 128 * 4), dtype=np.float32).copy()

    def attend(self, query: np.ndarray, key: np.ndarray, value: np.ndarray) -> np.ndarray:
        tokens = key.shape[0]
        self.scores = self._grow(self.scores, 24 * tokens * 4)
        self.attend_k = self._grow(self.attend_k, key.nbytes)
        self.attend_v = self._grow(self.attend_v, value.nbytes)
        self._write_f32(self.attend_q, query)
        self._write_f32(self.attend_k, key)
        self._write_f32(self.attend_v, value)
        scale = struct.pack("<f", 1.0 / math.sqrt(256.0))
        self._encode(
            self.attend_pipe,
            (self.attend_q, self.attend_k, self.attend_v, self.scores, self.attend_out),
            (
                struct.pack("<I", tokens),
                struct.pack("<I", 2),
                struct.pack("<I", 256),
                scale,
            ),
            self.metal.MTLSize(24, 1, 1),
            self.metal.MTLSize(32, 1, 1),
        )
        return np.frombuffer(_buffer_bytes(self.attend_out, 24 * 256 * 4), dtype=np.float32).copy()

    def _grow(self, buffer: Any, nbytes: int) -> Any:
        if buffer.length() >= nbytes:
            return buffer
        size = 16
        while size < nbytes:
            size *= 2
        return self.alloc(size)

    def _write_f32(self, buffer: Any, array: np.ndarray) -> None:
        blob = np.ascontiguousarray(array, dtype=np.float32).tobytes()
        if buffer.length() < len(blob):
            message = f"buffer {buffer.length()} is smaller than {len(blob)}"
            raise RuntimeError(message)
        _buffer_view(buffer, len(blob))[:] = blob

    def _encode(
        self,
        pipeline: Any,
        buffers: tuple[Any, ...],
        constants: tuple[bytes, ...],
        groups: Any,
        threads: Any,
        buffer_offsets: tuple[int, ...] | None = None,
    ) -> None:
        self.flush()
        started = time.perf_counter()
        command = self.queue.commandBuffer()
        encoder = command.computeCommandEncoder()
        encoder.setComputePipelineState_(pipeline)
        offsets = buffer_offsets or ((0,) * len(buffers))
        for index, buffer in enumerate(buffers):
            encoder.setBuffer_offset_atIndex_(buffer, offsets[index], index)
        for offset, blob in enumerate(constants):
            encoder.setBytes_length_atIndex_(blob, len(blob), len(buffers) + offset)
        encoder.dispatchThreadgroups_threadsPerThreadgroup_(groups, threads)
        encoder.endEncoding()
        command.commit()
        command.waitUntilCompleted()
        self.launch_count += 1
        self.command_count += 1
        self.encode_seconds += time.perf_counter() - started
        gpu_start = float(command.GPUStartTime())
        gpu_end = float(command.GPUEndTime())
        if gpu_end > gpu_start:
            self.gpu_seconds += gpu_end - gpu_start
        if int(command.status()) != 4:
            message = f"Metal command status {command.status()} {command.error()}"
            raise RuntimeError(message)

    def _size(self, width: int, height: int = 1, depth: int = 1) -> Any:
        return self.metal.MTLSize(width, height, depth)

    def dispatch(
        self,
        pipeline: Any,
        bindings: Sequence[tuple[Any, int]],
        constants: Sequence[bytes],
        groups: Any,
        threads: Any,
    ) -> None:
        if self._command is None:
            self._command = self.queue.commandBuffer()
            self._batch_launches = 0
        if self._encoder is None:
            self._encoder = self._command.computeCommandEncoder()
        encoder = self._encoder
        encoder.setComputePipelineState_(pipeline)
        for index, (buffer, offset) in enumerate(bindings):
            encoder.setBuffer_offset_atIndex_(buffer, int(offset), index)
        base = len(bindings)
        for index, blob in enumerate(constants):
            self._held_constants.append(blob)
            encoder.setBytes_length_atIndex_(blob, len(blob), base + index)
        encoder.dispatchThreadgroups_threadsPerThreadgroup_(groups, threads)
        self._batch_launches += 1

    def flush(self) -> None:
        if self._encoder is not None:
            self._encoder.endEncoding()
            self._encoder = None
        if self._command is None:
            return
        started = time.perf_counter()
        command = self._command
        command.commit()
        command.waitUntilCompleted()
        self.launch_count += self._batch_launches
        self.command_count += 1
        self._batch_launches = 0
        self.encode_seconds += time.perf_counter() - started
        gpu_start = float(command.GPUStartTime())
        gpu_end = float(command.GPUEndTime())
        if gpu_end > gpu_start:
            self.gpu_seconds += gpu_end - gpu_start
        self._command = None
        self._retain.clear()
        self._held_constants.clear()
        pending = self._io_pending
        self._io_pending = []
        if int(command.status()) != 4:
            message = f"Metal command status {command.status()} {command.error()}"
            raise RuntimeError(message)
        for io_command in pending:
            if int(io_command.status()) != 3:
                message = f"Metal IO status {io_command.status()} {io_command.error()}"
                raise RuntimeError(message)

    def tmp(self, n: int) -> Dev:
        buf = self.alloc(n * 4)
        self._retain.append(buf)
        return Dev(buf, n)

    def dev_from(self, array: np.ndarray) -> Dev:
        flat = np.ascontiguousarray(array, dtype=np.float32).reshape(-1)
        buf = self.upload(flat.tobytes())
        self._retain.append(buf)
        return Dev(buf, int(flat.size))

    def dev_u32(self, values: Sequence[int]) -> Dev:
        raw = np.asarray(list(values), dtype=np.uint32)
        buf = self.upload(raw.tobytes())
        self._retain.append(buf)
        return Dev(buf, int(raw.size))

    def read_f32(self, src: Dev) -> np.ndarray:
        self.flush()
        nbytes = (src.offset + src.n) * 4
        raw = _buffer_bytes(src.buf, nbytes)
        return np.frombuffer(raw, dtype=np.float32, count=src.n, offset=src.offset * 4).copy()

    def read_u32(self, src: Dev) -> np.ndarray:
        self.flush()
        nbytes = (src.offset + src.n) * 4
        raw = _buffer_bytes(src.buf, nbytes)
        return np.frombuffer(raw, dtype=np.uint32, count=src.n, offset=src.offset * 4).copy()

    def _ew(
        self,
        name: str,
        bindings: Sequence[tuple[Any, int]],
        constants: Sequence[bytes],
        n: int,
        width: int = 256,
    ) -> None:
        groups = max((n + width - 1) // width, 1)
        self.dispatch(self.act[name], bindings, constants, self._size(groups), self._size(width))

    def _unary(self, name: str, src: Dev) -> Dev:
        dst = self.tmp(src.n)
        self._ew(name, (_dev_bind(src), _dev_bind(dst)), (struct.pack("<I", src.n),), src.n)
        return dst

    def silu(self, src: Dev) -> Dev:
        return self._unary("silu_f32", src)

    def sigmoid(self, src: Dev) -> Dev:
        return self._unary("sigmoid_f32", src)

    def softplus(self, src: Dev) -> Dev:
        return self._unary("softplus_f32", src)

    def scale(self, src: Dev, factor: float) -> Dev:
        dst = self.tmp(src.n)
        self._ew(
            "scale_f32",
            (_dev_bind(src), _dev_bind(dst)),
            (struct.pack("<I", src.n), struct.pack("<f", float(factor))),
            src.n,
        )
        return dst

    def mul(self, left: Dev, right: Dev) -> Dev:
        if left.n != right.n:
            message = f"mul lengths {left.n} and {right.n}"
            raise ValueError(message)
        dst = self.tmp(left.n)
        self._ew(
            "mul_f32",
            (_dev_bind(left), _dev_bind(right), _dev_bind(dst)),
            (struct.pack("<I", left.n),),
            left.n,
        )
        return dst

    def add(self, left: Dev, right: Dev) -> Dev:
        dst = self.tmp(left.n)
        self._ew(
            "add_f32",
            (_dev_bind(left), _dev_bind(right), _dev_bind(dst)),
            (struct.pack("<I", left.n),),
            left.n,
        )
        return dst

    def add_into(self, dst: Dev, src: Dev) -> None:
        self._ew(
            "add_into_f32",
            (_dev_bind(dst), _dev_bind(src)),
            (struct.pack("<I", src.n),),
            src.n,
        )

    def copy_into(self, src: Dev, dst: Dev, dst_shift: int) -> None:
        self._ew(
            "copy_f32",
            (_dev_bind(src), (dst.buf, (dst.offset + dst_shift) * 4)),
            (struct.pack("<I", src.n),),
            src.n,
        )

    def copy_dev(self, src: Dev) -> Dev:
        dst = self.tmp(src.n)
        self.copy_into(src, dst, 0)
        return dst

    def copy_span(self, src: Dev, src_shift: int, dst_buf: Any, dst_float: int, n: int) -> None:
        view = src.view(src_shift, n)
        self._ew(
            "copy_f32",
            (_dev_bind(view), (dst_buf, dst_float * 4)),
            (struct.pack("<I", n),),
            n,
        )

    def axpy(self, dst: Dev, src: Dev, scales: Dev, slot: int) -> None:
        self._ew(
            "axpy_slot",
            (_dev_bind(dst), _dev_bind(src), _dev_bind(scales)),
            (struct.pack("<I", dst.n), struct.pack("<I", slot)),
            dst.n,
        )

    def scale_rows(self, values: Dev, scales: Dev, count: int, cols: int) -> Dev:
        dst = self.tmp(count * cols)
        self._ew(
            "scale_rows_f32",
            (_dev_bind(values), _dev_bind(scales), _dev_bind(dst)),
            (struct.pack("<I", count), struct.pack("<I", cols)),
            count * cols,
        )
        return dst

    def rms(
        self,
        src: Dev,
        weight: Any,
        weight_byte: int,
        cols: int,
        rows: int,
        eps: float,
        zero_centered: bool,
        weight_rows: int,
    ) -> Dev:
        dst = self.tmp(rows * cols)
        centered = 1 if zero_centered else 0
        self.dispatch(
            self.act["rms_rows"],
            (_dev_bind(src), (weight, weight_byte), _dev_bind(dst)),
            (
                struct.pack("<I", cols),
                struct.pack("<f", float(eps)),
                struct.pack("<I", centered),
                struct.pack("<I", weight_rows),
            ),
            self._size(rows),
            self._size(32),
        )
        return dst

    def l2(self, src: Dev, rows: int, cols: int, eps: float) -> Dev:
        dst = self.tmp(rows * cols)
        self.dispatch(
            self.act["l2_rows"],
            (_dev_bind(src), _dev_bind(dst)),
            (struct.pack("<I", cols), struct.pack("<f", float(eps))),
            self._size(rows),
            self._size(32),
        )
        return dst

    def rope(
        self, src: Dev, rows: int, cols: int, rotary: int, position: int, theta: float
    ) -> Dev:
        dst = self.tmp(rows * cols)
        self._ew(
            "rope_rows",
            (_dev_bind(src), _dev_bind(dst)),
            (
                struct.pack("<I", cols),
                struct.pack("<I", rotary),
                struct.pack("<I", position),
                struct.pack("<f", float(theta)),
                struct.pack("<I", rows),
            ),
            rows * cols,
        )
        return dst

    def causal_conv(
        self,
        history: Any,
        current: Dev,
        weight: Any,
        weight_byte: int,
        channels: int,
        kernel: int,
    ) -> Dev:
        dst = self.tmp(channels)
        self._ew(
            "causal_conv",
            ((history, 0), _dev_bind(current), (weight, weight_byte), _dev_bind(dst)),
            (struct.pack("<I", channels), struct.pack("<I", kernel)),
            channels,
        )
        return dst

    def dilated_conv(
        self,
        history: Any,
        current: Dev,
        weight: Any,
        weight_byte: int,
        channels: int,
        kernel: int,
        dilation: int,
    ) -> Dev:
        dst = self.tmp(channels)
        self._ew(
            "dilated_conv",
            ((history, 0), _dev_bind(current), (weight, weight_byte), _dev_bind(dst)),
            (
                struct.pack("<I", channels),
                struct.pack("<I", kernel),
                struct.pack("<I", dilation),
            ),
            channels,
        )
        return dst

    def repeat_rows(self, src: Dev, rows: int, cols: int, times: int) -> Dev:
        total = rows * times * cols
        dst = self.tmp(total)
        self._ew(
            "repeat_rows",
            (_dev_bind(src), _dev_bind(dst)),
            (struct.pack("<I", rows), struct.pack("<I", cols), struct.pack("<I", times)),
            total,
        )
        return dst

    def residual_write(
        self, residual: Dev, scales: Dev, mixed: Dev, count: int, branches: int, hidden: int
    ) -> None:
        self._ew(
            "residual_write",
            (_dev_bind(residual), _dev_bind(scales), _dev_bind(mixed)),
            (struct.pack("<I", count), struct.pack("<I", branches), struct.pack("<I", hidden)),
            count * branches * hidden,
        )

    def branch_mean(self, gate: Dev, hats: Dev, count: int, branches: int, hidden: int) -> Dev:
        dst = self.tmp(count * hidden)
        self._ew(
            "branch_mean",
            (_dev_bind(gate), _dev_bind(hats), _dev_bind(dst)),
            (struct.pack("<I", count), struct.pack("<I", branches), struct.pack("<I", hidden)),
            count * hidden,
        )
        return dst

    def gdn_alpha(self, raw: Dev, dt_bias: Any, dt_byte: int, a_log: Any, a_byte: int) -> Dev:
        dst = self.tmp(raw.n)
        self._ew(
            "gdn_alpha",
            (_dev_bind(raw), (dt_bias, dt_byte), (a_log, a_byte), _dev_bind(dst)),
            (struct.pack("<I", raw.n),),
            raw.n,
        )
        return dst

    def ple_gate(self, query: Dev, keyed: Dev, cols: int, rows: int) -> Dev:
        dst = self.tmp(rows)
        self.dispatch(
            self.act["ple_gate"],
            (_dev_bind(query), _dev_bind(keyed), _dev_bind(dst)),
            (struct.pack("<I", cols), struct.pack("<f", 1.0 / math.sqrt(cols))),
            self._size(rows),
            self._size(32),
        )
        return dst

    def gate_broadcast(self, gate: Dev, value: Dev, branches: int, hidden: int) -> Dev:
        dst = self.tmp(branches * hidden)
        self._ew(
            "gate_broadcast",
            (_dev_bind(gate), _dev_bind(value), _dev_bind(dst)),
            (struct.pack("<I", branches), struct.pack("<I", hidden)),
            branches * hidden,
        )
        return dst

    def split_q(self, packed: Dev, heads: int = 24, dim: int = 256) -> tuple[Dev, Dev]:
        query = self.tmp(heads * dim)
        gate = self.tmp(heads * dim)
        self._ew(
            "split_q",
            (_dev_bind(packed), _dev_bind(query), _dev_bind(gate)),
            (struct.pack("<I", heads), struct.pack("<I", dim)),
            heads * dim,
        )
        return query, gate

    def sigmoid_mul(self, left: Dev, gate: Dev) -> Dev:
        dst = self.tmp(left.n)
        self._ew(
            "sigmoid_mul",
            (_dev_bind(left), _dev_bind(gate), _dev_bind(dst)),
            (struct.pack("<I", left.n),),
            left.n,
        )
        return dst

    def topk(self, logits: Dev, n: int, k: int) -> tuple[Dev, Dev]:
        if n > 512:
            message = f"router width {n} exceeds the top-k shader"
            raise ValueError(message)
        index = self.tmp(k)
        weight = self.tmp(k)
        self.dispatch(
            self.act["topk_softmax"],
            (_dev_bind(logits), _dev_bind(index), _dev_bind(weight), (self._topk_probs, 0)),
            (struct.pack("<I", n), struct.pack("<I", k)),
            self._size(1),
            self._size(1),
        )
        return index, weight

    def argmax(self, logits: Dev) -> int:
        grid = 1024
        self.dispatch(
            self.act["argmax_partial"],
            (_dev_bind(logits), (self._argmax_peaks, 0), (self._argmax_index, 0)),
            (struct.pack("<I", logits.n), struct.pack("<I", grid)),
            self._size(grid // 256),
            self._size(256),
        )
        self.dispatch(
            self.act["argmax_final"],
            ((self._argmax_peaks, 0), (self._argmax_index, 0), (self._argmax_out, 0)),
            (struct.pack("<I", grid),),
            self._size(1),
            self._size(1),
        )
        self.flush()
        raw = _buffer_bytes(self._argmax_out, 4)
        return int(np.frombuffer(raw, dtype=np.uint32)[0])

    def sample(self, logits: Dev, temperature: float, top_p: float, top_k: int, rng: Any) -> int:
        if temperature <= 0.0:
            return self.argmax(logits)
        if self._sample_scratch is None or self._sample_scratch.length() < logits.n * 4:
            self._sample_scratch = self.alloc(logits.n * 4)
        uniform = float(rng.random())
        self.dispatch(
            self.act["sample_f32"],
            (_dev_bind(logits), (self._sample_scratch, 0), (self._argmax_out, 0)),
            (
                struct.pack("<I", logits.n),
                struct.pack("<f", float(temperature)),
                struct.pack("<f", float(top_p)),
                struct.pack("<I", int(top_k)),
                struct.pack("<f", uniform),
            ),
            self._size(1),
            self._size(1),
        )
        self.flush()
        raw = _buffer_bytes(self._argmax_out, 4)
        return int(np.frombuffer(raw, dtype=np.uint32)[0])

    def _cast_half(self, src: Dev, logical: int, stored: int) -> Any:
        half = self.alloc(stored * 2)
        self._retain.append(half)
        self._ew(
            "cast_f16_pad",
            (_dev_bind(src), (half, 0)),
            (struct.pack("<I", logical), struct.pack("<I", stored)),
            stored,
        )
        return half

    def gemv_into(self, matrix: Any, src: Dev, dst: Dev, dst_row: int) -> None:
        if src.n != matrix.inner:
            message = f"gemv source {src.n} does not match inner {matrix.inner}"
            raise ValueError(message)
        rows = int(matrix.rows)
        dst_off = (dst.offset + dst_row * rows) * 4
        if isinstance(matrix, F32Matrix):
            self.dispatch(
                self.f32_pipe,
                (_dev_bind(src), (matrix.weights, 0), (dst.buf, dst_off)),
                (struct.pack("<I", rows), struct.pack("<I", matrix.inner)),
                self._size(rows),
                self._size(32),
            )
            return
        if isinstance(matrix, F16Matrix):
            half = self._cast_half(src, matrix.inner, matrix.inner)
            self.dispatch(
                self.f16_pipe,
                ((half, 0), (matrix.weights, matrix.weight_off), (dst.buf, dst_off)),
                (struct.pack("<I", rows), struct.pack("<I", matrix.inner)),
                self._size(rows),
                self._size(32),
            )
            return
        half = self._cast_half(src, matrix.inner, matrix.stored_inner)
        self.dispatch(
            self.gemm.decode,
            (
                (half, 0),
                (matrix.weights, matrix.weight_off),
                (matrix.scales, matrix.scale_off),
                (dst.buf, dst_off),
            ),
            (struct.pack("<I", rows), struct.pack("<I", matrix.stored_inner)),
            self._size(rows),
            self._size(32),
        )

    def linear(self, matrix: Any, src: Dev, count: int, inner: int) -> Dev:
        if src.n != count * inner:
            message = f"linear source {src.n} is not {count} by {inner}"
            raise ValueError(message)
        gemm = (
            count > 1
            and isinstance(matrix, Int4Matrix)
            and matrix.rows % 32 == 0
            and matrix.inner == inner
        )
        if not gemm:
            dst = self.tmp(count * int(matrix.rows))
            for index in range(count):
                self.gemv_into(matrix, src.view(index * inner, inner), dst, index)
            return dst
        return self._gemm_int4(matrix, src, count, inner)

    def _gemm_int4(self, matrix: Int4Matrix, src: Dev, count: int, inner: int) -> Dev:
        padded = ((count + 63) // 64) * 64
        stored = matrix.stored_inner
        cols = matrix.rows
        act = self.alloc(padded * stored * 2)
        self._retain.append(act)
        wide = self.tmp(padded * cols)
        _fill_buffer(wide.buf, padded * cols * 4, 0)
        self._ew(
            "cast_prefill",
            (_dev_bind(src), (act, 0)),
            (
                struct.pack("<I", count),
                struct.pack("<I", inner),
                struct.pack("<I", padded),
                struct.pack("<I", stored),
            ),
            padded * stored,
        )
        self.dispatch(
            self.gemm.prefill,
            (
                (act, 0),
                (matrix.weights, matrix.weight_off),
                (matrix.scales, matrix.scale_off),
                _dev_bind(wide),
            ),
            (
                struct.pack("<I", padded),
                struct.pack("<I", cols),
                struct.pack("<I", stored),
            ),
            self._size(cols // 32, padded // 64, 1),
            self._size(32 * 4, 1, 1),
        )
        packed = self.tmp(count * matrix.rows)
        self._ew(
            "compact_rows",
            (_dev_bind(wide), _dev_bind(packed)),
            (
                struct.pack("<I", count),
                struct.pack("<I", matrix.rows),
                struct.pack("<I", cols),
            ),
            count * matrix.rows,
        )
        return packed

    def gdn_dev(
        self, state: Any, query: Dev, key: Dev, value: Dev, alpha: Dev, beta: Dev
    ) -> Dev:
        dst = self.tmp(48 * 128)
        self.dispatch(
            self.gdn_pipe,
            (
                (state, 0),
                _dev_bind(query),
                _dev_bind(key),
                _dev_bind(value),
                _dev_bind(alpha),
                _dev_bind(beta),
                _dev_bind(dst),
            ),
            (),
            self._size(48),
            self._size(128),
        )
        return dst

    def attend_dev(self, query: Dev, key_buf: Any, value_buf: Any, tokens: int) -> Dev:
        scores = self.alloc(24 * tokens * 4)
        self._retain.append(scores)
        dst = self.tmp(24 * 256)
        self.dispatch(
            self.attend_pipe,
            (
                _dev_bind(query),
                (key_buf, 0),
                (value_buf, 0),
                (scores, 0),
                _dev_bind(dst),
            ),
            (
                struct.pack("<I", tokens),
                struct.pack("<I", 2),
                struct.pack("<I", 256),
                struct.pack("<f", 1.0 / math.sqrt(256.0)),
            ),
            self._size(24),
            self._size(32),
        )
        return dst

    def broadcast(self, src: Dev, dst: Dev, dst_shift: int, rows: int, cols: int) -> None:
        self._ew(
            "broadcast_rows",
            (_dev_bind(src), (dst.buf, (dst.offset + dst_shift) * 4)),
            (struct.pack("<I", cols), struct.pack("<I", rows)),
            cols,
        )

    def embed_gather(
        self,
        image: Any,
        image_off: int,
        tokens: Dev,
        dst: Dev,
        count: int,
        logical: int,
        data_row: int,
        scale_row: int,
        data_base: int,
        branches: int,
    ) -> None:
        self._ew(
            "embed_gather",
            ((image, image_off), _dev_bind(tokens), _dev_bind(dst)),
            (
                struct.pack("<I", count),
                struct.pack("<I", logical),
                struct.pack("<I", data_row),
                struct.pack("<I", scale_row),
                struct.pack("<I", data_base),
                struct.pack("<I", branches),
            ),
            count * logical,
        )

    def ensure_io(self) -> None:
        if self._io is not None:
            return
        desc = self.metal.MTLIOCommandQueueDescriptor.alloc().init()
        queue, error = _objc_pair(self.device.newIOCommandQueueWithDescriptor_error_(desc, None))
        if queue is None:
            message = f"Metal IO queue failed: {error}"
            raise RuntimeError(message)
        self._io = queue
        self._event = self.device.newSharedEvent()

    def io_read(
        self,
        handle: Any,
        loads: Sequence[tuple[Any, int, int, int]],
        *,
        gpu_wait: bool = False,
    ) -> None:
        """Load file ranges straight into buffers. ``loads`` is dest, dest byte, size, source."""
        self.ensure_io()
        started = time.perf_counter()
        command = self._io.commandBuffer()
        issued = 0
        for buffer, dest_off, size, src_off in loads:
            # One load above 256MB returns MTLIOStatusError on this GPU.
            offset = 0
            while offset < size:
                take = min(IO_CHUNK, size - offset)
                command.loadBuffer_offset_size_sourceHandle_sourceHandleOffset_(
                    buffer, dest_off + offset, take, handle, src_off + offset
                )
                offset += take
                issued += 1
        self._event_value += 1
        command.signalEvent_value_(self._event, self._event_value)
        command.commit()
        self.io_loads += issued
        if gpu_wait:
            self._io_pending.append(command)
            self._compute_waits_for(self._event_value)
            self.io_seconds += time.perf_counter() - started
            return
        command.waitUntilCompleted()
        self.io_seconds += time.perf_counter() - started
        # MTLIOStatusComplete is 3. It is not MTLCommandBufferStatusCompleted.
        if int(command.status()) != 3:
            message = f"Metal IO status {command.status()} {command.error()}"
            raise RuntimeError(message)

    def _compute_waits_for(self, value: int) -> None:
        if self._encoder is not None:
            self._encoder.endEncoding()
            self._encoder = None
        if self._command is None:
            self._command = self.queue.commandBuffer()
            self._batch_launches = 0
        self._command.encodeWaitForEvent_value_(self._event, value)

    def ngram_dequant(self, handle: Any, rows: Sequence[int]) -> Dev:
        raw = self.alloc(len(rows) * NGRAM_SLOT)
        self._retain.append(raw)
        loads = [
            (raw, index * NGRAM_SLOT, NGRAM_SLOT, int(row) * NGRAM_SLOT)
            for index, row in enumerate(rows)
        ]
        self.io_read(handle, loads, gpu_wait=True)
        dst = self.tmp(len(rows) * NGRAM_WIDTH)
        self._ew(
            "dequant_ngram",
            ((raw, 0), _dev_bind(dst)),
            (struct.pack("<I", NGRAM_WIDTH), struct.pack("<I", NGRAM_SLOT)),
            len(rows) * NGRAM_WIDTH,
        )
        return dst
