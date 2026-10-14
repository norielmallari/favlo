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

from qwen38_flash_next.formats import (
    BLOCK,
    IO_CHUNK,
    NGRAM_SLOT,
    NGRAM_WIDTH,
    PAGE,
    int4_plane_sizes,
    kv_cache_name,
    kv_store_groups,
)
from qwen38_flash_next.shader_text import shader_text

VECTOR_SHADER = shader_text("vector.metal")
ACT_SHADER = shader_text("act.metal", "scan.metal", "moe.metal")
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


def _aligned_offset(used: int, align: int) -> int:
    return (used + align - 1) & ~(align - 1)


def _require_page(offset: int, size: int) -> None:
    """Uncached SSD reads stay off the bounce buffer only when both ends match the VM page."""
    if offset % PAGE or size % PAGE:
        message = f"Metal IO range {offset}+{size} is not a multiple of {PAGE}"
        raise RuntimeError(message)


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


def _bind_any(src: Any) -> tuple[Any, int]:
    if isinstance(src, Dev):
        return _dev_bind(src)
    if isinstance(src, Span):
        return src.buf, src.offset
    return src, 0


class Span:
    """A byte range in a buffer that outlives one command buffer."""

    def __init__(self, buf: Any, offset: int) -> None:
        self.buf = buf
        self.offset = offset


def _zero_dev(src: Dev) -> None:
    start = src.offset * 4
    nbytes = src.n * 4
    _buffer_view(src.buf, start + nbytes)[start : start + nbytes] = bytes(nbytes)


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
        self.host_encode = 0.0
        self.wait_seconds = 0.0
        self._arena = None
        self._arenas: list[Any] = []
        self._arena_i = 0
        self._arena_used = 0
        self._inflight: list[tuple[Any, list[Any], list[bytes]]] = []
        self._spare: list[Any] = []
        self._io_chain = 0
        self.io_seconds = 0.0
        self.io_loads = 0
        self._encoder = None
        self._command = None
        self._batch_launches = 0
        self._command_wait = 0
        self._retain: list[Any] = []
        self._held_constants: list[bytes] = []
        self._io = None
        self._event = None
        self._event_value = 0
        self._io_pending: list[tuple[Any, int]] = []
        self._tier_pad = self.alloc(16)
        self._cache_mode = self.alloc(16)
        self._cache_mode_held = -1
        self._presence_ids: Any = None
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
            "axpy_matching",
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
            "dequant_ngram_packed",
            "broadcast_rows",
            "sample_f32",
            "suppress_logit",
            "presence_logits",
            "sample_topk_partial",
            "sample_topk_draw",
            "copy_bytes",
            "gemv_f32_batch",
            "gemv_f16_batch",
            "decode_int4_batch",
            "topk_batch",
            "gdn_conv_scan",
            "gdn_state_scan",
            "split_q_batch",
            "rope_batch",
            "kv_store",
            "indexer_write",
            "indexer_block_score",
            "indexer_select",
            "qsa_attend",
            "qsa_attend_selected",
            "gather_f32",
            "gather_half",
            "moe_grouped",
            "moe_activate",
            "moe_scatter",
            "moe_add_weighted",
            "moe_decode_gate",
            "moe_decode_down",
            "moe_grouped_int2",
            "moe_decode_gate_int2",
            "moe_decode_down_int2",
            "gemv_int4_f32",
            "gemv_int8_f32",
            "gemv_int8_half",
        )
        self.act = {name: self._pipeline(library, name) for name in act_names}
        self._kv_pad = self.alloc(64)
        self._argmax_peaks = self.alloc(1024 * 4)
        self._argmax_index = self.alloc(1024 * 4)
        self._argmax_out = self.alloc(4)
        self._sample_top_s = self.alloc(1024 * 32 * 4)
        self._sample_top_i = self.alloc(1024 * 32 * 4)
        self._topk_probs = self.alloc(512 * 4)
        self._sample_scratch = None
        self._ngram_pages: dict[int, Any] = {}
        self._ngram_order: list[int] = []
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

    def begin_scratch(self, nbytes: int, *, copies: int = 1) -> None:
        copies = 2 if copies > 1 else 1
        ready = len(self._arenas) == copies and all(
            buf is not None and int(buf.length()) >= nbytes for buf in self._arenas
        )
        if not ready:
            # Drop the previous arenas before the new allocation. A long prefill
            # scratch kept alive across this call does not fit beside the atlas.
            self._arena = None
            self._arenas = []
            arenas = [self.alloc(nbytes) for _ in range(copies)]
            if any(buf is None for buf in arenas):
                message = f"scratch arena of {nbytes} bytes was not allocated"
                raise RuntimeError(message)
            self._arenas = arenas
        self._arena_i = 0
        self._arena = self._arenas[0]
        self._arena_used = 0

    def rewind_scratch(self) -> None:
        self._arena_used = 0

    def _bump(self, nbytes: int, *, align: int = 256) -> int:
        if self._arena is None:
            message = "scratch arena was not allocated"
            raise RuntimeError(message)
        off = _aligned_offset(self._arena_used, align)
        need = off + max(nbytes, 1)
        if need > int(self._arena.length()):
            message = f"scratch arena is {int(self._arena.length())} bytes, need {need}"
            raise RuntimeError(message)
        self._arena_used = need
        return off

    def span(self, nbytes: int) -> Span:
        off = self._bump(nbytes)
        return Span(self._arena, off)

    def span_page(self, nbytes: int) -> Span:
        """A scratch span whose offset is a multiple of the VM page.

        Uncached Metal IO rejects any other destination offset.
        """
        off = self._bump(nbytes, align=PAGE)
        return Span(self._arena, off)

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
        late: Sequence[tuple[Any, int, int]] = (),
    ) -> None:
        started = time.perf_counter()
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
        for buffer, offset, index in late:
            encoder.setBuffer_offset_atIndex_(buffer, int(offset), index)
        encoder.dispatchThreadgroups_threadsPerThreadgroup_(groups, threads)
        self._batch_launches += 1
        self.host_encode += time.perf_counter() - started

    def _enqueue_open(self) -> None:
        if self._encoder is not None:
            self._encoder.endEncoding()
            self._encoder = None
        if self._command is None:
            return
        command = self._command
        command.commit()
        self._inflight.append((command, self._retain, self._held_constants, self._command_wait))
        self._retain = []
        self._held_constants = []
        self._command_wait = 0
        self.launch_count += self._batch_launches
        self.command_count += 1
        self._batch_launches = 0
        self._command = None

    def _flip_scratch(self) -> None:
        if len(self._arenas) < 2:
            return
        self._arena_i ^= 1
        self._arena = self._arenas[self._arena_i]
        self._arena_used = 0

    def submit(self) -> None:
        """Commit the open buffer and keep encoding into the other scratch arena."""
        self._enqueue_open()
        self._flip_scratch()

    def barrier(self) -> None:
        """Order buffer reads and writes for the dispatches encoded after this."""
        if self._encoder is None:
            return
        self._encoder.memoryBarrierWithScope_(self.metal.MTLBarrierScopeBuffers)

    def flush(self) -> None:
        self._enqueue_open()
        if not self._inflight:
            self._retain.clear()
            self._held_constants.clear()
            return
        started = time.perf_counter()
        pending = self._io_pending
        self._io_pending = []
        inflight = self._inflight
        self._inflight = []
        ready = 0
        for command, retained, _constants, waited_for in inflight:
            waited = time.perf_counter()
            command.waitUntilCompleted()
            self.wait_seconds += time.perf_counter() - waited
            gpu_start = float(command.GPUStartTime())
            gpu_end = float(command.GPUEndTime())
            if gpu_end > gpu_start:
                self.gpu_seconds += gpu_end - gpu_start
            self._recycle(retained)
            if waited_for > ready:
                ready = waited_for
            if int(command.status()) != 4:
                message = f"Metal command status {command.status()} {command.error()}"
                raise RuntimeError(message)
        self.encode_seconds += time.perf_counter() - started
        self._retain.clear()
        self._held_constants.clear()
        for io_command, value in pending:
            # Loads submitted for a later layer are still in flight. The host
            # reads expert ids before that layer's fence, so Pending is expected.
            if value > ready:
                self._io_pending.append((io_command, value))
                continue
            status = int(io_command.status())
            if status == 0:
                io_command.waitUntilCompleted()
                status = int(io_command.status())
            if status != 3:
                message = f"Metal IO status {status} {io_command.error()}"
                raise RuntimeError(message)

    def tmp(self, n: int) -> Dev:
        off = self._bump(n * 4)
        return Dev(self._arena, n, off // 4)

    def dev_from(self, array: np.ndarray) -> Dev:
        flat = np.ascontiguousarray(array, dtype=np.float32).reshape(-1)
        buf = self.upload(flat.tobytes())
        self._retain.append(buf)
        return Dev(buf, int(flat.size))

    def dev_u32(self, values: Sequence[int]) -> Dev:
        raw = np.asarray(list(values), dtype=np.uint32)
        blob = raw.tobytes()
        buf = self._checkout(len(blob))
        if blob:
            _buffer_view(buf, len(blob))[:] = blob
        self._retain.append(buf)
        return Dev(buf, int(raw.size))

    def _checkout(self, nbytes: int) -> Any:
        for index, buf in enumerate(self._spare):
            if int(buf.length()) >= nbytes:
                return self._spare.pop(index)
        return self.alloc(max(nbytes, 64))

    def _recycle(self, buffers: list[Any]) -> None:
        for buf in buffers:
            if len(self._spare) >= 32 or int(buf.length()) > 4096:
                continue
            self._spare.append(buf)

    def read_f32(self, src: Dev) -> np.ndarray:
        self.flush()
        raw = _buffer_slice(src.buf, src.offset * 4, src.n * 4)
        return np.frombuffer(raw, dtype=np.float32).copy()

    def read_u32(self, src: Dev) -> np.ndarray:
        self.flush()
        raw = _buffer_slice(src.buf, src.offset * 4, src.n * 4)
        return np.frombuffer(raw, dtype=np.uint32).copy()

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

    def axpy_matching(self, dst: Dev, src: Dev, scales: Dev, ids: Dev, expert: int) -> None:
        """Add ``src`` into ``dst`` for every slot whose id is ``expert``."""
        self._ew(
            "axpy_matching",
            (_dev_bind(dst), _dev_bind(src), _dev_bind(scales), _dev_bind(ids)),
            (
                struct.pack("<I", dst.n),
                struct.pack("<I", ids.n),
                struct.pack("<I", expert),
            ),
            dst.n,
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
        keep = int(top_k)
        uniform = float(rng.random())
        if keep <= 0 or keep > 32:
            if self._sample_scratch is None or self._sample_scratch.length() < logits.n * 4:
                self._sample_scratch = self.alloc(logits.n * 4)
            self.dispatch(
                self.act["sample_f32"],
                (_dev_bind(logits), (self._sample_scratch, 0), (self._argmax_out, 0)),
                (
                    struct.pack("<I", logits.n),
                    struct.pack("<f", float(temperature)),
                    struct.pack("<f", float(top_p)),
                    struct.pack("<I", keep),
                    struct.pack("<f", uniform),
                ),
                self._size(1),
                self._size(1),
            )
            self.flush()
            raw = _buffer_bytes(self._argmax_out, 4)
            return int(np.frombuffer(raw, dtype=np.uint32)[0])
        grid = 1024
        self.dispatch(
            self.act["sample_topk_partial"],
            (
                _dev_bind(logits),
                (self._sample_top_s, 0),
                (self._sample_top_i, 0),
            ),
            (
                struct.pack("<I", logits.n),
                struct.pack("<I", keep),
                struct.pack("<f", float(temperature)),
                struct.pack("<I", grid),
            ),
            self._size(grid // 256),
            self._size(256),
        )
        self.dispatch(
            self.act["sample_topk_draw"],
            (
                (self._sample_top_s, 0),
                (self._sample_top_i, 0),
                (self._argmax_out, 0),
            ),
            (
                struct.pack("<I", grid),
                struct.pack("<I", keep),
                struct.pack("<f", float(top_p)),
                struct.pack("<f", uniform),
            ),
            self._size(1),
            self._size(1),
        )
        self.flush()
        raw = _buffer_bytes(self._argmax_out, 4)
        return int(np.frombuffer(raw, dtype=np.uint32)[0])

    def suppress_logit(self, logits: Dev, index: int, value: float = -1.0e30) -> None:
        self._ew(
            "suppress_logit",
            (_dev_bind(logits),),
            (struct.pack("<I", int(index)), struct.pack("<f", float(value))),
            1,
            1,
        )

    def presence(self, logits: Dev, ids: Sequence[int], penalty: float) -> None:
        seen = [int(token) for token in ids if 0 <= int(token) < logits.n]
        if not seen or penalty == 0.0:
            return
        blob = np.asarray(seen, dtype=np.uint32).tobytes()
        if self._presence_ids is None or int(self._presence_ids.length()) < len(blob):
            self._presence_ids = self.alloc(max(len(blob), 1024))
        _buffer_view(self._presence_ids, len(blob))[:] = blob
        self._ew(
            "presence_logits",
            (_dev_bind(logits), (self._presence_ids, 0)),
            (struct.pack("<I", len(seen)), struct.pack("<f", float(penalty))),
            1,
        )

    def _cast_half(self, src: Dev, logical: int, stored: int) -> Span:
        half = self.span(stored * 2)
        self._ew(
            "cast_f16_pad",
            (_dev_bind(src), (half.buf, half.offset)),
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
                ((half.buf, half.offset), (matrix.weights, matrix.weight_off), (dst.buf, dst_off)),
                (struct.pack("<I", rows), struct.pack("<I", matrix.inner)),
                self._size(rows),
                self._size(32),
            )
            return
        half = self._cast_half(src, matrix.inner, matrix.stored_inner)
        self.dispatch(
            self.gemm.decode,
            (
                (half.buf, half.offset),
                (matrix.weights, matrix.weight_off),
                (matrix.scales, matrix.scale_off),
                (dst.buf, dst_off),
            ),
            (struct.pack("<I", rows), struct.pack("<I", matrix.stored_inner)),
            self._size(rows),
            self._size(32),
        )

    def linear(
        self,
        matrix: Any,
        src: Dev,
        count: int,
        inner: int,
        *,
        epilogue: int = 0,
        epilogue_scale: float = 1.0,
    ) -> Dev:
        if src.n != count * inner:
            message = f"linear source {src.n} is not {count} by {inner}"
            raise ValueError(message)
        if count == 1 and isinstance(matrix, Int4Matrix):
            return self._gemv_int4(matrix, src, epilogue, epilogue_scale)
        gemm = (
            count > 1
            and isinstance(matrix, Int4Matrix)
            and matrix.rows % 32 == 0
            and matrix.inner == inner
        )
        if gemm:
            dst = self._gemm_int4(matrix, src, count, inner)
        elif count == 1:
            dst = self.tmp(int(matrix.rows))
            self.gemv_into(matrix, src, dst, 0)
            return self._apply_epilogue(dst, epilogue, epilogue_scale)
        else:
            dst = self._gemv_batch(matrix, src, count, inner)
        return self._apply_epilogue(dst, epilogue, epilogue_scale)

    def _apply_epilogue(self, src: Dev, epilogue: int, scale: float) -> Dev:
        if epilogue == 0:
            return src
        if epilogue == 1:
            return self.silu(src)
        if epilogue == 2:
            return self.sigmoid(src)
        if epilogue == 3:
            return self.silu(self.scale(src, scale))
        if epilogue == 4:
            return self.scale(self.sigmoid(self.scale(src, 0.25)), 2.0)
        message = f"unknown linear epilogue {epilogue}"
        raise ValueError(message)

    def _gemv_int4(self, matrix: Int4Matrix, src: Dev, epilogue: int, scale: float) -> Dev:
        rows = int(matrix.rows)
        dst = self.tmp(rows)
        self.dispatch(
            self.act["gemv_int4_f32"],
            (
                _dev_bind(src),
                (matrix.weights, matrix.weight_off),
                (matrix.scales, matrix.scale_off),
                _dev_bind(dst),
            ),
            (
                struct.pack("<I", rows),
                struct.pack("<I", int(matrix.stored_inner)),
                struct.pack("<I", int(matrix.inner)),
                struct.pack("<I", int(epilogue)),
                struct.pack("<f", float(scale)),
            ),
            self._size((rows + 15) // 16),
            self._size(128),
        )
        return dst

    def _gemv_batch(self, matrix: Any, src: Dev, count: int, inner: int) -> Dev:
        rows = int(matrix.rows)
        dst = self.tmp(count * rows)
        if isinstance(matrix, F32Matrix):
            self.dispatch(
                self.act["gemv_f32_batch"],
                (_dev_bind(src), (matrix.weights, 0), _dev_bind(dst)),
                (
                    struct.pack("<I", count),
                    struct.pack("<I", rows),
                    struct.pack("<I", inner),
                ),
                self._size(count * rows),
                self._size(32),
            )
            return dst
        if isinstance(matrix, F16Matrix):
            stored = inner
            weights = matrix.weights
            weight_off = matrix.weight_off
            pipe = self.act["gemv_f16_batch"]
            scales = None
            scale_off = 0
        else:
            stored = int(matrix.stored_inner)
            weights = matrix.weights
            weight_off = matrix.weight_off
            pipe = self.act["decode_int4_batch"]
            scales = matrix.scales
            scale_off = matrix.scale_off
        half = self.span(count * stored * 2)
        self._ew(
            "cast_prefill",
            (_dev_bind(src), (half.buf, half.offset)),
            (
                struct.pack("<I", count),
                struct.pack("<I", inner),
                struct.pack("<I", count),
                struct.pack("<I", stored),
            ),
            count * stored,
        )
        bindings = [(half.buf, half.offset), (weights, weight_off)]
        if scales is not None:
            bindings.append((scales, scale_off))
        bindings.append(_dev_bind(dst))
        self.dispatch(
            pipe,
            bindings,
            (
                struct.pack("<I", count),
                struct.pack("<I", rows),
                struct.pack("<I", stored),
            ),
            self._size(count * rows),
            self._size(32),
        )
        return dst

    def _gemm_int4(self, matrix: Int4Matrix, src: Dev, count: int, inner: int) -> Dev:
        padded = ((count + 63) // 64) * 64
        stored = matrix.stored_inner
        cols = matrix.rows
        act = self.span(padded * stored * 2)
        wide = self.tmp(padded * cols)
        _zero_dev(wide)
        self._ew(
            "cast_prefill",
            (_dev_bind(src), (act.buf, act.offset)),
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
                (act.buf, act.offset),
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

    def gemm_int4_half(self, matrix: Int4Matrix, act: Span, count: int) -> Dev:
        """int4 matmul whose activation is already half, one row per token."""
        if count < 1 or matrix.rows % 32 != 0:
            message = f"matmul count {count} rows {matrix.rows} cannot use the prefill tile"
            raise RuntimeError(message)
        padded = ((count + 63) // 64) * 64
        stored = matrix.stored_inner
        if padded != count:
            wide_act = self.span(padded * stored * 2)
            self.copy_bytes(
                act.buf, act.offset, wide_act.buf, wide_act.offset, count * stored * 2
            )
            act = wide_act
        cols = matrix.rows
        wide = self.tmp(padded * cols)
        self.dispatch(
            self.gemm.prefill,
            (
                (act.buf, act.offset),
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

    def gemv_int8(
        self,
        activation: Any,
        image: Any,
        cols: int,
        stored: int,
        weight_off: int,
        scale_off: int,
        logical: int,
        *,
        half: bool,
    ) -> Dev:
        dst = self.tmp(cols)
        kernel = "gemv_int8_half" if half else "gemv_int8_f32"
        self.dispatch(
            self.act[kernel],
            (
                _bind_any(activation),
                (image, weight_off),
                (image, scale_off),
                _dev_bind(dst),
            ),
            (
                struct.pack("<I", cols),
                struct.pack("<I", stored),
                struct.pack("<I", logical),
            ),
            self._size(cols),
            self._size(32),
        )
        return dst

    def upload_resident(self, blob: bytes) -> Any:
        """Copy one super-expert image into a buffer that stays resident."""
        staging = _shared_buffer(self.device, blob)
        private = self.device.newBufferWithLength_options_(
            len(blob), self.metal.MTLResourceStorageModePrivate
        )
        if private is None:
            return staging
        command = self.queue.commandBuffer()
        blit = command.blitCommandEncoder()
        blit.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size_(
            staging, 0, private, 0, len(blob)
        )
        blit.endEncoding()
        command.commit()
        command.waitUntilCompleted()
        return private

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
        # MTLIOPriorityHigh. A decode token queues one load per layer before the
        # mixer, so the queue has to accept that many buffers without blocking.
        desc.setPriority_(0)
        desc.setMaxCommandBufferCount_(128)
        queue, error = _objc_pair(self.device.newIOCommandQueueWithDescriptor_error_(desc, None))
        if queue is None:
            message = f"Metal IO queue failed: {error}"
            raise RuntimeError(message)
        self._io = queue
        self._event = self.device.newSharedEvent()

    def _open_io(self) -> Any:
        self.ensure_io()
        command = self._io.commandBuffer()
        if command is None:
            message = "Metal IO command buffer was not vended"
            raise RuntimeError(message)
        # The queue is concurrent. Each buffer waits for the previous signal
        # so a higher event value means the earlier loads have finished.
        if self._io_chain:
            command.waitForEvent_value_(self._event, self._io_chain)
        return command

    def _commit_io(self, command: Any) -> int:
        self._event_value += 1
        value = self._event_value
        command.signalEvent_value_(self._event, value)
        command.commit()
        self._io_chain = value
        return value

    def io_submit(self, handle: Any, loads: Sequence[tuple[Any, int, int, int]]) -> int:
        """Start loads and return the shared-event value compute should wait for."""
        if not loads:
            return 0
        started = time.perf_counter()
        command = self._open_io()
        issued = 0
        for buffer, dest_off, size, src_off in loads:
            _require_page(dest_off, size)
            _require_page(src_off, size)
            offset = 0
            while offset < size:
                take = min(IO_CHUNK, size - offset)
                command.loadBuffer_offset_size_sourceHandle_sourceHandleOffset_(
                    buffer, dest_off + offset, take, handle, src_off + offset
                )
                offset += take
                issued += 1
        value = self._commit_io(command)
        self.io_loads += issued
        self._io_pending.append((command, value))
        self.io_seconds += time.perf_counter() - started
        return int(value)

    def io_submit_mixed(
        self, groups: Sequence[tuple[Any, Sequence[tuple[Any, int, int, int]]]]
    ) -> int:
        """One command buffer for every tier, so the fence covers both files."""
        pending = [(handle, loads) for handle, loads in groups if loads]
        if not pending:
            return 0
        started = time.perf_counter()
        command = self._open_io()
        issued = 0
        for handle, loads in pending:
            for buffer, dest_off, size, src_off in loads:
                _require_page(dest_off, size)
                _require_page(src_off, size)
                offset = 0
                while offset < size:
                    take = min(IO_CHUNK, size - offset)
                    command.loadBuffer_offset_size_sourceHandle_sourceHandleOffset_(
                        buffer, dest_off + offset, take, handle, src_off + offset
                    )
                    offset += take
                    issued += 1
        value = self._commit_io(command)
        self.io_loads += issued
        self._io_pending.append((command, value))
        self.io_seconds += time.perf_counter() - started
        return int(value)

    def io_fence(self, value: int) -> None:
        if value:
            self._compute_waits_for(value)

    def io_read(
        self,
        handle: Any,
        loads: Sequence[tuple[Any, int, int, int]],
        *,
        gpu_wait: bool = False,
    ) -> None:
        """Load file ranges straight into buffers. ``loads`` is dest, dest byte, size, source."""
        started = time.perf_counter()
        command = self._open_io()
        issued = 0
        for buffer, dest_off, size, src_off in loads:
            # One load above 256MB returns MTLIOStatusError on this GPU.
            _require_page(dest_off, size)
            _require_page(src_off, size)
            offset = 0
            while offset < size:
                take = min(IO_CHUNK, size - offset)
                command.loadBuffer_offset_size_sourceHandle_sourceHandleOffset_(
                    buffer, dest_off + offset, take, handle, src_off + offset
                )
                offset += take
                issued += 1
        value = self._commit_io(command)
        self.io_loads += issued
        if gpu_wait:
            self._io_pending.append((command, value))
            self._compute_waits_for(value)
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
        if value > self._command_wait:
            self._command_wait = value

    def ngram_dequant(self, handle: Any, rows: Sequence[int]) -> Dev:
        if len(rows) > 16:
            return self._ngram_dequant_packed(handle, rows)
        raw = self.span(len(rows) * NGRAM_SLOT)
        loads: list[tuple[Any, int, int, int]] = []
        copies: list[tuple[Any, int, int]] = []
        needed: set[int] = set()
        for index, row in enumerate(rows):
            byte = int(row) * NGRAM_SLOT
            page = byte // PAGE
            local = byte - page * PAGE
            buf = self._ngram_pages.get(page)
            if buf is None:
                buf = self.alloc(PAGE)
                self._ngram_pages[page] = buf
                self._ngram_order.append(page)
                loads.append((buf, 0, PAGE, page * PAGE))
                needed.add(page)
            copies.append((buf, local, index * NGRAM_SLOT))
        while len(self._ngram_order) > 4096:
            old = self._ngram_order.pop(0)
            if old in needed:
                self._ngram_order.append(old)
                break
            self._ngram_pages.pop(old, None)
        if loads:
            self.io_read(handle, loads, gpu_wait=True)
        for buf, local, dest in copies:
            self.copy_bytes(buf, local, raw.buf, raw.offset + dest, NGRAM_SLOT)
        dst = self.tmp(len(rows) * NGRAM_WIDTH)
        self._ew(
            "dequant_ngram",
            ((raw.buf, raw.offset), _dev_bind(dst)),
            (struct.pack("<I", NGRAM_WIDTH), struct.pack("<I", NGRAM_SLOT)),
            len(rows) * NGRAM_WIDTH,
        )
        return dst

    def _ngram_dequant_packed(self, handle: Any, rows: Sequence[int]) -> Dev:
        """Load the pages a chunk needs and expand every slot in one shader."""
        dst = self.tmp(len(rows) * NGRAM_WIDTH)
        step = 4096
        for start in range(0, len(rows), step):
            part = rows[start : start + step]
            slots: dict[int, int] = {}
            order: list[int] = []
            offs: list[int] = []
            for row in part:
                byte = int(row) * NGRAM_SLOT
                page = byte // PAGE
                local = byte - page * PAGE
                slot = slots.get(page)
                if slot is None:
                    slot = len(order)
                    slots[page] = slot
                    order.append(page)
                offs.append(slot * PAGE + local)
            pack = self.span_page(max(len(order), 1) * PAGE)
            loads = [
                (pack.buf, pack.offset + slot * PAGE, PAGE, page * PAGE)
                for page, slot in slots.items()
            ]
            if loads:
                self.io_read(handle, loads, gpu_wait=True)
            raw = np.ascontiguousarray(np.asarray(offs, dtype=np.uint32)).tobytes()
            index = self.span(len(raw))
            _buffer_view(index.buf, index.offset + len(raw))[
                index.offset : index.offset + len(raw)
            ] = raw
            part_dst = self.tmp(len(part) * NGRAM_WIDTH)
            self._ew(
                "dequant_ngram_packed",
                ((pack.buf, pack.offset), (index.buf, index.offset), _dev_bind(part_dst)),
                (struct.pack("<I", NGRAM_WIDTH), struct.pack("<I", len(part))),
                len(part) * NGRAM_WIDTH,
            )
            self.copy_bytes(
                part_dst.buf,
                part_dst.offset * 4,
                dst.buf,
                dst.offset * 4 + start * NGRAM_WIDTH * 4,
                part_dst.n * 4,
            )
        return dst

    def copy_bytes(self, src: Any, src_off: int, dst: Any, dst_off: int, n: int) -> None:
        self._ew(
            "copy_bytes",
            ((src, src_off), (dst, dst_off)),
            (struct.pack("<I", n),),
            n,
        )

    def topk_batch(self, logits: Dev, count: int, n: int, k: int) -> tuple[Dev, Dev]:
        index = self.tmp(count * k)
        weight = self.tmp(count * k)
        scratch = self.span(count * n * 4)
        self.dispatch(
            self.act["topk_batch"],
            (_dev_bind(logits), _dev_bind(index), _dev_bind(weight), (scratch.buf, scratch.offset)),
            (struct.pack("<I", count), struct.pack("<I", n), struct.pack("<I", k)),
            self._size(count),
            self._size(1),
        )
        return index, weight

    def gdn_conv_scan(
        self,
        history: Any,
        qkv: Dev,
        weight: Any,
        weight_byte: int,
        mixed: Dev,
        count: int,
    ) -> None:
        self._ew(
            "gdn_conv_scan",
            ((history, 0), _dev_bind(qkv), (weight, weight_byte), _dev_bind(mixed)),
            (struct.pack("<I", count), struct.pack("<I", 10240), struct.pack("<I", 4)),
            10240,
        )

    def gdn_state_scan(
        self,
        state: Any,
        mixed: Dev,
        raw_a: Dev,
        raw_b: Dev,
        z: Dev,
        dt_bias: Any,
        dt_byte: int,
        a_log: Any,
        a_byte: int,
        norm: Any,
        norm_byte: int,
        heads: Dev,
        count: int,
        eps: float,
    ) -> None:
        self.dispatch(
            self.act["gdn_state_scan"],
            (
                (state, 0),
                _dev_bind(mixed),
                _dev_bind(raw_a),
                _dev_bind(raw_b),
                _dev_bind(z),
                (dt_bias, dt_byte),
                (a_log, a_byte),
                (norm, norm_byte),
                _dev_bind(heads),
            ),
            (struct.pack("<I", count), struct.pack("<f", float(eps))),
            self._size(48),
            self._size(128),
        )

    def split_q_batch(
        self, packed: Dev, count: int, heads: int = 24, dim: int = 256
    ) -> tuple[Dev, Dev]:
        query = self.tmp(count * heads * dim)
        gate = self.tmp(count * heads * dim)
        self._ew(
            "split_q_batch",
            (_dev_bind(packed), _dev_bind(query), _dev_bind(gate)),
            (struct.pack("<I", count), struct.pack("<I", heads), struct.pack("<I", dim)),
            count * heads * dim,
        )
        return query, gate

    def rope_batch(
        self,
        src: Dev,
        count: int,
        heads: int,
        cols: int,
        rotary: int,
        position: int,
        theta: float,
    ) -> Dev:
        dst = self.tmp(count * heads * cols)
        self._ew(
            "rope_batch",
            (_dev_bind(src), _dev_bind(dst)),
            (
                struct.pack("<I", cols),
                struct.pack("<I", rotary),
                struct.pack("<I", position),
                struct.pack("<f", float(theta)),
                struct.pack("<I", heads),
                struct.pack("<I", count),
            ),
            count * heads * cols,
        )
        return dst

    def kv_store(
        self,
        key: Dev,
        value: Dev,
        key_body: Any,
        value_body: Any,
        key_tail: Any,
        value_tail: Any,
        count: int,
        base: int,
        cache: str,
        phase: int,
    ) -> None:
        groups = kv_store_groups(count, base, cache, phase)
        if groups <= 0:
            return
        mode = 0 if kv_cache_name(cache) == "bf16" else 1
        self.dispatch(
            self.act["kv_store"],
            (
                _dev_bind(key),
                _dev_bind(value),
                (key_body, 0),
                (value_body, 0),
                (key_tail, 0),
                (value_tail, 0),
            ),
            (
                struct.pack("<I", count),
                struct.pack("<I", base),
                struct.pack("<I", mode),
                struct.pack("<I", phase),
            ),
            self._size(groups),
            self._size(32),
        )

    def indexer_write(self, index_qk: Dev, cache: Any, count: int, base: int) -> None:
        self._ew(
            "indexer_write",
            (_dev_bind(index_qk), (cache, 0)),
            (struct.pack("<I", count), struct.pack("<I", base)),
            count * 128,
        )

    def qsa_attend(
        self,
        query: Dev,
        gate: Dev,
        key: Any,
        value: Any,
        count: int,
        base: int,
        *,
        mode: int,
        complete: int,
        key_tail: Any = None,
        value_tail: Any = None,
    ) -> Dev:
        dst = self.tmp(count * 24 * 256)
        self.dispatch(
            self.act["qsa_attend"],
            (
                _dev_bind(query),
                _dev_bind(gate),
                (key, 0),
                (value, 0),
                (self._kv_pad if key_tail is None else key_tail, 0),
                (self._kv_pad if value_tail is None else value_tail, 0),
                _dev_bind(dst),
            ),
            (
                struct.pack("<I", count),
                struct.pack("<I", base),
                struct.pack("<f", 1.0 / math.sqrt(256.0)),
                struct.pack("<I", 16),
                struct.pack("<I", mode),
                struct.pack("<I", complete),
            ),
            self._size(24, max(count, 1)),
            self._size(512),
        )
        return dst

    def indexer_block_score(
        self, index_qk: Dev, cache: Any, scores: Dev, count: int, base: int, n_blocks: int
    ) -> None:
        self.dispatch(
            self.act["indexer_block_score"],
            (_dev_bind(index_qk), (cache, 0), _dev_bind(scores)),
            (
                struct.pack("<I", count),
                struct.pack("<I", base),
                struct.pack("<I", n_blocks),
            ),
            self._size(n_blocks),
            self._size(128),
        )

    def indexer_select(
        self,
        scores: Dev,
        indices: Dev,
        counts: Dev,
        count: int,
        base: int,
        n_blocks: int,
        stride: int,
    ) -> None:
        self.dispatch(
            self.act["indexer_select"],
            (_dev_bind(scores), _dev_bind(indices), _dev_bind(counts)),
            (
                struct.pack("<I", count),
                struct.pack("<I", base),
                struct.pack("<I", n_blocks),
                struct.pack("<I", stride),
            ),
            self._size(max(count, 1)),
            self._size(256),
        )

    def qsa_attend_selected(
        self,
        query: Dev,
        gate: Dev,
        key: Any,
        value: Any,
        indices: Dev,
        counts: Dev,
        count: int,
        stride: int,
        *,
        mode: int,
        complete: int,
        key_tail: Any = None,
        value_tail: Any = None,
    ) -> Dev:
        dst = self.tmp(count * 24 * 256)
        if self._cache_mode_held != mode:
            _write_u32(self._cache_mode, mode)
            self._cache_mode_held = mode
        self.dispatch(
            self.act["qsa_attend_selected"],
            (
                _dev_bind(query),
                _dev_bind(gate),
                (key, 0),
                (value, 0),
                (self._kv_pad if key_tail is None else key_tail, 0),
                (self._kv_pad if value_tail is None else value_tail, 0),
                _dev_bind(indices),
                _dev_bind(counts),
                _dev_bind(dst),
            ),
            (
                struct.pack("<I", count),
                struct.pack("<I", stride),
                struct.pack("<f", 1.0 / math.sqrt(256.0)),
                struct.pack("<I", 16),
                struct.pack("<I", mode),
                struct.pack("<I", complete),
            ),
            self._size(24, max(count, 1)),
            self._size(512),
            ((self._cache_mode, 0, 15),),
        )
        return dst

    def gather_half(self, src: Dev, tokens: Any, n: int, logical: int, stored: int) -> Span:
        dst = self.span(max(n, 1) * stored * 2)
        self._ew(
            "gather_half",
            (_dev_bind(src), _bind_any(tokens), (dst.buf, dst.offset)),
            (
                struct.pack("<I", n),
                struct.pack("<I", logical),
                struct.pack("<I", stored),
            ),
            max(n * stored, 1),
        )
        return dst

    def gather_f32(self, src: Dev, index: Dev, dst: Dev, n: int) -> None:
        self._ew(
            "gather_f32",
            (_dev_bind(src), _dev_bind(index), _dev_bind(dst)),
            (struct.pack("<I", n),),
            max(n, 1),
        )

    def moe_grouped(
        self,
        activation: Any,
        atlas: Any,
        table: Any,
        group_expert: Any,
        group_offset: Any,
        destination: Any,
        miss: Any,
        n_groups: int,
        table_base: int,
        cols: int,
        inner: int,
        weight_off: int,
        scale_off: int,
        miss_index: int,
        *,
        kernel: str = "moe_grouped",
    ) -> None:
        if n_groups < 1:
            return
        self.dispatch(
            self.act[kernel],
            (
                _bind_any(activation),
                (atlas, 0),
                (table, 0),
                _bind_any(group_expert),
                _bind_any(group_offset),
                _bind_any(destination),
                (miss, 0),
            ),
            (
                struct.pack("<I", table_base),
                struct.pack("<I", cols),
                struct.pack("<I", inner),
                struct.pack("<I", weight_off),
                struct.pack("<I", scale_off),
                struct.pack("<I", miss_index),
            ),
            self._size(n_groups * cols),
            self._size(32),
        )

    def moe_activate(self, gate_up: Any, n: int, moe: int, stored: int) -> Span:
        dst = self.span(max(n, 1) * stored * 2)
        self._ew(
            "moe_activate",
            (_bind_any(gate_up), (dst.buf, dst.offset)),
            (struct.pack("<I", n), struct.pack("<I", moe), struct.pack("<I", stored)),
            max(n * stored, 1),
        )
        return dst

    def moe_scatter(
        self,
        shared: Dev,
        partial: Any,
        inverse: Any,
        weights: Dev,
        count: int,
        topk: int,
        hidden: int,
    ) -> Dev:
        dst = self.tmp(count * hidden)
        self._ew(
            "moe_scatter",
            (
                _dev_bind(shared),
                _bind_any(partial),
                _bind_any(inverse),
                _dev_bind(weights),
                _dev_bind(dst),
            ),
            (
                struct.pack("<I", count),
                struct.pack("<I", topk),
                struct.pack("<I", hidden),
            ),
            count * hidden,
        )
        return dst

    def moe_add_weighted(
        self,
        down: Dev,
        tokens: Any,
        slots: Any,
        weight: Dev,
        dst: Dev,
        n: int,
        hidden: int,
        topk: int,
    ) -> None:
        if n < 1:
            return
        self._ew(
            "moe_add_weighted",
            (
                _dev_bind(down),
                _bind_any(tokens),
                _bind_any(slots),
                _dev_bind(weight),
                _dev_bind(dst),
            ),
            (
                struct.pack("<I", n),
                struct.pack("<I", hidden),
                struct.pack("<I", topk),
            ),
            n * hidden,
        )

    def moe_decode_gate(
        self,
        activation: Dev,
        atlas: Any,
        table: Any,
        ids: Dev,
        topk: int,
        table_base: int,
        cols: int,
        inner: int,
        weight_off: int,
        scale_off: int,
        miss: Any,
        miss_index: int,
        *,
        kernel: str = "moe_decode_gate",
        want: int = 255,
        tier: Any = None,
    ) -> Dev:
        dst = self.tmp(topk * cols)
        tier_buf = self._tier_pad if tier is None else tier
        self.dispatch(
            self.act[kernel],
            (
                _dev_bind(activation),
                (atlas, 0),
                (table, 0),
                _dev_bind(ids),
                _dev_bind(dst),
                (miss, 0),
                (tier_buf, 0),
            ),
            (
                struct.pack("<I", topk),
                struct.pack("<I", table_base),
                struct.pack("<I", cols),
                struct.pack("<I", inner),
                struct.pack("<I", weight_off),
                struct.pack("<I", scale_off),
                struct.pack("<I", miss_index),
                struct.pack("<I", want),
            ),
            self._size(topk * cols),
            self._size(32),
        )
        return dst

    def moe_decode_down(
        self,
        activation: Any,
        atlas: Any,
        table: Any,
        ids: Dev,
        weights: Dev,
        shared: Dev,
        topk: int,
        table_base: int,
        cols: int,
        inner: int,
        weight_off: int,
        scale_off: int,
        miss: Any,
        miss_index: int,
        *,
        kernel: str = "moe_decode_down",
        want: int = 255,
        tier: Any = None,
    ) -> Dev:
        dst = self.tmp(cols)
        tier_buf = self._tier_pad if tier is None else tier
        self.dispatch(
            self.act[kernel],
            (
                _bind_any(activation),
                (atlas, 0),
                (table, 0),
                _dev_bind(ids),
                _dev_bind(weights),
                _dev_bind(shared),
                _dev_bind(dst),
                (miss, 0),
                (tier_buf, 0),
            ),
            (
                struct.pack("<I", topk),
                struct.pack("<I", table_base),
                struct.pack("<I", cols),
                struct.pack("<I", inner),
                struct.pack("<I", weight_off),
                struct.pack("<I", scale_off),
                struct.pack("<I", miss_index),
                struct.pack("<I", want),
            ),
            self._size(cols),
            self._size(32),
        )
        return dst

    def moe_grouped_int2(self, *args: Any, **kwargs: Any) -> None:
        self.moe_grouped(*args, kernel="moe_grouped_int2", **kwargs)

    def moe_decode_gate_int2(self, *args: Any, **kwargs: Any) -> Dev:
        return self.moe_decode_gate(*args, kernel="moe_decode_gate_int2", **kwargs)

    def moe_decode_down_int2(self, *args: Any, **kwargs: Any) -> Dev:
        return self.moe_decode_down(*args, kernel="moe_decode_down_int2", **kwargs)
