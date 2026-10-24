"""Compile the VoxCPM2 shader library and dispatch its kernels."""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

from favlo.memory import memory

_SHADER = Path(__file__).resolve().parent / "shaders" / "kernels.metal"
TILE_M = 64
TILE_N = 32
K_ALIGN = 256
_NAMES = (
    "rmsnorm_row",
    "matmul_f16",
    "rope_half",
    "gqa_attention",
    "causal_conv1d",
    "causal_transpose1d",
)


def _pad(value: int, tile: int) -> int:
    return (int(value) + tile - 1) // tile * tile


class Resident:
    """A Linear weight already padded and uploaded as half."""

    def __init__(self, handle: object, rows: int, cols: int, rows_pad: int, cols_pad: int) -> None:
        self.handle = handle
        self.rows = rows
        self.cols = cols
        self.rows_pad = rows_pad
        self.cols_pad = cols_pad


class Session:
    """One owner, one queue, the kernels in ``shaders/kernels.metal``."""

    def __init__(self, owner: str) -> None:
        import Metal

        self._owner = owner
        source = _SHADER.read_text(encoding="utf-8")
        device = memory.device
        options = Metal.MTLCompileOptions.alloc().init()
        options.setLanguageVersion_((4 << 16) + 1)
        library, error = device.newLibraryWithSource_options_error_(source, options, None)
        if library is None:
            message = f"shader library failed to compile: {error}"
            raise RuntimeError(message)
        self._metal = Metal
        self._queue = memory.make_queue()
        self._pipes: dict[str, object] = {}
        for name in _NAMES:
            function = library.newFunctionWithName_(name)
            if function is None:
                message = f"{name} is not in the shader library"
                raise RuntimeError(message)
            pipeline, pipe_error = device.newComputePipelineStateWithFunction_error_(function, None)
            if pipeline is None:
                message = f"{name} pipeline failed: {pipe_error}"
                raise RuntimeError(message)
            self._pipes[name] = pipeline
        width = int(self._pipes["matmul_f16"].threadExecutionWidth())
        self._matmul_threads = (width * 4, 1, 1)

    def close(self) -> None:
        memory.sweep(self._owner)
        self._queue = None

    def upload(self, raw: bytes) -> object:
        return memory.upload(raw, owner=self._owner)

    def resident(self, weight: np.ndarray) -> Resident:
        """Pad a ``[out, in]`` matrix to the matmul tile and upload it as half."""
        matrix = np.asarray(weight)
        if matrix.ndim != 2:
            message = f"a linear weight has rank {matrix.ndim}"
            raise ValueError(message)
        out, inner = int(matrix.shape[0]), int(matrix.shape[1])
        out_pad = _pad(out, TILE_N)
        inner_pad = _pad(inner, K_ALIGN)
        padded = np.zeros((out_pad, inner_pad), dtype=np.float16)
        padded[:out, :inner] = np.ascontiguousarray(matrix, dtype=np.float16)
        return Resident(self.upload(padded.tobytes()), out, inner, out_pad, inner_pad)

    def rmsnorm(self, hidden: np.ndarray, weight: np.ndarray, eps: float) -> np.ndarray:
        """RMSNorm over the last axis. The result is float16."""
        data = np.ascontiguousarray(hidden, dtype=np.float16)
        flat = data.reshape(-1, data.shape[-1])
        rows, width = int(flat.shape[0]), int(flat.shape[1])
        scale = np.ascontiguousarray(weight, dtype=np.float16).reshape(-1)
        if int(scale.shape[0]) != width:
            message = f"rmsnorm row is {width}, the weight is {scale.shape[0]}"
            raise ValueError(message)
        source = self.upload(flat.tobytes())
        gains = self.upload(scale.tobytes())
        dest = memory.allocate(flat.nbytes, owner=self._owner, zero=True)
        try:
            raw = self._dispatch(
                "rmsnorm_row",
                [source, gains, dest],
                [struct.pack("<I", width), struct.pack("<f", float(eps))],
                (rows, 1, 1),
                (1, 1, 1),
                read=(dest, flat.nbytes),
            )
        finally:
            memory.release(source)
            memory.release(gains)
            memory.release(dest)
        result = np.frombuffer(raw, dtype=np.float16).copy().reshape(flat.shape)
        return result.reshape(data.shape)

    def linear(
        self,
        hidden: np.ndarray,
        weight: np.ndarray | Resident,
        bias: np.ndarray | None = None,
    ) -> np.ndarray:
        """``hidden @ weight.T`` plus an optional bias. ``weight`` is ``[out, in]``."""
        data = np.asarray(hidden, dtype=np.float32)
        squeeze = data.ndim == 1
        if squeeze:
            data = data.reshape(1, -1)
        if data.ndim != 2:
            message = f"a linear input has rank {data.ndim}"
            raise ValueError(message)
        rows, inner = int(data.shape[0]), int(data.shape[1])
        resident = weight if isinstance(weight, Resident) else self.resident(weight)
        if resident.cols != inner:
            message = f"linear inner extent is {inner}, the weight expects {resident.cols}"
            raise ValueError(message)
        rows_pad = _pad(rows, TILE_M)
        act = np.zeros((rows_pad, resident.cols_pad), dtype=np.float16)
        act[:rows, :inner] = data.astype(np.float16)
        source = self.upload(np.ascontiguousarray(act).tobytes())
        dest_count = rows_pad * resident.rows_pad
        dest = memory.allocate(dest_count * 4, owner=self._owner, zero=True)
        try:
            raw = self._dispatch(
                "matmul_f16",
                [source, resident.handle, dest],
                [
                    struct.pack("<I", rows_pad),
                    struct.pack("<I", resident.rows_pad),
                    struct.pack("<I", resident.cols_pad),
                ],
                (resident.rows_pad // TILE_N, rows_pad // TILE_M, 1),
                self._matmul_threads,
                read=(dest, dest_count * 4),
            )
        finally:
            memory.release(source)
            memory.release(dest)
            if not isinstance(weight, Resident):
                memory.release(resident.handle)
        out = np.frombuffer(raw, dtype=np.float32).copy().reshape(rows_pad, resident.rows_pad)
        out = out[:rows, : resident.rows]
        if bias is not None:
            out = out + np.asarray(bias, dtype=np.float32).reshape(1, -1)
        if squeeze:
            return out.reshape(-1)
        return out

    def rope(self, hidden: np.ndarray, cos: np.ndarray, sin: np.ndarray) -> np.ndarray:
        """Rotate-half on ``[seq, heads, dim]``."""
        data = np.ascontiguousarray(hidden, dtype=np.float32)
        seq, heads, dim = (int(data.shape[0]), int(data.shape[1]), int(data.shape[2]))
        cos_tab = np.ascontiguousarray(cos, dtype=np.float32)
        sin_tab = np.ascontiguousarray(sin, dtype=np.float32)
        source = self.upload(data.tobytes())
        cos_buf = self.upload(cos_tab.tobytes())
        sin_buf = self.upload(sin_tab.tobytes())
        dest = memory.allocate(data.nbytes, owner=self._owner, zero=True)
        try:
            raw = self._dispatch(
                "rope_half",
                [source, cos_buf, sin_buf, dest],
                [struct.pack("<I", heads), struct.pack("<I", dim)],
                _groups(dim // 2, heads, seq, 8, 4),
                (8, 4, 1),
                read=(dest, data.nbytes),
            )
        finally:
            memory.release(source)
            memory.release(cos_buf)
            memory.release(sin_buf)
            memory.release(dest)
        return np.frombuffer(raw, dtype=np.float32).copy().reshape(data.shape)

    def attention(
        self,
        query: np.ndarray,
        key: np.ndarray,
        value: np.ndarray,
        *,
        causal: bool,
    ) -> np.ndarray:
        """Grouped-query attention. Query is ``[q, heads, dim]``."""
        q = np.ascontiguousarray(query, dtype=np.float32)
        k = np.ascontiguousarray(key, dtype=np.float32)
        v = np.ascontiguousarray(value, dtype=np.float32)
        q_len, heads, dim = (int(q.shape[0]), int(q.shape[1]), int(q.shape[2]))
        kv_len, kv_heads, _ = (int(k.shape[0]), int(k.shape[1]), int(k.shape[2]))
        scratch_count = q_len * heads * kv_len
        q_buf = self.upload(q.tobytes())
        k_buf = self.upload(k.tobytes())
        v_buf = self.upload(v.tobytes())
        dest = memory.allocate(q.nbytes, owner=self._owner, zero=True)
        scratch = memory.allocate(max(scratch_count, 1) * 4, owner=self._owner, zero=True)
        try:
            raw = self._dispatch(
                "gqa_attention",
                [q_buf, k_buf, v_buf, dest, scratch],
                [
                    struct.pack("<I", q_len),
                    struct.pack("<I", kv_len),
                    struct.pack("<I", heads),
                    struct.pack("<I", kv_heads),
                    struct.pack("<I", dim),
                    struct.pack("<I", 1 if causal else 0),
                ],
                (heads, q_len, 1),
                (1, 1, 1),
                read=(dest, q.nbytes),
            )
        finally:
            memory.release(q_buf)
            memory.release(k_buf)
            memory.release(v_buf)
            memory.release(dest)
            memory.release(scratch)
        return np.frombuffer(raw, dtype=np.float32).copy().reshape(q.shape)

    def conv1d(
        self,
        source: np.ndarray,
        weight: np.ndarray,
        bias: np.ndarray,
        *,
        left: int,
        dilation: int,
        groups: int,
    ) -> np.ndarray:
        """Causal conv1d. Source is ``[in, time]``, weight is ``[out, in/groups, kernel]``."""
        data = np.ascontiguousarray(source, dtype=np.float32)
        kernel = np.ascontiguousarray(weight, dtype=np.float32)
        shift = np.ascontiguousarray(bias, dtype=np.float32).reshape(-1)
        cin, time = int(data.shape[0]), int(data.shape[1])
        cout, cin_g, width = (int(kernel.shape[0]), int(kernel.shape[1]), int(kernel.shape[2]))
        if cin // int(groups) != cin_g:
            message = f"conv groups {groups} do not match weight {kernel.shape}"
            raise ValueError(message)
        out = np.empty((cout, time), dtype=np.float32)
        src = self.upload(data.tobytes())
        wbuf = self.upload(kernel.tobytes())
        bbuf = self.upload(shift.tobytes())
        dest = memory.allocate(out.nbytes, owner=self._owner, zero=True)
        try:
            raw = self._dispatch(
                "causal_conv1d",
                [src, wbuf, bbuf, dest],
                [
                    struct.pack("<I", cin),
                    struct.pack("<I", cout),
                    struct.pack("<I", time),
                    struct.pack("<I", width),
                    struct.pack("<I", int(dilation)),
                    struct.pack("<I", int(left)),
                    struct.pack("<I", int(groups)),
                ],
                _groups(cout, time, 1, 16, 16),
                (16, 16, 1),
                read=(dest, out.nbytes),
            )
        finally:
            memory.release(src)
            memory.release(wbuf)
            memory.release(bbuf)
            memory.release(dest)
        return np.frombuffer(raw, dtype=np.float32).copy().reshape(out.shape)

    def conv_transpose1d(
        self,
        source: np.ndarray,
        weight: np.ndarray,
        bias: np.ndarray,
        *,
        stride: int,
    ) -> np.ndarray:
        """Causal transposed conv. Weight is ``[in, out, kernel]``. Length is ``time * stride``."""
        data = np.ascontiguousarray(source, dtype=np.float32)
        kernel = np.ascontiguousarray(weight, dtype=np.float32)
        shift = np.ascontiguousarray(bias, dtype=np.float32).reshape(-1)
        cin, tin = int(data.shape[0]), int(data.shape[1])
        _, cout, width = (int(kernel.shape[0]), int(kernel.shape[1]), int(kernel.shape[2]))
        if int(kernel.shape[0]) != cin:
            message = f"transpose in-channels are {kernel.shape[0]}, the input has {cin}"
            raise ValueError(message)
        tout = tin * int(stride)
        out = np.empty((cout, tout), dtype=np.float32)
        src = self.upload(data.tobytes())
        wbuf = self.upload(kernel.tobytes())
        bbuf = self.upload(shift.tobytes())
        dest = memory.allocate(out.nbytes, owner=self._owner, zero=True)
        try:
            raw = self._dispatch(
                "causal_transpose1d",
                [src, wbuf, bbuf, dest],
                [
                    struct.pack("<I", cin),
                    struct.pack("<I", cout),
                    struct.pack("<I", tin),
                    struct.pack("<I", tout),
                    struct.pack("<I", width),
                    struct.pack("<I", int(stride)),
                ],
                _groups(cout, tout, 1, 16, 16),
                (16, 16, 1),
                read=(dest, out.nbytes),
            )
        finally:
            memory.release(src)
            memory.release(wbuf)
            memory.release(bbuf)
            memory.release(dest)
        return np.frombuffer(raw, dtype=np.float32).copy().reshape(out.shape)

    def _dispatch(
        self,
        name: str,
        handles: list[object],
        constants: list[bytes],
        groups: tuple[int, int, int],
        threads: tuple[int, int, int],
        *,
        read: tuple[object, int],
    ) -> bytes:
        command = memory.command_buffer(self._queue)
        encoder = command.computeCommandEncoder()
        encoder.setComputePipelineState_(self._pipes[name])
        for index, handle in enumerate(handles):
            memory.bind(encoder, index, handle, 0)
        for offset, packed in enumerate(constants):
            encoder.setBytes_length_atIndex_(packed, len(packed), len(handles) + offset)
        encoder.dispatchThreadgroups_threadsPerThreadgroup_(
            self._metal.MTLSize(int(groups[0]), int(groups[1]), int(groups[2])),
            self._metal.MTLSize(int(threads[0]), int(threads[1]), int(threads[2])),
        )
        encoder.endEncoding()
        command.commit()
        command.waitUntilCompleted()
        status = int(command.status())
        error = command.error()
        raw = b""
        if status == 4:
            handle, nbytes = read
            raw = bytes(memoryview(handle.contents().as_buffer(int(nbytes))))
        memory.retire(command)
        if status != 4:
            message = f"{name} command status {status} {error}"
            raise RuntimeError(message)
        return raw


def _groups(x: int, y: int, z: int, tx: int, ty: int) -> tuple[int, int, int]:
    return ((int(x) + tx - 1) // tx, (int(y) + ty - 1) // ty, int(z))
