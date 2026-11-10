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
    "pack_half",
    "bias_pack",
    "cast_f32_f16",
    "cast_f16_f32",
    "silu_f32",
    "swiglu_f32",
    "add_f32",
    "copy_rows",
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


class Plane:
    """A contiguous float32 matrix on the device. ``rows`` by ``cols``."""

    def __init__(self, handle: object, rows: int, cols: int) -> None:
        self.handle = handle
        self.rows = rows
        self.cols = cols


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
        self._enc = None
        self._cmd = None
        self._half_a = None
        self._half_b = None
        self._half_count = 0
        self._wide = None
        self._wide_count = 0
        self._attn = None
        self._attn_count = 0

    def close(self) -> None:
        try:
            self.flush()
        finally:
            memory.sweep(self._owner)
            self._queue = None
            self._half_a = None
            self._half_b = None
            self._wide = None
            self._attn = None
            self._half_count = 0
            self._wide_count = 0
            self._attn_count = 0

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

    def plane(self, rows: int, cols: int, *, zero: bool = False) -> Plane:
        """A float32 matrix. The caller releases it after the command buffer finishes."""
        if rows < 1 or cols < 1:
            message = f"a device tensor is {rows} by {cols}"
            raise ValueError(message)
        handle = memory.allocate(int(rows) * int(cols) * 4, owner=self._owner, zero=zero)
        return Plane(handle, int(rows), int(cols))

    def load(self, values: np.ndarray) -> Plane:
        """Upload a float32 matrix. A vector becomes one row."""
        data = np.asarray(values, dtype=np.float32)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        if data.ndim != 2:
            message = f"a device tensor has rank {data.ndim}"
            raise ValueError(message)
        data = np.ascontiguousarray(data)
        plane = self.plane(int(data.shape[0]), int(data.shape[1]))
        memory.store(plane.handle, 0, data.tobytes())
        return plane

    def read(self, plane: Plane) -> np.ndarray:
        """Finish the open command buffer and copy the matrix back."""
        self.flush()
        raw = memory.load(plane.handle, 0, plane.rows * plane.cols * 4)
        return np.frombuffer(raw, dtype=np.float32).copy().reshape(plane.rows, plane.cols)

    def release_all(self, planes: list[Plane], handles: list[object] | None = None) -> None:
        """Finish any open command buffer, then release these tensors."""
        self.flush()
        for plane in planes:
            memory.release(plane.handle)
        for handle in handles or ():
            memory.release(handle)

    def prepare_linear(self, rows: int, inner: int, outer: int, attn: int) -> None:
        """Size the scratch for a stack step. Call this before the command buffer opens."""
        rows_pad = _pad(rows, TILE_M)
        self._grow_half(rows_pad * _pad(inner, K_ALIGN))
        self._grow_wide(rows_pad * _pad(outer, TILE_N))
        self._grow_attn(max(int(attn), 1))

    def project(self, src: Plane, weight: Resident, bias: object | None, dst: Plane) -> None:
        """``src @ weight.T`` plus an optional bias, encoded on the open command buffer."""
        rows, inner = src.rows, src.cols
        if weight.cols != inner or dst.rows != rows or dst.cols != weight.rows:
            message = (
                f"project {rows}x{inner} -> {dst.rows}x{dst.cols} "
                f"does not match weight {weight.rows}x{weight.cols}"
            )
            raise ValueError(message)
        rows_pad = _pad(rows, TILE_M)
        if (
            self._half_a is None
            or self._wide is None
            or self._half_count < rows_pad * weight.cols_pad
            or self._wide_count < rows_pad * weight.rows_pad
        ):
            message = "the matmul scratch is short of this linear"
            raise RuntimeError(message)
        self._encode(
            "pack_half",
            [src.handle, self._half_a],
            [_u(rows), _u(inner), _u(rows_pad), _u(weight.cols_pad)],
            _groups(weight.cols_pad, rows_pad, 1, 16, 16),
            (16, 16, 1),
        )
        self._encode(
            "matmul_f16",
            [self._half_a, weight.handle, self._wide],
            [_u(rows_pad), _u(weight.rows_pad), _u(weight.cols_pad)],
            (weight.rows_pad // TILE_N, rows_pad // TILE_M, 1),
            self._matmul_threads,
        )
        self._encode(
            "bias_pack",
            [self._wide, dst.handle if bias is None else bias, dst.handle],
            [_u(rows), _u(dst.cols), _u(weight.rows_pad), _u(0 if bias is None else 1)],
            _groups(dst.cols, rows, 1, 16, 16),
            (16, 16, 1),
        )

    def norm(self, src: Plane, gain: object, dst: Plane, eps: float) -> None:
        """RMSNorm. The row is cast to half, normalized, and stored as float32."""
        _same(src, dst, "rmsnorm")
        count = src.rows * src.cols
        if self._half_a is None or self._half_b is None or self._half_count < count:
            message = "the half scratch is short of this norm"
            raise RuntimeError(message)
        groups, threads = _count(count)
        self._encode("cast_f32_f16", [src.handle, self._half_a], [_u(count)], groups, threads)
        self._encode(
            "rmsnorm_row",
            [self._half_a, gain, self._half_b],
            [_u(src.cols), _f(eps)],
            (src.rows, 1, 1),
            (1, 1, 1),
        )
        self._encode("cast_f16_f32", [self._half_b, dst.handle], [_u(count)], groups, threads)

    def silu(self, src: Plane, dst: Plane) -> None:
        _same(src, dst, "silu")
        self._map("silu_f32", [src.handle, dst.handle], src.rows * src.cols)

    def swiglu(self, gate: Plane, up: Plane, dst: Plane) -> None:
        _same(gate, up, "swiglu")
        _same(gate, dst, "swiglu")
        self._map("swiglu_f32", [gate.handle, up.handle, dst.handle], gate.rows * gate.cols)

    def add(self, left: Plane, right: Plane, dst: Plane) -> None:
        _same(left, right, "add")
        _same(left, dst, "add")
        self._map("add_f32", [left.handle, right.handle, dst.handle], left.rows * left.cols)

    def copy(self, src: Plane, dst: Plane, rows: int, src_row: int, dst_row: int) -> None:
        """Copy ``rows`` from ``src_row`` into ``dst`` at ``dst_row``."""
        if src.cols != dst.cols:
            message = f"copy width is {src.cols} and {dst.cols}"
            raise ValueError(message)
        if rows < 1 or src_row < 0 or dst_row < 0:
            message = "copy rows start below zero"
            raise ValueError(message)
        if src_row + rows > src.rows or dst_row + rows > dst.rows:
            message = "copy rows do not fit"
            raise ValueError(message)
        self._encode(
            "copy_rows",
            [src.handle, dst.handle],
            [_u(rows), _u(src.cols), _u(src_row), _u(dst_row)],
            _groups(src.cols, rows, 1, 16, 16),
            (16, 16, 1),
        )

    def rotate(self, hidden: Plane, cos: object, sin: object, heads: int, dim: int) -> None:
        """In-place rotate-half. ``hidden`` is ``[seq, heads * dim]``."""
        if hidden.cols != heads * dim:
            message = f"rope width is {hidden.cols}, the heads cover {heads * dim}"
            raise ValueError(message)
        if dim % 2:
            message = f"rope dim {dim} is odd"
            raise ValueError(message)
        self._encode(
            "rope_half",
            [hidden.handle, cos, sin, hidden.handle],
            [_u(heads), _u(dim)],
            _groups(dim // 2, heads, hidden.rows, 8, 4),
            (8, 4, 1),
        )

    def attend(
        self,
        query: Plane,
        key: Plane,
        value: Plane,
        dest: Plane,
        *,
        kv_len: int,
        heads: int,
        kv_heads: int,
        dim: int,
        causal: bool,
    ) -> None:
        """Grouped-query attention. ``kv_len`` is the filled prefix, not the allocation."""
        if query.cols != heads * dim or dest.rows != query.rows or dest.cols != query.cols:
            message = f"attention query is {query.rows}x{query.cols}"
            raise ValueError(message)
        if key.cols != kv_heads * dim or value.cols != key.cols:
            message = f"attention cache width is {key.cols}"
            raise ValueError(message)
        if kv_len < 1 or kv_len > key.rows or kv_len > value.rows:
            message = f"attention reads {kv_len} keys from {key.rows}"
            raise ValueError(message)
        need = query.rows * heads * kv_len
        if self._attn is None or self._attn_count < need:
            message = f"attention scratch holds {self._attn_count}, this step needs {need}"
            raise RuntimeError(message)
        self._encode(
            "gqa_attention",
            [query.handle, key.handle, value.handle, dest.handle, self._attn],
            [
                _u(query.rows),
                _u(kv_len),
                _u(heads),
                _u(kv_heads),
                _u(dim),
                _u(1 if causal else 0),
            ],
            (heads, query.rows, 1),
            (1, 1, 1),
        )

    def flush(self) -> None:
        """Wait for the open command buffer. A closed session has nothing to wait for."""
        encoder = self._enc
        command = self._cmd
        if encoder is None or command is None:
            return
        self._enc = None
        self._cmd = None
        status = 0
        error = None
        try:
            encoder.endEncoding()
            command.commit()
            command.waitUntilCompleted()
            status = int(command.status())
            error = command.error()
        finally:
            memory.retire(command)
        if status != 4:
            message = f"command status {status} {error}"
            raise RuntimeError(message)

    def _map(self, name: str, handles: list[object], count: int) -> None:
        groups, threads = _count(count)
        self._encode(name, handles, [_u(count)], groups, threads)

    def _grow_half(self, count: int) -> None:
        if self._half_count >= count:
            return
        if self._enc is not None:
            message = "a scratch buffer grew while a command buffer was open"
            raise RuntimeError(message)
        for slot in ("_half_a", "_half_b"):
            old = getattr(self, slot)
            if old is not None:
                memory.release(old)
            setattr(self, slot, memory.allocate(count * 2, owner=self._owner, zero=False))
        self._half_count = count

    def _grow_wide(self, count: int) -> None:
        self._grow("_wide", "_wide_count", count, 4)

    def _grow_attn(self, count: int) -> None:
        self._grow("_attn", "_attn_count", count, 4)

    def _grow(self, slot: str, count_slot: str, count: int, width: int) -> None:
        if getattr(self, count_slot) >= count:
            return
        if self._enc is not None:
            message = "a scratch buffer grew while a command buffer was open"
            raise RuntimeError(message)
        old = getattr(self, slot)
        if old is not None:
            memory.release(old)
        setattr(self, slot, memory.allocate(count * width, owner=self._owner, zero=False))
        setattr(self, count_slot, count)

    def _encoder(self) -> object:
        if self._enc is None:
            self._cmd = memory.command_buffer(self._queue)
            self._enc = self._cmd.computeCommandEncoder()
        return self._enc

    def _encode(
        self,
        name: str,
        handles: list[object],
        constants: list[bytes],
        groups: tuple[int, int, int],
        threads: tuple[int, int, int],
    ) -> None:
        self._bind(self._encoder(), name, handles, constants, groups, threads)

    def _bind(
        self,
        encoder: object,
        name: str,
        handles: list[object],
        constants: list[bytes],
        groups: tuple[int, int, int],
        threads: tuple[int, int, int],
    ) -> None:
        encoder.setComputePipelineState_(self._pipes[name])
        for index, handle in enumerate(handles):
            memory.bind(encoder, index, handle, 0)
        for offset, packed in enumerate(constants):
            encoder.setBytes_length_atIndex_(packed, len(packed), len(handles) + offset)
        encoder.dispatchThreadgroups_threadsPerThreadgroup_(
            self._metal.MTLSize(int(groups[0]), int(groups[1]), int(groups[2])),
            self._metal.MTLSize(int(threads[0]), int(threads[1]), int(threads[2])),
        )

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
        if self._enc is not None:
            message = f"{name} started while a command buffer was open"
            raise RuntimeError(message)
        command = memory.command_buffer(self._queue)
        encoder = command.computeCommandEncoder()
        self._bind(encoder, name, handles, constants, groups, threads)
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


def _count(items: int) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    width = 256
    return ((int(items) + width - 1) // width, 1, 1), (width, 1, 1)


def _same(src: Plane, dst: Plane, name: str) -> None:
    if src.rows != dst.rows or src.cols != dst.cols:
        message = f"{name} shape is {src.rows}x{src.cols} and {dst.rows}x{dst.cols}"
        raise ValueError(message)


def _u(value: int) -> bytes:
    return struct.pack("<I", int(value))


def _f(value: float) -> bytes:
    return struct.pack("<f", float(value))
