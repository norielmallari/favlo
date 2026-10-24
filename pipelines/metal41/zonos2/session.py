"""Metal matmul2d. Left is half, the right plane is fp4, fp8, or half, and the dest is float."""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import objc

from favlo.memory import memory
from zonos2.formats import dequant_fp4, dequant_fp8, quantize_fp4, quantize_fp8

_TILE = 64
_FP4 = 148
_FP8 = 142
_E8M0 = 145
_LANGUAGE = (4 << 16) + 1
_SOURCE = Path(__file__).resolve().parent / "shaders" / "kernels.metal"

_libobjc = ctypes.CDLL("/usr/lib/libobjc.A.dylib")
_msg = _libobjc.objc_msgSend
_msg.restype = ctypes.c_void_p
_msg.argtypes = [
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_ulong,
    ctypes.POINTER(ctypes.c_long),
]
_sel = _libobjc.sel_registerName
_sel.restype = ctypes.c_void_p
_sel.argtypes = [ctypes.c_char_p]
_EXTENTS = _sel(b"initWithRank:values:")
_COMMIT = _sel(b"commit:count:")


def _extents(*values: int):
    import Metal

    raw = (ctypes.c_long * len(values))(*values)
    allocated = Metal.MTLTensorExtents.alloc()
    pointer = _msg(objc.pyobjc_id(allocated), _EXTENTS, len(values), raw)
    return objc.objc_object(c_void_p=pointer)


def _pad(count: int) -> int:
    return ((count + _TILE - 1) // _TILE) * _TILE


@dataclass
class Weight:
    """One right-hand matrix. ``rows`` is the unpadded N. Storage is K-innermost."""

    tensor: object
    buffers: tuple[object, ...]
    rows: int
    inner: int
    padded_rows: int
    kind: str


class Session:
    """One device, one queue, and the three matmul pipelines."""

    def __init__(self) -> None:
        import Metal

        self._owner = f"zonos:{id(self)}"
        self.device = memory.device
        options = Metal.MTLCompileOptions.alloc().init()
        options.setLanguageVersion_(_LANGUAGE)
        library, error = self.device.newLibraryWithSource_options_error_(
            _SOURCE.read_text(encoding="utf-8"), options, None
        )
        if library is None:
            message = str(error) if error is not None else "shader library failed"
            raise RuntimeError(message)
        self.pipelines = {}
        for name in ("gemm_f16", "gemm_fp4", "gemm_fp8", "attend"):
            function = library.newFunctionWithName_(name)
            if function is None:
                message = f"{name} is missing from the shader library"
                raise RuntimeError(message)
            pipeline, error = self.device.newComputePipelineStateWithFunction_error_(function, None)
            if pipeline is None:
                message = str(error) if error is not None else f"{name} did not compile"
                raise RuntimeError(message)
            self.pipelines[name] = pipeline
        self.width = self.pipelines["gemm_f16"].threadExecutionWidth()
        table = Metal.MTL4ArgumentTableDescriptor.alloc().init()
        table.setMaxBufferBindCount_(3)
        self.table, error = self.device.newArgumentTableWithDescriptor_error_(table, None)
        if self.table is None:
            message = str(error) if error is not None else "argument table failed"
            raise RuntimeError(message)
        residency, error = self.device.newResidencySetWithDescriptor_error_(
            Metal.MTLResidencySetDescriptor.alloc().init(), None
        )
        if residency is None:
            message = str(error) if error is not None else "residency set failed"
            raise RuntimeError(message)
        self.residency = residency
        self.classic = memory.make_queue()
        memory.watch(self.residency)
        self.queue = self.device.newMTL4CommandQueue()
        self.queue.addResidencySet_(self.residency)
        self.allocator = self.device.newCommandAllocator()
        self._signal = 0
        self.event = self.device.newSharedEvent()
        self._left: dict[tuple[int, int], Weight] = {}
        self._dest: dict[tuple[int, int], Weight] = {}
        self._commit = ctypes.CFUNCTYPE(
            None,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_ulong,
        )(("objc_msgSend", _libobjc))

    def half(self, matrix: np.ndarray) -> Weight:
        """Load a row-major ``[N, K]`` half matrix."""
        array = np.ascontiguousarray(matrix, dtype=np.float16)
        if array.ndim != 2:
            message = "a half weight is a matrix"
            raise ValueError(message)
        return self._plain(array, "f16", 0)

    def fp4(self, packed: np.ndarray, scales: np.ndarray, inner: int) -> Weight:
        """Load K-innermost fp4 bytes ``[N, K/2]`` and E8M0 scales ``[N, K/32]``."""
        return self._quant(packed, scales, inner, "fp4", _FP4)

    def fp8(self, data: np.ndarray, scales: np.ndarray, inner: int) -> Weight:
        """Load K-innermost fp8 bytes ``[N, K]`` and E8M0 scales ``[N, K/32]``."""
        return self._quant(data, scales, inner, "fp8", _FP8)

    def close(self) -> None:
        """Sweep this session's spans and drop the queues."""
        memory.sweep(getattr(self, "_owner", ""))
        self.pipelines = {}
        self.queue = None
        self.classic = None
        self.device = None

    def gemm(self, left: np.ndarray, weight: Weight) -> np.ndarray:
        """``left @ weight.T`` for ``left`` shaped ``[M, K]``. The dest starts at zero."""
        import Metal

        matrix = np.ascontiguousarray(left, dtype=np.float16)
        if matrix.ndim != 2 or matrix.shape[1] != weight.inner:
            message = "the left matrix K does not match the weight"
            raise ValueError(message)
        rows = matrix.shape[0]
        padded_rows = _pad(rows)
        staged = np.zeros((padded_rows, weight.inner), dtype=np.float16)
        staged[:rows] = matrix
        activation = self._reuse(self._left, staged, "f16")
        dest_rows = weight.padded_rows
        destination = self._reuse(
            self._dest, np.zeros((padded_rows, dest_rows), dtype=np.float32), "f32"
        )
        self.table.setResource_atBufferIndex_(activation.tensor.gpuResourceID(), 0)
        self.table.setResource_atBufferIndex_(weight.tensor.gpuResourceID(), 1)
        self.table.setResource_atBufferIndex_(destination.tensor.gpuResourceID(), 2)
        self.residency.commit()
        self.residency.requestResidency()
        command = self.device.newCommandBuffer()
        command.beginCommandBufferWithAllocator_(self.allocator)
        command.useResidencySet_(self.residency)
        encoder = command.computeCommandEncoder()
        encoder.setComputePipelineState_(self.pipelines[f"gemm_{weight.kind}"])
        encoder.setArgumentTable_(self.table)
        encoder.dispatchThreadgroups_threadsPerThreadgroup_(
            Metal.MTLSize(dest_rows // _TILE, padded_rows // _TILE, 1),
            Metal.MTLSize(self.width * 4, 1, 1),
        )
        encoder.endEncoding()
        command.endCommandBuffer()
        ids = (ctypes.c_void_p * 1)(objc.pyobjc_id(command))
        self._commit(objc.pyobjc_id(self.queue), _COMMIT, ids, 1)
        self._signal += 1
        self.queue.signalEvent_value_(self.event, self._signal)
        if not self.event.waitUntilSignaledValue_timeoutMS_(self._signal, 10000):
            message = "matmul did not finish"
            raise RuntimeError(message)
        view = destination.buffers[0].contents().as_buffer(padded_rows * dest_rows * 4)
        got = np.frombuffer(view, dtype=np.float32).reshape(padded_rows, dest_rows).copy()
        return got[:rows, : weight.rows]

    def attend(self, query: np.ndarray, key: np.ndarray, value: np.ndarray) -> np.ndarray:
        """GQA over ``query`` [16, 128], ``key`` and ``value`` [steps, 4, 128]."""
        import Metal

        heads = np.ascontiguousarray(query, dtype=np.float32)
        keys = np.ascontiguousarray(key, dtype=np.float32)
        values = np.ascontiguousarray(value, dtype=np.float32)
        steps = keys.shape[0]
        if heads.shape != (16, 128) or keys.shape != (steps, 4, 128) or values.shape != keys.shape:
            message = "attention shapes do not match this model"
            raise ValueError(message)
        out = np.zeros((16, 128), dtype=np.float32)
        scores = np.zeros((16, steps), dtype=np.float32)
        with objc.autorelease_pool():
            buffers = [self._bytes(item) for item in (heads, keys, values, out, scores)]
            steps_buf = self._bytes(np.array([steps], dtype=np.uint32))
            command = memory.command_buffer(self.classic)
            encoder = command.computeCommandEncoder()
            encoder.setComputePipelineState_(self.pipelines["attend"])
            for index, buf in enumerate(buffers):
                memory.bind(encoder, index, buf, 0)
            memory.bind(encoder, 5, steps_buf, 0)
            encoder.dispatchThreads_threadsPerThreadgroup_(
                Metal.MTLSize(16, 1, 1), Metal.MTLSize(16, 1, 1)
            )
            encoder.endEncoding()
            command.commit()
            command.waitUntilCompleted()
            view = buffers[3].contents().as_buffer(out.nbytes)
            copied = np.frombuffer(view, dtype=np.float32).reshape(16, 128).copy()
            memory.retire(command)
            return copied

    def _reuse(self, cache: dict[tuple[int, int], Weight], matrix: np.ndarray, kind: str) -> Weight:
        key = (matrix.shape[0], matrix.shape[1])
        found = cache.get(key)
        if found is None:
            found = self._plain(matrix, kind, 0)
            cache[key] = found
            return found
        raw = np.ascontiguousarray(matrix)
        view = found.buffers[0].contents().as_buffer(raw.nbytes)
        np.frombuffer(view, dtype=raw.dtype)[:] = raw.ravel()
        return found

    def _quant(
        self,
        data: np.ndarray,
        scales: np.ndarray,
        inner: int,
        kind: str,
        data_type: int,
    ) -> Weight:
        import Metal

        packed = np.ascontiguousarray(data)
        exponents = np.ascontiguousarray(scales, dtype=np.uint8)
        rows = packed.shape[0]
        padded = _pad(rows)
        width = inner // 32
        if exponents.shape != (rows, width):
            message = "the scale plane does not match the data plane"
            raise ValueError(message)
        data_buf, scale_buf = memory.upload_group(
            (_fit(packed, padded), _fit(exponents, padded)),
            owner=self._owner,
        )
        desc = Metal.MTLTensorDescriptor.alloc().init()
        desc.setDimensions_(_extents(inner, padded))
        desc.setStrides_(_extents(1, inner))
        desc.setDataType_(data_type)
        desc.setUsage_(Metal.MTLTensorUsageCompute)
        plane = Metal.MTLTensorAuxiliaryPlaneDescriptor.alloc().init()
        plane.setDataType_(_E8M0)
        plane.setBlockFactors_(_extents(32, 1))
        planes = Metal.MTLTensorAuxiliaryPlaneDescriptorMap.alloc().init()
        planes.setDescriptor_forPlane_(plane, 1)
        desc.setAuxiliaryPlanes_(planes)
        attachments = Metal.MTLTensorBufferAttachments.alloc().init()
        memory.bind_plane(attachments, 0, data_buf, 0)
        memory.bind_plane(attachments, 1, scale_buf, 0)
        tensor = self.device.newTensorWithDescriptor_attachments_error_(desc, attachments, None)
        if tensor is None:
            message = f"{kind} tensor failed"
            raise RuntimeError(message)
        return Weight(tensor, (data_buf, scale_buf), rows, inner, padded, kind)

    def _plain(self, matrix: np.ndarray, kind: str, data_type: int) -> Weight:
        import Metal

        array = np.ascontiguousarray(matrix)
        rows, inner = array.shape
        padded = _pad(rows)
        buf = self._bytes(_fit(array, padded))
        desc = Metal.MTLTensorDescriptor.alloc().init()
        desc.setDimensions_(_extents(inner, padded))
        desc.setStrides_(_extents(1, inner))
        if kind == "f32":
            desc.setDataType_(Metal.MTLTensorDataTypeFloat32)
        else:
            desc.setDataType_(Metal.MTLTensorDataTypeFloat16)
        del data_type
        desc.setUsage_(Metal.MTLTensorUsageCompute)
        attachments = Metal.MTLTensorBufferAttachments.alloc().init()
        memory.bind_plane(attachments, 0, buf, 0)
        tensor = self.device.newTensorWithDescriptor_attachments_error_(desc, attachments, None)
        if tensor is None:
            message = f"{kind} tensor failed"
            raise RuntimeError(message)
        stored = "f16" if kind == "f16" else kind
        return Weight(tensor, (buf,), rows, inner, padded, stored)

    def _bytes(self, array: np.ndarray):
        return memory.upload(np.ascontiguousarray(array), owner=self._owner)


def _fit(array: np.ndarray, rows: int) -> np.ndarray:
    if array.shape[0] == rows:
        return array
    fitted = np.zeros((rows, array.shape[1]), dtype=array.dtype)
    fitted[: array.shape[0]] = array
    return fitted


def check_gemms() -> None:
    """One padded tile of each plane against the host dequant."""
    session = Session()
    rng = np.random.default_rng(2)
    left = rng.standard_normal((3, 256)).astype(np.float32)
    weight = rng.standard_normal((70, 256)).astype(np.float32)
    rounded = left.astype(np.float16).astype(np.float32)
    packed, scales = quantize_fp4(weight)
    fp4 = session.fp4(packed, scales, 256)
    got = session.gemm(left, fp4)
    expect = rounded @ dequant_fp4(packed, scales, 70, 256).T
    _close(got, expect, "fp4")
    data, scales = quantize_fp8(weight)
    fp8 = session.fp8(data, scales, 256)
    got = session.gemm(left, fp8)
    expect = rounded @ dequant_fp8(data, scales, 70, 256).T
    _close(got, expect, "fp8")
    half = session.half(weight.astype(np.float16))
    got = session.gemm(left, half)
    expect = rounded @ weight.astype(np.float16).astype(np.float32).T
    _close(got, expect, "f16")
    _check_attend(session)


def _check_attend(session: Session) -> None:
    rng = np.random.default_rng(3)
    query = rng.standard_normal((16, 128)).astype(np.float32)
    key = rng.standard_normal((5, 4, 128)).astype(np.float32)
    value = rng.standard_normal((5, 4, 128)).astype(np.float32)
    got = session.attend(query, key, value)
    scale = np.float32(1.0 / np.sqrt(128.0))
    expect = np.zeros_like(query)
    for head in range(16):
        kv = head // 4
        scores = key[:, kv] @ query[head] * scale
        scores = scores - scores.max()
        weight = np.exp(scores)
        weight /= weight.sum()
        expect[head] = weight @ value[:, kv]
    _close(got, expect, "attend")


def _close(got: np.ndarray, expect: np.ndarray, name: str) -> None:
    if not np.allclose(got, expect, rtol=1e-3, atol=1e-3):
        delta = float(np.max(np.abs(got - expect)))
        message = f"{name} matmul disagrees with the host ({delta:.6g})"
        raise RuntimeError(message)
