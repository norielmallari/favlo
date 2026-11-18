"""Host checkpoint packing. NumPy on the CPU. No Metal and no CUDA.

Windows, macOS, and Linux run the same functions. Serving does not import
this module. The pool size is one worker per CPU, and each worker keeps one
BLAS thread.
"""

from __future__ import annotations

import os
import threading
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor

import numpy as np

FP8_QMAX = 448.0
FP8_BLOCK = 32
INT8_QMAX = 127.0
INT4_QMAX = 7.0

_IO = threading.Lock()


def worker_count() -> int:
    """One worker per logical CPU on this machine.

    ``cpu_count`` is the machine. ``process_cpu_count`` is included so a
    process that can see more threads than that report still uses them.
    """
    machine = os.cpu_count() or 0
    visible = os.process_cpu_count() or 0
    return max(1, machine, visible)


def limit_blas_threads() -> None:
    """Stop one worker from starting its own BLAS team."""
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"


def parallel_map[T, R](fn: Callable[[T], R], tasks: Iterable[T]) -> list[R]:
    """Run ``fn`` on each task. The worker limits BLAS before the call."""

    def run(task: T) -> R:
        limit_blas_threads()
        return fn(task)

    items = list(tasks)
    if len(items) <= 1:
        return [run(item) for item in items]
    with ThreadPoolExecutor(max_workers=worker_count()) as pool:
        return list(pool.map(run, items))


def binary_fd(fd: int) -> None:
    """Stop Windows from expanding ``\\n`` to ``\\r\\n`` on this handle."""
    if os.name != "nt":
        return
    import msvcrt

    msvcrt.setmode(fd, os.O_BINARY)


def write_at(fd: int, data: bytes | memoryview, offset: int) -> None:
    """Write ``data`` at ``offset``. Uses ``pwrite`` where the OS has it."""
    binary_fd(fd)
    pwrite = getattr(os, "pwrite", None)
    if pwrite is not None:
        pwrite(fd, data, offset)
        return
    with _IO:
        os.lseek(fd, offset, os.SEEK_SET)
        view = memoryview(data)
        while view:
            wrote = os.write(fd, view)
            if wrote <= 0:
                message = f"short write at {offset}"
                raise OSError(message)
            view = view[wrote:]


def read_at(fd: int, nbytes: int, offset: int) -> bytes:
    """Read ``nbytes`` at ``offset``. Uses ``pread`` where the OS has it."""
    binary_fd(fd)
    pread = getattr(os, "pread", None)
    if pread is not None:
        return pread(fd, nbytes, offset)
    with _IO:
        os.lseek(fd, offset, os.SEEK_SET)
        chunks: list[bytes] = []
        left = nbytes
        while left:
            got = os.read(fd, left)
            if not got:
                break
            chunks.append(got)
            left -= len(got)
        return b"".join(chunks)


def bf16_to_f32(raw: np.ndarray) -> np.ndarray:
    """BF16, FP16, or FP32 values as float32."""
    if raw.dtype == np.float32:
        return np.ascontiguousarray(raw)
    if raw.dtype == np.float16:
        return raw.astype(np.float32)
    view = np.ascontiguousarray(raw).view(np.uint16)
    shifted = view.astype(np.uint32) << np.uint32(16)
    return shifted.view(np.float32)


to_f32 = bf16_to_f32


def half_bytes(weight: np.ndarray) -> bytes:
    values = np.ascontiguousarray(weight, dtype=np.float32)
    return values.astype(np.float16).tobytes()


def biased_e8m0(amax: np.ndarray, qmax: float) -> tuple[np.ndarray, np.ndarray]:
    """``2^ceil(log2(amax / qmax))``. A zero block stores scale byte 0."""
    positive = amax > 0.0
    ratio = np.ones(amax.shape, dtype=np.float64)
    np.divide(amax, qmax, out=ratio, where=positive)
    power = np.ceil(np.log2(np.maximum(ratio, np.finfo(np.float64).tiny)))
    biased = np.clip(power + 127.0, 0, 254).astype(np.uint8)
    biased = np.where(positive, biased, np.uint8(0))
    scale = np.ldexp(np.ones(amax.shape, dtype=np.float32), biased.astype(np.int32) - 127)
    return biased, scale


def enc_e4m3(values: np.ndarray) -> np.ndarray:
    """OCP E4M3FN. Finite max is 448 (``0x7E``). ``0x7F`` is NaN."""
    original = np.ascontiguousarray(values, dtype=np.float32)
    bits = np.frombuffer(original.tobytes(), dtype=np.uint32).reshape(original.shape)
    sign = ((bits >> np.uint32(24)) & np.uint32(0x80)).astype(np.uint8)
    exp = ((bits >> np.uint32(23)) & np.uint32(0xFF)).astype(np.int32)
    mant = bits & np.uint32(0x7FFFFF)
    mag = np.abs(original)
    sub = np.minimum(np.rint(mag * np.float32(512.0)).astype(np.int32), 7).astype(np.uint8)
    out = sign.copy()
    out = np.where(mag >= np.float32(FP8_QMAX), sign | np.uint8(0x7E), out)
    tiny = (mag > np.float32(0.0)) & (mag < np.float32(FP8_QMAX)) & (
        (exp == 0) | (mag < np.float32(0.015625))
    )
    out = np.where(tiny, sign | sub, out)
    normal = (
        (mag >= np.float32(0.015625))
        & (mag < np.float32(FP8_QMAX))
        & (exp != 0)
        & (exp != 255)
    )
    top = (mant >> np.uint32(20)).astype(np.int32)
    rem = mant & np.uint32(0xFFFFF)
    round_up = (rem > np.uint32(0x80000)) | ((rem == np.uint32(0x80000)) & ((top & 1) != 0))
    top = top + round_up.astype(np.int32)
    biased = exp - 127 + 7
    carry = top >= 8
    top = np.where(carry, 0, top)
    biased = np.where(carry, biased + 1, biased)
    code = (sign.astype(np.int32) | ((biased & 15) << 3) | (top & 7)).astype(np.uint8)
    code = np.where(biased > 15, sign | np.uint8(0x7E), code)
    code = np.where((biased == 15) & (top > 6), sign | np.uint8(0x7E), code)
    code = np.where(biased <= 0, sign | sub, code)
    return np.where(normal, code, out).astype(np.uint8)


def quantize_fp8(weight: np.ndarray) -> tuple[bytes, bytes]:
    """Pack ``[rows, inner]`` float32. ``inner`` must be a multiple of 128."""
    matrix = np.ascontiguousarray(weight, dtype=np.float32)
    rows, inner = matrix.shape
    if inner % 128:
        message = f"inner dimension {inner} is not a multiple of 128"
        raise ValueError(message)
    values = matrix.reshape(rows, inner // FP8_BLOCK, FP8_BLOCK)
    amax = np.max(np.abs(values), axis=-1)
    biased, _scale = biased_e8m0(amax, FP8_QMAX)
    scaled = (values / _scale[..., None]).reshape(rows, inner)
    codes = np.ascontiguousarray(enc_e4m3(scaled)).tobytes()
    scales = np.ascontiguousarray(biased).tobytes()
    return codes, scales


def quantize_int8(weight: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Signed int8 codes and one fp32 scale per row. qmax is 127."""
    values = np.ascontiguousarray(weight, dtype=np.float32)
    if values.ndim != 2:
        message = "int8 weight must be a matrix"
        raise ValueError(message)
    scale = _row_scale(values, INT8_QMAX)
    codes = _codes(values, scale, -127, 127).astype(np.int8)
    return codes, scale


def quantize_int4(weight: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Packed int4 codes, shape ``(rows, inner // 2)``, and one fp32 scale per row."""
    values = np.ascontiguousarray(weight, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] % 2:
        message = "int4 weight must be a matrix with an even inner dimension"
        raise ValueError(message)
    scale = _row_scale(values, INT4_QMAX)
    signed = _codes(values, scale, -7, 7).astype(np.int8)
    low = signed[:, 0::2] & np.int8(0x0F)
    high = signed[:, 1::2] & np.int8(0x0F)
    packed = (low | (high << np.int8(4))).view(np.uint8)
    return np.ascontiguousarray(packed), scale


def _row_scale(values: np.ndarray, qmax: float) -> np.ndarray:
    amax = np.max(np.abs(values), axis=1)
    scale = np.zeros(values.shape[0], dtype=np.float32)
    positive = amax > 0.0
    scale[positive] = amax[positive] / np.float32(qmax)
    return scale


def _codes(values: np.ndarray, scale: np.ndarray, low: int, high: int) -> np.ndarray:
    safe = np.where(scale == 0.0, np.float32(1.0), scale)
    scaled = values / safe[:, None]
    return np.clip(np.rint(scaled), low, high)
