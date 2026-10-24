"""Compile this package's shader library and dispatch the formula checks."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import objc

from favlo.memory import memory
from zonos2.checks import rms_norm, rope_pairs, softcap, swiglu
from zonos2.codec import check_codec
from zonos2.layout import NORM_EPS, ROPE_THETA, SOFTCAP
from zonos2.session import check_gemms

_SOURCE = Path(__file__).resolve().parent / "shaders" / "kernels.metal"
_LANGUAGE = (4 << 16) + 1


def _buffer(device, array: np.ndarray, owner: str):
    del device
    return memory.upload(np.ascontiguousarray(array), owner=owner)


def _scalar(device, value, dtype: np.dtype, owner: str):
    return _buffer(device, np.array([value], dtype=dtype), owner)


def _read(buffer, count: int, dtype: np.dtype) -> np.ndarray:
    item = np.dtype(dtype).itemsize
    view = buffer.contents().as_buffer(count * item)
    return np.frombuffer(view, dtype=dtype).copy()


def compile_library(device=None):
    import Metal

    chosen = device or memory.device
    if chosen is None:
        message = "this runtime cannot run here"
        raise RuntimeError(message)
    options = Metal.MTLCompileOptions.alloc().init()
    options.setLanguageVersion_(_LANGUAGE)
    library, error = chosen.newLibraryWithSource_options_error_(
        _SOURCE.read_text(encoding="utf-8"), options, None
    )
    if library is None:
        message = str(error) if error is not None else "shader library failed"
        raise RuntimeError(message)
    names = ("rms_norm", "rope_pairs", "swiglu", "softcap")
    pipelines = {}
    for name in names:
        function = library.newFunctionWithName_(name)
        if function is None:
            message = f"{name} is missing from the shader library"
            raise RuntimeError(message)
        pipeline, error = chosen.newComputePipelineStateWithFunction_error_(function, None)
        if pipeline is None:
            message = str(error) if error is not None else f"{name} did not compile"
            raise RuntimeError(message)
        pipelines[name] = pipeline
    return chosen, pipelines


def _encode(queue, pipeline, buffers, grid: tuple[int, int, int], group: tuple[int, int, int], out, count: int):
    import Metal

    with objc.autorelease_pool():
        command = memory.command_buffer(queue)
        encoder = command.computeCommandEncoder()
        encoder.setComputePipelineState_(pipeline)
        for index, buffer in enumerate(buffers):
            memory.bind(encoder, index, buffer, 0)
        encoder.dispatchThreads_threadsPerThreadgroup_(
            Metal.MTLSize(*grid), Metal.MTLSize(*group)
        )
        encoder.endEncoding()
        command.commit()
        command.waitUntilCompleted()
        memory.retire(command)
        return _read(out, count, np.float32)


def run_checks() -> None:
    """Dispatch each formula kernel and compare it with the host formula."""
    holder = object()
    owner = f"zonos2:{id(holder)}"
    try:
        device, pipelines = compile_library()
        queue = memory.make_queue()
        values = np.linspace(-1.5, 2.0, 32, dtype=np.float32)
        weight = np.linspace(0.5, 1.5, 32, dtype=np.float32)
        out = _buffer(device, np.zeros_like(values), owner)
        got = _encode(
            queue,
            pipelines["rms_norm"],
            (
                _buffer(device, values, owner),
                _buffer(device, weight, owner),
                out,
                _scalar(device, np.uint32(values.size), np.uint32, owner),
                _scalar(device, np.float32(NORM_EPS), np.float32, owner),
            ),
            (32, 1, 1),
            (32, 1, 1),
            out,
            values.size,
        )
        _close(got, rms_norm(values, weight), "rms_norm")
        rotated = _buffer(device, np.zeros_like(values), owner)
        got = _encode(
            queue,
            pipelines["rope_pairs"],
            (
                _buffer(device, values, owner),
                rotated,
                _scalar(device, np.uint32(values.size), np.uint32, owner),
                _scalar(device, np.uint32(3), np.uint32, owner),
                _scalar(device, np.float32(ROPE_THETA), np.float32, owner),
            ),
            (values.size // 2, 1, 1),
            (values.size // 2, 1, 1),
            rotated,
            values.size,
        )
        _close(got, rope_pairs(values, 3), "rope_pairs")
        gate = np.linspace(-2.0, 2.0, 16, dtype=np.float32)
        up = np.linspace(0.25, 1.25, 16, dtype=np.float32)
        mixed = _buffer(device, np.zeros_like(gate), owner)
        got = _encode(
            queue,
            pipelines["swiglu"],
            (_buffer(device, gate, owner), _buffer(device, up, owner), mixed),
            (gate.size, 1, 1),
            (gate.size, 1, 1),
            mixed,
            gate.size,
        )
        _close(got, swiglu(gate, up), "swiglu")
        capped = _buffer(device, np.zeros_like(values), owner)
        got = _encode(
            queue,
            pipelines["softcap"],
            (
                _buffer(device, values, owner),
                capped,
                _scalar(device, np.float32(SOFTCAP), np.float32, owner),
            ),
            (values.size, 1, 1),
            (32, 1, 1),
            capped,
            values.size,
        )
        _close(got, softcap(values), "softcap")
        check_gemms()
        check_codec()
    finally:
        memory.sweep(owner)


def _close(got: np.ndarray, expected: np.ndarray, name: str) -> None:
    if not np.allclose(got, expected, rtol=1e-5, atol=1e-5):
        message = f"{name} disagrees with the host formula"
        raise RuntimeError(message)
