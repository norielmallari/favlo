"""Metal int8 encoder. One sequence, one weight image."""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Any

import Metal
import numpy as np
import objc

from bge_m3.pack import TILE_M, TILE_N
from bge_m3.text import HEADS, HIDDEN, INTERMEDIATE, LIMIT
from favlo.memory import memory

_SHADER = Path(__file__).resolve().parent / "shaders" / "kernels.metal"
_NAMES = (
    "quantize_i8",
    "matmul_i8",
    "scale_i8",
    "embed_f32",
    "layernorm_affine",
    "residual_norm",
    "gelu_erf",
    "attend_head",
    "copy_f32",
)
_EPS = np.float32(1e-5)
_LAYERS = 24
_PROJECTIONS = ("q", "k", "v", "o", "up", "down")


def workspace_bytes(length: int) -> int:
    """Scratch for one sequence of ``length`` rows, padded to the matmul tile."""
    rows = int(length)
    narrow = rows * HIDDEN * 4
    wide = rows * INTERMEDIATE * 4
    return rows * 4 * 4 + narrow * 7 + wide + rows * INTERMEDIATE + wide


class Gpu:
    """Compiled kernels and the int8 forward."""

    def __init__(self, owner: str) -> None:
        self.owner = owner
        self.device = memory.device
        self.queue = memory.make_queue()
        self.pipes: dict[str, Any] = {}
        with objc.autorelease_pool():
            source = _SHADER.read_text(encoding="utf-8")
            options = Metal.MTLCompileOptions.alloc().init()
            options.setLanguageVersion_((4 << 16) + 1)
            library, error = self.device.newLibraryWithSource_options_error_(source, options, None)
            if library is None:
                message = f"shader compile failed: {error}"
                raise RuntimeError(message)
            for name in _NAMES:
                fn = library.newFunctionWithName_(name)
                if fn is None:
                    message = f"shader is missing {name}"
                    raise RuntimeError(message)
                pipe, pipe_error = self.device.newComputePipelineStateWithFunction_error_(fn, None)
                if pipe is None:
                    message = f"pipeline {name} failed: {pipe_error}"
                    raise RuntimeError(message)
                self.pipes[name] = pipe
        width = int(self.pipes["matmul_i8"].threadExecutionWidth())
        self._matmul_threads = (width * 4, 1, 1)
        self._cmd: Any = None
        self._enc: Any = None
        self._held: list[bytes] = []
        self._inflight: list[Any] = []
        self._handles: list[Any] = []

    def close(self) -> None:
        self.end()
        for handle in self._handles:
            handle.release()
        self._handles.clear()
        memory.sweep(self.owner)

    def end(self) -> None:
        if self._enc is not None:
            memory.handoff(self._enc, self._cmd)
            self._enc.endEncoding()
            self._cmd.commit()
            self._inflight.append(self._cmd)
            self._enc = None
            self._cmd = None
            self._held.clear()
        pending = self._inflight
        self._inflight = []
        for cmd in pending:
            status, error = memory.retire(cmd)
            if status == int(Metal.MTLCommandBufferStatusError):
                message = f"metal command failed: {error}"
                raise RuntimeError(message)

    def upload(self, array: np.ndarray) -> Any:
        raw = np.ascontiguousarray(array)
        handle = memory.allocate(max(raw.nbytes, 16), owner=self.owner, storage="shared")
        self._handles.append(handle)
        view = np.frombuffer(memory.map(handle, 0, raw.nbytes), dtype=raw.dtype)
        view[...] = raw.reshape(-1)
        return handle

    def scratch(self, nbytes: int, *, zero: bool = False) -> Any:
        handle = memory.allocate(
            max(int(nbytes), 16), owner=self.owner, storage="shared", zero=zero
        )
        self._handles.append(handle)
        return handle

    def encode(
        self,
        weights: dict[str, Any],
        ids: np.ndarray,
        positions: np.ndarray,
        mask: np.ndarray,
        work: dict[str, Any],
    ) -> np.ndarray:
        """CLS row, float32, before the L2 norm. ``ids`` is the padded length."""
        rows = int(ids.shape[0])
        self._copy_array(work["ids"], ids.astype(np.int32, copy=False))
        self._copy_array(work["pos"], positions.astype(np.int32, copy=False))
        self._copy_array(work["mask"], mask.astype(np.float32, copy=False))
        self._span(
            "embed_f32",
            [
                weights["word"],
                weights["position"],
                weights["type"],
                work["ids"],
                work["pos"],
                work["cur"],
            ],
            [struct.pack("I", rows), struct.pack("I", HIDDEN)],
            (rows * HIDDEN + 255) // 256,
        )
        self._norm(weights["embed_w"], weights["embed_b"], work["cur"], work["mix"], rows)
        self._copy(work["mix"], work["cur"], rows * HIDDEN)
        for layer in range(_LAYERS):
            pack = weights["layers"][layer]
            self._project(work["cur"], rows, HIDDEN, pack["q"], work["q"])
            self._project(work["cur"], rows, HIDDEN, pack["k"], work["k"])
            self._project(work["cur"], rows, HIDDEN, pack["v"], work["v"])
            self._attend(work, rows)
            self._project(work["attn"], rows, HIDDEN, pack["o"], work["mix"])
            self._residual(
                work["cur"], work["mix"], pack["attn_w"], pack["attn_b"], work["nxt"], rows, HIDDEN
            )
            self._project(work["nxt"], rows, HIDDEN, pack["up"], work["wide"])
            self._span(
                "gelu_erf",
                [work["wide"]],
                [struct.pack("I", rows * INTERMEDIATE)],
                (rows * INTERMEDIATE + 255) // 256,
            )
            self._project(work["wide"], rows, INTERMEDIATE, pack["down"], work["mix"])
            self._residual(
                work["nxt"], work["mix"], pack["ffn_w"], pack["ffn_b"], work["cur"], rows, HIDDEN
            )
        self.end()
        return self._read(work["cur"], (HIDDEN,), np.float32)

    def _project(
        self,
        src: Any,
        rows: int,
        k: int,
        linear: dict[str, Any],
        dest: Any,
    ) -> None:
        n = int(linear["n"])
        self._dispatch(
            "quantize_i8",
            [src, self._codes, self._row_scale],
            [
                struct.pack("I", rows),
                struct.pack("I", k),
                struct.pack("I", k),
                struct.pack("I", k),
            ],
            (rows, 1, 1),
            (128, 1, 1),
        )
        groups = (n // TILE_N, rows // TILE_M, 1)
        self._dispatch(
            "matmul_i8",
            [self._codes, linear["codes"], self._acc],
            [struct.pack("I", rows), struct.pack("I", n), struct.pack("I", k)],
            groups,
            self._matmul_threads,
        )
        count = rows * n
        self._dispatch(
            "scale_i8",
            [self._acc, self._row_scale, linear["scale"], linear["bias"], dest],
            [struct.pack("I", rows), struct.pack("I", n), struct.pack("I", n)],
            ((count + 255) // 256, 1, 1),
            (256, 1, 1),
        )

    def bind_workspace(self, work: dict[str, Any]) -> None:
        self._codes = work["codes"]
        self._row_scale = work["row_scale"]
        self._acc = work["acc"]

    def _attend(self, work: dict[str, Any], rows: int) -> None:
        self._dispatch(
            "attend_head",
            [work["q"], work["k"], work["v"], work["mask"], work["attn"]],
            [struct.pack("I", rows), struct.pack("I", HIDDEN)],
            (rows, HEADS, 1),
            (32, 1, 1),
        )

    def _norm(self, weight: Any, bias: Any, src: Any, dest: Any, rows: int) -> None:
        self._dispatch(
            "layernorm_affine",
            [src, weight, bias, dest],
            [struct.pack("I", rows), struct.pack("I", HIDDEN), struct.pack("<f", float(_EPS))],
            (rows, 1, 1),
            (128, 1, 1),
        )

    def _residual(
        self,
        left: Any,
        right: Any,
        weight: Any,
        bias: Any,
        dest: Any,
        rows: int,
        width: int,
    ) -> None:
        self._dispatch(
            "residual_norm",
            [left, right, weight, bias, dest],
            [
                struct.pack("I", rows),
                struct.pack("I", width),
                struct.pack("I", width),
                struct.pack("<f", float(_EPS)),
            ],
            (rows, 1, 1),
            (128, 1, 1),
        )

    def _copy(self, src: Any, dest: Any, count: int) -> None:
        self._span("copy_f32", [src, dest], [struct.pack("I", count)], (count + 255) // 256)

    def _span(self, name: str, buffers: list[Any], constants: list[bytes], groups: int) -> None:
        self._dispatch(name, buffers, constants, (groups, 1, 1), (256, 1, 1))

    def _dispatch(
        self,
        name: str,
        buffers: list[Any],
        constants: list[bytes],
        grid: tuple[int, int, int],
        threads: tuple[int, int, int],
    ) -> None:
        if self._enc is None:
            self._cmd = memory.command_buffer(self.queue)
            self._enc = self._cmd.computeCommandEncoder()
        enc = self._enc
        enc.setComputePipelineState_(self.pipes[name])
        for index, buf in enumerate(buffers):
            memory.bind(enc, index, buf, 0)
        for index, blob in enumerate(constants):
            self._held.append(blob)
            enc.setBytes_length_atIndex_(blob, len(blob), len(buffers) + index)
        enc.dispatchThreadgroups_threadsPerThreadgroup_(
            Metal.MTLSizeMake(int(grid[0]), int(grid[1]), int(grid[2])),
            Metal.MTLSizeMake(int(threads[0]), int(threads[1]), int(threads[2])),
        )

    def _copy_array(self, handle: Any, array: np.ndarray) -> None:
        raw = np.ascontiguousarray(array)
        view = np.frombuffer(memory.map(handle, 0, raw.nbytes), dtype=raw.dtype)
        view[...] = raw.reshape(-1)

    def _read(self, handle: Any, shape: tuple[int, ...], dtype: Any) -> np.ndarray:
        count = int(np.prod(shape))
        view = np.frombuffer(memory.map(handle, 0, count * np.dtype(dtype).itemsize), dtype=dtype)
        return np.array(view[:count], copy=True).reshape(shape)


def check_kernels() -> None:
    """One int8 tile and one attention head against the host formula."""
    gpu = Gpu("bge-m3-kernels")
    try:
        _check_matmul(gpu)
        _check_attention(gpu)
    finally:
        gpu.close()
    print("kernels ok", flush=True)


def _check_matmul(gpu: Gpu) -> None:
    rng = np.random.default_rng(0)
    rows, k, n = TILE_M, 256, TILE_N
    activation = rng.standard_normal((rows, k)).astype(np.float32)
    weight = rng.standard_normal((n, k)).astype(np.float32) * 0.02
    from bge_m3.pack import pack_int8

    codes, scale = pack_int8(weight)
    bias = rng.standard_normal(n).astype(np.float16)
    expected = _host_project(activation, codes, scale, bias)
    src = gpu.upload(activation)
    packed = gpu.upload(codes)
    factors = gpu.upload(scale)
    bias_h = gpu.upload(bias)
    dest = gpu.scratch(rows * n * 4)
    codes_b = gpu.scratch(rows * k)
    row_scale = gpu.scratch(rows * 4)
    acc = gpu.scratch(rows * n * 4)
    gpu._codes = codes_b
    gpu._row_scale = row_scale
    gpu._acc = acc
    gpu._project(src, rows, k, {"n": n, "codes": packed, "scale": factors, "bias": bias_h}, dest)
    gpu.end()
    got = gpu._read(dest, (rows, n), np.float32)
    if float(np.max(np.abs(got - expected))) > 1e-3:
        message = "int8 matmul disagrees with the host product"
        raise RuntimeError(message)


def _host_project(
    activation: np.ndarray, codes: np.ndarray, scale: np.ndarray, bias: np.ndarray
) -> np.ndarray:
    rows, k = activation.shape
    act_codes = np.zeros((rows, k), dtype=np.int32)
    act_scale = np.zeros(rows, dtype=np.float32)
    for row in range(rows):
        peak = float(np.max(np.abs(activation[row])))
        if peak == 0.0:
            continue
        act_scale[row] = peak / 127.0
        act_codes[row] = np.clip(np.rint(activation[row] * (127.0 / peak)), -127, 127)
    product = act_codes.astype(np.int32) @ codes.astype(np.int32).T
    return product.astype(np.float32) * act_scale[:, None] * scale[None, :] + bias.astype(
        np.float32
    )


def _check_attention(gpu: Gpu) -> None:
    rng = np.random.default_rng(1)
    rows = 4
    query = rng.standard_normal((rows, HIDDEN)).astype(np.float32)
    key = rng.standard_normal((rows, HIDDEN)).astype(np.float32)
    value = rng.standard_normal((rows, HIDDEN)).astype(np.float32)
    mask = np.zeros(rows, dtype=np.float32)
    mask[-1] = -40000.0
    dest = gpu.scratch(rows * HIDDEN * 4, zero=True)
    gpu._dispatch(
        "attend_head",
        [gpu.upload(query), gpu.upload(key), gpu.upload(value), gpu.upload(mask), dest],
        [struct.pack("I", rows), struct.pack("I", HIDDEN)],
        (rows, HEADS, 1),
        (32, 1, 1),
    )
    gpu.end()
    got = gpu._read(dest, (rows, HIDDEN), np.float32)
    expected = _host_attention(query, key, value, mask)
    if float(np.max(np.abs(got - expected))) > 1e-3:
        message = "attention disagrees with the host softmax"
        raise RuntimeError(message)


def _host_attention(
    query: np.ndarray, key: np.ndarray, value: np.ndarray, mask: np.ndarray
) -> np.ndarray:
    out = np.zeros_like(query)
    scale = np.float32(0.125)
    for head in range(HEADS):
        start = head * 64
        stop = start + 64
        scores = query[:, start:stop] @ key[:, start:stop].T
        scores = scores * scale + mask[None, :]
        scores = scores - np.max(scores, axis=1, keepdims=True)
        weights = np.exp(scores)
        weights /= np.sum(weights, axis=1, keepdims=True)
        out[:, start:stop] = weights @ value[:, start:stop]
    return out


def fit_length(resident: int, cap: int | None) -> int:
    """Longest tile-aligned window whose scratch fits. 64 is the minimum."""
    ceiling = memory.budget()
    if cap is not None:
        ceiling = min(ceiling, cap)
    room = max(0, ceiling - memory.live - memory.pending)
    length = LIMIT
    while length >= TILE_M:
        if resident + workspace_bytes(length) <= room:
            return length
        length //= 2
    message = "int8 embeddings do not fit"
    raise RuntimeError(message)
