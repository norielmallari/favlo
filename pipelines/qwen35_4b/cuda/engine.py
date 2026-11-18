"""Load the language-step library and check each call."""

from __future__ import annotations

import ctypes
from typing import Any

import numpy as np

_LIB: ctypes.CDLL | None = None


def _checks() -> Any:
    import importlib.util
    import sys
    from pathlib import Path

    path = Path(__file__).resolve().parent / "checks.py"
    name = "qwen35_cuda_checks"
    module = sys.modules.get(name)
    if module is not None:
        return module
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        message = f"cannot load {path}"
        raise ImportError(message)
    loaded = importlib.util.module_from_spec(spec)
    sys.modules[name] = loaded
    spec.loader.exec_module(loaded)
    return loaded


def library() -> ctypes.CDLL:
    """The shipped engine. The driver supplies the CUDA runtime. This does not compile."""
    global _LIB
    if _LIB is not None:
        return _LIB
    lib = ctypes.CDLL(str(_checks().shipped_engine()))
    _bind(lib)
    _LIB = lib
    return lib


def _bind(lib: ctypes.CDLL) -> None:
    void = ctypes.c_void_p
    integer = ctypes.c_int
    number = ctypes.c_float
    size = ctypes.c_size_t
    lib.engine_open.restype = integer
    lib.engine_open.argtypes = []
    lib.engine_close.restype = None
    lib.engine_close.argtypes = []
    lib.engine_alloc.restype = void
    lib.engine_alloc.argtypes = [size]
    lib.engine_free.restype = None
    lib.engine_free.argtypes = [void]
    lib.engine_on_device.restype = integer
    lib.engine_on_device.argtypes = [void]
    for name in ("engine_h2d", "engine_d2h"):
        fn = getattr(lib, name)
        fn.restype = integer
        fn.argtypes = [void, void, size]
    lib.engine_zero.restype = integer
    lib.engine_zero.argtypes = [void, size]
    lib.engine_gemv.restype = integer
    lib.engine_gemv.argtypes = [void, void, void, void, integer, integer]
    lib.engine_gemm.restype = integer
    lib.engine_gemm.argtypes = [void, void, void, void, integer, integer, integer]
    lib.engine_gemm_pair.restype = integer
    lib.engine_gemm_pair.argtypes = [
        void,
        void,
        void,
        integer,
        void,
        void,
        void,
        integer,
        void,
        integer,
        integer,
    ]
    lib.engine_embed.restype = integer
    lib.engine_embed.argtypes = [void, void, integer, void, integer]
    lib.engine_rmsnorm.restype = integer
    lib.engine_rmsnorm.argtypes = [void, void, void, integer, number, number, integer]
    lib.engine_add.restype = integer
    lib.engine_add.argtypes = [void, void, integer, integer]
    lib.engine_silu_mul.restype = integer
    lib.engine_silu_mul.argtypes = [void, void, void, integer, integer]
    lib.engine_rope.restype = integer
    lib.engine_rope.argtypes = [void, integer, integer, integer, integer, number, integer]
    lib.engine_head_norm.restype = integer
    lib.engine_head_norm.argtypes = [void, void, integer, integer, integer, number, number, integer]
    lib.engine_conv_silu.restype = integer
    lib.engine_conv_silu.argtypes = [void, void, void, integer, integer]
    lib.engine_gdn.restype = integer
    lib.engine_gdn.argtypes = [void, void, void, void, void, void, void, integer, integer, integer]
    lib.engine_gated_norm.restype = integer
    lib.engine_gated_norm.argtypes = [void, void, void, integer, integer, number, integer]
    lib.engine_store_kv.restype = integer
    lib.engine_store_kv.argtypes = [void, void, void, integer, integer, integer, integer]
    lib.engine_attend.restype = integer
    lib.engine_attend.argtypes = [
        void,
        void,
        void,
        void,
        void,
        void,
        integer,
        integer,
        integer,
        integer,
        number,
        integer,
    ]
    lib.engine_sample.restype = integer
    lib.engine_sample.argtypes = [
        void,
        void,
        void,
        integer,
        integer,
        number,
        number,
        number,
        number,
        number,
    ]
    lib.engine_sync.restype = integer
    lib.engine_sync.argtypes = []
    lib.engine_reserve.restype = integer
    lib.engine_reserve.argtypes = [integer, integer]
    lib.engine_set_step.restype = integer
    lib.engine_set_step.argtypes = [integer, integer, integer]
    lib.engine_capture_begin.restype = integer
    lib.engine_capture_begin.argtypes = [integer]
    lib.engine_capture_end.restype = integer
    lib.engine_capture_end.argtypes = [integer]
    lib.engine_replay.restype = integer
    lib.engine_replay.argtypes = [integer]


class Engine:
    """One device and the scratch the int8 product uses."""

    def __init__(self) -> None:
        self.lib = library()
        status = self.lib.engine_open()
        if status == 1001:
            message = "qwen35_4b cuda needs an RTX 30 or later (compute capability 8.6)"
            raise RuntimeError(message)
        self._check("engine_open", status)

    def close(self) -> None:
        self.lib.engine_close()

    def alloc(self, nbytes: int) -> int:
        pointer = self.lib.engine_alloc(nbytes)
        if not pointer:
            message = f"cuda alloc failed for {nbytes} bytes"
            raise MemoryError(message)
        return int(pointer)

    def free(self, pointer: int) -> None:
        if pointer:
            self.lib.engine_free(pointer)

    def on_device(self, pointer: int) -> bool:
        """True when the pointer is dedicated device memory and has no host mapping."""
        return int(self.lib.engine_on_device(pointer)) == 1

    def h2d(self, destination: int, host: np.ndarray) -> None:
        raw = np.ascontiguousarray(host)
        self._check("engine_h2d", self.lib.engine_h2d(destination, raw.ctypes.data, raw.nbytes))

    def d2h(self, host: np.ndarray, source: int) -> None:
        raw = np.ascontiguousarray(host)
        if raw.ctypes.data != host.ctypes.data:
            message = "device read needs a contiguous buffer"
            raise ValueError(message)
        self._check("engine_d2h", self.lib.engine_d2h(raw.ctypes.data, source, raw.nbytes))

    def zero(self, pointer: int, nbytes: int) -> None:
        self._check("engine_zero", self.lib.engine_zero(pointer, nbytes))

    def gemv(
        self, weight: int, scale: int, source: int, destination: int, rows: int, inner: int
    ) -> None:
        self._check(
            "engine_gemv",
            self.lib.engine_gemv(weight, scale, source, destination, rows, inner),
        )

    def gemm(
        self,
        weight: int,
        scale: int,
        source: int,
        destination: int,
        rows: int,
        inner: int,
        tokens: int,
    ) -> None:
        self._check(
            "engine_gemm",
            self.lib.engine_gemm(weight, scale, source, destination, rows, inner, tokens),
        )

    def gemm_pair(
        self,
        weight_a: int,
        scale_a: int,
        destination_a: int,
        rows_a: int,
        weight_b: int,
        scale_b: int,
        destination_b: int,
        rows_b: int,
        source: int,
        inner: int,
        tokens: int,
    ) -> None:
        """Two products of one activation. The second product uses the side stream."""
        self._check(
            "engine_gemm_pair",
            self.lib.engine_gemm_pair(
                weight_a,
                scale_a,
                destination_a,
                rows_a,
                weight_b,
                scale_b,
                destination_b,
                rows_b,
                source,
                inner,
                tokens,
            ),
        )

    def embed(self, table: int, scale: int, token: int, destination: int, inner: int) -> None:
        self._check("engine_embed", self.lib.engine_embed(table, scale, token, destination, inner))

    def rmsnorm(
        self,
        source: int,
        gain: int,
        destination: int,
        width: int,
        eps: float,
        shift: float,
        count: int = 1,
    ) -> None:
        self._check(
            "engine_rmsnorm",
            self.lib.engine_rmsnorm(source, gain, destination, width, eps, shift, count),
        )

    def add(self, stream: int, delta: int, width: int, count: int = 1) -> None:
        self._check("engine_add", self.lib.engine_add(stream, delta, width, count))

    def silu_mul(
        self, gate: int, up: int, destination: int, width: int, count: int = 1
    ) -> None:
        self._check(
            "engine_silu_mul", self.lib.engine_silu_mul(gate, up, destination, width, count)
        )

    def rope(
        self,
        data: int,
        heads: int,
        stride: int,
        rope_dim: int,
        position: int,
        theta: float,
        count: int = 1,
    ) -> None:
        self._check(
            "engine_rope",
            self.lib.engine_rope(data, heads, stride, rope_dim, position, theta, count),
        )

    def head_norm(
        self,
        data: int,
        gain: int,
        heads: int,
        stride: int,
        dim: int,
        eps: float,
        shift: float,
        count: int = 1,
    ) -> None:
        self._check(
            "engine_head_norm",
            self.lib.engine_head_norm(data, gain, heads, stride, dim, eps, shift, count),
        )

    def conv_silu(
        self, taps: int, mixed: int, history: int, channels: int, count: int = 1
    ) -> None:
        self._check(
            "engine_conv_silu", self.lib.engine_conv_silu(taps, mixed, history, channels, count)
        )

    def gdn(
        self,
        state: int,
        mixed: int,
        proj_a: int,
        proj_b: int,
        a_log: int,
        dt_bias: int,
        output: int,
        key_heads: int,
        value_heads: int,
        count: int = 1,
    ) -> None:
        self._check(
            "engine_gdn",
            self.lib.engine_gdn(
                state,
                mixed,
                proj_a,
                proj_b,
                a_log,
                dt_bias,
                output,
                key_heads,
                value_heads,
                count,
            ),
        )

    def gated_norm(
        self, data: int, gate: int, gain: int, heads: int, dim: int, eps: float, count: int = 1
    ) -> None:
        self._check(
            "engine_gated_norm",
            self.lib.engine_gated_norm(data, gate, gain, heads, dim, eps, count),
        )

    def store_kv(
        self,
        source: int,
        data: int,
        scale: int,
        position: int,
        heads: int,
        dim: int,
        count: int = 1,
    ) -> None:
        self._check(
            "engine_store_kv",
            self.lib.engine_store_kv(source, data, scale, position, heads, dim, count),
        )

    def attend(
        self,
        query: int,
        key_data: int,
        key_scale: int,
        value_data: int,
        value_scale: int,
        output: int,
        length: int,
        heads: int,
        kv_heads: int,
        dim: int,
        scale: float,
        count: int = 1,
    ) -> None:
        self._check(
            "engine_attend",
            self.lib.engine_attend(
                query,
                key_data,
                key_scale,
                value_data,
                value_scale,
                output,
                length,
                heads,
                kv_heads,
                dim,
                scale,
                count,
            ),
        )

    def reserve(self, tokens: int, inner: int) -> None:
        self._check("engine_reserve", self.lib.engine_reserve(tokens, inner))

    def set_step(self, token: int, position: int, rope: int | None = None) -> None:
        origin = position if rope is None else rope
        self._check("engine_set_step", self.lib.engine_set_step(token, position, origin))

    def capture_begin(self, slot: int = 0) -> None:
        self._check("engine_capture_begin", self.lib.engine_capture_begin(slot))

    def capture_end(self, slot: int = 0) -> None:
        self._check("engine_capture_end", self.lib.engine_capture_end(slot))

    def replay(self, slot: int = 0) -> None:
        self._check("engine_replay", self.lib.engine_replay(slot))

    def sample(
        self,
        logits: int,
        counts: int,
        token_out: int,
        vocab: int,
        top_k: int,
        temperature: float,
        top_p: float,
        presence: float,
        frequency: float,
        draw: float,
    ) -> None:
        """Draw one token on device. ``top_k`` is 1 to 64. Temperature 0 is argmax."""
        self._check(
            "engine_sample",
            self.lib.engine_sample(
                logits,
                counts,
                token_out,
                vocab,
                top_k,
                temperature,
                top_p,
                presence,
                frequency,
                draw,
            ),
        )

    def sync(self) -> None:
        self._check("engine_sync", self.lib.engine_sync())

    @staticmethod
    def _check(name: str, status: int) -> None:
        if status:
            message = f"{name} cuda error {status}"
            raise RuntimeError(message)
