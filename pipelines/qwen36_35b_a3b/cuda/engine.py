"""Load the language-step library."""

from __future__ import annotations

import ctypes
import os
from pathlib import Path
from typing import Any

import numpy as np

_LIB: ctypes.CDLL | None = None


def _checks() -> Any:
    import importlib.util
    import sys
    from pathlib import Path

    path = Path(__file__).resolve().parent / "checks.py"
    name = "qwen36_cuda_checks"
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
    """The compiled step. Windows loads the engine.dll in this checkout."""
    global _LIB
    if _LIB is not None:
        return _LIB
    if os.name == "nt":
        path = Path(__file__).resolve().parent / "engine.dll"
        if not path.is_file():
            message = "engine.dll is missing from the checkout"
            raise RuntimeError(message)
    else:
        path = _checks().ensure_engine()
    lib = ctypes.CDLL(str(path))
    _bind(lib)
    code = lib.engine_open()
    if code != 0:
        message = f"CUDA device open failed ({code})"
        raise RuntimeError(message)
    _LIB = lib
    return lib


def _bind(lib: ctypes.CDLL) -> None:
    void = ctypes.c_void_p
    integer = ctypes.c_int
    number = ctypes.c_float
    size = ctypes.c_size_t
    unsigned = ctypes.c_uint
    lib.engine_open.restype = integer
    lib.engine_open.argtypes = []
    lib.engine_close.restype = None
    lib.engine_close.argtypes = []
    lib.engine_alloc.restype = void
    lib.engine_alloc.argtypes = [size]
    lib.engine_free.restype = None
    lib.engine_free.argtypes = [void]
    lib.engine_pin.restype = void
    lib.engine_pin.argtypes = [size]
    lib.engine_unpin.restype = None
    lib.engine_unpin.argtypes = [void]
    lib.engine_host_alloc.restype = void
    lib.engine_host_alloc.argtypes = [size]
    lib.engine_host_free.restype = None
    lib.engine_host_free.argtypes = [void]
    lib.engine_register.restype = integer
    lib.engine_register.argtypes = [void, size]
    lib.engine_register_ro.restype = integer
    lib.engine_register_ro.argtypes = [void, size]
    lib.engine_unregister.restype = None
    lib.engine_unregister.argtypes = [void]
    for name in ("engine_h2d", "engine_d2h"):
        fn = getattr(lib, name)
        fn.restype = integer
        fn.argtypes = [void, void, size]
    lib.engine_copy_async.restype = integer
    lib.engine_copy_async.argtypes = [void, void, size, ctypes.POINTER(integer)]
    lib.engine_wait_copy.restype = integer
    lib.engine_wait_copy.argtypes = [integer]
    lib.engine_zero.restype = integer
    lib.engine_zero.argtypes = [void, size]
    lib.engine_sync.restype = integer
    lib.engine_sync.argtypes = []
    lib.engine_mem_info.restype = integer
    lib.engine_mem_info.argtypes = [ctypes.POINTER(size), ctypes.POINTER(size)]
    lib.engine_quant.restype = integer
    lib.engine_quant.argtypes = [void, void, void, integer, integer, integer]
    lib.engine_gemv.restype = integer
    lib.engine_gemv.argtypes = [void, void, void, void, void, integer, integer, integer, integer]
    lib.engine_gemv_f16.restype = integer
    lib.engine_gemv_f16.argtypes = [void, void, void, integer, integer]
    lib.engine_embed.restype = integer
    lib.engine_embed.argtypes = [void, void, integer, void, integer]
    lib.engine_rmsnorm.restype = integer
    lib.engine_rmsnorm.argtypes = [void, void, void, integer, number, integer]
    lib.engine_add.restype = integer
    lib.engine_add.argtypes = [void, void, integer]
    lib.engine_silu_mul.restype = integer
    lib.engine_silu_mul.argtypes = [void, void, void, integer]
    lib.engine_scale_add.restype = integer
    lib.engine_scale_add.argtypes = [void, void, number, integer]
    lib.engine_scale_add_at.restype = integer
    lib.engine_scale_add_at.argtypes = [void, void, void, integer]
    lib.engine_sig_add.restype = integer
    lib.engine_sig_add.argtypes = [void, void, void, integer]
    lib.engine_l2norm.restype = integer
    lib.engine_l2norm.argtypes = [void, integer, integer]
    lib.engine_l2norm_tokens.restype = integer
    lib.engine_l2norm_tokens.argtypes = [void, integer, integer, integer, integer]
    lib.engine_repeat2.restype = integer
    lib.engine_repeat2.argtypes = [void, void, integer, integer]
    lib.engine_repeat2_tokens.restype = integer
    lib.engine_repeat2_tokens.argtypes = [void, void, integer, integer, integer, integer, integer]
    lib.engine_conv_silu.restype = integer
    lib.engine_conv_silu.argtypes = [void, void, void, void, integer, integer]
    lib.engine_gdn.restype = integer
    lib.engine_gdn.argtypes = [
        void, void, void, void, void, void, void, void, void, void, void, number, integer, integer,
            integer,
    ]
    lib.engine_split_q.restype = integer
    lib.engine_split_q.argtypes = [void, void, void, integer]
    lib.engine_rope.restype = integer
    lib.engine_rope.argtypes = [void, integer, void, integer, integer, number, integer]
    lib.engine_store_kv.restype = integer
    lib.engine_store_kv.argtypes = [void, void, integer, integer, integer, integer, integer,
        integer]
    for name in ("engine_gemm", "engine_reserve", "engine_set_step", "engine_capture_begin",
        "engine_capture_end", "engine_replay"):
        fn = getattr(lib, name)
        fn.restype = integer
    lib.engine_reserve.argtypes = [integer, integer]
    lib.engine_gemm.argtypes = [void, void, void, void, integer, integer, integer]
    lib.engine_gemm_i4.restype = integer
    lib.engine_gemm_i4.argtypes = [void, void, void, void, integer, integer, integer]
    lib.engine_gather_rows.restype = integer
    lib.engine_gather_rows.argtypes = [void, void, void, integer, integer]
    lib.engine_silu_rows.restype = integer
    lib.engine_silu_rows.argtypes = [void, void, integer, integer, integer]
    lib.engine_scatter_scale.restype = integer
    lib.engine_scatter_scale.argtypes = [void, void, void, void, void, integer, integer, integer]
    lib.engine_sig_rows.restype = integer
    lib.engine_sig_rows.argtypes = [void, void, void, integer, integer]
    lib.engine_gemv_f16_tokens.restype = integer
    lib.engine_gemv_f16_tokens.argtypes = [void, void, void, integer, integer, integer]
    lib.engine_topk_tokens.restype = integer
    lib.engine_topk_tokens.argtypes = [void, void, void, integer, integer, integer]
    lib.engine_grouped_mlp.restype = integer
    lib.engine_grouped_mlp.argtypes = [
        void, void, void, void, void, void, void, void, void, void, integer,
        void, void, void, void, void, void, void, void, void,
        integer, integer, integer, integer, integer,
    ]
    lib.engine_gemm_pair.restype = integer
    lib.engine_gemm_pair.argtypes = [
        void, void, void, integer, void, void, void, integer, void, integer, integer,
    ]
    lib.engine_attend_prefill.restype = integer
    lib.engine_attend_prefill.argtypes = [
        void, void, void, void, void, integer, integer, integer, number, integer,
    ]
    lib.engine_set_step.argtypes = [integer, integer, integer]
    lib.engine_capture_begin.argtypes = [integer]
    lib.engine_capture_end.argtypes = [integer]
    lib.engine_replay.argtypes = [integer]
    lib.engine_attend.restype = integer
    lib.engine_attend.argtypes = [void, void, void, void, integer, integer, integer, number]
    lib.engine_gate_mul.restype = integer
    lib.engine_gate_mul.argtypes = [void, void, integer]
    lib.engine_zero_f32.restype = integer
    lib.engine_zero_f32.argtypes = [void, integer]
    lib.engine_topk.restype = integer
    lib.engine_topk.argtypes = [void, void, void, integer, integer]
    lib.engine_argmax.restype = integer
    lib.engine_argmax.argtypes = [void, void, void, void, integer]
    lib.engine_sample.restype = integer
    lib.engine_sample.argtypes = [
        void, void, void, integer, number, number, integer, number, number, unsigned,
    ]
    lib.engine_count.restype = integer
    lib.engine_count.argtypes = [void, void]
    lib.engine_penalize.restype = integer
    lib.engine_penalize.argtypes = [void, void, number, number, integer]


class Engine:
    """One CUDA device and the language-step library."""

    def __init__(self) -> None:
        self.lib = library()

    def _check(self, code: int) -> None:
        if code != 0:
            message = f"CUDA call failed ({code})"
            raise RuntimeError(message)

    def alloc(self, nbytes: int) -> int:
        pointer = self.lib.engine_alloc(max(int(nbytes), 16))
        if not pointer:
            message = f"device allocation of {nbytes} bytes failed"
            raise RuntimeError(message)
        return int(pointer)

    def free(self, pointer: int) -> None:
        if pointer:
            self.lib.engine_free(pointer)

    def pin(self, nbytes: int) -> int:
        pointer = self.lib.engine_pin(int(nbytes))
        if not pointer:
            message = f"pinned allocation of {nbytes} bytes failed"
            raise RuntimeError(message)
        return int(pointer)

    def unpin(self, pointer: int) -> None:
        if pointer:
            self.lib.engine_unpin(pointer)

    def host_alloc(self, nbytes: int) -> int:
        """Write-combined pinned bytes. Zero when the driver refuses."""
        pointer = self.lib.engine_host_alloc(max(int(nbytes), 16))
        return int(pointer or 0)

    def host_free(self, pointer: int) -> None:
        if pointer:
            self.lib.engine_host_free(pointer)

    def register(self, pointer: int, nbytes: int) -> bool:
        """Pin existing host bytes. False when the driver refuses."""
        if not pointer or nbytes < 1:
            return False
        return int(self.lib.engine_register(pointer, int(nbytes))) == 0

    def register_ro(self, pointer: int, nbytes: int) -> bool:
        """Register host bytes the device only reads. False when the driver refuses."""
        if not pointer or nbytes < 1:
            return False
        return int(self.lib.engine_register_ro(pointer, int(nbytes))) == 0

    def unregister(self, pointer: int) -> None:
        if pointer:
            self.lib.engine_unregister(pointer)

    def zero(self, pointer: int, nbytes: int) -> None:
        self._check(self.lib.engine_zero(pointer, int(nbytes)))

    def sync(self) -> None:
        self._check(self.lib.engine_sync())

    def mem_info(self) -> tuple[int, int]:
        """``(free, total)`` from the runtime that owns ``alloc``."""
        free = ctypes.c_size_t()
        total = ctypes.c_size_t()
        self._check(self.lib.engine_mem_info(ctypes.byref(free), ctypes.byref(total)))
        return int(free.value), int(total.value)

    def h2d(self, pointer: int, host: np.ndarray | bytes) -> None:
        array = np.frombuffer(host, dtype=np.uint8) if isinstance(host, bytes | bytearray) else host
        array = np.ascontiguousarray(array)
        self._check(self.lib.engine_h2d(pointer, array.ctypes.data, array.nbytes))

    def h2d_ptr(self, destination: int, source: int, nbytes: int) -> None:
        self._check(self.lib.engine_h2d(destination, source, int(nbytes)))

    def copy_async(self, destination: int, source: int, nbytes: int) -> int:
        """Copy on the side stream. The returned event is waited on the compute stream."""
        event = ctypes.c_int()
        self._check(self.lib.engine_copy_async(destination, source, int(nbytes), ctypes.byref(event)))
        return int(event.value)

    def wait_copy(self, event: int) -> None:
        self._check(self.lib.engine_wait_copy(int(event)))

    def d2h(self, host: np.ndarray | ctypes.Array[Any], pointer: int) -> None:
        if isinstance(host, np.ndarray):
            array = np.ascontiguousarray(host)
            self._check(self.lib.engine_d2h(array.ctypes.data, pointer, array.nbytes))
            return
        self._check(self.lib.engine_d2h(ctypes.addressof(host), pointer, ctypes.sizeof(host)))

    def quant(self, source: int, codes: int, scales: int, inner: int, bits: int,
        tokens: int = 1) -> None:
        self._check(self.lib.engine_quant(source, codes, scales, inner, bits, tokens))

    def gemv(
        self,
        weight: int,
        scale: int,
        codes: int,
        ascales: int,
        destination: int,
        rows: int,
        inner: int,
        bits: int,
        tokens: int = 1,
    ) -> None:
        self._check(
            self.lib.engine_gemv(weight, scale, codes, ascales, destination, rows, inner, bits,
                tokens)
        )

    def gemv_f16(self, weight: int, source: int, destination: int, rows: int, inner: int) -> None:
        self._check(self.lib.engine_gemv_f16(weight, source, destination, rows, inner))

    def gemv_f16_tokens(
        self, weight: int, source: int, destination: int, rows: int, inner: int, tokens: int
    ) -> None:
        self._check(self.lib.engine_gemv_f16_tokens(weight, source, destination, rows, inner, tokens))

    def topk_tokens(self, scores: int, ids: int, weights: int, n: int, k: int, tokens: int) -> None:
        self._check(self.lib.engine_topk_tokens(scores, ids, weights, n, k, tokens))

    def embed(self, table: int, scale: int, token: int, destination: int, inner: int) -> None:
        self._check(self.lib.engine_embed(table, scale, token, destination, inner))

    def rmsnorm(
        self, source: int, weight: int, destination: int, width: int, eps: float, heads: int = 1
    ) -> None:
        self._check(self.lib.engine_rmsnorm(source, weight, destination, width, float(eps), heads))

    def add(self, stream: int, delta: int, width: int) -> None:
        self._check(self.lib.engine_add(stream, delta, width))

    def silu_mul(self, gate: int, up: int, destination: int, width: int) -> None:
        self._check(self.lib.engine_silu_mul(gate, up, destination, width))

    def scale_add(self, destination: int, source: int, scale: float, width: int) -> None:
        self._check(self.lib.engine_scale_add(destination, source, float(scale), width))

    def scale_add_at(self, destination: int, source: int, scale: int, width: int) -> None:
        """Add ``source`` scaled by the single device float at ``scale``."""
        self._check(self.lib.engine_scale_add_at(destination, source, scale, width))

    def sig_add(self, destination: int, source: int, coeff: int, width: int) -> None:
        self._check(self.lib.engine_sig_add(destination, source, coeff, width))

    def l2norm(self, values: int, heads: int, dim: int) -> None:
        self._check(self.lib.engine_l2norm(values, heads, dim))

    def l2norm_tokens(self, values: int, heads: int, dim: int, stride: int, tokens: int) -> None:
        self._check(self.lib.engine_l2norm_tokens(values, heads, dim, stride, tokens))

    def repeat2(self, source: int, destination: int, heads: int, dim: int) -> None:
        self._check(self.lib.engine_repeat2(source, destination, heads, dim))

    def repeat2_tokens(
        self, source: int, destination: int, heads: int, dim: int, source_stride: int, dest_stride:
        int, tokens: int
    ) -> None:
        self._check(
            self.lib.engine_repeat2_tokens(
                source, destination, heads, dim, source_stride, dest_stride, tokens
            )
        )

    def conv_silu(
        self, history: int, taps: int, source: int, destination: int, channels: int, count: int = 1
    ) -> None:
        self._check(self.lib.engine_conv_silu(history, taps, source, destination, channels, count))

    def gdn(
        self,
        state: int,
        query: int,
        key: int,
        value: int,
        a_proj: int,
        b_proj: int,
        a_log: int,
        dt_bias: int,
        norm: int,
        gate: int,
        destination: int,
        eps: float,
        norm_span: int,
        count: int = 1,
        value_stride: int = 4096,
    ) -> None:
        self._check(
            self.lib.engine_gdn(
                state, query, key, value, a_proj, b_proj, a_log, dt_bias, norm, gate,
                destination, float(eps), norm_span, count, value_stride,
            )
        )

    def split_q(self, packed: int, query: int, gate: int, heads: int) -> None:
        self._check(self.lib.engine_split_q(packed, query, gate, heads))

    def rope(
        self, query: int, q_heads: int, key: int, k_heads: int, position: int, theta: float, count:
        int = 1
    ) -> None:
        self._check(self.lib.engine_rope(query, q_heads, key, k_heads, position, float(theta),
            count))

    def store_kv(
        self,
        cache: int,
        source: int,
        tokens: int,
        layer: int,
        heads: int,
        token: int,
        full: int,
        count: int = 1,
    ) -> None:
        self._check(self.lib.engine_store_kv(cache, source, tokens, layer, heads, token, full,
            count))

    def attend(
        self,
        k_cache: int,
        v_cache: int,
        query: int,
        destination: int,
        count: int,
        layer: int,
        full: int,
        scale: float,
    ) -> None:
        self._check(
            self.lib.engine_attend(k_cache, v_cache, query, destination, count, layer, full,
                float(scale))
        )

    def gate_mul(self, values: int, gate: int, width: int) -> None:
        self._check(self.lib.engine_gate_mul(values, gate, width))

    def zero_f32(self, destination: int, width: int) -> None:
        self._check(self.lib.engine_zero_f32(destination, width))

    def topk(self, scores: int, ids: int, weights: int, n: int, k: int) -> None:
        self._check(self.lib.engine_topk(scores, ids, weights, n, k))

    def argmax(self, logits: int, partial_v: int, partial_i: int, token: int, n: int) -> None:
        self._check(self.lib.engine_argmax(logits, partial_v, partial_i, token, n))

    def sample(
        self,
        logits: int,
        counts: int,
        token: int,
        n: int,
        temperature: float,
        top_p: float,
        top_k: int,
        presence: float,
        frequency: float,
        seed: int,
    ) -> None:
        self._check(
            self.lib.engine_sample(
                logits, counts, token, n, float(temperature), float(top_p), top_k,
                float(presence), float(frequency), seed & 0xFFFFFFFF,
            )
        )

    def count(self, counts: int, token: int) -> None:
        self._check(self.lib.engine_count(counts, token))

    def penalize(self, logits: int, counts: int, presence: float, frequency: float, n: int) -> None:
        self._check(self.lib.engine_penalize(logits, counts, float(presence), float(frequency), n))

    def gemm(
        self, weight: int, scale: int, source: int, destination: int, rows: int, inner: int, tokens:
        int
    ) -> None:
        self._check(self.lib.engine_gemm(weight, scale, source, destination, rows, inner, tokens))

    def gemm_i4(
        self, weight: int, scale: int, source: int, destination: int, rows: int, inner: int, tokens:
        int
    ) -> None:
        """Int4 weight, expanded to int8, times an int4-ranged activation tile."""
        self._check(self.lib.engine_gemm_i4(weight, scale, source, destination, rows, inner, tokens))

    def gather_rows(self, source: int, destination: int, index: int, rows: int, width: int) -> None:
        self._check(self.lib.engine_gather_rows(source, destination, index, rows, width))

    def silu_rows(self, packed: int, destination: int, width: int, stride: int, tokens: int) -> None:
        self._check(self.lib.engine_silu_rows(packed, destination, width, stride, tokens))

    def scatter_scale(
        self, destination: int, source: int, token: int, slot: int, route: int, rows: int, width:
        int, topk: int
    ) -> None:
        self._check(
            self.lib.engine_scatter_scale(destination, source, token, slot, route, rows, width, topk)
        )

    def sig_rows(self, destination: int, source: int, coeff: int, tokens: int, width: int) -> None:
        self._check(self.lib.engine_sig_rows(destination, source, coeff, tokens, width))

    def grouped_mlp(
        self,
        device: np.ndarray,
        host_src: np.ndarray,
        copy_bytes: np.ndarray,
        bits: np.ndarray,
        token_index: np.ndarray,
        slot_index: np.ndarray,
        group_n: np.ndarray,
        gate_scale: np.ndarray,
        down: np.ndarray,
        down_scale: np.ndarray,
        normed: int,
        gather: int,
        up: int,
        silu: int,
        proj: int,
        moe: int,
        route: int,
        token_buf: int,
        slot_buf: int,
        hidden: int,
        gate_rows: int,
        down_inner: int,
        down_rows: int,
        topk: int,
    ) -> None:
        """One host call for every expert in the tile. Copies land just before each product."""
        held = [
            np.ascontiguousarray(array)
            for array in (
                device, host_src, copy_bytes, bits, token_index, slot_index, group_n, gate_scale, down,
                down_scale,
            )
        ]
        groups = int(held[6].shape[0])
        self._check(
            self.lib.engine_grouped_mlp(
                *(item.ctypes.data for item in held),
                groups,
                normed, gather, up, silu, proj, moe, route, token_buf, slot_buf, hidden, gate_rows,
                down_inner, down_rows, topk,
            )
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
        self._check(
            self.lib.engine_gemm_pair(
                weight_a, scale_a, destination_a, rows_a, weight_b, scale_b, destination_b, rows_b,
                source, inner, tokens,
            )
        )

    def attend_prefill(
        self,
        k_cache: int,
        v_cache: int,
        query: int,
        gate: int,
        destination: int,
        length: int,
        layer: int,
        full: int,
        scale: float,
        count: int,
    ) -> None:
        self._check(
            self.lib.engine_attend_prefill(
                k_cache, v_cache, query, gate, destination, length, layer, full, float(scale), count
            )
        )

    def reserve(self, tokens: int, inner: int) -> None:
        self._check(self.lib.engine_reserve(tokens, inner))

    def set_step(self, token: int, position: int, rope: int) -> None:
        self._check(self.lib.engine_set_step(token, position, rope))

    def capture_begin(self, slot: int) -> None:
        self._check(self.lib.engine_capture_begin(slot))

    def capture_end(self, slot: int) -> None:
        self._check(self.lib.engine_capture_end(slot))

    def replay(self, slot: int) -> None:
        self._check(self.lib.engine_replay(slot))

    def close(self) -> None:
        self.lib.engine_close()
