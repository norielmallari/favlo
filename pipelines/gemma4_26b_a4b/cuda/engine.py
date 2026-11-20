"""Load the Gemma 4 CUDA step."""

from __future__ import annotations

import ctypes
import os
from typing import Any

import numpy as np

from favlo.host import cuda_home

_LIB: ctypes.CDLL | None = None


def _checks() -> Any:
    import importlib.util
    import sys
    from pathlib import Path

    path = Path(__file__).resolve().parent / "checks.py"
    name = "gemma4_cuda_checks"
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
    """The compiled step. The toolkit directory is on the loader path."""
    global _LIB
    if _LIB is not None:
        return _LIB
    home = cuda_home()
    if home is not None:
        os.add_dll_directory(str(home / "bin" / "x64"))
    lib = ctypes.CDLL(str(_checks().ensure_engine()))
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
    lib.engine_open.restype = integer
    lib.engine_open.argtypes = []
    lib.engine_close.restype = None
    lib.engine_close.argtypes = []
    lib.engine_alloc.restype = void
    lib.engine_alloc.argtypes = [size]
    lib.engine_free.restype = None
    lib.engine_free.argtypes = [void]
    lib.engine_host_alloc.restype = void
    lib.engine_host_alloc.argtypes = [size]
    lib.engine_host_free.restype = None
    lib.engine_host_free.argtypes = [void]
    for name in ("engine_h2d", "engine_d2h", "engine_h2d_ptr", "engine_d2d"):
        fn = getattr(lib, name)
        fn.restype = integer
        fn.argtypes = [void, void, size]
    lib.engine_copy_async.restype = integer
    lib.engine_copy_async.argtypes = [void, void, size, ctypes.POINTER(integer)]
    lib.engine_wait_copy.restype = integer
    lib.engine_wait_copy.argtypes = [integer]
    lib.engine_register.restype = integer
    lib.engine_register.argtypes = [void, size]
    lib.engine_register_ro.restype = integer
    lib.engine_register_ro.argtypes = [void, size]
    lib.engine_unregister.restype = None
    lib.engine_unregister.argtypes = [void]
    lib.engine_sync.restype = integer
    lib.engine_sync.argtypes = []
    lib.engine_mem_info.restype = integer
    lib.engine_mem_info.argtypes = [ctypes.POINTER(size), ctypes.POINTER(size)]
    lib.engine_zero.restype = integer
    lib.engine_zero.argtypes = [void, size]
    lib.engine_linear.restype = integer
    lib.engine_linear.argtypes = [void, void, void, void, integer, integer, integer, integer]
    lib.engine_rmsnorm.restype = integer
    lib.engine_rmsnorm.argtypes = [void, void, void, integer, number, integer, integer]
    lib.engine_head_norm.restype = integer
    lib.engine_head_norm.argtypes = [void, void, integer, integer, number, integer]
    lib.engine_rope.restype = integer
    lib.engine_rope.argtypes = [void, void, integer, integer, integer, integer, number, integer]
    lib.engine_store_kv.restype = integer
    lib.engine_store_kv.argtypes = [void, void, integer, integer, integer]
    lib.engine_attend.restype = integer
    lib.engine_attend.argtypes = [void, void, void, void, integer, integer, integer, integer, integer]
    lib.engine_rope_span.restype = integer
    lib.engine_rope_span.argtypes = [void, void, integer, integer, integer, integer, number, integer, integer]
    lib.engine_store_span.restype = integer
    lib.engine_store_span.argtypes = [void, void, integer, integer, integer, integer]
    lib.engine_attend_span.restype = integer
    lib.engine_attend_span.argtypes = [void, void, void, void, integer, integer, integer, integer, integer, integer]
    lib.engine_set_step.restype = integer
    lib.engine_set_step.argtypes = [integer]
    lib.engine_capture_begin.restype = integer
    lib.engine_capture_begin.argtypes = [integer]
    lib.engine_capture_end.restype = integer
    lib.engine_capture_end.argtypes = [integer]
    lib.engine_replay.restype = integer
    lib.engine_replay.argtypes = [integer]
    lib.engine_reset_graphs.restype = None
    lib.engine_reset_graphs.argtypes = []
    lib.engine_add.restype = integer
    lib.engine_add.argtypes = [void, void, void, integer]
    lib.engine_scale.restype = integer
    lib.engine_scale.argtypes = [void, number, integer]
    lib.engine_copy.restype = integer
    lib.engine_copy.argtypes = [void, void, integer]
    lib.engine_geglu.restype = integer
    lib.engine_geglu.argtypes = [void, void, integer]
    lib.engine_geglu_rows.restype = integer
    lib.engine_geglu_rows.argtypes = [void, void, integer, integer]
    lib.engine_geglu_pair.restype = integer
    lib.engine_geglu_pair.argtypes = [void, void, void, integer, integer]
    lib.engine_router.restype = integer
    lib.engine_router.argtypes = [void, void, void, void, integer, integer]
    lib.engine_router_tokens.restype = integer
    lib.engine_router_tokens.argtypes = [void, void, void, void, integer, integer, integer]
    lib.engine_mix.restype = integer
    lib.engine_mix.argtypes = [void, void, void, integer, integer]
    lib.engine_embed.restype = integer
    lib.engine_embed.argtypes = [void, void, integer, void, integer, number]
    lib.engine_softcap.restype = integer
    lib.engine_softcap.argtypes = [void, number, integer]
    lib.engine_argmax.restype = integer
    lib.engine_argmax.argtypes = [void, void, integer]
    lib.engine_penalize.restype = integer
    lib.engine_penalize.argtypes = [void, void, number, integer]
    lib.engine_count.restype = integer
    lib.engine_count.argtypes = [void, void]
    lib.engine_gemv_f16.restype = integer
    lib.engine_gemv_f16.argtypes = [void, void, void, integer, integer]
    lib.engine_gemv_f16_tokens.restype = integer
    lib.engine_gemv_f16_tokens.argtypes = [void, void, void, integer, integer, integer]
    lib.engine_gather_f16.restype = integer
    lib.engine_gather_f16.argtypes = [void, void, void, integer, integer]
    lib.engine_scatter_add_f16.restype = integer
    lib.engine_scatter_add_f16.argtypes = [void, void, void, void, void, integer, integer, integer]
    lib.engine_geglu_strided.restype = integer
    lib.engine_geglu_strided.argtypes = [void, void, integer, integer, integer]
    lib.engine_sample.restype = integer
    lib.engine_sample.argtypes = [void, void, integer, number, integer, number, number]


class Engine:
    """One CUDA device and the language step."""

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

    def host_alloc(self, nbytes: int) -> int:
        pointer = self.lib.engine_host_alloc(max(int(nbytes), 16))
        if not pointer:
            message = f"pinned allocation of {nbytes} bytes failed"
            raise RuntimeError(message)
        return int(pointer)

    def host_free(self, pointer: int) -> None:
        if pointer:
            self.lib.engine_host_free(pointer)

    def zero(self, pointer: int, nbytes: int) -> None:
        self._check(self.lib.engine_zero(pointer, int(nbytes)))

    def sync(self) -> None:
        self._check(self.lib.engine_sync())

    def mem_info(self) -> tuple[int, int]:
        free = ctypes.c_size_t()
        total = ctypes.c_size_t()
        self._check(self.lib.engine_mem_info(ctypes.byref(free), ctypes.byref(total)))
        return int(free.value), int(total.value)

    def h2d(self, pointer: int, host: np.ndarray | bytes) -> None:
        array = np.frombuffer(host, dtype=np.uint8) if isinstance(host, bytes | bytearray) else host
        array = np.ascontiguousarray(array)
        self._check(self.lib.engine_h2d(pointer, array.ctypes.data, array.nbytes))

    def h2d_ptr(self, destination: int, source: int, nbytes: int) -> None:
        self._check(self.lib.engine_h2d_ptr(destination, source, int(nbytes)))

    def copy_async(self, destination: int, source: int, nbytes: int) -> int:
        event = ctypes.c_int()
        self._check(self.lib.engine_copy_async(destination, source, int(nbytes), ctypes.byref(event)))
        return int(event.value)

    def wait_copy(self, event: int) -> None:
        self._check(self.lib.engine_wait_copy(int(event)))

    def d2d(self, destination: int, source: int, nbytes: int) -> None:
        self._check(self.lib.engine_d2d(destination, source, int(nbytes)))

    def d2h(self, host: np.ndarray | ctypes.Array[Any], pointer: int) -> None:
        if isinstance(host, np.ndarray):
            array = np.ascontiguousarray(host)
            self._check(self.lib.engine_d2h(array.ctypes.data, pointer, array.nbytes))
            return
        self._check(self.lib.engine_d2h(ctypes.addressof(host), pointer, ctypes.sizeof(host)))

    def register(self, pointer: int, nbytes: int) -> bool:
        if not pointer or nbytes < 1:
            return False
        return int(self.lib.engine_register(pointer, int(nbytes))) == 0

    def register_ro(self, pointer: int, nbytes: int) -> bool:
        if not pointer or nbytes < 1:
            return False
        return int(self.lib.engine_register_ro(pointer, int(nbytes))) == 0

    def unregister(self, pointer: int) -> None:
        if pointer:
            self.lib.engine_unregister(pointer)

    def linear(
        self, codes: int, scales: int, source: int, destination: int, rows: int, inner: int, tokens: int, bits: int
    ) -> None:
        self._check(
            self.lib.engine_linear(codes, scales, source, destination, rows, inner, tokens, bits)
        )

    def rmsnorm(
        self, source: int, weight: int, destination: int, width: int, eps: float, tokens: int, unit: int = 0
    ) -> None:
        self._check(self.lib.engine_rmsnorm(source, weight, destination, width, float(eps), tokens, unit))

    def head_norm(self, values: int, weight: int, heads: int, dim: int, eps: float, unit: int) -> None:
        self._check(self.lib.engine_head_norm(values, weight, heads, dim, float(eps), unit))

    def rope(
        self,
        query: int,
        key: int,
        query_heads: int,
        key_heads: int,
        dim: int,
        rotating: int,
        theta: float,
        position: int,
    ) -> None:
        self._check(
            self.lib.engine_rope(query, key, query_heads, key_heads, dim, rotating, float(theta), position)
        )

    def store_kv(self, source: int, cache: int, heads: int, dim: int, position: int) -> None:
        self._check(self.lib.engine_store_kv(source, cache, heads, dim, position))

    def attend(
        self,
        query: int,
        key: int,
        value: int,
        destination: int,
        query_heads: int,
        key_heads: int,
        dim: int,
        position: int,
        window: int,
    ) -> None:
        self._check(
            self.lib.engine_attend(
                query, key, value, destination, query_heads, key_heads, dim, position, window
            )
        )

    def rope_span(
        self,
        query: int,
        key: int,
        query_heads: int,
        key_heads: int,
        dim: int,
        rotating: int,
        theta: float,
        base: int,
        tokens: int,
    ) -> None:
        self._check(
            self.lib.engine_rope_span(
                query, key, query_heads, key_heads, dim, rotating, float(theta), base, tokens
            )
        )

    def store_span(self, source: int, cache: int, heads: int, dim: int, base: int, tokens: int) -> None:
        self._check(self.lib.engine_store_span(source, cache, heads, dim, base, tokens))

    def set_step(self, position: int) -> None:
        self._check(self.lib.engine_set_step(int(position)))

    def capture_begin(self, slot: int) -> None:
        self._check(self.lib.engine_capture_begin(int(slot)))

    def capture_end(self, slot: int) -> None:
        self._check(self.lib.engine_capture_end(int(slot)))

    def replay(self, slot: int) -> None:
        self._check(self.lib.engine_replay(int(slot)))

    def reset_graphs(self) -> None:
        self.lib.engine_reset_graphs()

    def attend_span(
        self,
        query: int,
        key: int,
        value: int,
        destination: int,
        query_heads: int,
        key_heads: int,
        dim: int,
        base: int,
        tokens: int,
        window: int,
    ) -> None:
        self._check(
            self.lib.engine_attend_span(
                query, key, value, destination, query_heads, key_heads, dim, base, tokens, window
            )
        )

    def add(self, left: int, right: int, destination: int, count: int) -> None:
        self._check(self.lib.engine_add(left, right, destination, count))

    def scale(self, values: int, factor: float, count: int) -> None:
        self._check(self.lib.engine_scale(values, float(factor), count))

    def copy(self, source: int, destination: int, count: int) -> None:
        self._check(self.lib.engine_copy(source, destination, count))

    def geglu(self, fused: int, destination: int, width: int) -> None:
        self._check(self.lib.engine_geglu(fused, destination, width))

    def geglu_pair(self, gate: int, up: int, destination: int, width: int, rows: int) -> None:
        self._check(self.lib.engine_geglu_pair(gate, up, destination, width, rows))

    def geglu_rows(self, fused: int, destination: int, width: int, rows: int) -> None:
        self._check(self.lib.engine_geglu_rows(fused, destination, width, rows))

    def router(self, logits: int, expert_scale: int, ids: int, weights: int, count: int, keep: int) -> None:
        self._check(self.lib.engine_router(logits, expert_scale, ids, weights, count, keep))

    def router_tokens(
        self, logits: int, expert_scale: int, ids: int, weights: int, count: int, keep: int, tokens: int
    ) -> None:
        self._check(self.lib.engine_router_tokens(logits, expert_scale, ids, weights, count, keep, tokens))

    def mix(self, acc: int, row: int, weight: int, slot: int, width: int) -> None:
        self._check(self.lib.engine_mix(acc, row, weight, slot, width))

    def embed(self, table: int, scale: int, token: int, destination: int, inner: int, gain: float) -> None:
        self._check(self.lib.engine_embed(table, scale, token, destination, inner, float(gain)))

    def softcap(self, logits: int, cap: float, count: int) -> None:
        self._check(self.lib.engine_softcap(logits, float(cap), count))

    def penalize(self, logits: int, counts: int, presence: float, vocab: int) -> None:
        self._check(self.lib.engine_penalize(logits, counts, float(presence), vocab))

    def count(self, counts: int, token: int) -> None:
        self._check(self.lib.engine_count(counts, token))

    def gemv_f16(self, weight: int, source: int, destination: int, rows: int, inner: int) -> None:
        self._check(self.lib.engine_gemv_f16(weight, source, destination, rows, inner))

    def gemv_f16_tokens(
        self, weight: int, source: int, destination: int, rows: int, inner: int, tokens: int
    ) -> None:
        self._check(self.lib.engine_gemv_f16_tokens(weight, source, destination, rows, inner, tokens))

    def gather_f16(self, source: int, destination: int, index: int, rows: int, width: int) -> None:
        self._check(self.lib.engine_gather_f16(source, destination, index, rows, width))

    def scatter_add_f16(
        self, destination: int, source: int, token: int, slot: int, route: int, rows: int, width: int, topk: int
    ) -> None:
        self._check(self.lib.engine_scatter_add_f16(destination, source, token, slot, route, rows, width, topk))

    def geglu_strided(self, fused: int, destination: int, width: int, stride: int, rows: int) -> None:
        self._check(self.lib.engine_geglu_strided(fused, destination, width, stride, rows))

    def sample(
        self, logits: int, destination: int, vocab: int, temperature: float, top_k: int, top_p: float, uniform: float
    ) -> None:
        self._check(
            self.lib.engine_sample(
                logits, destination, vocab, float(temperature), int(top_k), float(top_p), float(uniform)
            )
        )

    def read_int(self, pointer: int) -> int:
        slot = ctypes.c_int()
        self.d2h(slot, pointer)
        return int(slot.value)

    def read_half(self, pointer: int) -> float:
        slot = ctypes.c_uint16()
        self._check(self.lib.engine_d2h(ctypes.addressof(slot), pointer, 2))
        return float(np.array([slot.value], dtype=np.uint16).view(np.float16)[0])
