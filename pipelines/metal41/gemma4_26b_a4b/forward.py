"""Text forward for Gemma 4 26B-A4B.

The block matches the published Gemma 4 text layer. Attention and both
feed-forward branches add at scale one. ``layer_scalar`` multiplies the
hidden state after the feed-forward residual. The dense MLP uses
``pre_feedforward_layernorm`` and ``post_feedforward_layernorm_1``. Routed
experts use ``pre_feedforward_layernorm_2`` and
``post_feedforward_layernorm_2``. The router norm is the full router scale
times ``1/sqrt(hidden)``. The embedding row is multiplied by ``sqrt(hidden)``.
Decode reads packed q4_0 one token at a time, except the routed experts.
Those are fp4 E2M1 with an E8M0 scale, one simd per output row.
Prefill attends a chunk of 1024 tokens in one kernel. Routed rows are
sorted by expert and padded to 64. One fp4 matmul covers every slot of
a projection. A dense batch of eight or fewer
tokens uses that dot. A longer dense batch dequantizes one matrix to
float16 and multiplies it with matmul2d. Attention scores are the raw
dot. Values are RMSNorm with no
scale: a global value is that norm of the key projection, without RoPE.
Global layers rotate the first quarter of the head pairs (p-RoPE). Local
layers use full RoPE and a sliding window. Final logits use the tied
embedding and the checkpoint soft-cap.
"""

from __future__ import annotations

import array
import random
import struct
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from favlo.memory import memory
from favlo.prefix import (
    DEFAULT_BUDGET,
    PrefixSession,
    PrefixStore,
    checkpoint_digest,
    default_prefix_dir,
)
from gemma4_26b_a4b.geometry import (
    DENSE_WIDTH,
    EXPERT_WIDTH,
    GLOBAL_DIM,
    GLOBAL_HEADS,
    GLOBAL_LAYERS,
    HIDDEN,
    LAYERS,
    LOCAL_DIM,
    LOCAL_HEADS,
    Q_HEADS,
    RMS_EPS,
    ROPE_GLOBAL,
    ROPE_LOCAL,
    ROPE_PARTIAL,
    SOFTCAP,
    TOP_K,
    VOCAB,
    WINDOW,
    local_layers,
)
from gemma4_26b_a4b.planes import FP4, Q40, align_k, matrix_nbytes
from gemma4_26b_a4b.prefix import RuntimeAdapter
from gemma4_26b_a4b.session import Session

_ROWS = 8


_CHUNK = 1024
_SLOT = 64
_SLOTS = 256
_MAX_K = 8192
_MAX_N = VOCAB
_PICK = 65536
_SEEN = 8192
_STOP = (1, 106, 50)


@dataclass
class Matrix:
    data: object
    scales: object | None
    kind: str
    rows: int
    k: int
    k_pad: int
    origin: int = 0


@dataclass
class Layer:
    scalar: float
    input_norm: object
    post_attn: object
    pre: object
    pre2: object
    post: object
    post1: object
    post2: object
    q_norm: object
    k_norm: object
    q: Matrix
    k: Matrix
    v: Matrix | None
    o: Matrix
    router: Matrix
    router_scale: object
    expert_scale: object
    gate: Matrix
    up: Matrix
    down: Matrix
    global_attn: bool


class Gemma4Runtime:
    """One packed record. The expert atlas is resident."""

    def __init__(
        self,
        path: Path,
        *,
        owner: str,
        kv_cache: str = "e4m3",
        console: object = None,
        max_memory: str | int | None = None,
    ) -> None:
        del console
        from favlo.memory import memory

        ceiling = memory.budget() if not isinstance(max_memory, int) else max_memory
        resident = (path / "resident.bin").stat().st_size
        atlas = sum((path / "experts" / f"{layer}.bin").stat().st_size for layer in range(LAYERS))
        if resident + atlas > ceiling:
            message = "the record does not fit the memory budget"
            raise RuntimeError(message)
        self.root = path
        self.owner = owner
        self._owner = owner
        self.kv_cache = kv_cache
        self._stop = False
        self.session = Session(owner)
        self.filled = 0
        self.capacity = 256
        self._load(path)
        self._buffers()
        self._cache()
        self.prefix = PrefixSession(
            PrefixStore(
                default_prefix_dir(),
                model="gemma4-26b-a4b",
                digest=checkpoint_digest(path),
                tag=kv_cache,
                budget=DEFAULT_BUDGET,
            ),
            RuntimeAdapter(self),
        )

    def close(self) -> None:
        try:
            self.prefix.flush()
        except OSError:
            pass
        self.session.close()

    def reset(self) -> None:
        self.filled = 0
        self._stop = False

    def generate(
        self,
        prompt: list[int],
        limit: int,
        temperature: float,
        top_p: float,
        top_k: int,
        presence: float,
        *,
        seed: int = 0,
    ) -> tuple[list[int], str, str]:
        """Sample after ``prompt``. Returns tokens, the prefill line, and the decode line."""
        rng = random.Random(seed)
        hit = self.prefix.align(prompt)
        if self.prefix.loaded:
            print(f"prefix load {hit} tokens", flush=True)
        print("cache hit" if hit else "cache miss", flush=True)
        start = time.perf_counter()
        self._prefill(prompt[hit:])
        self.session.flush()
        prefill_s = time.perf_counter() - start
        prompt_n = len(prompt) - hit
        self.prefix.note(prompt[: self.filled])
        try:
            self.prefix.flush()
        except OSError:
            pass
        drawn: list[int] = []
        seen: list[int] = []
        decode_start = time.perf_counter()
        if hit == len(prompt):
            self._logits()
        for _ in range(limit):
            if self._stop:
                break
            token = self._draw(temperature, top_p, top_k, presence, seen, rng)
            if token in _STOP:
                break
            drawn.append(token)
            seen.append(token)
            self._step(token)
        decode_s = time.perf_counter() - decode_start
        self.prefix.note(prompt + drawn)
        try:
            self.prefix.flush()
        except OSError:
            pass
        return drawn, _rate("prefill", prompt_n, prefill_s), _rate("decode", len(drawn), decode_s)

    def complete_text(
        self,
        messages: list[dict[str, object]],
        *,
        thinking: bool = False,
        limit: int = 384,
        temperature: float | None = None,
        seed: int = 0,
    ) -> str:
        from gemma4_26b_a4b.chat import decode_text, encode_chat, reply_text
        from gemma4_26b_a4b.sampling import VARIANTS

        card = VARIANTS["thinking" if thinking else "instruct"]
        prompt = encode_chat(self.root, messages, thinking)
        chosen = float(card["temperature"]) if temperature is None else temperature
        tokens, prefill_line, decode_line = self.generate(
            prompt,
            limit,
            chosen,
            float(card["top_p"]),
            int(card["top_k"]),
            float(card["presence_penalty"]),
            seed=seed,
        )
        print(prefill_line, flush=True)
        print(decode_line, flush=True)
        _reasoning, answer = reply_text(decode_text(self.root, tokens))
        return answer

    def export_local(self) -> tuple[np.ndarray, np.ndarray]:
        count = self.filled
        layers = len(local_layers())
        keys = np.zeros((layers, LOCAL_HEADS, count, LOCAL_DIM), dtype=np.float32)
        values = np.zeros((layers, LOCAL_HEADS, count, LOCAL_DIM), dtype=np.float32)
        for index, layer in enumerate(local_layers()):
            keys[index] = self._read_cache(self.local_k[layer], LOCAL_HEADS, LOCAL_DIM, count)
            values[index] = self._read_cache(self.local_v[layer], LOCAL_HEADS, LOCAL_DIM, count)
        return keys, values

    def export_global(self) -> tuple[np.ndarray, np.ndarray]:
        count = self.filled
        shape = (len(GLOBAL_LAYERS), GLOBAL_HEADS, count, GLOBAL_DIM)
        keys = np.zeros(shape, dtype=np.float32)
        values = np.zeros(shape, dtype=np.float32)
        for index, layer in enumerate(GLOBAL_LAYERS):
            keys[index] = self._read_cache(self.global_k[layer], GLOBAL_HEADS, GLOBAL_DIM, count)
            values[index] = self._read_cache(self.global_v[layer], GLOBAL_HEADS, GLOBAL_DIM, count)
        return keys, values

    def import_local(self, keys: np.ndarray, values: np.ndarray) -> None:
        count = int(keys.shape[2])
        self._ensure(count)
        for index, layer in enumerate(local_layers()):
            self._write_cache(self.local_k[layer], keys[index], LOCAL_HEADS, LOCAL_DIM)
            self._write_cache(self.local_v[layer], values[index], LOCAL_HEADS, LOCAL_DIM)
        self.filled = count

    def import_global(self, keys: np.ndarray, values: np.ndarray) -> None:
        for index, layer in enumerate(GLOBAL_LAYERS):
            self._write_cache(self.global_k[layer], keys[index], GLOBAL_HEADS, GLOBAL_DIM)
            self._write_cache(self.global_v[layer], values[index], GLOBAL_HEADS, GLOBAL_DIM)

    def _read_cache(self, handle: object, heads: int, dim: int, count: int) -> np.ndarray:
        raw = memory.load(handle, 0, count * heads * dim * 2)
        values = np.frombuffer(raw, dtype=np.float16).astype(np.float32)
        return values.reshape(count, heads, dim).transpose(1, 0, 2)

    def _write_cache(self, handle: object, plane: np.ndarray, heads: int, dim: int) -> None:
        # plane is [heads, count, dim]
        ordered = np.ascontiguousarray(plane.transpose(1, 0, 2).astype(np.float16))
        memory.store(handle, 0, ordered.tobytes())

    def _step(self, token: int) -> None:
        self.session.flush()
        self._ensure(self.filled + 1)
        self._embed(token)
        for layer in range(LAYERS):
            self._block(self.layers[layer], layer)
        self.filled += 1
        self._logits()

    def _prefill(self, tokens: list[int]) -> None:
        if self._stop or not tokens:
            return
        for start in range(0, len(tokens), _CHUNK):
            if self._stop:
                return
            span = tokens[start : start + _CHUNK]
            self._span(span, start + len(span) == len(tokens))

    def _span(self, tokens: list[int], last: bool) -> None:
        count = len(tokens)
        self._ensure(self.filled + count)
        self._embed_span(tokens)
        for index, layer in enumerate(self.layers):
            self._block_span(layer, index, count)
        self._copy_at(self.span, (count - 1) * HIDDEN, self.hidden, HIDDEN)
        self.filled += count
        if last:
            self._logits()

    def _embed_span(self, tokens: list[int]) -> None:
        plane = self.embed
        blocks = plane.k_pad // 32
        for index, token in enumerate(tokens):
            self.session.encode(
                "embed_q4_0",
                [plane.data, self.span],
                [
                    self.session._u32(token),
                    self.session._u32(plane.k_pad),
                    self.session._u32(plane.k),
                    self.session._f32(HIDDEN**0.5),
                ],
                *self._grid(blocks),
                origins=[0, index * HIDDEN * 4],
            )

    def _block_span(self, layer: Layer, index: int, count: int) -> None:
        q_dim = GLOBAL_DIM if layer.global_attn else LOCAL_DIM
        kv_heads = GLOBAL_HEADS if layer.global_attn else LOCAL_HEADS
        q_width = Q_HEADS * q_dim
        kv_width = kv_heads * q_dim
        self._norm_rows(self.span, layer.input_norm, self.span_n, count)
        self._gemm(self.span_n, layer.q, HIDDEN, count, self.span_q)
        self._gemm(self.span_n, layer.k, HIDDEN, count, self.span_k)
        if layer.v is not None:
            self._gemm(self.span_n, layer.v, HIDDEN, count, self.span_v)
        else:
            self._copy_at(self.span_k, 0, self.span_v, count * kv_width)
        self._norm_heads(self.span_v, self.unit, count * kv_heads, q_dim)
        self._norm_heads(self.span_q, layer.q_norm, count * Q_HEADS, q_dim)
        self._norm_heads(self.span_k, layer.k_norm, count * kv_heads, q_dim)
        rotating = int(q_dim / 2 * ROPE_PARTIAL) if layer.global_attn else q_dim // 2
        theta = ROPE_GLOBAL if layer.global_attn else ROPE_LOCAL
        self.session.encode(
            "rope_rows",
            [self.span_q, self.span_k],
            [
                self.session._u32(Q_HEADS),
                self.session._u32(kv_heads),
                self.session._u32(q_dim),
                self.session._u32(rotating),
                self.session._f32(theta),
                self.session._u32(self.filled),
                self.session._u32(count),
            ],
            (count, 1, 1),
            (256, 1, 1),
        )
        cache_k = self.global_k[index] if layer.global_attn else self.local_k[index]
        cache_v = self.global_v[index] if layer.v is None else self.local_v[index]
        self._store_rows(self.span_k, cache_k, kv_heads, q_dim, count)
        self._store_rows(self.span_v, cache_v, kv_heads, q_dim, count)
        window = 0 if layer.global_attn else WINDOW
        group = Q_HEADS // kv_heads
        attend = "attend_span4" if q_dim == LOCAL_DIM else "attend_span"
        groups = ((count + 3) // 4, Q_HEADS, 1) if q_dim == LOCAL_DIM else (count, Q_HEADS, 1)
        self.session.encode(
            attend,
            [self.span_q, cache_k, cache_v, self.span_a],
            [
                self.session._u32(Q_HEADS),
                self.session._u32(kv_heads),
                self.session._u32(q_dim),
                self.session._u32(group),
                self.session._u32(self.filled),
                self.session._u32(count),
                self.session._u32(window),
            ],
            groups,
            (32, 1, 1),
        )
        self._gemm(self.span_a, layer.o, q_width, count, self.span_m)
        self._norm_rows(self.span_m, layer.post_attn, self.span_n, count)
        self._axpy(self.span, self.span_n, 1.0, count * HIDDEN)
        self._feed_span(layer, index, count)

    def _feed_span(self, layer: Layer, index: int, count: int) -> None:
        ids, weights = self._route_span(layer, count)
        self._norm_rows(self.span, layer.pre, self.span_n, count)
        self._gemm(self.span_n, layer.gate, HIDDEN, count, self.span_w)
        self._gemm(self.span_n, layer.up, HIDDEN, count, self.span_d)
        self.session.encode(
            "geglu_pair",
            [self.span_w, self.span_d, self.span_m],
            [self.session._u32(count * DENSE_WIDTH)],
            *self._grid(count * DENSE_WIDTH),
        )
        self._gemm(self.span_m, layer.down, DENSE_WIDTH, count, self.span_f)
        self._norm_rows(self.span_f, layer.post1, self.span_r, count)
        self._norm_rows(self.span, layer.pre2, self.span_n, count)
        self._zero(self.span_m, count * HIDDEN * 4)
        self._experts_span(index, ids, weights, count)
        self._norm_rows(self.span_m, layer.post2, self.span_f, count)
        self._zero(self.span_m, count * HIDDEN * 4)
        self._axpy(self.span_m, self.span_r, 1.0, count * HIDDEN)
        self._axpy(self.span_m, self.span_f, 1.0, count * HIDDEN)
        self._norm_rows(self.span_m, layer.post, self.span_n, count)
        self._axpy(self.span, self.span_n, 1.0, count * HIDDEN)
        self.session.encode(
            "scale_copy",
            [self.span, self.span],
            [self.session._f32(layer.scalar), self.session._u32(count * HIDDEN)],
            *self._grid(count * HIDDEN),
        )

    def _route_span(self, layer: Layer, count: int) -> tuple[np.ndarray, np.ndarray]:
        self._norm_rows(self.span, layer.router_scale, self.span_n, count)
        padded = (count + 7) & ~7
        router = layer.router
        self.session.encode(
            "write_rows",
            [self.span_n, self.span_h],
            [
                self.session._u32(HIDDEN),
                self.session._u32(router.k_pad),
                self.session._u32(count),
            ],
            ((router.k_pad + 255) // 256, padded, 1),
            (256, 1, 1),
        )
        self.session.matmul(
            router.kind,
            self.span_h,
            router.data,
            router.scales,
            self.span_p,
            padded,
            router.rows,
            router.k_pad,
            commit=False,
        )
        self.session.encode(
            "router_topk",
            [self.span_p, layer.expert_scale, self.route_i, self.route_w],
            [self.session._u32(128), self.session._u32(TOP_K)],
            (count, 1, 1),
            (128, 1, 1),
        )
        raw_ids = self._bytes(self.route_i, count * TOP_K * 4)
        raw_weights = self._bytes(self.route_w, count * TOP_K * 4)
        ids = np.frombuffer(raw_ids, dtype=np.uint32).reshape(count, TOP_K).copy()
        weights = np.frombuffer(raw_weights, dtype=np.float32).reshape(count, TOP_K).copy()
        return ids, weights

    def _experts_span(self, layer: int, ids: np.ndarray, weights: np.ndarray, count: int) -> None:
        groups: list[list[tuple[int, float]]] = [[] for _ in range(128)]
        for token in range(count):
            for slot in range(TOP_K):
                groups[int(ids[token, slot])].append((token, float(weights[token, slot])))
        slots: list[tuple[int, list[tuple[int, float]]]] = []
        for expert, rows in enumerate(groups):
            for begin in range(0, len(rows), _SLOT):
                slots.append((expert, rows[begin : begin + _SLOT]))
        for begin in range(0, len(slots), _SLOTS):
            self._expert_slots(layer, slots[begin : begin + _SLOTS])

    def _expert_slots(self, layer: int, slots: list[tuple[int, list[tuple[int, float]]]]) -> None:
        count = len(slots)
        if count == 0:
            return
        indices = np.full(count * _SLOT, 0xFFFFFFFF, dtype=np.uint32)
        which = np.zeros(count, dtype=np.uint32)
        source_row: list[int] = []
        token: list[int] = []
        scale: list[float] = []
        for slot, (expert, rows) in enumerate(slots):
            which[slot] = expert
            for local, (row_token, weight) in enumerate(rows):
                indices[slot * _SLOT + local] = row_token
                source_row.append(slot * _SLOT + local)
                token.append(row_token)
                scale.append(weight)
        memory.store(self.slot_index, 0, indices.tobytes())
        memory.store(self.slot_which, 0, which.tobytes())
        memory.store(self.slot_row, 0, np.asarray(source_row, dtype=np.uint32).tobytes())
        memory.store(self.slot_token, 0, np.asarray(token, dtype=np.uint32).tobytes())
        memory.store(self.slot_scale, 0, np.asarray(scale, dtype=np.float32).tobytes())
        padded = count * _SLOT
        k_gate = align_k(HIDDEN, FP4)
        self.session.encode(
            "gather_half_slots",
            [self.span_n, self.slot_act, self.slot_index],
            [self.session._u32(HIDDEN), self.session._u32(k_gate), self.session._u32(padded)],
            ((k_gate + 255) // 256, padded, 1),
            (256, 1, 1),
        )
        self._fp4_batch(
            self.fp4_gate_code[layer],
            self.fp4_gate_scale[layer],
            padded,
            count,
            2 * EXPERT_WIDTH,
            k_gate,
        )
        self.session.encode(
            "geglu_rows",
            [self.slot_out, self.slot_mid],
            [self.session._u32(EXPERT_WIDTH), self.session._u32(padded)],
            *self._grid(padded * EXPERT_WIDTH),
        )
        k_down = align_k(EXPERT_WIDTH, FP4)
        self.session.encode(
            "pad_half_rows",
            [self.slot_mid, self.slot_act],
            [
                self.session._u32(EXPERT_WIDTH),
                self.session._u32(k_down),
                self.session._u32(EXPERT_WIDTH),
                self.session._u32(padded),
            ],
            ((k_down + 255) // 256, padded, 1),
            (256, 1, 1),
        )
        self._fp4_batch(
            self.fp4_down_code[layer],
            self.fp4_down_scale[layer],
            padded,
            count,
            HIDDEN,
            k_down,
        )
        self.session.encode(
            "scatter_slot_rows",
            [self.span_m, self.slot_out, self.slot_row, self.slot_token, self.slot_scale],
            [self.session._u32(HIDDEN), self.session._u32(len(token))],
            *self._grid(len(token) * HIDDEN),
        )

    def _fp4_batch(
        self,
        codes: object,
        scales: object,
        rows: int,
        slots: int,
        cols: int,
        inner: int,
    ) -> None:
        pipe = self.session.pipes["matmul_fp4_64_batch"]
        width = int(pipe.threadExecutionWidth())
        self.session.encode(
            "matmul_fp4_64_batch",
            [self.slot_act, codes, scales, self.slot_out, self.slot_which],
            [self.session._u32(rows // slots), self.session._u32(cols), self.session._u32(inner)],
            (cols // 64, (rows // slots) // 64, slots),
            (width * 4, 1, 1),
        )

    def _gather(self, at: int, count: int) -> None:
        self.session.encode(
            "gather_rows",
            [self.span_n, self.span_g, self.gather_i],
            [self.session._u32(HIDDEN), self.session._u32(count)],
            *self._grid(count * HIDDEN),
            origins=[0, 0, at * 4],
        )

    def _scatter(self, at: int, count: int) -> None:
        self.session.encode(
            "scatter_axpy",
            [self.span_m, self.span_f, self.gather_i, self.gather_s],
            [self.session._u32(HIDDEN), self.session._u32(count)],
            *self._grid(count * HIDDEN),
            origins=[0, 0, at * 4, at * 4],
        )

    def _store_rows(self, source: object, cache: object, heads: int, dim: int, count: int) -> None:
        self.session.encode(
            "store_kv_rows",
            [source, cache],
            [
                self.session._u32(heads),
                self.session._u32(dim),
                self.session._u32(self.filled),
                self.session._u32(count),
            ],
            *self._grid(count * heads * dim),
        )

    def _norm_rows(self, source: object, weight: object, destination: object, count: int) -> None:
        self.session.encode(
            "rmsnorm_rows",
            [source, weight, destination],
            [self.session._u32(HIDDEN), self.session._f32(RMS_EPS), self.session._u32(count)],
            (count, 1, 1),
            (256, 1, 1),
        )

    def _gemm(
        self,
        source: object,
        matrix: Matrix,
        count: int,
        tokens: int,
        destination: object,
    ) -> None:
        if tokens <= 0:
            return
        active = min(count, matrix.k)
        if tokens <= 8:
            self.session.encode(
                "gemm_q4_0_n",
                [matrix.data, source, destination],
                [
                    self.session._u32(matrix.k_pad),
                    self.session._u32(matrix.rows),
                    self.session._u32(tokens),
                    self.session._u32(active),
                ],
                ((matrix.rows + 7) // 8, 1, 1),
                (64, 1, 1),
                origins=[matrix.origin, 0, 0],
            )
            return
        blocks = matrix.rows * (matrix.k_pad // 32)
        self.session.encode(
            "dequant_q4_0",
            [matrix.data, self.half_w],
            [self.session._u32(matrix.k_pad), self.session._u32(matrix.rows)],
            *self._grid(blocks),
            origins=[matrix.origin, 0],
        )
        wide = tokens >= 32 and matrix.rows % 64 == 0
        padded = ((tokens + 63) & ~63) if wide else ((tokens + 7) & ~7)
        self.session.encode(
            "write_rows",
            [source, self.span_ah],
            [
                self.session._u32(active),
                self.session._u32(matrix.k_pad),
                self.session._u32(tokens),
            ],
            ((matrix.k_pad + 255) // 256, padded, 1),
            (256, 1, 1),
        )
        if wide:
            pipe = self.session.pipes["matmul_f16_64"]
            width = int(pipe.threadExecutionWidth())
            self.session.encode(
                "matmul_f16_64",
                [self.span_ah, self.half_w, destination],
                [
                    self.session._u32(padded),
                    self.session._u32(matrix.rows),
                    self.session._u32(matrix.k_pad),
                ],
                (matrix.rows // 64, padded // 64, 1),
                (width * 4, 1, 1),
            )
            return
        self.session.matmul(
            "matmul_f16",
            self.span_ah,
            self.half_w,
            None,
            destination,
            padded,
            matrix.rows,
            matrix.k_pad,
            commit=False,
        )

    def _copy_at(
        self,
        source: object,
        src_index: int,
        destination: object,
        count: int,
        dest_index: int = 0,
    ) -> None:
        self.session.encode(
            "scale_copy",
            [source, destination],
            [self.session._f32(1.0), self.session._u32(count)],
            *self._grid(count),
            origins=[src_index * 4, dest_index * 4],
        )

    def _logits(self) -> None:
        self._norm(self.hidden, self.final_norm, self.norm)
        self._linear(self.norm, self.embed, HIDDEN)
        self.session.encode(
            "softcap",
            [self.out],
            [self.session._f32(SOFTCAP), self.session._u32(VOCAB)],
            *self._grid(VOCAB),
        )

    def _embed(self, token: int) -> None:
        plane = self.embed
        blocks = plane.k_pad // 32
        self.session.encode(
            "embed_q4_0",
            [plane.data, self.hidden],
            [
                self.session._u32(token),
                self.session._u32(plane.k_pad),
                self.session._u32(plane.k),
                self.session._f32(HIDDEN**0.5),
            ],
            *self._grid(blocks),
        )

    def _block(self, layer: Layer, index: int) -> None:
        self._norm(self.hidden, layer.input_norm, self.norm)
        q_dim = GLOBAL_DIM if layer.global_attn else LOCAL_DIM
        kv_heads = GLOBAL_HEADS if layer.global_attn else LOCAL_HEADS
        self._linear(self.norm, layer.q, HIDDEN)
        self._copy(self.out, self.query, Q_HEADS * q_dim)
        self._linear(self.norm, layer.k, HIDDEN)
        self._copy(self.out, self.key, kv_heads * q_dim)
        if layer.v is not None:
            self._linear(self.norm, layer.v, HIDDEN)
            self._copy(self.out, self.value, kv_heads * q_dim)
        else:
            self._copy(self.key, self.value, kv_heads * q_dim)
        self._norm_heads(self.value, self.unit, kv_heads, q_dim)
        self._qk_norm(layer, q_dim, kv_heads)
        rotating = int(q_dim / 2 * ROPE_PARTIAL) if layer.global_attn else q_dim // 2
        theta = ROPE_GLOBAL if layer.global_attn else ROPE_LOCAL
        self.session.encode(
            "rope_pairs",
            [self.query, self.key],
            [
                self.session._u32(Q_HEADS),
                self.session._u32(kv_heads),
                self.session._u32(q_dim),
                self.session._u32(rotating),
                self.session._f32(theta),
                self.session._u32(self.filled),
            ],
            *self._grid(Q_HEADS * (q_dim // 2) + kv_heads * (q_dim // 2)),
        )
        cache_k = self.global_k[index] if layer.global_attn else self.local_k[index]
        self.session.encode(
            "store_kv",
            [self.key, cache_k],
            [
                self.session._u32(kv_heads),
                self.session._u32(q_dim),
                self.session._u32(self.filled),
            ],
            *self._grid(kv_heads * q_dim),
        )
        cache_v = self.global_v[index] if layer.v is None else self.local_v[index]
        self.session.encode(
            "store_kv",
            [self.value, cache_v],
            [
                self.session._u32(kv_heads),
                self.session._u32(q_dim),
                self.session._u32(self.filled),
            ],
            *self._grid(kv_heads * q_dim),
        )
        values = cache_v
        window = 0 if layer.global_attn else WINDOW
        group = Q_HEADS // kv_heads
        self.session.encode(
            "attend_span",
            [self.query, cache_k, values, self.attn],
            [
                self.session._u32(Q_HEADS),
                self.session._u32(kv_heads),
                self.session._u32(q_dim),
                self.session._u32(group),
                self.session._u32(self.filled),
                self.session._u32(1),
                self.session._u32(window),
            ],
            (1, Q_HEADS, 1),
            (32, 1, 1),
        )
        self._linear(self.attn, layer.o, Q_HEADS * q_dim)
        self._copy(self.out, self.mixed, HIDDEN)
        self._norm(self.mixed, layer.post_attn, self.norm)
        self._axpy(self.hidden, self.norm, 1.0, HIDDEN)
        self._feed(layer, index)

    def _feed(self, layer: Layer, index: int) -> None:
        self._route_device(layer)
        self._norm(self.hidden, layer.pre, self.norm)
        self._linear(self.norm, layer.gate, HIDDEN)
        self._copy(self.out, self.wide, DENSE_WIDTH)
        self._linear(self.norm, layer.up, HIDDEN)
        self.session.encode(
            "geglu_pair",
            [self.wide, self.out, self.mid],
            [self.session._u32(DENSE_WIDTH)],
            *self._grid(DENSE_WIDTH),
        )
        self._linear(self.mid, layer.down, DENSE_WIDTH)
        self._copy(self.out, self.ffn, HIDDEN)
        self._norm(self.ffn, layer.post1, self.routed)
        self._norm(self.hidden, layer.pre2, self.norm)
        self._zero(self.mixed, HIDDEN * 4)
        self._experts_device(index)
        self._norm(self.mixed, layer.post2, self.ffn)
        self._zero(self.mixed, HIDDEN * 4)
        self._axpy(self.mixed, self.routed, 1.0, HIDDEN)
        self._axpy(self.mixed, self.ffn, 1.0, HIDDEN)
        self._norm(self.mixed, layer.post, self.norm)
        self._axpy(self.hidden, self.norm, 1.0, HIDDEN)
        self.session.encode(
            "scale_copy",
            [self.hidden, self.hidden],
            [self.session._f32(layer.scalar), self.session._u32(HIDDEN)],
            *self._grid(HIDDEN),
        )

    def _route_device(self, layer: Layer) -> None:
        self._norm(self.hidden, layer.router_scale, self.routed)
        self._linear(self.routed, layer.router, HIDDEN)
        self.session.encode(
            "router_topk",
            [self.out, layer.expert_scale, self.expert_index, self.expert_weight],
            [self.session._u32(128), self.session._u32(TOP_K)],
            (1, 1, 1),
            (128, 1, 1),
        )

    def _experts_device(self, layer: int) -> None:
        gate_rows = 2 * EXPERT_WIDTH
        self._fp4_gemv(
            self.fp4_gate_code[layer],
            self.fp4_gate_scale[layer],
            self.norm,
            self.expert_up,
            gate_rows,
            align_k(HIDDEN, FP4),
            HIDDEN,
            0,
        )
        self.session.encode(
            "geglu_rows",
            [self.expert_up, self.expert_hid],
            [self.session._u32(EXPERT_WIDTH), self.session._u32(TOP_K)],
            *self._grid(TOP_K * EXPERT_WIDTH),
        )
        self._fp4_gemv(
            self.fp4_down_code[layer],
            self.fp4_down_scale[layer],
            self.expert_hid,
            self.expert_down,
            HIDDEN,
            align_k(EXPERT_WIDTH, FP4),
            EXPERT_WIDTH,
            EXPERT_WIDTH,
        )
        self.session.encode(
            "mix_experts",
            [self.mixed, self.expert_down, self.expert_weight],
            [self.session._u32(HIDDEN), self.session._u32(TOP_K)],
            *self._grid(HIDDEN),
        )

    def _fp4_gemv(
        self,
        codes: object,
        scales: object,
        activation: object,
        destination: object,
        rows: int,
        k_pad: int,
        active: int,
        act_stride: int,
    ) -> None:
        groups = (rows + 7) // 8
        self.session.encode(
            "gemv_fp4_routed",
            [codes, scales, activation, destination, self.expert_index],
            [
                self.session._u32(k_pad),
                self.session._u32(rows),
                self.session._u32(active),
                self.session._u32(act_stride),
            ],
            (groups * TOP_K, 1, 1),
            (64, 1, 1),
        )

    def _grid(self, count: int) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
        return ((count + 255) // 256, 1, 1), (256, 1, 1)

    def _zero(self, handle: object, nbytes: int) -> None:
        self.session.encode(
            "zero_buffer",
            [handle],
            [self.session._u32(nbytes)],
            *self._grid(nbytes),
        )

    def _bytes(self, handle: object, nbytes: int) -> bytes:
        self.session.flush()
        return memory.load(handle, 0, nbytes)

    def _linear(self, source: object, matrix: Matrix, count: int) -> None:
        if matrix.kind == Q40:
            self.session.encode(
                "gemv_q4_0",
                [matrix.data, source, self.out],
                [
                    self.session._u32(matrix.k_pad),
                    self.session._u32(matrix.rows),
                    self.session._u32(min(count, matrix.k)),
                ],
                ((matrix.rows + 7) // 8, 1, 1),
                (64, 1, 1),
                origins=[matrix.origin, 0, 0],
            )
            return
        width = min(count, matrix.k)
        self.session.encode(
            "write_row",
            [source, self.act],
            [self.session._u32(width), self.session._u32(matrix.k_pad)],
            *self._grid(matrix.k_pad),
        )
        weight = matrix.data
        kind = matrix.kind
        scales = matrix.scales
        self.session.matmul(
            kind,
            self.act,
            weight,
            scales,
            self.out,
            _ROWS,
            matrix.rows,
            matrix.k_pad,
            commit=False,
        )

    def _norm(self, source: object, weight: object, destination: object) -> None:
        self.session.encode(
            "rmsnorm",
            [source, weight, destination],
            [self.session._u32(HIDDEN), self.session._f32(RMS_EPS)],
            (1, 1, 1),
            (256, 1, 1),
        )

    def _qk_norm(self, layer: Layer, dim: int, kv_heads: int) -> None:
        self._norm_heads(self.query, layer.q_norm, Q_HEADS, dim)
        self._norm_heads(self.key, layer.k_norm, kv_heads, dim)

    def _norm_heads(self, values: object, weight: object, heads: int, dim: int) -> None:
        self.session.encode(
            "rmsnorm_heads",
            [values, weight],
            [self.session._u32(dim), self.session._f32(RMS_EPS)],
            (heads, 1, 1),
            (min(dim, 256), 1, 1),
        )

    def _copy(self, source: object, destination: object, count: int) -> None:
        self.session.encode(
            "scale_copy",
            [source, destination],
            [self.session._f32(1.0), self.session._u32(count)],
            *self._grid(count),
        )

    def _axpy(self, hidden: object, update: object, scale: float, count: int) -> None:
        self.session.encode(
            "residual_scale",
            [hidden, update],
            [self.session._f32(scale), self.session._u32(count)],
            *self._grid(count),
        )

    def _ensure(self, count: int) -> None:
        if count <= self.capacity:
            return
        larger = max(self.capacity * 2, count)
        self.local_k = [
            self._grow(handle, LOCAL_HEADS, LOCAL_DIM, larger) for handle in self.local_k
        ]
        self.local_v = [
            self._grow(handle, LOCAL_HEADS, LOCAL_DIM, larger) for handle in self.local_v
        ]
        self.global_k = [
            self._grow(handle, GLOBAL_HEADS, GLOBAL_DIM, larger) for handle in self.global_k
        ]
        self.global_v = [
            self._grow(handle, GLOBAL_HEADS, GLOBAL_DIM, larger) for handle in self.global_v
        ]
        self.capacity = larger

    def _grow(self, handle: object, heads: int, dim: int, capacity: int) -> object:
        if handle is None:
            return None
        fresh = self.session.zeros(capacity * heads * dim * 2)
        memory.copy(handle, 0, fresh, 0, self.filled * heads * dim * 2)
        memory.release(handle)
        return fresh

    def _buffers(self) -> None:
        session = self.session
        self.hidden = session.zeros(HIDDEN * 4)
        self.unit = session.zeros(GLOBAL_DIM * 4)
        memory.store(self.unit, 0, np.ones(GLOBAL_DIM, dtype=np.float32).tobytes())
        self.span = session.zeros(_CHUNK * HIDDEN * 4)
        self.span_n = session.zeros(_CHUNK * HIDDEN * 4)
        self.span_r = session.zeros(_CHUNK * HIDDEN * 4)
        self.span_f = session.zeros(_CHUNK * HIDDEN * 4)
        self.span_m = session.zeros(_CHUNK * HIDDEN * 4)
        self.span_g = session.zeros(_CHUNK * HIDDEN * 4)
        self.span_w = session.zeros(_CHUNK * DENSE_WIDTH * 4)
        self.span_d = session.zeros(_CHUNK * DENSE_WIDTH * 4)
        self.span_q = session.zeros(_CHUNK * Q_HEADS * GLOBAL_DIM * 4)
        self.span_k = session.zeros(_CHUNK * LOCAL_HEADS * GLOBAL_DIM * 4)
        self.span_v = session.zeros(_CHUNK * LOCAL_HEADS * GLOBAL_DIM * 4)
        self.span_a = session.zeros(_CHUNK * Q_HEADS * GLOBAL_DIM * 4)
        self.span_p = session.zeros(_CHUNK * Q_HEADS * GLOBAL_DIM * 4)
        self.span_h = session.zeros(_CHUNK * HIDDEN * 2)
        self.span_ah = session.zeros(_CHUNK * _MAX_K * 2)
        self.half_w = session.zeros(8192 * HIDDEN * 2)
        self.gather_i = session.zeros(_CHUNK * TOP_K * 4)
        self.gather_s = session.zeros(_CHUNK * TOP_K * 4)
        self.qmm_jobs = session.zeros(_CHUNK * 16)
        self.qmm_origins = session.zeros(128 * 8)
        self.slot_act = session.zeros(_SLOTS * _SLOT * align_k(HIDDEN, FP4) * 2)
        self.slot_out = session.zeros(_SLOTS * _SLOT * HIDDEN * 4)
        self.slot_mid = session.zeros(_SLOTS * _SLOT * EXPERT_WIDTH * 4)
        self.slot_which = session.zeros(_SLOTS * 4)
        self.slot_index = session.zeros(_SLOTS * _SLOT * 4)
        self.slot_row = session.zeros(_SLOTS * _SLOT * 4)
        self.slot_token = session.zeros(_SLOTS * _SLOT * 4)
        self.slot_scale = session.zeros(_SLOTS * _SLOT * 4)
        self.route_i = session.zeros(_CHUNK * TOP_K * 4)
        self.route_w = session.zeros(_CHUNK * TOP_K * 4)
        self.norm = session.zeros(HIDDEN * 4)
        self.routed = session.zeros(HIDDEN * 4)
        self.mixed = session.zeros(HIDDEN * 4)
        self.ffn = session.zeros(HIDDEN * 4)
        self.wide = session.zeros(DENSE_WIDTH * 4)
        self.mid = session.zeros(DENSE_WIDTH * 4)
        self.query = session.zeros(Q_HEADS * GLOBAL_DIM * 4)
        self.key = session.zeros(LOCAL_HEADS * GLOBAL_DIM * 4)
        self.value = session.zeros(LOCAL_HEADS * LOCAL_DIM * 4)
        self.attn = session.zeros(Q_HEADS * GLOBAL_DIM * 4)
        self.act = session.zeros(_ROWS * _MAX_K * 2)
        self.out = session.zeros(_ROWS * _MAX_N * 4)
        self.expert_index = session.zeros(TOP_K * 4)
        self.expert_weight = session.zeros(TOP_K * 4)
        self.expert_up = session.zeros(TOP_K * 2 * EXPERT_WIDTH * 4)
        self.expert_hid = session.zeros(TOP_K * EXPERT_WIDTH * 4)
        self.expert_down = session.zeros(TOP_K * HIDDEN * 4)
        self.pick_score = [session.zeros(_PICK * 4), session.zeros(_PICK * 4)]
        self.pick_id = [session.zeros(_PICK * 4), session.zeros(_PICK * 4)]
        self.chosen = session.zeros(4)
        self.seen_ids = session.zeros(_SEEN * 4)

    def _cache(self) -> None:
        self.local_k = []
        self.local_v = []
        self.global_k = []
        self.global_v = []
        for _layer in range(LAYERS):
            if _layer in GLOBAL_LAYERS:
                self.local_k.append(None)
                self.local_v.append(None)
                self.global_k.append(
                    self.session.zeros(self.capacity * GLOBAL_HEADS * GLOBAL_DIM * 2)
                )
                self.global_v.append(
                    self.session.zeros(self.capacity * GLOBAL_HEADS * GLOBAL_DIM * 2)
                )
            else:
                self.global_k.append(None)
                self.global_v.append(None)
                self.local_k.append(self.session.zeros(self.capacity * LOCAL_HEADS * LOCAL_DIM * 2))
                self.local_v.append(self.session.zeros(self.capacity * LOCAL_HEADS * LOCAL_DIM * 2))

    def _load(self, path: Path) -> None:
        import json

        index = json.loads((path / "resident.json").read_text(encoding="utf-8"))
        blob = path / "resident.bin"
        found: dict[str, Matrix | object] = {}
        with blob.open("rb") as handle:
            for row in index:
                handle.seek(int(row["offset"]))
                raw = handle.read(int(row["nbytes"]))
                shape = tuple(int(item) for item in row["shape"])
                plane = str(row["plane"])
                name = str(row["name"])
                if plane == "fp16" and len(shape) == 1:
                    found[name] = self._expand_fp16(raw)
                    if name.endswith("router.scale"):
                        self._scale_vector(found[name], HIDDEN**-0.5)
                elif plane == "fp16" and len(shape) == 2:
                    found[str(row["name"])] = _matrix_from(self.session, raw, shape, "fp16")
                elif plane == Q40 and len(shape) == 2:
                    found[str(row["name"])] = _matrix_from(self.session, raw, shape, plane)
        self._map = found
        self.embed = found["model.language_model.embed_tokens.weight"]
        self.final_norm = found["model.language_model.norm.weight"]
        self.layers = []
        for layer in range(LAYERS):
            base = f"model.language_model.layers.{layer}"
            item = Layer(
                scalar=struct.unpack("<f", memory.load(found[f"{base}.layer_scalar"], 0, 4))[0],
                input_norm=found[f"{base}.input_layernorm.weight"],
                post_attn=found[f"{base}.post_attention_layernorm.weight"],
                pre=found[f"{base}.pre_feedforward_layernorm.weight"],
                pre2=found[f"{base}.pre_feedforward_layernorm_2.weight"],
                post=found[f"{base}.post_feedforward_layernorm.weight"],
                post1=found[f"{base}.post_feedforward_layernorm_1.weight"],
                post2=found[f"{base}.post_feedforward_layernorm_2.weight"],
                q_norm=found[f"{base}.self_attn.q_norm.weight"],
                k_norm=found[f"{base}.self_attn.k_norm.weight"],
                q=found[f"{base}.self_attn.q_proj.weight"],
                k=found[f"{base}.self_attn.k_proj.weight"],
                v=found.get(f"{base}.self_attn.v_proj.weight"),
                o=found[f"{base}.self_attn.o_proj.weight"],
                router=found[f"{base}.router.proj.weight"],
                router_scale=found[f"{base}.router.scale"],
                expert_scale=found[f"{base}.router.per_expert_scale"],
                gate=found[f"{base}.mlp.gate_proj.weight"],
                up=found[f"{base}.mlp.up_proj.weight"],
                down=found[f"{base}.mlp.down_proj.weight"],
                global_attn=layer in GLOBAL_LAYERS,
            )
            self.layers.append(item)
        offsets = array.array("Q")
        offsets.frombytes((path / "experts" / "offsets.bin").read_bytes())
        self.offsets = offsets
        self.expert_offsets = self.session.upload(offsets.tobytes())
        self.fp4_gate_code = []
        self.fp4_gate_scale = []
        self.fp4_down_code = []
        self.fp4_down_scale = []
        gate_rows = 2 * EXPERT_WIDTH
        for layer in range(LAYERS):
            raw = (path / "experts" / f"{layer}.bin").read_bytes()
            atlas = self.session.upload(raw)
            del raw
            self._repack_layer(atlas, layer, gate_rows)
            memory.release(atlas)

    def _repack_layer(self, atlas: object, layer: int, gate_rows: int) -> None:
        k_gate_q4 = align_k(HIDDEN, Q40)
        k_down_q4 = align_k(EXPERT_WIDTH, Q40)
        k_gate = align_k(HIDDEN, FP4)
        k_down = align_k(EXPERT_WIDTH, FP4)
        gate_code = self.session.zeros(128 * gate_rows * (k_gate // 2))
        gate_scale = self.session.zeros(128 * gate_rows * (k_gate // 32))
        down_code = self.session.zeros(128 * HIDDEN * (k_down // 2))
        down_scale = self.session.zeros(128 * HIDDEN * (k_down // 32))
        extra = matrix_nbytes(gate_rows, HIDDEN, Q40)
        self._repack(atlas, layer, gate_code, gate_scale, k_gate_q4, k_gate, gate_rows, 0)
        self._repack(atlas, layer, down_code, down_scale, k_down_q4, k_down, HIDDEN, extra)
        self.session.flush()
        self.fp4_gate_code.append(gate_code)
        self.fp4_gate_scale.append(gate_scale)
        self.fp4_down_code.append(down_code)
        self.fp4_down_scale.append(down_scale)

    def _repack(
        self,
        atlas: object,
        layer: int,
        codes: object,
        scales: object,
        k_q4: int,
        k_fp4: int,
        rows: int,
        extra: int,
    ) -> None:
        blocks = rows * (k_q4 // 32)
        self.session.encode(
            "repack_q4_fp4",
            [atlas, self.expert_offsets, codes, scales],
            [
                self.session._u32(k_q4),
                self.session._u32(k_fp4),
                self.session._u32(rows),
                self.session._u32(extra),
            ],
            ((blocks + 255) // 256, 128, 1),
            (256, 1, 1),
            origins=[0, layer * 128 * 8, 0, 0],
        )

    def _scale_vector(self, handle: object, factor: float) -> None:
        raw = bytearray(memory.load(handle, 0, HIDDEN * 4))
        for index in range(HIDDEN):
            value = struct.unpack_from("<f", raw, index * 4)[0] * factor
            struct.pack_into("<f", raw, index * 4, value)
        memory.store(handle, 0, bytes(raw))

    def _expand_fp16(self, raw: bytes) -> object:
        count = len(raw) // 2
        source = self.session.upload(raw)
        dest = self.session.zeros(count * 4)
        self.session.dispatch(
            "expand_fp16",
            [source, dest],
            [self.session._u32(count)],
            *self._grid(count),
        )
        memory.release(source)
        return dest

    def _draw(
        self,
        temperature: float,
        top_p: float,
        top_k: int,
        presence: float,
        seen: list[int],
        rng: random.Random,
    ) -> int:
        if presence:
            unique = list(dict.fromkeys(seen))
            if len(unique) > _SEEN:
                message = "presence window exceeded the device buffer"
                raise RuntimeError(message)
            packed = struct.pack(f"<{len(unique)}I", *unique)
            memory.store(self.seen_ids, 0, packed)
            self.session.encode(
                "presence_penalty",
                [self.out, self.seen_ids],
                [self.session._u32(len(unique)), self.session._f32(presence)],
                *self._grid(len(unique)),
            )
        if temperature == 0.0:
            self._reduce_top(self.out, VOCAB, 1, 1.0)
            return struct.unpack("<I", self._bytes(self._kept_ids, 4))[0]
        if not 0 < top_k <= 64:
            message = "top_k must be from 1 to 64"
            raise ValueError(message)
        self._reduce_top(self.out, VOCAB, top_k, temperature)
        self.session.encode(
            "sample_topk",
            [self._kept_scores, self._kept_ids, self.chosen],
            [
                self.session._u32(self._kept_count),
                self.session._f32(top_p),
                self.session._f32(rng.random()),
            ],
            (1, 1, 1),
            (1, 1, 1),
        )
        return struct.unpack("<I", self._bytes(self.chosen, 4))[0]

    def _reduce_top(self, scores: object, count: int, keep: int, temperature: float) -> None:
        src_scores = scores
        src_ids = self.pick_id[0]
        slot = 0
        first = True
        while count > keep:
            groups = (count + 255) // 256
            produced = groups * keep
            if produced > _PICK:
                message = "top-k scratch is smaller than this vocabulary"
                raise RuntimeError(message)
            self.session.encode(
                "topk_groups",
                [src_scores, src_ids, self.pick_score[slot], self.pick_id[slot]],
                [
                    self.session._u32(count),
                    self.session._u32(keep),
                    self.session._f32(temperature if first else 1.0),
                    self.session._u32(1 if first else 0),
                ],
                (groups, 1, 1),
                (256, 1, 1),
            )
            src_scores = self.pick_score[slot]
            src_ids = self.pick_id[slot]
            slot = 1 - slot
            count = produced
            first = False
        self._kept_scores = src_scores
        self._kept_ids = src_ids
        self._kept_count = count


def _matrix_from(session: Session, raw: bytes, shape: tuple[int, ...], plane: str) -> Matrix:
    rows, k = int(shape[0]), int(shape[1])
    if plane == "fp16":
        return Matrix(session.upload(raw), None, "matmul_f16", rows, k, k)
    return Matrix(session.upload(raw), None, Q40, rows, k, align_k(k, Q40))


def _rate(phase: str, count: int, elapsed: float) -> str:
    rate = 0.0 if elapsed <= 0.0 else count / elapsed
    return f"{phase} {rate:.2f} tok/s ({count} tok in {elapsed:.2f}s)"
