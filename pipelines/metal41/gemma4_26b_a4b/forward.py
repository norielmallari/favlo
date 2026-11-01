"""Text forward for Gemma 4 26B-A4B.

The block matches the published Gemma 4 text layer. Attention and both
feed-forward branches add at scale one. ``layer_scalar`` multiplies the
hidden state after the feed-forward residual. The dense MLP uses
``pre_feedforward_layernorm`` and ``post_feedforward_layernorm_1``. Routed
experts use ``pre_feedforward_layernorm_2`` and
``post_feedforward_layernorm_2``. The router norm is the full router scale
times ``1/sqrt(hidden)``. The embedding row is multiplied by ``sqrt(hidden)``.
Decode reads packed q4_0 one token at a time, including the routed experts.
Those stay the q4_0 layer file. A full budget keeps every layer resident. A short budget keeps as many as fit and reads a miss into the oldest slot. One threadgroup owns eight output rows.
Prefill attends a chunk of 1024 tokens in one kernel. A device kernel
groups the routed rows into slots of 64. The matmul grid is an indirect
dispatch, so the host does not read the expert ids. Every occupied slot
is multiplied with ``matmul_fp4_64_batch``. The down projection pads K
from 704 to 768. A chunk submits every ten layers. The prefill count
stays short of the prompt until the last of those buffers has finished.
A dense batch of eight or fewer
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
import ctypes
import fcntl
import os
import random
import struct
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from favlo.console import format_note, format_rate
from favlo.memory import memory
from favlo.prefix import (
    DEFAULT_BUDGET,
    PrefixSession,
    PrefixStore,
    checkpoint_digest,
    default_prefix_dir,
)
from gemma4_26b_a4b.geometry import (
    CONTEXT,
    DENSE_WIDTH,
    EXPERT_WIDTH,
    GLOBAL_DIM,
    GLOBAL_HEADS,
    GLOBAL_LAYERS,
    HIDDEN,
    IMAGE_TOKEN,
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
from gemma4_26b_a4b.atlas import ExpertAtlas, kv_bytes, kv_margin, layer_slots
from gemma4_26b_a4b.planes import Q40, align_k, matrix_nbytes
from gemma4_26b_a4b.prefix import RuntimeAdapter
from gemma4_26b_a4b.session import Session

_ROWS = 8


_CHUNK = 1024
_SLOT = 64
_BATCH = 128
_FP4_DOWN = align_k(EXPERT_WIDTH, "fp4")
_SLOTS = 256
_MAX_K = 8192
_MAX_N = VOCAB
_PICK = 65536
_SEEN = 8192
_STOP = (1, 106, 50)


def _contains_stop(root: Path, tokens: list[int], stops: list[str]) -> bool:
    from gemma4_26b_a4b.chat import decode_text

    text = decode_text(root, tokens)
    return any(stop in text for stop in stops)


# Darwin fcntl. A whole expert file is read once, so the page cache is skipped.
_F_RDAHEAD = 45
_F_NOCACHE = 48
_PAGE = 4096
_READ = 64 << 20


def _page_aligned(view: memoryview) -> bool:
    if len(view) % _PAGE:
        return False
    try:
        address = ctypes.addressof(ctypes.c_char.from_buffer(view))
    except (TypeError, BufferError):
        return False
    return address % _PAGE == 0


def _fill(raw: object, view: memoryview, start: int) -> None:
    """Copy ``len(view)`` bytes from ``start`` into a shared Metal span."""
    nocache = False
    try:
        fcntl.fcntl(raw.fileno(), _F_NOCACHE, 0)
        fcntl.fcntl(raw.fileno(), _F_RDAHEAD, 1)
    except OSError:
        pass
    if _page_aligned(view) and start % _PAGE == 0:
        try:
            fcntl.fcntl(raw.fileno(), _F_NOCACHE, 1)
            nocache = True
        except OSError:
            nocache = False
    done = 0
    size = len(view)
    try:
        _fill_view(raw, view, start, nocache, done, size)
    finally:
        try:
            fcntl.fcntl(raw.fileno(), _F_NOCACHE, 0)
        except OSError:
            pass


def _fill_view(raw: object, view: memoryview, start: int, nocache: bool, done: int, size: int) -> None:
    while done < size:
        n = min(_READ, size - done)
        window = view[done : done + n]
        if nocache and (n % _PAGE or not _page_aligned(window)):
            fcntl.fcntl(raw.fileno(), _F_NOCACHE, 0)
            nocache = False
        try:
            got = raw.readinto(window)
        except OSError:
            if not nocache:
                raise
            fcntl.fcntl(raw.fileno(), _F_NOCACHE, 0)
            nocache = False
            raw.seek(start + done)
            continue
        except (TypeError, BufferError):
            data = raw.read(n)
            if not data:
                got = 0
            else:
                window[: len(data)] = data
                got = len(data)
        if not got:
            message = "short read"
            raise EOFError(message)
        done += got


def _read_span(handle: object, path: Path, start: int = 0, size: int | None = None) -> None:
    total = path.stat().st_size if size is None else size
    view = memory.map(handle, 0, total)
    raw = open(path, "rb", buffering=0)
    try:
        if start:
            raw.seek(start)
        _fill(raw, view, start)
    finally:
        raw.close()


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
    """One packed record. Expert layers stay resident while they fit."""

    def __init__(
        self,
        path: Path,
        *,
        owner: str,
        kv_cache: str = "e4m3",
        console: object = None,
        max_memory: str | int | None = None,
    ) -> None:
        from favlo.memory import memory

        self.console = console
        self._request_id = 0
        self._prefill_progress = None
        self._load_progress = None
        self._budget = memory.budget() if not isinstance(max_memory, int) else max_memory
        self._closed = False
        self.root = path
        self.owner = owner
        self._owner = owner
        self.kv_cache = kv_cache
        self._stop = False
        self.session = Session(owner)
        self.filled = 0
        self.prompt_tokens = 0
        self._vision = None
        self.capacity = 256
        self.atlas = None
        self.prefix = None
        resident_bytes = (path / "resident.bin").stat().st_size
        layer_bytes = (path / "experts" / "0.bin").stat().st_size
        self._load_progress = self._begin_load(resident_bytes + LAYERS * layer_bytes)
        try:
            self._load(path)
            self._buffers()
            self._cache()
            self._open_atlas(path, resident_bytes, layer_bytes)
            self._cache_experts()
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
        except BaseException:
            self._finish_load("")
            self.close()
            raise

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        prefix = getattr(self, "prefix", None)
        if prefix is not None:
            try:
                prefix.flush()
            except OSError:
                pass
        session = getattr(self, "session", None)
        if session is not None:
            session.close()

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
        on_token: Any = None,
        images: list[bytes] | None = None,
        stops: list[str] | None = None,
    ) -> tuple[list[int], str, str]:
        """Sample after ``prompt``. Returns tokens, the prefill line, and the decode line."""
        pictures = list(images or [])
        halt_at = [item for item in (stops or []) if item]
        visual = bool(pictures) or IMAGE_TOKEN in prompt
        self._hit_limit = False
        rng = random.Random(seed)
        self._note("prompt", tokens=len(prompt))
        if visual:
            cut = prompt.index(IMAGE_TOKEN) if IMAGE_TOKEN in prompt else len(prompt)
            hit = self.prefix.align(prompt[:cut])
        else:
            hit = self.prefix.align(prompt)
        if self.prefix.loaded:
            self._note("prefix_load", tokens=hit)
        self._note("cache", hit=hit)
        pending = self._visual_tokens(prompt, hit, pictures) if visual else len(prompt) - hit
        prefill_row = self._begin_phase("prefill", max(pending, 0))
        self._prefill_progress = prefill_row
        decode_row = None
        try:
            start = time.perf_counter()
            if visual:
                self._prefill_visual(prompt, hit, pictures)
            else:
                self._prefill(prompt[hit:])
            self.session.flush()
            prompt_n = self.filled - hit if not visual else pending
            self.prompt_tokens = self.filled
            self._note_prefill(prompt_n)
            prefill_s = time.perf_counter() - start
            prefill_line = format_rate("prefill", prompt_n, prefill_s)
            self._prefill_progress = None
            self._end_phase(prefill_row, prefill_line)
            prefill_row = None
            if not visual:
                self.prefix.note(prompt[: self.filled])
                self._store_prefix()
            drawn: list[int] = []
            seen: list[int] = []
            decode_row = self._begin_phase("decode", limit, live=True)
            decode_start = time.perf_counter()
            if hit == len(prompt):
                self._logits()
            halted = False
            for _ in range(limit):
                if self._stop:
                    halted = True
                    break
                token = self._draw(temperature, top_p, top_k, presence, seen, rng)
                if token in _STOP:
                    halted = True
                    break
                drawn.append(token)
                seen.append(token)
                stopped = bool(halt_at) and _contains_stop(self.root, drawn, halt_at)
                if not stopped and on_token is not None:
                    on_token(token)
                if decode_row is not None:
                    decode_row.done = len(drawn)
                self._step(token)
                if stopped:
                    halted = True
                    break
            if not halted:
                self._hit_limit = True
            decode_s = time.perf_counter() - decode_start
            if not visual:
                self.prefix.note(prompt + drawn)
                self._store_prefix()
            decode_line = format_rate("decode", len(drawn), decode_s)
            self._end_phase(decode_row, decode_line)
            decode_row = None
            return drawn, prefill_line, decode_line
        finally:
            self._prefill_progress = None
            if visual:
                self.prefix.tokens = []
                self.prefix.dirty = False
                self.prefix.loaded = False
            if prefill_row is not None:
                self._end_phase(prefill_row, "")
            if decode_row is not None:
                self._end_phase(decode_row, "")

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

    def _prefill(self, tokens: list[int], *, finish: bool = True) -> None:
        if self._stop or not tokens:
            return
        for start in range(0, len(tokens), _CHUNK):
            if self._stop:
                return
            span = tokens[start : start + _CHUNK]
            last = finish and start + len(span) == len(tokens)
            self._span(span, last, noted=start)

    def _prefill_visual(self, prompt: list[int], hit: int, images: list[bytes]) -> None:
        from gemma4_26b_a4b.vision import VisionTower

        if prompt.count(IMAGE_TOKEN) != len(images):
            message = "image parts do not match the image tokens"
            raise RuntimeError(message)
        tower = self._vision
        if tower is None:
            tower = VisionTower(self.session, self.root, self._owner)
            self._vision = tower
        pos = hit
        image_i = 0
        while pos < len(prompt):
            if self._stop:
                return
            if prompt[pos] == IMAGE_TOKEN:
                handle, count = tower.encode(images[image_i])
                image_i += 1
                pos += 1
                self._accept_rows(handle, count, finish=pos == len(prompt))
                continue
            end = pos + 1
            while end < len(prompt) and prompt[end] != IMAGE_TOKEN:
                end += 1
            self._prefill(prompt[pos:end], finish=end == len(prompt))
            pos = end

    def _visual_tokens(self, prompt: list[int], hit: int, images: list[bytes]) -> int:
        from gemma4_26b_a4b.vision import image_soft_tokens

        tail = prompt[hit:]
        if tail.count(IMAGE_TOKEN) != len(images):
            message = "image parts do not match the image tokens"
            raise RuntimeError(message)
        soft = sum(image_soft_tokens(blob) for blob in images)
        return len(tail) - len(images) + soft

    def _span(self, tokens: list[int], last: bool, *, noted: int) -> None:
        count = len(tokens)
        self._ensure(self.filled + count)
        self._embed_span(tokens)
        self._run_span(count, last, noted=noted)

    def _accept_rows(self, source: object, count: int, *, finish: bool) -> None:
        if count > _CHUNK:
            message = f"soft tokens {count} exceed the prefill chunk"
            raise RuntimeError(message)
        self._ensure(self.filled + count)
        self.session.encode(
            "scale_copy",
            [source, self.span],
            [self.session._f32(1.0), self.session._u32(count * HIDDEN)],
            *self._grid(count * HIDDEN),
        )
        self._run_span(count, finish, noted=self.filled)

    def _run_span(self, count: int, last: bool, *, noted: int) -> None:
        for index, layer in enumerate(self.layers):
            self._block_span(layer, index, count, noted=noted)
            if (index + 1) % 10 == 0 and index + 1 != len(self.layers):
                self.session.flush()
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

    def _block_span(self, layer: Layer, index: int, count: int, *, noted: int) -> None:
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
        self._feed_span(layer, index, count, noted=noted)

    def _feed_span(self, layer: Layer, index: int, count: int, *, noted: int) -> None:
        self._route_span(layer, count)
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
        self._experts_span(index, count, noted=noted)
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

    def _route_span(self, layer: Layer, count: int) -> None:
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

    def _experts_span(self, layer: int, count: int, *, noted: int) -> None:
        """Group the route on device and multiply every occupied slot.

        ``which`` selects the expert id inside the full 128-expert pack.
        Each slot is padded to 64 tokens. Gate K is 2816. Down K is 768.
        The grid is indirect, so this span does not read the ids back.
        """
        start = noted + (count * layer) // LAYERS
        end = noted + (count * (layer + 1)) // LAYERS
        self._note_prefill_open(start)
        if layer >= len(self.fp4_gate_code):
            self._repack_experts(layer)
        rows = _SLOTS * _SLOT
        self.session.encode(
            "group_expert_slots",
            [
                self.route_i,
                self.route_w,
                self.slot_which,
                self.slot_index,
                self.slot_scale,
                self.slot_grid,
            ],
            [
                self.session._u32(count),
                self.session._u32(HIDDEN),
                self.session._u32(2 * EXPERT_WIDTH),
                self.session._u32(EXPERT_WIDTH),
                self.session._u32(_FP4_DOWN),
            ],
            (1, 1, 1),
            (256, 1, 1),
        )
        self.session.split()
        threads = (256, 1, 1)
        self.session.encode_indirect(
            "gather_half_slots",
            [self.span_n, self.expert_ah, self.slot_index],
            [self.session._u32(HIDDEN), self.session._u32(HIDDEN), self.session._u32(rows)],
            self.slot_grid,
            0,
            threads,
        )
        self._matmul_fp4_indirect(layer, 2 * EXPERT_WIDTH, HIDDEN, gate=True, record=1)
        self.session.encode_indirect(
            "geglu_rows",
            [self.slot_out, self.slot_mid],
            [self.session._u32(EXPERT_WIDTH), self.session._u32(rows)],
            self.slot_grid,
            32,
            threads,
        )
        self.session.encode_indirect(
            "pad_half_rows",
            [self.slot_mid, self.expert_ah],
            [
                self.session._u32(EXPERT_WIDTH),
                self.session._u32(_FP4_DOWN),
                self.session._u32(EXPERT_WIDTH),
                self.session._u32(rows),
            ],
            self.slot_grid,
            48,
            threads,
        )
        self._matmul_fp4_indirect(layer, HIDDEN, _FP4_DOWN, gate=False, record=4)
        self.session.encode_indirect(
            "scatter_slot_rows",
            [self.span_m, self.slot_out, self.slot_index, self.slot_scale],
            [self.session._u32(HIDDEN), self.session._u32(rows)],
            self.slot_grid,
            80,
            threads,
        )
        self._note_prefill_open(end)

    def _matmul_fp4_indirect(
        self, layer: int, cols: int, inner: int, *, gate: bool, record: int
    ) -> None:
        width = int(self.session.pipes["matmul_fp4_64_batch"].threadExecutionWidth())
        cached = layer < len(self.fp4_gate_code)
        if gate:
            codes = self.fp4_gate_code[layer] if cached else self.gate_code
            scales = self.fp4_gate_scale[layer] if cached else self.gate_scale
        else:
            codes = self.fp4_down_code[layer] if cached else self.down_code
            scales = self.fp4_down_scale[layer] if cached else self.down_scale
        self.session.encode_indirect(
            "matmul_fp4_64_batch",
            [self.expert_ah, codes, scales, self.slot_out, self.slot_which],
            [self.session._u32(_SLOT), self.session._u32(cols), self.session._u32(inner)],
            self.slot_grid,
            record * 16,
            (width * 4, 1, 1),
        )

    def _repack_experts(self, layer: int) -> None:
        packed = self.atlas.bind(self.session, layer)
        gate_rows = 2 * EXPERT_WIDTH
        self._repack(
            packed, layer, self.gate_code, self.gate_scale, HIDDEN, HIDDEN, gate_rows, 0
        )
        self._repack(
            packed,
            layer,
            self.down_code,
            self.down_scale,
            EXPERT_WIDTH,
            _FP4_DOWN,
            HIDDEN,
            self._gate_bytes,
        )

    def _repack(
        self,
        packed: object,
        layer: int,
        codes: object,
        scales: object,
        k_q4: int,
        k_fp4: int,
        rows: int,
        extra: int,
    ) -> None:
        blocks = rows * (align_k(k_q4, Q40) // 32)
        self.session.encode(
            "repack_q4_fp4",
            [packed, self.expert_offsets, codes, scales],
            [
                self.session._u32(align_k(k_q4, Q40)),
                self.session._u32(k_fp4),
                self.session._u32(rows),
                self.session._u32(extra),
            ],
            ((blocks + 255) // 256, 128, 1),
            (256, 1, 1),
            origins=[0, layer * 128 * 8, 0, 0],
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
        keys = self.filled + 1
        if window:
            keys = min(keys, window)
        simds = 8 if keys >= 1024 else 4 if keys >= 256 else 1
        self.session.encode(
            "attend_span" if simds == 1 else "attend_decode",
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
            (32 * simds, 1, 1),
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
        self._q4_gemv(layer, self.norm, self.expert_up, gate_rows, HIDDEN, HIDDEN, 0, 0)
        self.session.encode(
            "geglu_rows",
            [self.expert_up, self.expert_hid],
            [self.session._u32(EXPERT_WIDTH), self.session._u32(TOP_K)],
            *self._grid(TOP_K * EXPERT_WIDTH),
        )
        self._q4_gemv(
            layer,
            self.expert_hid,
            self.expert_down,
            HIDDEN,
            EXPERT_WIDTH,
            EXPERT_WIDTH,
            self._gate_bytes,
            EXPERT_WIDTH,
        )
        self.session.encode(
            "mix_experts",
            [self.mixed, self.expert_down, self.expert_weight],
            [self.session._u32(HIDDEN), self.session._u32(TOP_K)],
            *self._grid(HIDDEN),
        )

    def _q4_gemv(
        self,
        layer: int,
        activation: object,
        destination: object,
        rows: int,
        k_pad: int,
        active: int,
        extra: int,
        act_stride: int,
    ) -> None:
        groups = (rows + 7) // 8
        self.session.encode(
            "gemv_q4_0_routed",
            [
                self.atlas.bind(self.session, layer),
                activation,
                destination,
                self.expert_offsets,
                self.expert_index,
            ],
            [
                self.session._u32(k_pad),
                self.session._u32(rows),
                self.session._u32(active),
                self.session._u32(groups),
                self.session._u32(extra),
                self.session._u32(act_stride),
            ],
            (groups * TOP_K, 1, 1),
            (64, 1, 1),
            origins=[0, 0, 0, layer * 128 * 8, 0],
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
        if count > CONTEXT:
            message = f"prompt length {count} is past {CONTEXT}"
            raise RuntimeError(message)
        larger = min(CONTEXT, max(self.capacity * 2, count))
        room = self._atlas_room(larger, margin=kv_margin(larger))
        if self.atlas is None or layer_slots(room, self.atlas.layer_bytes, LAYERS) < 1:
            message = "the key/value cache does not fit the memory budget"
            raise RuntimeError(message)
        self.atlas.retune(self.session, room)
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
        self._note("context", tokens=larger, atlas=self.atlas.slots)

    def _atlas_room(self, capacity: int, *, margin: int = 0) -> int:
        owner = memory.owner_bytes().get(self._owner, 0)
        atlas = 0 if self.atlas is None else self.atlas.nbytes
        fixed = owner - atlas - kv_bytes(self.capacity)
        return self._budget - fixed - kv_bytes(capacity) - margin

    def _open_atlas(self, path: Path, resident_bytes: int, layer_bytes: int) -> None:
        room = self._atlas_room(self.capacity)
        slots = layer_slots(room, layer_bytes, LAYERS)
        if slots < 1:
            message = "the record does not fit the memory budget"
            raise RuntimeError(message)
        progress = self._load_progress
        if progress is not None:
            progress.total = resident_bytes + slots * layer_bytes
        self.atlas = ExpertAtlas(self._owner, path, layer_bytes, slots, _read_span)
        summary = self.atlas.load(self._note_load)
        self._finish_load(summary)
        self._note("context", tokens=self.capacity, atlas=self.atlas.slots)

    def _cache_experts(self) -> None:
        """Repack every layer that still fits after the atlas and the cache."""
        gate_code_n = 128 * (2 * EXPERT_WIDTH) * (HIDDEN // 2)
        gate_scale_n = 128 * (2 * EXPERT_WIDTH) * (HIDDEN // 32)
        down_code_n = 128 * HIDDEN * (_FP4_DOWN // 2)
        down_scale_n = 128 * HIDDEN * (_FP4_DOWN // 32)
        per = gate_code_n + gate_scale_n + down_code_n + down_scale_n
        owner = memory.owner_bytes().get(self._owner, 0)
        reserve = kv_bytes(8192) + self.atlas.layer_bytes + kv_margin(8192)
        room = self._budget - owner - reserve
        count = 0 if per <= 0 or room < per else min(LAYERS, room // per)
        self.fp4_gate_code = []
        self.fp4_gate_scale = []
        self.fp4_down_code = []
        self.fp4_down_scale = []
        for layer in range(count):
            gate_code = self.session.zeros(gate_code_n)
            gate_scale = self.session.zeros(gate_scale_n)
            down_code = self.session.zeros(down_code_n)
            down_scale = self.session.zeros(down_scale_n)
            packed = self.atlas.bind(self.session, layer)
            gate_rows = 2 * EXPERT_WIDTH
            self._repack(packed, layer, gate_code, gate_scale, HIDDEN, HIDDEN, gate_rows, 0)
            self._repack(
                packed,
                layer,
                down_code,
                down_scale,
                EXPERT_WIDTH,
                _FP4_DOWN,
                HIDDEN,
                self._gate_bytes,
            )
            self.fp4_gate_code.append(gate_code)
            self.fp4_gate_scale.append(gate_scale)
            self.fp4_down_code.append(down_code)
            self.fp4_down_scale.append(down_scale)
        self.session.flush()

    def _store_prefix(self) -> None:
        try:
            path = self.prefix.flush()
        except OSError as exc:
            self._note("prefix_store_failed", error=exc)
            return
        if path is not None:
            self._note("prefix_store", tokens=len(self.prefix.tokens))

    def _status(self, line: str) -> None:
        console = self.console
        if console is None:
            print(line, flush=True)
            return
        console.log(line)

    def _begin_load(self, total: int) -> object:
        console = self.console
        if console is None or total <= 0 or not getattr(console, "shows_bars", False):
            return None
        detail = getattr(console, "load_detail", "") or self.root.name
        return console.begin("load", total=total, unit="bytes", detail=detail)

    def _note_load(self, step: int) -> None:
        progress = self._load_progress
        if progress is None or step <= 0:
            return
        progress.done = min(progress.total, progress.done + step)

    def _finish_load(self, summary: str) -> None:
        progress = self._load_progress
        self._load_progress = None
        if progress is not None and not getattr(progress, "closed", False):
            console = self.console
            if console is not None:
                console.end(progress, summary)
                return
        if summary:
            self._status(summary)

    def _note_prefill_open(self, done: int) -> None:
        progress = self._prefill_progress
        if progress is None or progress.total <= 1:
            self._note_prefill(done)
            return
        self._note_prefill(min(done, progress.total - 1))

    def _note_prefill(self, done: int) -> None:
        progress = self._prefill_progress
        if progress is None:
            return
        capped = min(progress.total, done)
        if capped > progress.done:
            progress.done = capped

    def _note(self, topic: str, **fields: object) -> None:
        console = self.console
        note = getattr(console, "note", None)
        if callable(note):
            note(topic, **fields)
            return
        self._status(format_note(topic, **fields))

    def _begin_phase(self, phase: str, total: int, *, live: bool = False) -> object:
        console = self.console
        if console is None:
            return None
        if live:
            if not getattr(console, "live_decode", False):
                return None
            total = 0
        elif total <= 0:
            return None
        detail = f"req #{self._request_id}" if self._request_id else ""
        return console.begin(
            phase,
            total=total,
            unit="tok",
            detail=detail,
            request=self._request_id or None,
        )

    def _end_phase(self, progress: object, summary: str) -> None:
        if progress is None:
            if summary and self.console is not None:
                self._status(summary)
            return
        if getattr(progress, "closed", False):
            return
        console = self.console
        if console is None:
            return
        console.end(progress, summary)

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
        memory.store(self.unit, 0, struct.pack("<f", 1.0) * GLOBAL_DIM)
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
        self.expert_ah = session.zeros(_SLOTS * _SLOT * HIDDEN * 2)
        self.slot_grid = session.zeros(6 * 16)
        self.gate_code = session.zeros(_BATCH * (2 * EXPERT_WIDTH) * (HIDDEN // 2))
        self.gate_scale = session.zeros(_BATCH * (2 * EXPERT_WIDTH) * (HIDDEN // 32))
        self.down_code = session.zeros(_BATCH * HIDDEN * (_FP4_DOWN // 2))
        self.down_scale = session.zeros(_BATCH * HIDDEN * (_FP4_DOWN // 32))
        self.gather_i = session.zeros(_CHUNK * TOP_K * 4)
        self.gather_s = session.zeros(_CHUNK * TOP_K * 4)
        self.slot_out = session.zeros(_SLOTS * _SLOT * HIDDEN * 4)
        self.slot_mid = session.zeros(_SLOTS * _SLOT * EXPERT_WIDTH * 4)
        self.slot_which = session.zeros(_SLOTS * 4)
        self.slot_index = session.zeros(_SLOTS * _SLOT * 4)
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
        raw_file = open(blob, "rb", buffering=0)
        try:
            for row in index:
                nbytes = int(row["nbytes"])
                offset = int(row["offset"])
                shape = tuple(int(item) for item in row["shape"])
                plane = str(row["plane"])
                name = str(row["name"])
                if "vision_tower" in name or "embed_vision" in name:
                    self._note_load(nbytes)
                    continue
                handle = memory.allocate(nbytes, owner=self._owner)
                if nbytes:
                    raw_file.seek(offset)
                    _fill(raw_file, memory.map(handle, 0, nbytes), offset)
                    self._note_load(nbytes)
                if plane == "fp16" and len(shape) == 1:
                    found[name] = self._expand_fp16(handle, nbytes)
                    if name.endswith("router.scale"):
                        self._scale_vector(found[name], HIDDEN**-0.5)
                elif plane == "fp16" and len(shape) == 2:
                    found[name] = _matrix_from(handle, shape, "fp16")
                elif plane == Q40 and len(shape) == 2:
                    found[name] = _matrix_from(handle, shape, plane)
        finally:
            raw_file.close()
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
        if len(offsets) != LAYERS * 128:
            message = "expert offset table does not match this geometry"
            raise RuntimeError(message)
        gate_bytes = matrix_nbytes(2 * EXPERT_WIDTH, HIDDEN, Q40)
        stride = gate_bytes + matrix_nbytes(HIDDEN, EXPERT_WIDTH, Q40)
        self._gate_bytes = gate_bytes
        self.offsets = offsets
        self.expert_offsets = self.session.upload(offsets.tobytes())
        down = array.array("Q", (item + gate_bytes for item in offsets))
        self.expert_down_offsets = self.session.upload(down.tobytes())
        sample = path / "experts" / "0.bin"
        if sample.stat().st_size != 128 * stride:
            message = "expert layer is not the q4_0 layout"
            raise RuntimeError(message)

    def _scale_vector(self, handle: object, factor: float) -> None:
        raw = bytearray(memory.load(handle, 0, HIDDEN * 4))
        for index in range(HIDDEN):
            value = struct.unpack_from("<f", raw, index * 4)[0] * factor
            struct.pack_into("<f", raw, index * 4, value)
        memory.store(handle, 0, bytes(raw))

    def _expand_fp16(self, source: object, nbytes: int) -> object:
        count = nbytes // 2
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


def _matrix_from(handle: object, shape: tuple[int, ...], plane: str) -> Matrix:
    rows, k = int(shape[0]), int(shape[1])
    if plane == "fp16":
        return Matrix(handle, None, "matmul_f16", rows, k, k)
    return Matrix(handle, None, Q40, rows, k, align_k(k, Q40))

