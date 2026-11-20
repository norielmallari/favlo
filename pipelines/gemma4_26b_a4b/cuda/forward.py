"""Text forward for Gemma 4 on CUDA.

Activations, norms, attention, and the cache are fp16. int8 and int4 weights
are dequantized in the step. Down K is stored as 768. The expert atlas streams
the eight chosen images.
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from favlo.console import format_rate
from favlo.memory import memory
from favlo.prefix import aim_session
from gemma4_26b_a4b.cuda.atlas import ExpertAtlas
from gemma4_26b_a4b.cuda.engine import Engine
from gemma4_26b_a4b.cuda.layout import (
    CONTEXT_STEP,
    DENSE_WIDTH,
    EOS,
    EOT,
    EXPERT_K,
    EXPERT_WIDTH,
    EXPERTS,
    GATE_ROWS,
    GLOBAL_DIM,
    GLOBAL_HEADS,
    GLOBAL_LAYERS,
    HIDDEN,
    LAYERS,
    LOCAL_DIM,
    LOCAL_HEADS,
    PREFILL_TILE,
    Q_HEADS,
    RMS_EPS,
    ROPE_GLOBAL,
    ROPE_LOCAL,
    ROPE_PARTIAL,
    SOFTCAP,
    TOP_K,
    VOCAB,
    WINDOW,
    expert_parts,
)
from gemma4_26b_a4b.cuda import progress as progress_mod
from gemma4_26b_a4b.cuda.prefix import open_session

_STOP = {EOS, EOT}
_GAIN = HIDDEN**0.5
_ROUTER_GAIN = HIDDEN**-0.5
_PARTS = {8: expert_parts("int8"), 4: expert_parts("int4")}


@dataclass
class Matrix:
    codes: int
    scales: int
    rows: int
    inner: int
    bits: int


@dataclass
class Layer:
    scalar: float
    input_norm: int
    post_attn: int
    pre: int
    pre2: int
    post: int
    post1: int
    post2: int
    q_norm: int
    k_norm: int
    router_scale: int
    expert_scale: int
    router: int
    q: Matrix
    k: Matrix
    v: Matrix | None
    o: Matrix
    gate: Matrix
    up: Matrix
    down: Matrix
    global_attn: bool


def _at(base: int, elements: int) -> int:
    return base + elements * 2


def _ints(base: int, count: int) -> int:
    return base + count * 4


class Gemma4Runtime:
    """One packed record. Dense weights are int8. Routed experts are int4 and stream."""

    def __init__(
        self,
        path: Path,
        *,
        owner: str,
        kv_cache: str = "half",
        console: object = None,
    ) -> None:
        del kv_cache
        self.console = console
        self.root = path
        self.owner = owner
        self._owner = owner
        self.kv_cache = "half"
        self._stop: Any = False
        self.filled = 0
        self.prompt_tokens = 0
        self._cached_tokens = 0
        self._hit_limit = False
        self.capacity = 0
        self._closed = False
        self._budget = memory.budget()
        self.eng = Engine()
        self._graph_on = [False] * 128
        self._held: list[int] = []
        resident = (path / "resident.bin").stat().st_size
        expert_bytes = sum((path / "experts" / f"{layer}.bin").stat().st_size for layer in range(LAYERS))
        self._load_row = progress_mod.begin_load(console, resident + expert_bytes)
        try:
            self._load(path)
            self._scratch()
            self.atlas = ExpertAtlas(path, self.eng)
            progress_mod.note_load(self._load_row, expert_bytes)
            self._ensure(CONTEXT_STEP)
            self._product_room = False
            self.prefix = open_session(self)
        except BaseException:
            progress_mod.end_phase(self.console, self._load_row, "")
            self.close()
            raise
        progress_mod.end_phase(self.console, self._load_row, "")
        self._load_row = None

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
        atlas = getattr(self, "atlas", None)
        if atlas is not None:
            atlas.close()
        eng = getattr(self, "eng", None)
        if eng is not None:
            for pointer in self._held:
                eng.free(pointer)
            self._held.clear()

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
        channel_open: bool = False,
        thought_budget: int | None = None,
        fill: bool = False,
    ) -> tuple[list[int], str, str]:
        """Sample after ``prompt``. Returns tokens, the prefill line, and the decode line."""
        del images
        halt_at = [item for item in (stops or []) if item]
        self._hit_limit = False
        rng = random.Random(seed)
        aim_session(
            self.prefix,
            budget=getattr(self, "_prefix_budget", None),
            scratch=getattr(self, "_prefix_scratch", None),
        )
        progress_mod.note(self.console, "prompt", tokens=len(prompt))
        hit = self.prefix.align(prompt)
        if self.prefix.loaded:
            progress_mod.note(self.console, "prefix_load", tokens=hit)
        self._cached_tokens = hit
        progress_mod.note(self.console, "cache", hit=hit)
        pending = len(prompt) - hit
        self._ensure(len(prompt) + max(int(limit), 1))
        prefill_row = progress_mod.begin_phase(self.console, "prefill", max(pending, 0))
        self._prefill_row = prefill_row
        decode_row = None
        try:
            start = time.perf_counter()
            if pending:
                self._prefill(prompt[hit:])
            elif prompt:
                self.filled -= 1
                self._token(prompt[-1], self.filled)
                self.filled += 1
                self._logits()
            self.prompt_tokens = self.filled
            prefill_s = time.perf_counter() - start
            prefill_line = format_rate("prefill", pending, prefill_s)
            progress_mod.end_phase(self.console, prefill_row, prefill_line)
            prefill_row = None
            self.prefix.note(prompt[: self.filled])
            self._store_prefix()
            drawn: list[int] = []
            self.eng.zero(self.counts, VOCAB * 4)
            decode_row = progress_mod.begin_phase(self.console, "decode", limit, live=True)
            decode_start = time.perf_counter()
            open_ids: list[int] = []
            close_ids: list[int] = []
            budget = int(thought_budget or 0)
            if budget > 0:
                from gemma4_26b_a4b.cuda.chat import advance_thought, encode_text

                open_ids = encode_text(self.root, "<|channel>thought\n")
                close_ids = encode_text(self.root, "<channel|>")
            inside = bool(channel_open)
            reasoned = 0
            halted = False
            for _ in range(limit):
                if self._halted() or len(drawn) >= limit:
                    halted = True
                    break
                token = self._draw(temperature, top_p, top_k, presence, rng)
                if token in _STOP and not fill:
                    halted = True
                    break
                drawn.append(token)
                stopped = bool(halt_at) and self._stopped(drawn, halt_at)
                if on_token is not None:
                    on_token(token)
                self._token(token, self.filled)
                self.filled += 1
                self._logits()
                if stopped:
                    halted = True
                    break
                if budget > 0 and close_ids:
                    from gemma4_26b_a4b.cuda.chat import advance_thought

                    inside, reasoned, inject = advance_thought(
                        drawn,
                        inside=inside,
                        reasoned=reasoned,
                        budget=budget,
                        open_ids=open_ids,
                        close_ids=close_ids,
                    )
                    for piece in inject:
                        if len(drawn) >= limit or (piece in _STOP and not fill):
                            break
                        drawn.append(piece)
                        if on_token is not None:
                            on_token(piece)
                        self._token(piece, self.filled)
                        self.filled += 1
                        self._logits()
            if not halted:
                self._hit_limit = True
            decode_s = time.perf_counter() - decode_start
            self.prefix.note((prompt + drawn)[: self.filled])
            decode_line = format_rate("decode", len(drawn), decode_s)
            progress_mod.end_phase(self.console, decode_row, decode_line)
            decode_row = None
            self._store_prefix()
            return drawn, prefill_line, decode_line
        finally:
            if prefill_row is not None:
                progress_mod.end_phase(self.console, prefill_row, "")
            if decode_row is not None:
                progress_mod.end_phase(self.console, decode_row, "")

    def complete_text(
        self,
        messages: list[dict[str, object]],
        *,
        thinking: bool = False,
        limit: int = 384,
        temperature: float | None = None,
        seed: int = 0,
    ) -> str:
        from gemma4_26b_a4b.cuda.chat import THOUGHT_BUDGET, decode_text, encode_chat, reply_text
        from gemma4_26b_a4b.cuda.sampling import DEFAULT_VARIANT, VARIANTS

        card = VARIANTS[DEFAULT_VARIANT]
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
            thought_budget=THOUGHT_BUDGET,
        )
        print(prefill_line, flush=True)
        print(decode_line, flush=True)
        _reasoning, answer = reply_text(decode_text(self.root, tokens))
        return answer

    def capture_cache(self) -> tuple[bytes, bytes]:
        keys: list[bytes] = []
        values: list[bytes] = []
        for layer in range(LAYERS):
            heads, dim, key, value = self._plane(layer)
            nbytes = self.filled * heads * dim * 2
            keys.append(self._read(key, nbytes))
            values.append(self._read(value, nbytes))
        return b"".join(keys), b"".join(values)

    def install_cache(self, keys: bytes, values: bytes, count: int) -> bool:
        self._ensure(count)
        key_at = 0
        value_at = 0
        for layer in range(LAYERS):
            heads, dim, key, value = self._plane(layer)
            nbytes = count * heads * dim * 2
            if key_at + nbytes > len(keys) or value_at + nbytes > len(values):
                return False
            self.eng.h2d(key, keys[key_at : key_at + nbytes])
            self.eng.h2d(value, values[value_at : value_at + nbytes])
            key_at += nbytes
            value_at += nbytes
        if key_at != len(keys) or value_at != len(values):
            return False
        self.filled = count
        self.eng.sync()
        return True

    def _load(self, path: Path) -> None:
        index = json.loads((path / "resident.json").read_text(encoding="utf-8"))
        blob = (path / "resident.bin").read_bytes()
        base = self._alloc(len(blob))
        self.eng.h2d(base, blob)
        progress_mod.note_load(self._load_row, len(blob))
        found: dict[str, Any] = {}
        for name, row in index.items():
            offset = int(row["offset"])
            shape = [int(item) for item in row["shape"]]
            kind = str(row["kind"])
            pointer = base + offset
            if kind == "fp16":
                found[name] = pointer
                if name.endswith("router.scale"):
                    self.eng.scale(pointer, _ROUTER_GAIN, HIDDEN)
                continue
            rows, inner = int(shape[0]), int(shape[1])
            found[name] = Matrix(pointer, pointer + rows * inner, rows, inner, 8)
        self.embed = found["embed"]
        self.final_norm = found["final_norm"]
        self.layers: list[Layer] = []
        for layer in range(LAYERS):
            stem = f"model.language_model.layers.{layer}"
            global_attn = layer in GLOBAL_LAYERS
            self.layers.append(
                Layer(
                    scalar=self.eng.read_half(found[f"{stem}.layer_scalar"]),
                    input_norm=found[f"{stem}.input_layernorm.weight"],
                    post_attn=found[f"{stem}.post_attention_layernorm.weight"],
                    pre=found[f"{stem}.pre_feedforward_layernorm.weight"],
                    pre2=found[f"{stem}.pre_feedforward_layernorm_2.weight"],
                    post=found[f"{stem}.post_feedforward_layernorm.weight"],
                    post1=found[f"{stem}.post_feedforward_layernorm_1.weight"],
                    post2=found[f"{stem}.post_feedforward_layernorm_2.weight"],
                    q_norm=found[f"{stem}.self_attn.q_norm.weight"],
                    k_norm=found[f"{stem}.self_attn.k_norm.weight"],
                    router_scale=found[f"{stem}.router.scale"],
                    expert_scale=found[f"{stem}.router.per_expert_scale"],
                    router=found[f"{stem}.router.proj.weight"],
                    q=found[f"{stem}.self_attn.q_proj.weight"],
                    k=found[f"{stem}.self_attn.k_proj.weight"],
                    v=None if global_attn else found[f"{stem}.self_attn.v_proj.weight"],
                    o=found[f"{stem}.self_attn.o_proj.weight"],
                    gate=found[f"{stem}.mlp.gate_proj.weight"],
                    up=found[f"{stem}.mlp.up_proj.weight"],
                    down=found[f"{stem}.mlp.down_proj.weight"],
                    global_attn=global_attn,
                )
            )
        self.eng.sync()

    def _scratch(self) -> None:
        tile = PREFILL_TILE
        q_span = Q_HEADS * GLOBAL_DIM
        k_span = LOCAL_HEADS * LOCAL_DIM
        self.hidden = self._alloc(tile * HIDDEN * 2)
        self.norm = self._alloc(tile * HIDDEN * 2)
        self.mixed = self._alloc(tile * HIDDEN * 2)
        self.ffn = self._alloc(tile * HIDDEN * 2)
        self.routed = self._alloc(tile * HIDDEN * 2)
        self.act = self._alloc(HIDDEN * 2)
        self.query = self._alloc(tile * q_span * 2)
        self.key = self._alloc(tile * k_span * 2)
        self.value = self._alloc(tile * k_span * 2)
        self.attn = self._alloc(tile * q_span * 2)
        self.wide = self._alloc(tile * DENSE_WIDTH * 2)
        self.up = self._alloc(tile * DENSE_WIDTH * 2)
        self.mid = self._alloc(tile * DENSE_WIDTH * 2)
        tile = PREFILL_TILE
        self.expert_up = self._alloc(tile * GATE_ROWS * 2)
        self.expert_hid = self._alloc(tile * EXPERT_K * 2)
        self.expert_down = self._alloc(tile * HIDDEN * 2)
        self.gather = self._alloc(tile * HIDDEN * 2)
        self.router_logits = self._alloc(tile * EXPERTS * 2)
        self.logits = self._alloc(VOCAB * 2)
        self.ids = self._alloc(tile * TOP_K * 4)
        self.weights = self._alloc(tile * TOP_K * 4)
        self.token_index = self._alloc(tile * TOP_K * 4)
        self.slot_index = self._alloc(tile * TOP_K * 4)
        self.token_slot = self._alloc(4)
        self.counts = self._alloc(VOCAB * 4)
        self.local_k = [0] * LAYERS
        self.local_v = [0] * LAYERS
        self.global_k = [0] * LAYERS
        self.global_v = [0] * LAYERS
        self._last_hidden = self.hidden

    def _token(self, token: int, position: int) -> None:
        self.eng.embed(self.embed.codes, self.embed.scales, int(token), self.hidden, HIDDEN, _GAIN)
        held = self.filled
        self.filled = position
        for layer in range(LAYERS):
            self._block(layer, 1)
        self.filled = held
        self._last_hidden = self.hidden

    def _graph_slot(self, kind: str, layer: int, count: int) -> int:
        base = 0 if count == 1 else 60
        if kind == "mlp":
            base += 30
        return base + layer

    def _with_graph(self, slot: int, body) -> None:
        self.eng.set_step(self.filled)
        if self._graph_on[slot]:
            self.eng.replay(slot)
            return
        self.eng.capture_begin(slot)
        try:
            body()
        finally:
            self.eng.capture_end(slot)
        self._graph_on[slot] = True
        self.eng.replay(slot)

    def _attn(self, layer: int, count: int, position: int) -> None:
        spec = self.layers[layer]
        eng = self.eng
        q_dim = GLOBAL_DIM if spec.global_attn else LOCAL_DIM
        kv_heads = GLOBAL_HEADS if spec.global_attn else LOCAL_HEADS
        q_rows = Q_HEADS * q_dim
        k_rows = kv_heads * q_dim
        eng.rmsnorm(self.hidden, spec.input_norm, self.norm, HIDDEN, RMS_EPS, count, 0)
        self._linear(spec.q, self.norm, self.query, q_rows, HIDDEN, count)
        self._linear(spec.k, self.norm, self.key, k_rows, HIDDEN, count)
        if spec.v is None:
            eng.copy(self.key, self.value, count * k_rows)
        else:
            self._linear(spec.v, self.norm, self.value, k_rows, HIDDEN, count)
        rotating = int(q_dim / 2 * ROPE_PARTIAL) if spec.global_attn else q_dim // 2
        theta = ROPE_GLOBAL if spec.global_attn else ROPE_LOCAL
        window = 0 if spec.global_attn else WINDOW
        _heads, _dim, cache_k, cache_v = self._plane(layer)
        if count > 1:
            eng.head_norm(self.value, self.act, kv_heads * count, q_dim, RMS_EPS, 1)
            eng.head_norm(self.query, spec.q_norm, Q_HEADS * count, q_dim, RMS_EPS, 0)
            eng.head_norm(self.key, spec.k_norm, kv_heads * count, q_dim, RMS_EPS, 0)
            eng.rope_span(self.query, self.key, Q_HEADS, kv_heads, q_dim, rotating, theta, position, count)
            eng.store_span(self.key, cache_k, kv_heads, q_dim, position, count)
            eng.store_span(self.value, cache_v, kv_heads, q_dim, position, count)
            eng.attend_span(
                self.query, cache_k, cache_v, self.attn, Q_HEADS, kv_heads, q_dim, position, count, window
            )
        else:
            eng.head_norm(self.value, self.act, kv_heads, q_dim, RMS_EPS, 1)
            eng.head_norm(self.query, spec.q_norm, Q_HEADS, q_dim, RMS_EPS, 0)
            eng.head_norm(self.key, spec.k_norm, kv_heads, q_dim, RMS_EPS, 0)
            eng.rope(self.query, self.key, Q_HEADS, kv_heads, q_dim, rotating, theta, position)
            eng.store_kv(self.key, cache_k, kv_heads, q_dim, position)
            eng.store_kv(self.value, cache_v, kv_heads, q_dim, position)
            eng.attend_span(
                self.query, cache_k, cache_v, self.attn, Q_HEADS, kv_heads, q_dim, position, 1, window
            )
        self._linear(spec.o, self.attn, self.mixed, HIDDEN, q_rows, count)
        eng.rmsnorm(self.mixed, spec.post_attn, self.norm, HIDDEN, RMS_EPS, count, 0)
        eng.add(self.hidden, self.norm, self.hidden, count * HIDDEN)

    def _block(self, layer: int, count: int) -> None:
        spec = self.layers[layer]
        if count == 1 or count == PREFILL_TILE:
            self._with_graph(self._graph_slot("attn", layer, count), lambda: self._attn(layer, count, -1))
            self._feed(spec, layer, count)
            return
        self._attn(layer, count, self.filled)
        self._feed(spec, layer, count)

    def _feed(self, spec: Layer, layer: int, count: int) -> None:
        eng = self.eng
        if count > 1:
            self._experts_tile(spec, layer, count)
        else:
            eng.zero(self.mixed, HIDDEN * 2)
            eng.rmsnorm(self.hidden, spec.router_scale, self.act, HIDDEN, RMS_EPS, 1, 0)
            eng.gemv_f16(spec.router, self.act, self.router_logits, EXPERTS, HIDDEN)
            eng.router(self.router_logits, spec.expert_scale, self.ids, self.weights, EXPERTS, TOP_K)
            eng.rmsnorm(self.hidden, spec.pre2, self.act, HIDDEN, RMS_EPS, 1, 0)
            self._experts(spec, layer, self.mixed)
        self._mlp_tail(spec, count)

    def _launch_proj(self, spec: Layer, layer: int, count: int) -> None:
        if count == 1 or count == PREFILL_TILE:
            self._with_graph(self._graph_slot("mlp", layer, count), lambda: self._mlp_proj(spec, count))
            return
        self._mlp_proj(spec, count)

    def _mlp_proj(self, spec: Layer, count: int) -> None:
        eng = self.eng
        eng.rmsnorm(self.hidden, spec.pre, self.norm, HIDDEN, RMS_EPS, count, 0)
        self._linear(spec.gate, self.norm, self.wide, DENSE_WIDTH, HIDDEN, count)
        self._linear(spec.up, self.norm, self.up, DENSE_WIDTH, HIDDEN, count)
        eng.geglu_pair(self.wide, self.up, self.mid, DENSE_WIDTH, count)
        self._linear(spec.down, self.mid, self.ffn, HIDDEN, DENSE_WIDTH, count)

    def _mlp_tail(self, spec: Layer, count: int) -> None:
        eng = self.eng
        eng.rmsnorm(self.ffn, spec.post1, self.routed, HIDDEN, RMS_EPS, count, 0)
        eng.rmsnorm(self.mixed, spec.post2, self.ffn, HIDDEN, RMS_EPS, count, 0)
        eng.zero(self.norm, count * HIDDEN * 2)
        eng.add(self.norm, self.routed, self.norm, count * HIDDEN)
        eng.add(self.norm, self.ffn, self.norm, count * HIDDEN)
        eng.rmsnorm(self.norm, spec.post, self.mixed, HIDDEN, RMS_EPS, count, 0)
        eng.add(self.hidden, self.mixed, self.hidden, count * HIDDEN)
        if spec.scalar != 1.0:
            eng.scale(self.hidden, spec.scalar, count * HIDDEN)

    def _experts(self, spec: Layer, layer: int, destination: int) -> None:
        ids = np.zeros(TOP_K, dtype=np.int32)
        self.eng.d2h(ids, self.ids)
        staged: list[tuple[int, int, int, int]] = []
        for slot, expert in enumerate(ids.tolist()):
            image, source, nbytes = self.atlas.bind(layer, int(expert))
            event = self.eng.copy_async(image, source, nbytes) if nbytes else -1
            bits = int(self.atlas.widths[layer, int(expert)])
            staged.append((slot, image, bits, event))
        self._launch_proj(spec, layer, 1)
        for slot, image, bits, event in staged:
            if event >= 0:
                self.eng.wait_copy(event)
            gate_span, gate_scale, down_codes, down_scale = _PARTS[bits]
            del gate_span
            self.eng.linear(
                image, image + gate_scale, self.act, self.expert_up, GATE_ROWS, HIDDEN, 1, bits
            )
            self.eng.zero(self.expert_hid, EXPERT_K * 2)
            self.eng.geglu(self.expert_up, self.expert_hid, EXPERT_WIDTH)
            self.eng.linear(
                image + down_codes,
                image + down_scale,
                self.expert_hid,
                self.expert_down,
                HIDDEN,
                EXPERT_K,
                1,
                bits,
            )
            self.eng.mix(destination, self.expert_down, self.weights, slot, HIDDEN)

    def _experts_tile(self, spec: Layer, layer: int, count: int) -> None:
        """Copy each routed expert once and apply it to every token in the tile that selected it."""
        eng = self.eng
        eng.rmsnorm(self.hidden, spec.router_scale, self.norm, HIDDEN, RMS_EPS, count, 0)
        eng.gemv_f16_tokens(spec.router, self.norm, self.router_logits, EXPERTS, HIDDEN, count)
        eng.router_tokens(self.router_logits, spec.expert_scale, self.ids, self.weights, EXPERTS, TOP_K, count)
        eng.rmsnorm(self.hidden, spec.pre2, self.routed, HIDDEN, RMS_EPS, count, 0)
        chosen = np.empty(count * TOP_K, dtype=np.int32)
        eng.d2h(chosen, self.ids)
        groups: dict[int, list[tuple[int, int]]] = {}
        order: list[int] = []
        for token in range(count):
            start = token * TOP_K
            for slot in range(TOP_K):
                expert = int(chosen[start + slot])
                if expert not in groups:
                    order.append(expert)
                    groups[expert] = []
                groups[expert].append((token, slot))
        token_index = np.empty(count * TOP_K, dtype=np.int32)
        slot_index = np.empty(count * TOP_K, dtype=np.int32)
        spans: list[tuple[int, int, int]] = []
        cursor = 0
        for expert in order:
            begin = cursor
            for token, slot in groups[expert]:
                token_index[cursor] = token
                slot_index[cursor] = slot
                cursor += 1
            spans.append((expert, begin, cursor - begin))
        eng.h2d(self.token_index, token_index[:cursor])
        eng.h2d(self.slot_index, slot_index[:cursor])
        eng.zero(self.mixed, count * HIDDEN * 2)
        if not self._product_room:
            self.atlas.release(64 << 20)
            self._product_room = True
        queued: list[tuple[int, int, int, int, int]] = []
        for expert, begin, rows in spans:
            image, source, nbytes = self.atlas.bind(layer, int(expert))
            event = eng.copy_async(image, source, nbytes) if nbytes else -1
            queued.append((int(expert), begin, rows, image, event))
        self._launch_proj(spec, layer, count)
        for expert, begin, rows, image, event in queued:
            if event >= 0:
                eng.wait_copy(event)
            picked = _ints(self.token_index, begin)
            slots = _ints(self.slot_index, begin)
            eng.gather_f16(self.routed, self.gather, picked, rows, HIDDEN)
            bits = int(self.atlas.widths[layer, int(expert)])
            _gate_span, gate_scale, down_codes, down_scale = _PARTS[bits]
            eng.linear(image, image + gate_scale, self.gather, self.expert_up, GATE_ROWS, HIDDEN, rows, bits)
            eng.zero(self.expert_hid, rows * EXPERT_K * 2)
            eng.geglu_strided(self.expert_up, self.expert_hid, EXPERT_WIDTH, EXPERT_K, rows)
            eng.linear(
                image + down_codes,
                image + down_scale,
                self.expert_hid,
                self.expert_down,
                HIDDEN,
                EXPERT_K,
                rows,
                bits,
            )
            eng.scatter_add_f16(self.mixed, self.expert_down, picked, slots, self.weights, rows, HIDDEN, TOP_K)

    def _logits(self) -> None:
        self.eng.rmsnorm(self._last_hidden, self.final_norm, self.norm, HIDDEN, RMS_EPS, 1, 0)
        self._linear(self.embed, self.norm, self.logits, VOCAB, HIDDEN, 1)
        self.eng.softcap(self.logits, SOFTCAP, VOCAB)

    def _draw(self, temperature: float, top_p: float, top_k: int, presence: float, rng: random.Random) -> int:
        if presence:
            self.eng.penalize(self.logits, self.counts, presence, VOCAB)
        self.eng.sample(self.logits, self.token_slot, VOCAB, temperature, top_k, top_p, rng.random())
        token = self.eng.read_int(self.token_slot)
        self.eng.count(self.counts, self.token_slot)
        return token

    def _linear(self, matrix: Matrix, source: int, destination: int, rows: int, inner: int, tokens: int) -> None:
        self.eng.linear(matrix.codes, matrix.scales, source, destination, rows, inner, tokens, matrix.bits)

    def _plane(self, layer: int) -> tuple[int, int, int, int]:
        if layer in GLOBAL_LAYERS:
            return GLOBAL_HEADS, GLOBAL_DIM, self.global_k[layer], self.global_v[layer]
        return LOCAL_HEADS, LOCAL_DIM, self.local_k[layer], self.local_v[layer]

    def _ensure(self, count: int) -> None:
        if count <= self.capacity:
            return
        grown = ((max(count, CONTEXT_STEP) + CONTEXT_STEP - 1) // CONTEXT_STEP) * CONTEXT_STEP
        local_n = grown * LOCAL_HEADS * LOCAL_DIM * 2
        global_n = grown * GLOBAL_HEADS * GLOBAL_DIM * 2
        need = (LAYERS - len(GLOBAL_LAYERS)) * 2 * local_n + len(GLOBAL_LAYERS) * 2 * global_n
        old = 0
        if self.capacity:
            old = (LAYERS - len(GLOBAL_LAYERS)) * 2 * self.capacity * LOCAL_HEADS * LOCAL_DIM * 2
            old += len(GLOBAL_LAYERS) * 2 * self.capacity * GLOBAL_HEADS * GLOBAL_DIM * 2
        free, _total = self.eng.mem_info()
        if free < need - old + (32 << 20):
            self.atlas.release(need - old - free + (32 << 20))
        local_n = grown * LOCAL_HEADS * LOCAL_DIM * 2
        global_n = grown * GLOBAL_HEADS * GLOBAL_DIM * 2
        try:
            for layer in range(LAYERS):
                if layer in GLOBAL_LAYERS:
                    key = self._alloc(global_n)
                    value = self._alloc(global_n)
                    if self.capacity and self.filled:
                        nbytes = self.filled * GLOBAL_HEADS * GLOBAL_DIM * 2
                        self.eng.d2d(key, self.global_k[layer], nbytes)
                        self.eng.d2d(value, self.global_v[layer], nbytes)
                    self._drop(self.global_k[layer])
                    self._drop(self.global_v[layer])
                    self.global_k[layer] = key
                    self.global_v[layer] = value
                    continue
                key = self._alloc(local_n)
                value = self._alloc(local_n)
                if self.capacity and self.filled:
                    nbytes = self.filled * LOCAL_HEADS * LOCAL_DIM * 2
                    self.eng.d2d(key, self.local_k[layer], nbytes)
                    self.eng.d2d(value, self.local_v[layer], nbytes)
                self._drop(self.local_k[layer])
                self._drop(self.local_v[layer])
                self.local_k[layer] = key
                self.local_v[layer] = value
        except RuntimeError as exc:
            message = f"prompt of {count} tokens does not fit"
            raise RuntimeError(message) from exc
        self.capacity = grown
        self._graph_on = [False] * 128
        self.eng.reset_graphs()

    def _logits_row(self, count: int) -> None:
        if count > 1:
            self.eng.copy(_at(self.hidden, (count - 1) * HIDDEN), self.hidden, HIDDEN)
        self._last_hidden = self.hidden

    def _prefill(self, tokens: list[int]) -> None:
        done = 0
        total = len(tokens)
        while done < total:
            chunk = tokens[done : done + PREFILL_TILE]
            count = len(chunk)
            for index, token in enumerate(chunk):
                self.eng.embed(
                    self.embed.codes,
                    self.embed.scales,
                    int(token),
                    _at(self.hidden, index * HIDDEN),
                    HIDDEN,
                    _GAIN,
                )
            for layer in range(LAYERS):
                self._block(layer, count)
            self._logits_row(count)
            self.filled += count
            done += count
            progress_mod.note_count(self.console, getattr(self, "_prefill_row", None), done)
        self._logits()

    def _alloc(self, nbytes: int) -> int:
        pointer = self.eng.alloc(nbytes)
        self._held.append(pointer)
        return pointer

    def _drop(self, pointer: int) -> None:
        if not pointer:
            return
        self.eng.free(pointer)
        self._held.remove(pointer)

    def _read(self, pointer: int, nbytes: int) -> bytes:
        if nbytes <= 0:
            return b""
        host = np.empty(nbytes, dtype=np.uint8)
        self.eng.d2h(host, pointer)
        return host.tobytes()

    def _store_prefix(self) -> None:
        try:
            self.prefix.flush()
        except OSError:
            progress_mod.note(self.console, "prefix_store_failed")

    def _halted(self) -> bool:
        flag = self._stop
        if callable(flag):
            return bool(flag())
        return bool(flag)

    def _stopped(self, tokens: list[int], stops: list[str]) -> bool:
        from gemma4_26b_a4b.cuda.chat import decode_text

        text = decode_text(self.root, tokens)
        return any(stop in text for stop in stops)
