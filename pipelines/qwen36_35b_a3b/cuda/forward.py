"""Language step.

Dense int8 projections use a 512-token prefill tile and a one-warp decode GEMV.
Each of those dense halves is captured once and replayed. The expert atlas stays
in host memory. Routing stays outside the graph. Leftover device memory keeps
the expert images last used, and a miss streams that image from the host atlas.
"""

from __future__ import annotations

import ctypes
import json
import time

import numpy as np
from pathlib import Path
from typing import Any

from favlo.host import device_budget
from favlo.memory import memory
from qwen36_35b_a3b.cuda import progress
from qwen36_35b_a3b.cuda.atlas import ExpertAtlas
from qwen36_35b_a3b.cuda.closed import RequestClosed
from qwen36_35b_a3b.cuda.engine import Engine
from qwen36_35b_a3b.cuda.layout import (
    ATTN_SCALE,
    CONTEXT_CAP,
    CONTEXT_FLOOR,
    CONTEXT_STEP,
    DOWN_INNER,
    DOWN_ROWS,
    FULL,
    GATE_ROWS,
    GDN_QKV,
    GDN_Z,
    HEAD_DIM,
    HIDDEN,
    KV_HEADS,
    KV_RECORD,
    KV_ROWS,
    LAYERS,
    LINEAR_HEAD_DIM,
    LINEAR_KEY_HEADS,
    LINEAR_VALUE_HEADS,
    O_INNER,
    PREFILL_TILE,
    Q_HEADS,
    Q_ROWS,
    RMS_EPS,
    STEM,
    THETA,
    TOP_K,
    VOCAB,
    cache_tokens,
    expert_parts,
    is_full,
    scratch_items,
)

_CHUNK = 8 << 20


class Qwen36Runtime:
    """Language weights on the device. Int8 experts stay resident. Int4 experts stream."""

    def __init__(self, root: Path, *, kv_cache: str, console: Any, owner: str | None,
        mtp: bool) -> None:
        if kv_cache not in {"int8", "fp8_e4m3", "e4m3", "fp8"}:
            message = f"qwen36_35b_a3b cuda cache is int8; {kv_cache} is not built"
            raise RuntimeError(message)
        if mtp:
            message = "qwen36_35b_a3b cuda does not draft"
            raise RuntimeError(message)
        self.root = Path(root)
        self.console = console
        self.kv_cache = "int8"
        self._owner = owner or f"{STEM}:{id(self)}"
        self._mtp = False
        self._stop: Any = None
        self._halted = False
        self._request_id: int | None = None
        self._cached_tokens = 0
        self._prefill_seconds = 0.0
        self._decode_seconds = 0.0
        self._output_ids: list[int] = []
        self._presence = 0.0
        self._frequency = 0.0
        self._temperature = 0.0
        self._top_p = 1.0
        self._top_k = 0
        self._decode_graph = [False] * LAYERS
        self._prefill_graph = [False] * LAYERS
        self._head_graph = False
        self.max_memory = memory.budget()
        self.pos = 0
        self.max_pos = CONTEXT_FLOOR
        self._owned: list[int] = []
        self.engine = Engine()
        self.atlas: ExpertAtlas | None = None
        self._product_room = False
        self.session: Any = None
        manifest = self.root / "manifest.json"
        weights = self.root / "resident.bin"
        if not manifest.is_file() or not weights.is_file():
            message = f"missing packed record at {self.root}"
            raise FileNotFoundError(message)
        body = json.loads(manifest.read_text(encoding="utf-8"))
        if body.get("complete") is not True:
            message = f"record at {self.root} is not complete"
            raise RuntimeError(message)
        self._tensors: dict[str, dict[str, Any]] = body["tensors"]
        self._language_bytes = int(body["language_bytes"])
        self.resident = self._language_bytes
        fit = cache_tokens(device_budget() or self.max_memory, resident=self.resident)
        if fit < CONTEXT_FLOOR:
            message = f"device budget cannot hold a {CONTEXT_FLOOR}-token int8 cache"
            raise RuntimeError(message)
        # The cap fits the formula and leaves no int4 image on an 8 GiB device.
        # Open at the floor. A longer prompt grows this and releases cache slots.
        self.max_pos = CONTEXT_FLOOR
        detail = getattr(console, "load_detail", "") or self.root.name
        row = progress.begin_load(console, self._language_bytes, detail)
        try:
            self.base = self._alloc(self._language_bytes)
            self._upload(weights, self._language_bytes, row)
            self._workspace()
            self._open_cache(self.max_pos)
            self._bind_layers()
            self.atlas = ExpertAtlas(self.root, self.engine)
            from qwen36_35b_a3b.cuda.prefix import open_session

            self.session = open_session(self)
        except Exception:
            self.close()
            raise
        finally:
            slots = self.atlas.slots if self.atlas is not None else 0
            progress.end_phase(
                console, row, f"loaded {self.root.name} context {self.max_pos} experts {slots}"
            )

    def close(self) -> None:
        session = self.session
        if session is not None:
            try:
                session.flush()
            except Exception:
                pass
            self.session = None
        atlas = self.atlas
        if atlas is not None:
            atlas.close()
            self.atlas = None
        engine = getattr(self, "engine", None)
        if engine is None:
            return
        for pointer in self._owned:
            engine.free(pointer)
        self._owned.clear()
        engine.close()

    def generate(
        self,
        prompt: list[int],
        *,
        temperature: float,
        top_p: float,
        top_k: int,
        presence: float,
        frequency: float,
        limit: int,
        on_token: Any,
    ) -> list[int]:
        """Apply ``prompt`` from the prefix hit, then sample up to ``limit`` tokens."""
        if not prompt:
            message = "prompt is empty"
            raise RuntimeError(message)
        self._temperature = float(temperature)
        self._top_p = float(top_p)
        self._top_k = int(top_k)
        self._presence = float(presence)
        self._frequency = float(frequency)
        self._output_ids = []
        self.engine.zero(self.counts, VOCAB * 4)
        hit = 0
        if self.session is not None:
            hit = int(self.session.align(prompt))
            self._cached_tokens = hit
        else:
            self._reset_sequence()
            self._cached_tokens = 0
        progress.note(self.console, "prompt", tokens=len(prompt))
        if getattr(self.session, "loaded", False):
            progress.note(self.console, "prefix_load", tokens=hit)
        progress.note(self.console, "cache", hit=hit)
        fresh = prompt[hit:]
        if not fresh:
            fresh = prompt[-1:]
            hit = len(prompt) - 1
        self._ensure(hit + len(fresh) + 1)
        console = self.console
        prefill = progress.begin_phase(console, "prefill", len(fresh))
        started = time.perf_counter()
        try:
            done = 0
            while done < len(fresh):
                count = min(PREFILL_TILE, len(fresh) - done)
                self._check()
                self._prefill(fresh[done : done + count], head=done + count == len(fresh))
                done += count
                progress.note_count(prefill, done)
            first = self._sample()
        finally:
            self._prefill_seconds = time.perf_counter() - started
            progress.end_phase(
                console, prefill, progress.rate("prefill", len(fresh), self._prefill_seconds)
            )
        if first in _STOP:
            self._keep(prompt, [])
            return []
        drawn = [first]
        if on_token is not None:
            on_token(first)
        decode = progress.begin_phase(console, "decode", 0, live=True)
        started = time.perf_counter()
        try:
            while len(drawn) < limit and self.pos < CONTEXT_CAP:
                self._check()
                self._step(drawn[-1], head=True)
                nxt = self._sample()
                if nxt in _STOP:
                    break
                drawn.append(nxt)
                if on_token is not None:
                    on_token(nxt)
        finally:
            self._decode_seconds = time.perf_counter() - started
            progress.end_phase(console, decode, progress.rate("decode", len(drawn),
                self._decode_seconds))
        self._keep(prompt, drawn)
        return drawn

    def _keep(self, prompt: list[int], drawn: list[int]) -> None:
        """Store the tokens the device has applied, including every returned token."""
        self._output_ids = drawn
        if self.session is None:
            return
        applied = list(prompt) + list(drawn)
        self.session.note(applied)
        try:
            self.session.flush()
        except OSError as exc:
            progress.note(self.console, "prefix_store_failed", error=exc)
            return
        progress.note(self.console, "prefix_store", tokens=len(applied))

    def _prefill(self, tokens: list[int], *, head: bool) -> None:
        count = len(tokens)
        self._ensure(self.pos + count)
        self.engine.reserve(max(count, 1), max(GDN_QKV, O_INNER, HIDDEN))
        data, scale, entry = self._at("model.language_model.embed_tokens.weight")
        width = int(entry["shape"][1])
        for index, token in enumerate(tokens):
            self.engine.embed(data, scale, int(token), self._slot(self.hidden, index, HIDDEN),
                width)
        graphed = count == PREFILL_TILE
        if graphed:
            self.engine.set_step(0, self.pos, self.pos)
        full_index = 0
        gdn_index = 0
        for layer in range(LAYERS):
            self._check()
            slot = LAYERS + 1 + layer
            position = -1 if graphed else self.pos
            if graphed and self._prefill_graph[layer]:
                self.engine.replay(slot)
            else:
                if graphed:
                    self.engine.capture_begin(slot)
                self._dense(layer, full_index, gdn_index, count, position)
                if graphed:
                    self.engine.capture_end(slot)
                    self._prefill_graph[layer] = True
                    self.engine.replay(slot)
            if is_full(layer):
                full_index += 1
            else:
                gdn_index += 1
            self._moe_tile(layer, count)
        self.pos += count
        if head:
            last = self._slot(self.hidden, count - 1, HIDDEN)
            self._norm("model.language_model.norm.weight", source=last, dest=self.normed)
            self._gemv(self._head_name(), self.normed, HIDDEN, self.logits, VOCAB)

    def _step(self, token: int, *, head: bool) -> None:
        self._ensure(self.pos + 1)
        self.engine.set_step(int(token), self.pos, self.pos)
        data, scale, entry = self._at("model.language_model.embed_tokens.weight")
        self.engine.embed(data, scale, int(token), self.hidden, int(entry["shape"][1]))
        full_index = 0
        gdn_index = 0
        for layer in range(LAYERS):
            self._check()
            if self._decode_graph[layer]:
                self.engine.replay(layer)
            else:
                self.engine.capture_begin(layer)
                self._dense(layer, full_index, gdn_index, 1, -1)
                self.engine.capture_end(layer)
                self._decode_graph[layer] = True
                self.engine.replay(layer)
            if is_full(layer):
                full_index += 1
            else:
                gdn_index += 1
            self._moe(layer, self.hidden, self.normed)
        self.pos += 1
        if head:
            self._head()

    def _head(self) -> None:
        if self._head_graph:
            self.engine.replay(LAYERS)
            return
        self.engine.capture_begin(LAYERS)
        self._norm("model.language_model.norm.weight")
        self._gemv(self._head_name(), self.normed, HIDDEN, self.logits, VOCAB)
        self.engine.capture_end(LAYERS)
        self._head_graph = True
        self.engine.replay(LAYERS)

    def _head_name(self) -> str:
        if "lm_head.weight" in self._tensors:
            return "lm_head.weight"
        return "model.language_model.embed_tokens.weight"

    def _dense(self, layer: int, full_index: int, gdn_index: int, count: int,
        position: int) -> None:
        self._norm(f"model.language_model.layers.{layer}.input_layernorm.weight", count=count)
        if is_full(layer):
            self._attend(layer, full_index, count, position)
        else:
            self._gdn(layer, gdn_index, count)
        self.engine.add(self.hidden, self.proj, HIDDEN * count)
        self._norm(
            f"model.language_model.layers.{layer}.post_attention_layernorm.weight", count=count
        )

    def _slot(self, base: int, index: int, width: int) -> int:
        return base + index * width * 4

    def _attend(self, layer: int, slot: int, count: int, position: int) -> None:
        base = f"model.language_model.layers.{layer}.self_attn."
        self._project_pair(
            self.normed, base + "q_proj.weight", self.wide, Q_ROWS, base + "k_proj.weight", self.kv,
                KV_ROWS, count
        )
        self._project(base + "v_proj.weight", self.normed, self.vv, KV_ROWS, HIDDEN, count)
        self.engine.split_q(self.wide, self.query, self.gate, Q_HEADS * count)
        q_norm, _scale, _entry = self._at(base + "q_norm.weight")
        k_norm, _scale, _entry = self._at(base + "k_norm.weight")
        self.engine.rmsnorm(self.query, q_norm, self.query, HEAD_DIM, RMS_EPS, Q_HEADS * count)
        self.engine.rmsnorm(self.kv, k_norm, self.kv, HEAD_DIM, RMS_EPS, KV_HEADS * count)
        self.engine.rope(self.query, Q_HEADS, self.kv, KV_HEADS, position, THETA, count)
        origin = position if position < 0 else position
        self.engine.store_kv(self.k_cache, self.kv, self.max_pos, slot, KV_HEADS, origin, FULL,
            count)
        self.engine.store_kv(self.v_cache, self.vv, self.max_pos, slot, KV_HEADS, origin, FULL,
            count)
        if count == 1:
            length = -1 if position < 0 else position + 1
            self.engine.attend(
                self.k_cache, self.v_cache, self.query, self.attn, length, slot, FULL, ATTN_SCALE
            )
            self.engine.gate_mul(self.attn, self.gate, O_INNER)
        else:
            length = -1 if position < 0 else position + 1
            self.engine.attend_prefill(
                self.k_cache, self.v_cache, self.query, self.gate, self.attn, length, slot, FULL,
                    ATTN_SCALE, count
            )
        self._project(base + "o_proj.weight", self.attn, self.proj, HIDDEN, O_INNER, count)

    def _gdn(self, layer: int, slot: int, count: int) -> None:
        base = f"model.language_model.layers.{layer}.linear_attn."
        self._project_pair(
            self.normed,
            base + "in_proj_qkv.weight",
            self.mixed,
            GDN_QKV,
            base + "in_proj_z.weight",
            self.z,
            GDN_Z,
            count,
        )
        self._project_pair(
            self.normed,
            base + "in_proj_a.weight",
            self.ab,
            LINEAR_VALUE_HEADS,
            base + "in_proj_b.weight",
            self.ab2,
            LINEAR_VALUE_HEADS,
            count,
        )
        taps, _scale, _entry = self._at(base + "conv1d.weight")
        history = self.history + slot * GDN_QKV * 3 * 2
        self.engine.conv_silu(history, taps, self.mixed, self.wide, GDN_QKV, count)
        q_span = LINEAR_KEY_HEADS * LINEAR_HEAD_DIM
        self.engine.l2norm_tokens(self.wide, LINEAR_KEY_HEADS, LINEAR_HEAD_DIM, GDN_QKV, count)
        self.engine.l2norm_tokens(
            self.wide + q_span * 4, LINEAR_KEY_HEADS, LINEAR_HEAD_DIM, GDN_QKV, count
        )
        self.engine.repeat2_tokens(
            self.wide, self.q_rep, LINEAR_KEY_HEADS, LINEAR_HEAD_DIM, GDN_QKV, GDN_Z, count
        )
        self.engine.repeat2_tokens(
            self.wide + q_span * 4, self.k_rep, LINEAR_KEY_HEADS, LINEAR_HEAD_DIM, GDN_QKV, GDN_Z, count
        )
        a_log, _scale, _entry = self._at(base + "A_log")
        bias, _scale, _entry = self._at(base + "dt_bias")
        norm, _scale, entry = self._at(base + "norm.weight")
        span = 128 if int(entry["shape"][-1] if entry["shape"] else 128) > LINEAR_HEAD_DIM else 0
        state = self.state + slot * LINEAR_VALUE_HEADS * LINEAR_HEAD_DIM * LINEAR_HEAD_DIM * 4
        value = self.wide + q_span * 2 * 4
        self.engine.gdn(
            state, self.q_rep, self.k_rep, value, self.ab, self.ab2, a_log, bias, norm, self.z,
            self.gdn_y, RMS_EPS, span, count, GDN_QKV,
        )
        self._project(base + "out_proj.weight", self.gdn_y, self.proj, HIDDEN, GDN_Z, count)

    def _moe(self, layer: int, hidden: int, normed: int) -> None:
        base = f"model.language_model.layers.{layer}.mlp."
        gate, _scale, _entry = self._at(base + "gate.weight")
        rows = int(self._tensors[base + "gate.weight"]["shape"][0])
        self.engine.gemv_f16(gate, normed, self.router, rows, HIDDEN)
        self.engine.topk(self.router, self.route_ids, self.route_w, rows, TOP_K)
        ids = (ctypes.c_int * TOP_K)()
        self.engine.d2h(ids, self.route_ids)
        assert self.atlas is not None
        chosen = [int(ids[slot]) for slot in range(TOP_K)]
        placed = [self.atlas.place(layer, expert, slot) for slot, expert in enumerate(chosen)]
        self.engine.zero_f32(self.moe, HIDDEN)
        for slot, (image, plane) in enumerate(placed):
            bits = 8 if plane == 8 else 4
            _gate_n, gate_scale, down, down_scale = expert_parts("int8" if bits == 8 else "int4")
            self._quant_gemv(image, image + gate_scale, normed, HIDDEN, self.up, GATE_ROWS, bits)
            self.engine.silu_mul(self.up, self.up + DOWN_INNER * 4, self.silu, DOWN_INNER)
            self._quant_gemv(image + down, image + down_scale, self.silu, DOWN_INNER, self.proj,
                DOWN_ROWS, bits)
            self.engine.scale_add_at(self.moe, self.proj, self.route_w + slot * 4, HIDDEN)
        shared = base + "shared_expert."
        self._gemv(shared + "gate_up", normed, HIDDEN, self.up, GATE_ROWS)
        self.engine.silu_mul(self.up, self.up + DOWN_INNER * 4, self.silu, DOWN_INNER)
        self._gemv(shared + "down_proj.weight", self.silu, DOWN_INNER, self.proj, DOWN_ROWS)
        coeff, _scale, _entry = self._at(base + "shared_expert_gate.weight")
        self.engine.gemv_f16(coeff, normed, self.coeff, 1, HIDDEN)
        self.engine.sig_add(self.moe, self.proj, self.coeff, HIDDEN)
        self.engine.add(hidden, self.moe, HIDDEN)

    def _moe_tile(self, layer: int, count: int) -> None:
        """Copy each routed expert once and apply it to every token in the tile that selected it."""
        base = f"model.language_model.layers.{layer}.mlp."
        gate, _scale, _entry = self._at(base + "gate.weight")
        rows = int(self._tensors[base + "gate.weight"]["shape"][0])
        self.engine.gemv_f16_tokens(gate, self.normed, self.moe_rows, rows, HIDDEN, count)
        self.engine.topk_tokens(self.moe_rows, self.route_ids_tile, self.route_w_tile, rows, TOP_K, count)
        chosen = np.empty(count * TOP_K, dtype=np.int32)
        self.engine.d2h(chosen, self.route_ids_tile)
        groups: dict[int, list[tuple[int, int]]] = {}
        for token in range(count):
            start = token * TOP_K
            for slot in range(TOP_K):
                groups.setdefault(int(chosen[start + slot]), []).append((token, slot))
        self.engine.zero_f32(self.moe_rows, HIDDEN * count)
        assert self.atlas is not None
        if not self._product_room:
            self.atlas.release(64 << 20)
            self._product_room = True
        width = len(groups)
        device = np.empty(width, dtype=np.int64)
        host_src = np.empty(width, dtype=np.int64)
        copy_bytes = np.empty(width, dtype=np.int32)
        bits = np.empty(width, dtype=np.int32)
        group_n = np.empty(width, dtype=np.int32)
        gate_scale = np.empty(width, dtype=np.int32)
        down = np.empty(width, dtype=np.int32)
        down_scale = np.empty(width, dtype=np.int32)
        token_index = np.empty(count * TOP_K, dtype=np.int32)
        slot_index = np.empty(count * TOP_K, dtype=np.int32)
        parts = {"8": expert_parts("int8"), "4": expert_parts("int4")}
        cursor = 0
        for index, (expert, members) in enumerate(groups.items()):
            pointer, source, nbytes = self.atlas.bind(layer, expert)
            plane = int(self.atlas.widths[layer, expert])
            _gate_n, gate_at, down_at, down_at_scale = parts["8" if plane == 8 else "4"]
            device[index] = pointer
            host_src[index] = source
            copy_bytes[index] = nbytes
            bits[index] = plane
            group_n[index] = len(members)
            gate_scale[index] = gate_at
            down[index] = down_at
            down_scale[index] = down_at_scale
            for token, slot in members:
                token_index[cursor] = token
                slot_index[cursor] = slot
                cursor += 1
        self.engine.grouped_mlp(
            device, host_src, copy_bytes, bits, token_index[:cursor], slot_index[:cursor], group_n,
            gate_scale, down, down_scale, self.normed, self.gather, self.up_rows, self.silu_rows,
            self.proj, self.moe_rows, self.route_w_tile, self.group_token, self.group_slot, HIDDEN,
            GATE_ROWS, DOWN_INNER, DOWN_ROWS, TOP_K,
        )
        shared = base + "shared_expert."
        self._project(shared + "gate_up", self.normed, self.up_rows, GATE_ROWS, HIDDEN, count)
        self.engine.silu_rows(self.up_rows, self.silu_rows, DOWN_INNER, GATE_ROWS, count)
        self._project(shared + "down_proj.weight", self.silu_rows, self.proj, HIDDEN, DOWN_INNER, count)
        coeff, _scale, _entry = self._at(base + "shared_expert_gate.weight")
        self.engine.gemv_f16_tokens(coeff, self.normed, self.coeff_tile, 1, HIDDEN, count)
        self.engine.sig_rows(self.moe_rows, self.proj, self.coeff_tile, count, HIDDEN)
        self.engine.add(self.hidden, self.moe_rows, HIDDEN * count)

    def _sample(self) -> int:
        if self._temperature <= 0.0:
            if self._presence or self._frequency:
                self.engine.penalize(self.logits, self.counts, self._presence, self._frequency,
                    VOCAB)
            self.engine.argmax(self.logits, self.partial_v, self.partial_i, self.picked, VOCAB)
        else:
            seed = int(time.perf_counter_ns() & 0xFFFFFFFF) or 1
            self.engine.sample(
                self.logits, self.counts, self.picked, VOCAB, self._temperature, self._top_p,
                self._top_k, self._presence, self._frequency, seed,
            )
        self.engine.count(self.counts, self.picked)
        self.engine.sync()
        token = ctypes.c_int()
        self.engine.d2h(token, self.picked)
        return int(token.value)

    def _norm(self, name: str, source: int | None = None, dest: int | None = None,
        count: int = 1) -> None:
        weight, _scale, entry = self._at(name)
        width = int(entry["shape"][-1])
        self.engine.rmsnorm(
            self.hidden if source is None else source,
            weight,
            self.normed if dest is None else dest,
            width,
            RMS_EPS,
            count,
        )

    def _project(
        self, name: str, source: int, destination: int, rows: int, inner: int, count: int
    ) -> None:
        if count == 1:
            self._gemv(name, source, inner, destination, rows)
            return
        data, scale, entry = self._at(name)
        if str(entry["dtype"]) == "fp16":
            self.engine.gemv_f16(data, source, destination, rows, inner)
            return
        self.engine.gemm(data, scale, source, destination, rows, inner, count)

    def _project_pair(
        self,
        source: int,
        first: str,
        destination_first: int,
        rows_first: int,
        second: str,
        destination_second: int,
        rows_second: int,
        count: int,
    ) -> None:
        if count == 1:
            self._gemv(first, source, HIDDEN, destination_first, rows_first)
            self._gemv(second, source, HIDDEN, destination_second, rows_second)
            return
        left, left_scale, _entry = self._at(first)
        right, right_scale, _entry = self._at(second)
        self.engine.gemm_pair(
            left,
            left_scale,
            destination_first,
            rows_first,
            right,
            right_scale,
            destination_second,
            rows_second,
            source, HIDDEN, count,
        )

    def _gemv(self, name: str, source: int, inner: int, destination: int, rows: int) -> None:
        data, scale, entry = self._at(name)
        plane = str(entry["dtype"])
        if plane == "fp16":
            self.engine.gemv_f16(data, source, destination, rows, inner)
            return
        bits = 8 if plane == "int8" else 4
        self._quant_gemv(data, scale, source, inner, destination, rows, bits)

    def _quant_gemv(
        self, weight: int, scale: int, source: int, inner: int, destination: int, rows: int, bits:
        int
    ) -> None:
        self.engine.quant(source, self.codes, self.ascales, inner, bits, 1)
        self.engine.gemv(weight, scale, self.codes, self.ascales, destination, rows, inner, bits, 1)

    def _at(self, name: str) -> tuple[int, int, dict[str, Any]]:
        entry = self._tensors.get(name)
        if entry is None:
            message = f"record is missing {name}"
            raise RuntimeError(message)
        if entry.get("file", "resident.bin") != "resident.bin":
            message = f"{name} is not a resident language weight"
            raise RuntimeError(message)
        offset = int(entry["offset"])
        data = int(entry["data_nbytes"])
        return self.base + offset, self.base + offset + data, entry

    def _workspace(self) -> None:
        for name, size in scratch_items():
            pointer = self._alloc(size, zero=True)
            setattr(self, name, pointer)

    def _open_cache(self, tokens: int) -> None:
        self.max_pos = tokens
        nbytes = tokens * FULL * KV_HEADS * KV_RECORD
        self.k_cache = self._alloc(nbytes, zero=True)
        self.v_cache = self._alloc(nbytes, zero=True)
        self._k_bytes = nbytes

    def _ensure(self, count: int) -> None:
        if count <= self.max_pos:
            return
        target = min(CONTEXT_CAP, ((count + CONTEXT_STEP - 1) // CONTEXT_STEP) * CONTEXT_STEP)
        plane = target * FULL * KV_HEADS * KV_RECORD
        # Both new planes are allocated before the old pair is released.
        need = plane * 2
        free, _total = self.engine.mem_info()
        if self.atlas is not None and free < need:
            self.atlas.release(need - free + (32 << 20))
        try:
            k_new = self._alloc(plane, zero=True)
            v_new = self._alloc(plane, zero=True)
        except RuntimeError as exc:
            message = f"device budget cannot hold {target} tokens"
            raise RuntimeError(message) from exc
        self._copy_cache(self.k_cache, k_new)
        self._copy_cache(self.v_cache, v_new)
        self.engine.free(self.k_cache)
        self.engine.free(self.v_cache)
        self._owned.remove(self.k_cache)
        self._owned.remove(self.v_cache)
        self.k_cache = k_new
        self.v_cache = v_new
        self._decode_graph = [False] * LAYERS
        self._prefill_graph = [False] * LAYERS
        self._k_bytes = plane
        self.max_pos = target

    def _copy_cache(self, source: int, destination: int) -> None:
        # Token-major records: the live prefix is a contiguous head of the buffer.
        live = self.pos * FULL * KV_HEADS * KV_RECORD
        if live:
            host = bytearray(live)
            view = (ctypes.c_uint8 * live).from_buffer(host)
            self.engine.d2h(view, source)
            self.engine.h2d(destination, host)

    def _reset_sequence(self) -> None:
        self.engine.zero(self.state, self._nbytes("state"))
        self.engine.zero(self.history, self._nbytes("history"))
        self.engine.zero(self.k_cache, self._k_bytes)
        self.engine.zero(self.v_cache, self._k_bytes)
        self.engine.zero(self.counts, VOCAB * 4)
        self.pos = 0

    def _nbytes(self, name: str) -> int:
        for item, size in scratch_items():
            if item == name:
                return size
        return 0

    def _bind_layers(self) -> None:
        for layer in range(LAYERS):
            if is_full(layer):
                self._at(f"model.language_model.layers.{layer}.self_attn.q_proj.weight")
            else:
                self._at(f"model.language_model.layers.{layer}.linear_attn.in_proj_qkv.weight")

    def _upload(self, path: Path, nbytes: int, row: Any) -> None:
        with path.open("rb") as handle:
            done = 0
            while done < nbytes:
                blob = handle.read(min(_CHUNK, nbytes - done))
                if not blob:
                    break
                self.engine.h2d(self.base + done, blob)
                done += len(blob)
                progress.note_load(row, len(blob))

    def _alloc(self, nbytes: int, *, zero: bool = False) -> int:
        pointer = self.engine.alloc(nbytes)
        self._owned.append(pointer)
        if zero:
            self.engine.zero(pointer, nbytes)
        return pointer

    def _check(self) -> None:
        if self._halted or (self._stop is not None and self._stop()):
            raise RequestClosed()

    def capture_regions(self) -> list[tuple[str, bytes]]:
        """Packed prefix of the cache plus the recurrent state."""
        live = self.pos * FULL * KV_HEADS * KV_RECORD
        return [
            ("k", self._download(self.k_cache, live)),
            ("v", self._download(self.v_cache, live)),
            ("state", self._download(self.state, self._nbytes("state"))),
            ("history", self._download(self.history, self._nbytes("history"))),
        ]

    def install_regions(self, blobs: dict[str, bytes], tokens: int) -> bool:
        live = tokens * FULL * KV_HEADS * KV_RECORD
        if len(blobs.get("k", b"")) != live or len(blobs.get("v", b"")) != live:
            return False
        if tokens > self.max_pos:
            self._ensure(tokens)
        self._reset_sequence()
        self.engine.h2d(self.k_cache, blobs["k"])
        self.engine.h2d(self.v_cache, blobs["v"])
        self.engine.h2d(self.state, blobs["state"])
        self.engine.h2d(self.history, blobs["history"])
        self.pos = tokens
        return True

    def _download(self, pointer: int, nbytes: int) -> bytes:
        if nbytes <= 0:
            return b""
        host = bytearray(nbytes)
        view = (ctypes.c_uint8 * nbytes).from_buffer(host)
        self.engine.d2h(view, pointer)
        return bytes(host)


_STOP = {248_044, 248_046}
