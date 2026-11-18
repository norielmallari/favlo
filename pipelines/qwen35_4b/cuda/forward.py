"""Token step for the CUDA language opening.

Weights stay int8: one fp32 scale per row, and the activation is quantized per
32 channels inside the product. Prefill projects 128 tokens with one
``mma.m16n8k32`` tile and replays that tile from a CUDA graph. Decode stays
one token and replays its own graph. Prefill attention stages one key tile
for the query heads that share a KV head.
Gated DeltaNet and the partial RoPE match the published recurrence. Temperature
0 is argmax on device. A top-k of 1 to 64 draws the nucleus on device.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from favlo.memory import memory
from qwen35_4b.cuda import progress
from qwen35_4b.cuda.closed import RequestClosed
from qwen35_4b.cuda.engine import Engine
from qwen35_4b.cuda.layout import (
    ATTN_HEADS,
    ATTN_SCALE,
    CONV_BYTES,
    FULL,
    GDN_QKV,
    GDN_STATE_BYTES,
    HEAD_DIM,
    HIDDEN,
    INTERMEDIATE,
    KV_HEADS,
    LAYERS,
    LINEAR_HEAD_DIM,
    LINEAR_KEY_HEADS,
    LINEAR_VALUE_HEADS,
    RMS_EPS,
    ROPE_DIM,
    STEM,
    THETA,
    VOCAB,
    is_full,
)

_MIN_CTX = 4096
_CHUNK = 32 * 1024 * 1024
_TILE = 128


class Qwen35Runtime:
    """One resident int8 record and the device buffers for a single sequence."""

    def __init__(
        self,
        root: Path,
        *,
        kv_cache: str,
        console: Any,
        owner: str | None,
        mtp: bool,
    ) -> None:
        if kv_cache != "int8":
            message = f"qwen35_4b cuda cache is int8; {kv_cache} is not built"
            raise RuntimeError(message)
        self.root = Path(root)
        self.console = console
        self.kv_cache = kv_cache
        self._owner = owner or f"{STEM}:{id(self)}"
        self._mtp = bool(mtp)
        self._stop: Callable[[], bool] | None = None
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
        self._counts = np.zeros(VOCAB, dtype=np.int32)
        self.max_memory = memory.budget()
        self.max_pos = _MIN_CTX
        self.pos = 0
        self._rope_delta = 0
        self._owned: list[int] = []
        self._decode_graph = False
        self._prefill_graph = False
        self._capturing = False
        self._cache: list[int] = []
        self.engine = Engine()
        manifest = self.root / "manifest.json"
        weights = self.root / "weights.bin"
        if not manifest.is_file() or not weights.is_file():
            message = f"missing packed record at {self.root}"
            raise FileNotFoundError(message)
        body = json.loads(manifest.read_text(encoding="utf-8"))
        if body.get("complete") is not True or body.get("weight") != "int8":
            message = f"qwen35_4b cuda serves the int8 record, not {body.get('weight')}"
            raise RuntimeError(message)
        self._tensors = body["tensors"]
        packed = int(body.get("context") or _MIN_CTX)
        context = _served_context(packed, bool(body.get("vision")))
        size = weights.stat().st_size
        detail = getattr(console, "load_detail", "") or self.root.name
        row = progress.begin_load(console, size, detail)
        summary = ""
        try:
            self.base = self._alloc(size)
            self._upload(weights, size, row)
            self._workspace()
            self._open_cache(context)
            self._require_device()
            self._index_layers()
            from qwen35_4b.cuda.prefix import open_session

            self.prefix = open_session(self)
            summary = (
                f"loaded {self.root.name} kv {self.kv_cache} "
                f"context {self.max_pos} memory {self.max_memory}"
            )
        except Exception:
            self.close()
            raise
        finally:
            progress.end_phase(console, row, summary)

    def refresh_budget(self) -> None:
        """Re-read the serve ledger when it has grown."""
        ceiling = memory.budget()
        if ceiling > self.max_memory:
            self.max_memory = ceiling

    def close(self) -> None:
        prefix = getattr(self, "prefix", None)
        if prefix is not None:
            try:
                prefix.flush()
            except OSError:
                pass
            self.prefix = None
        engine = getattr(self, "engine", None)
        if engine is not None:
            for pointer in self._cache + self._owned:
                engine.free(pointer)
            self._cache.clear()
            self._owned.clear()
            engine.close()
            self.engine = None

    def halt(self) -> None:
        self._halted = True

    def generate(
        self,
        prompt: list[int],
        max_tokens: int,
        stop: set[int],
        *,
        forced: list[int] | None = None,
        stop_after_forced: bool = False,
        tool_close: int | None = None,
        on_token: Callable[[int], None] | None = None,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 0,
        presence: float = 0.0,
        frequency: float = 0.0,
        mtp: bool = False,
        mtp_depth: int | None = None,
        fill: bool = False,
    ) -> list[int]:
        """Prefill the uncached suffix, then draw up to ``max_tokens``.

        ``fill`` keeps a stop id and continues until ``max_tokens``.
        """
        del mtp_depth
        if mtp or self._mtp:
            message = "qwen35_4b cuda has no draft model"
            raise RuntimeError(message)
        if self._halted or (self._stop is not None and self._stop()):
            raise RequestClosed()
        if not prompt:
            message = "prompt is empty"
            raise ValueError(message)
        if len(prompt) > self.max_pos:
            message = f"prompt length {len(prompt)} exceeds context {self.max_pos}"
            raise ValueError(message)
        self._presence = float(presence)
        self._frequency = float(frequency)
        self._temperature = float(temperature)
        self._top_p = float(top_p)
        self._top_k = int(top_k)
        self._counts[:] = 0
        self.engine.zero(self._count_dev, VOCAB * 4)
        self.refresh_budget()
        self._output_ids = []
        console = self.console
        from favlo.prefix import aim_session

        aim_session(
            self.prefix,
            budget=getattr(self, "_prefix_budget", None),
            scratch=getattr(self, "_prefix_scratch", None),
        )
        progress.note(console, "prompt", tokens=len(prompt))
        hit = self.prefix.align(prompt)
        if self.prefix.loaded:
            progress.note(console, "prefix_load", tokens=hit)
        progress.note(console, "cache", hit=hit)
        self._cached_tokens = hit
        suffix = prompt[hit:]
        uncached = len(suffix)
        boundary = False
        produced = 0
        self._prefill_seconds = 0.0
        self._decode_seconds = 0.0
        prefill_row = progress.begin_phase(console, "prefill", uncached, request=self._request_id)
        started = time.perf_counter()
        decode_row = None
        decode_started = started
        last_sample = started
        prefill_open = True
        carry: int | None = None
        pending = list(forced or [])
        close_after = bool(stop_after_forced) and bool(pending)
        close_call = int(tool_close) if tool_close is not None and pending else None
        try:
            done = 0
            from qwen35_4b.cuda.chat import generation_cut

            cuts: list[int] = []
            stable = getattr(self, "_stable_cut", None)
            self._stable_cut = None
            if isinstance(stable, int) and 0 < stable < len(prompt):
                cuts.append(stable)
            gen = generation_cut(self.root, prompt)
            if isinstance(gen, int) and gen not in cuts:
                cuts.append(gen)
            cuts.sort()
            # The user-turn cut is the file a later user message still shares.
            # A prompt that already ends on that user message uses the
            # generation suffix instead. Pin it. A tool-step generation cut
            # is stored and not pinned.
            held = stable if isinstance(stable, int) and stable in cuts else gen
            for cut in cuts:
                if hit >= cut:
                    if cut == held:
                        self._keep_boundary(prompt[:cut])
                    continue
                span = prompt[hit:cut]
                walked = 0
                while walked < len(span):
                    piece = span[walked : walked + _TILE]
                    walked += len(piece)
                    done += len(piece)
                    self._prefill_tile([int(token) for token in piece], logits=False)
                    progress.note_prefill(prefill_row, done)
                self.prefix.note(prompt[:cut])
                self._store_prefix(note=False)
                if cut == held:
                    self._keep_boundary(prompt[:cut])
                hit = cut
                suffix = prompt[hit:]
            tail = 0
            while tail < len(suffix):
                piece = suffix[tail : tail + _TILE]
                tail += len(piece)
                self._prefill_tile(
                    [int(token) for token in piece],
                    logits=tail >= len(suffix),
                )
                progress.note_prefill(prefill_row, done + tail)
            while produced < max_tokens:
                self._check()
                forced_token = False
                finished_forced = False
                if pending:
                    next_id = int(pending.pop(0))
                    finished_forced = close_after and not pending
                    forced_token = True
                    sampled_at = time.perf_counter()
                    carry = None
                elif carry is not None:
                    next_id = int(carry)
                    carry = None
                    sampled_at = time.perf_counter()
                else:
                    next_id = self._sample()
                    sampled_at = time.perf_counter()
                if prefill_open:
                    self._prefill_seconds = sampled_at - started
                    row = prefill_row
                    prefill_row = None
                    progress.end_phase(
                        console, row, progress.rate("prefill", uncached, self._prefill_seconds)
                    )
                    self.prefix.note(prompt)
                    self._store_prefix()
                    boundary = True
                    decode_started = sampled_at
                    last_sample = sampled_at
                    decode_row = progress.begin_phase(
                        console, "decode", 0, request=self._request_id, live=True
                    )
                    prefill_open = False
                if next_id in stop and not fill:
                    last_sample = sampled_at
                    break
                last_sample = sampled_at
                self._output_ids.append(next_id)
                produced += 1
                progress.note_decode(decode_row, produced)
                if on_token is not None:
                    on_token(next_id)
                closed_call = (
                    close_call is not None
                    and not pending
                    and not forced_token
                    and next_id == close_call
                )
                if (
                    finished_forced
                    or closed_call
                    or produced >= max_tokens
                    or self.pos >= self.max_pos
                ):
                    if finished_forced or closed_call:
                        self._step(next_id, logits=False)
                    break
                self._step(next_id, logits=True)
                if not forced_token:
                    carry = self._sample()
            if prefill_open:
                self._prefill_seconds = time.perf_counter() - started
                row = prefill_row
                prefill_row = None
                progress.end_phase(
                    console, row, progress.rate("prefill", uncached, self._prefill_seconds)
                )
                self.prefix.note(prompt)
                self._store_prefix()
                boundary = True
                self._decode_seconds = 0.0
            else:
                elapsed = last_sample - decode_started
                self._decode_seconds = elapsed if elapsed > 0.0 else 0.0
                progress.end_phase(
                    console, decode_row, progress.rate("decode", produced, self._decode_seconds)
                )
                decode_row = None
            self.prefix.note((prompt + self._output_ids)[: self.pos])
            self._store_prefix()
        except RequestClosed:
            if boundary:
                self.prefix.note((prompt + self._output_ids)[: self.pos])
                self._store_prefix()
            else:
                self.prefix.tokens = []
                self.prefix.dirty = False
                self._reset_state()
            raise
        finally:
            if prefill_row is not None:
                progress.end_phase(console, prefill_row, "")
            if decode_row is not None:
                elapsed = last_sample - decode_started
                self._decode_seconds = elapsed if elapsed > 0.0 else 0.0
                progress.end_phase(console, decode_row, "")
        return list(self._output_ids)

    def _keep_boundary(self, tokens: list[int]) -> None:
        """Pin the stored file for these tokens so eviction leaves it in place."""
        from favlo.prefix import object_name

        store = self.prefix.store
        name = object_name(store.model, store.digest, store.tag, tokens)
        path = store.directory / f"{name}.bin"
        if path.is_file():
            store.pin(path)

    def _store_prefix(self, *, note: bool = True) -> None:
        try:
            path = self.prefix.flush()
        except OSError as exc:
            progress.note(self.console, "prefix_store_failed", error=exc)
            return
        if path is not None and note:
            progress.note(self.console, "prefix_store", tokens=len(self.prefix.tokens))

    def _prefill_tile(self, tokens: list[int], *, logits: bool) -> None:
        count = len(tokens)
        if self.pos + count > self.max_pos:
            message = f"context {self.max_pos} is full"
            raise RuntimeError(message)
        for index, token in enumerate(tokens):
            self._check()
            self.engine.embed(
                self._weight("embedding"),
                self._scale("embedding"),
                int(token),
                self._slot(self.hidden, index, HIDDEN),
                HIDDEN,
            )
        self._layers(count)
        if logits:
            last = count - 1
            self.engine.rmsnorm(
                self._slot(self.hidden, last, HIDDEN),
                self._weight("final_norm"),
                self.normed,
                HIDDEN,
                RMS_EPS,
                1.0,
                1,
            )
            self.engine.gemv(
                self._weight("embedding"),
                self._scale("embedding"),
                self.normed,
                self.logits,
                VOCAB,
                HIDDEN,
            )
        self.pos += count

    def _tile_graph(self, count: int) -> bool:
        return count == _TILE and (self._prefill_graph or self._capturing)

    def _layers(self, count: int) -> None:
        """Run every block. A full 128-token tile is captured once and replayed."""
        if count != _TILE:
            for layer in range(LAYERS):
                self._layer(layer, count)
            return
        rope = max(0, self.pos + int(self._rope_delta))
        self.engine.set_step(0, self.pos, rope)
        if self._prefill_graph:
            self.engine.replay(1)
            return
        self.engine.reserve(_TILE, INTERMEDIATE)
        self.engine.capture_begin(1)
        self._capturing = True
        try:
            for layer in range(LAYERS):
                self._layer(layer, count)
        finally:
            self._capturing = False
            self.engine.capture_end(1)
        self._prefill_graph = True
        self.engine.replay(1)

    def _step(self, token: int, *, logits: bool) -> None:
        self._check()
        if self.pos >= self.max_pos:
            message = f"context {self.max_pos} is full"
            raise RuntimeError(message)
        rope = max(0, self.pos + int(self._rope_delta))
        self.engine.set_step(int(token), self.pos, rope)
        if logits and not self._decode_graph:
            self.engine.capture_begin()
            try:
                self._decode_launches()
            finally:
                self.engine.capture_end()
            self._decode_graph = True
        if logits:
            self.engine.replay()
        else:
            self._decode_launches()
            self.engine.sync()
        self.pos += 1

    def _decode_launches(self) -> None:
        self.engine.embed(
            self._weight("embedding"), self._scale("embedding"), -1, self.hidden, HIDDEN
        )
        for layer in range(LAYERS):
            self._layer(layer, 1)
        self.engine.rmsnorm(
            self.hidden, self._weight("final_norm"), self.normed, HIDDEN, RMS_EPS, 1.0
        )
        self.engine.gemv(
            self._weight("embedding"),
            self._scale("embedding"),
            self.normed,
            self.logits,
            VOCAB,
            HIDDEN,
        )

    def _layer(self, layer: int, count: int) -> None:
        self.engine.rmsnorm(
            self.hidden,
            self._weight(f"layer.{layer}.in_norm"),
            self.normed,
            HIDDEN,
            RMS_EPS,
            1.0,
            count,
        )
        if is_full(layer):
            self._full(layer, count)
        else:
            self._gdn(layer, count)
        self.engine.add(self.hidden, self.delta, HIDDEN, count)
        self.engine.rmsnorm(
            self.hidden,
            self._weight(f"layer.{layer}.post_norm"),
            self.normed,
            HIDDEN,
            RMS_EPS,
            1.0,
            count,
        )
        self._mlp(layer, count)
        self.engine.add(self.hidden, self.delta, HIDDEN, count)

    def _full(self, layer: int, count: int) -> None:
        slot = self._kv_slot[layer]
        data = slot * self.max_pos * KV_HEADS * HEAD_DIM
        scale = slot * self.max_pos * KV_HEADS * (HEAD_DIM // 32)
        q_width = ATTN_HEADS * HEAD_DIM * 2
        k_width = KV_HEADS * HEAD_DIM
        self._project_pair(
            self.normed,
            f"layer.{layer}.q",
            self.q,
            q_width,
            f"layer.{layer}.k",
            self.k,
            k_width,
            HIDDEN,
            count,
        )
        self._project(self.normed, f"layer.{layer}.v", self.v, k_width, HIDDEN, count)
        q_gain = self._weight(f"layer.{layer}.q_norm")
        k_gain = self._weight(f"layer.{layer}.k_norm")
        graphed = self._tile_graph(count)
        origin = -1 if count == 1 or graphed else self.pos
        rope_pos = -1 if count == 1 or graphed else max(0, self.pos + int(self._rope_delta))
        self.engine.head_norm(
            self.q, q_gain, ATTN_HEADS, HEAD_DIM * 2, HEAD_DIM, RMS_EPS, 1.0, count
        )
        self.engine.head_norm(self.k, k_gain, KV_HEADS, HEAD_DIM, HEAD_DIM, RMS_EPS, 1.0, count)
        self.engine.rope(self.q, ATTN_HEADS, HEAD_DIM * 2, ROPE_DIM, rope_pos, THETA, count)
        self.engine.rope(self.k, KV_HEADS, HEAD_DIM, ROPE_DIM, rope_pos, THETA, count)
        self.engine.store_kv(
            self.k, self.k_data + data, self.k_scale + scale, origin, KV_HEADS, HEAD_DIM, count
        )
        self.engine.store_kv(
            self.v, self.v_data + data, self.v_scale + scale, origin, KV_HEADS, HEAD_DIM, count
        )
        self.engine.attend(
            self.q,
            self.k_data + data,
            self.k_scale + scale,
            self.v_data + data,
            self.v_scale + scale,
            self.attn,
            -1 if count == 1 or self._tile_graph(count) else self.pos + 1,
            ATTN_HEADS,
            KV_HEADS,
            HEAD_DIM,
            ATTN_SCALE,
            count,
        )
        self._project(
            self.attn, f"layer.{layer}.o", self.delta, HIDDEN, ATTN_HEADS * HEAD_DIM, count
        )

    def _gdn(self, layer: int, count: int) -> None:
        index = self._gdn_index[layer]
        state = self.gdn_state + index * LINEAR_VALUE_HEADS * LINEAR_HEAD_DIM * LINEAR_HEAD_DIM * 4
        history = self.conv_hist + index * GDN_QKV * 3 * 4
        z_width = LINEAR_VALUE_HEADS * LINEAR_HEAD_DIM
        self._project_pair(
            self.normed,
            f"layer.{layer}.qkv",
            self.mixed,
            GDN_QKV,
            f"layer.{layer}.z",
            self.z,
            z_width,
            HIDDEN,
            count,
        )
        self._project_pair(
            self.normed,
            f"layer.{layer}.a",
            self.a,
            LINEAR_VALUE_HEADS,
            f"layer.{layer}.b",
            self.b,
            LINEAR_VALUE_HEADS,
            HIDDEN,
            count,
        )
        conv = self._weight(f"layer.{layer}.conv")
        a_log = self._weight(f"layer.{layer}.A_log")
        dt_bias = self._weight(f"layer.{layer}.dt_bias")
        gain = self._weight(f"layer.{layer}.gdn_norm")
        self.engine.conv_silu(conv, self.mixed, history, GDN_QKV, count)
        self.engine.gdn(
            state,
            self.mixed,
            self.a,
            self.b,
            a_log,
            dt_bias,
            self.gdn_out,
            LINEAR_KEY_HEADS,
            LINEAR_VALUE_HEADS,
            count,
        )
        self.engine.gated_norm(
            self.gdn_out,
            self.z,
            gain,
            LINEAR_VALUE_HEADS,
            LINEAR_HEAD_DIM,
            RMS_EPS,
            count,
        )
        self._project(self.gdn_out, f"layer.{layer}.out", self.delta, HIDDEN, z_width, count)

    def _mlp(self, layer: int, count: int) -> None:
        self._project_pair(
            self.normed,
            f"layer.{layer}.gate",
            self.gate,
            INTERMEDIATE,
            f"layer.{layer}.up",
            self.up,
            INTERMEDIATE,
            HIDDEN,
            count,
        )
        self.engine.silu_mul(self.gate, self.up, self.ff, INTERMEDIATE, count)
        self._project(self.ff, f"layer.{layer}.down", self.delta, HIDDEN, INTERMEDIATE, count)

    def _project(
        self, source: int, name: str, destination: int, rows: int, inner: int, count: int
    ) -> None:
        weight = self._weight(name)
        scale = self._scale(name)
        if count == 1:
            self.engine.gemv(weight, scale, source, destination, rows, inner)
            return
        self.engine.gemm(weight, scale, source, destination, rows, inner, count)

    def _project_pair(
        self,
        source: int,
        first: str,
        destination_first: int,
        rows_first: int,
        second: str,
        destination_second: int,
        rows_second: int,
        inner: int,
        count: int,
    ) -> None:
        """Project one activation into two tensors. Decode stays two separate products."""
        if count == 1:
            self._project(source, first, destination_first, rows_first, inner, 1)
            self._project(source, second, destination_second, rows_second, inner, 1)
            return
        self.engine.gemm_pair(
            self._weight(first),
            self._scale(first),
            destination_first,
            rows_first,
            self._weight(second),
            self._scale(second),
            destination_second,
            rows_second,
            source,
            inner,
            count,
        )

    def _slot(self, base: int, index: int, width: int) -> int:
        return base + index * width * 4

    def _sample(self) -> int:
        if not 0 < self._top_k <= 64:
            return self._sample_host()
        draw = float(np.random.random())
        self.engine.sample(
            self.logits,
            self._count_dev,
            self._picked,
            VOCAB,
            self._top_k,
            self._temperature,
            self._top_p,
            self._presence,
            self._frequency,
            draw,
        )
        token = np.empty(1, dtype=np.int32)
        self.engine.d2h(token, self._picked)
        chosen = int(token[0])
        self._counts[chosen] += 1
        return chosen

    def _sample_host(self) -> int:
        logits = np.empty(VOCAB, dtype=np.float32)
        self.engine.d2h(logits, self.logits)
        if self._presence or self._frequency:
            seen = self._counts > 0
            logits[seen] -= self._presence + self._frequency * self._counts[seen]
        if self._temperature <= 0.0:
            token = int(np.argmax(logits))
        else:
            token = _pick(logits, self._temperature, self._top_p, self._top_k)
        self._counts[token] += 1
        return token

    def _require_device(self) -> None:
        """Weights, activations, Gated DeltaNet state, and the cache stay in VRAM."""
        named = (
            ("weights", self.base),
            ("hidden", self.hidden),
            ("logits", self.logits),
            ("gdn_state", self.gdn_state),
            ("conv_hist", self.conv_hist),
            ("k", self.k_data),
            ("v", self.v_data),
            ("k_scale", self.k_scale),
            ("v_scale", self.v_scale),
        )
        for name, pointer in named:
            if not self.engine.on_device(pointer):
                message = f"qwen35_4b cuda {name} is not in device memory"
                raise RuntimeError(message)

    def _weight(self, name: str) -> int:
        return self.base + int(self._tensors[name]["offset"])

    def _scale(self, name: str) -> int:
        spec = self._tensors[name]
        rows, inner = spec["shape"]
        return self._weight(name) + int(rows) * int(inner)

    def _upload(self, path: Path, size: int, row: Any) -> None:
        mapping = np.memmap(path, dtype=np.uint8, mode="r")
        try:
            for start in range(0, size, _CHUNK):
                end = min(size, start + _CHUNK)
                chunk = np.array(mapping[start:end], dtype=np.uint8, copy=True)
                self.engine.h2d(self.base + start, chunk)
                progress.note_load(row, end - start)
        finally:
            del mapping

    def _workspace(self) -> None:
        self.hidden = self._f32(HIDDEN * _TILE)
        self.normed = self._f32(HIDDEN * _TILE)
        self.delta = self._f32(HIDDEN * _TILE)
        self.q = self._f32(ATTN_HEADS * HEAD_DIM * 2 * _TILE)
        self.k = self._f32(KV_HEADS * HEAD_DIM * _TILE)
        self.v = self._f32(KV_HEADS * HEAD_DIM * _TILE)
        self.attn = self._f32(ATTN_HEADS * HEAD_DIM * _TILE)
        self.mixed = self._f32(GDN_QKV * _TILE)
        self.z = self._f32(LINEAR_VALUE_HEADS * LINEAR_HEAD_DIM * _TILE)
        self.a = self._f32(LINEAR_VALUE_HEADS * _TILE)
        self.b = self._f32(LINEAR_VALUE_HEADS * _TILE)
        self.gdn_out = self._f32(LINEAR_VALUE_HEADS * LINEAR_HEAD_DIM * _TILE)
        self.gate = self._f32(INTERMEDIATE * _TILE)
        self.up = self._f32(INTERMEDIATE * _TILE)
        self.ff = self._f32(INTERMEDIATE * _TILE)
        self.logits = self._f32(VOCAB)
        self._count_dev = self._alloc(VOCAB * 4, zero=True)
        self._picked = self._alloc(4)
        self.gdn_state = self._alloc(GDN_STATE_BYTES, zero=True)
        self.conv_hist = self._alloc(CONV_BYTES, zero=True)

    def _open_cache(self, context: int) -> None:
        high = max(_MIN_CTX, context)
        low = _MIN_CTX
        fit = 0
        held = 0
        while low <= high:
            attempt = (low + high) // 2
            try:
                self._cache_at(attempt)
            except MemoryError:
                self._drop_cache()
                held = 0
                high = attempt - 1
                continue
            fit = attempt
            held = attempt
            if attempt == high:
                break
            self._drop_cache()
            held = 0
            low = attempt + 1
        if fit < _MIN_CTX:
            message = "qwen35_4b cache does not fit"
            raise MemoryError(message)
        if held != fit:
            self._cache_at(fit)
        self.max_pos = fit

    def _cache_at(self, context: int) -> None:
        data = FULL * context * KV_HEADS * HEAD_DIM
        scale = FULL * context * KV_HEADS * (HEAD_DIM // 32)
        self.k_data = self._alloc_cache(data, zero=True)
        self.v_data = self._alloc_cache(data, zero=True)
        self.k_scale = self._alloc_cache(scale, zero=True)
        self.v_scale = self._alloc_cache(scale, zero=True)
        self._kv_data_bytes = data
        self._kv_scale_bytes = scale

    def _index_layers(self) -> None:
        self._gdn_index: dict[int, int] = {}
        self._kv_slot: dict[int, int] = {}
        gdn = 0
        slot = 0
        for layer in range(LAYERS):
            if is_full(layer):
                self._kv_slot[layer] = slot
                slot += 1
            else:
                self._gdn_index[layer] = gdn
                gdn += 1

    def _reset_state(self) -> None:
        self.engine.zero(self.gdn_state, GDN_STATE_BYTES)
        self.engine.zero(self.conv_hist, CONV_BYTES)
        self.engine.zero(self.k_data, self._kv_data_bytes)
        self.engine.zero(self.v_data, self._kv_data_bytes)
        self.engine.zero(self.k_scale, self._kv_scale_bytes)
        self.engine.zero(self.v_scale, self._kv_scale_bytes)
        self.pos = 0

    def _device_bytes(self, pointer: int, nbytes: int) -> bytes:
        if nbytes <= 0:
            return b""
        host = np.empty(nbytes, dtype=np.uint8)
        self.engine.d2h(host, pointer)
        return host.tobytes()

    def _host_bytes(self, pointer: int, blob: bytes) -> None:
        if not blob:
            return
        host = np.frombuffer(blob, dtype=np.uint8)
        self.engine.h2d(pointer, host)

    def _f32(self, count: int) -> int:
        return self._alloc(count * 4)

    def _alloc(self, nbytes: int, *, zero: bool = False) -> int:
        pointer = self.engine.alloc(nbytes)
        self._owned.append(pointer)
        if zero:
            self.engine.zero(pointer, nbytes)
        return pointer

    def _alloc_cache(self, nbytes: int, *, zero: bool) -> int:
        pointer = self.engine.alloc(nbytes)
        self._cache.append(pointer)
        if zero:
            self.engine.zero(pointer, nbytes)
        return pointer

    def _drop_cache(self) -> None:
        engine = self.engine
        if engine is None:
            self._cache.clear()
            return
        for pointer in self._cache:
            engine.free(pointer)
        self._cache.clear()

    def _check(self) -> None:
        if self._halted or (self._stop is not None and self._stop()):
            raise RequestClosed()


def _served_context(packed: int, vision: bool) -> int:
    """Cache length for the registry device budget. The packed length is the fallback."""
    from favlo.host import device_budget
    from qwen35_4b.cuda.layout import cache_tokens

    budget = device_budget()
    if budget is None:
        return max(_MIN_CTX, packed)
    tokens = cache_tokens(budget, "int8", vision=vision)
    return max(_MIN_CTX, tokens)


def _pick(logits: np.ndarray, temperature: float, top_p: float, top_k: int) -> int:
    scores = np.array(logits, dtype=np.float32, copy=True)
    if top_k > 0 and top_k < scores.size:
        keep = np.argpartition(scores, -top_k)[-top_k:]
        masked = np.full(scores.shape, -np.inf, dtype=np.float32)
        masked[keep] = scores[keep]
        scores = masked
    scores /= np.float32(temperature)
    scores -= np.max(scores)
    probs = np.exp(scores)
    total = float(probs.sum())
    if total <= 0.0 or not np.isfinite(total):
        return int(np.argmax(logits))
    probs /= np.float32(total)
    if 0.0 < top_p < 1.0:
        order = np.argsort(probs)[::-1]
        ranked = np.cumsum(probs[order])
        cutoff = int(np.searchsorted(ranked, top_p, side="left"))
        chosen = order[: cutoff + 1]
        trimmed = np.zeros_like(probs)
        trimmed[chosen] = probs[chosen]
        trimmed /= np.float32(trimmed.sum())
        probs = trimmed
    return int(np.random.choice(probs.size, p=probs))
