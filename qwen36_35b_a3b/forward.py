"""Text forward for one token at a time. The MTP block is packed and not drafted."""

from __future__ import annotations

import json
import struct
import time
from pathlib import Path
from typing import Any

import numpy as np

from qwen36_35b_a3b.atlas import ExpertAtlas, available_bytes
from qwen36_35b_a3b.attn import attend_plane, bind_plane, plane_bytes
from qwen36_35b_a3b.cache import (
    kv_buffer_bytes,
    kv_mode,
    kv_record_bytes,
    max_position,
    parse_memory,
)
from qwen36_35b_a3b.convert import LANGUAGE_LAYERS, SLOT_MAP_BYTES
from qwen36_35b_a3b.formats import EXPERT_IMAGE
from qwen36_35b_a3b.prefill import workspace_bytes
from qwen36_35b_a3b.prefix import open_session
from qwen36_35b_a3b.session import Session, _clear

HIDDEN = 2048
VOCAB = 248320
HEAD = 256
Q_HEADS = 16
KV = 2
EPS = 1e-6
# Each head stores query then gate. The other split is the flat halves.
GATE_PER_HEAD = 1
# Paulus, Xiong, and Socher block a repeated n-gram. Eight tokens is one short clause.
_REPEAT_SPAN = 8


def _reserved_bytes(resident: int, kv_one: int, plane: int) -> int:
    """Bytes allocated besides the expert atlas, including the chunk workspace."""
    scratch = (
        HIDDEN * 4
        + HIDDEN * 4
        + 8192 * 2
        + 8192 * 4
        + 4096 * 4
        + 32 * 4
        + 32 * 4
        + 8192 * 4
        + 4096 * 4
        + 4096 * 4
        + 4096 * 4
        + 512 * 4
        + 512 * 4
        + 4096 * 4
        + 2048 * 4
        + 8 * 1024 * 4
        + 8 * 512 * 4
        + 8 * HIDDEN * 4
        + 8 * 4
        + 8 * 4
        + HIDDEN * 4
        + 256 * 4
        + 4
        + VOCAB * 4
        + 30 * 3 * 8192 * 2
        + 30 * 32 * 128 * 128 * 4
    )
    return resident + scratch + workspace_bytes() + 20 * kv_one + plane


def _u32(value: int) -> bytes:
    return struct.pack("<I", value)


def _f32(value: float) -> bytes:
    return struct.pack("<f", value)


def _file_buffer(session: Session, path: Path) -> Any:
    size = path.stat().st_size
    buffer = session.device.newBufferWithLength_options_(max(size, 16), session.shared)
    view = memoryview(buffer.contents().as_buffer(size))
    with path.open("rb") as handle:
        offset = 0
        while offset < size:
            chunk = handle.read(8 << 20)
            if not chunk:
                break
            view[offset : offset + len(chunk)] = chunk
            offset += len(chunk)
    return buffer


class RequestClosed(Exception):
    """The client closed, or the listener asked the forward to stop."""


class Qwen36Runtime:
    """One resident record. ``close`` drops its buffers."""

    def __init__(
        self,
        path: Path,
        *,
        kv_cache: str = "e4m3",
        console: Any = None,
        resident_percent: float | None = None,
        max_memory: str | int | None = None,
    ) -> None:
        del resident_percent
        self.root = Path(path)
        self.console = console
        self.kv_cache = kv_mode(kv_cache)
        self._request_id = 0
        self._hit_limit = False
        self._halted = False
        self._stop = None
        manifest_path = self.root / "manifest.json"
        if not manifest_path.is_file():
            message = f"{self.root.name} has no manifest.json"
            raise FileNotFoundError(message)
        self.manifest = json.loads(manifest_path.read_text())
        if not self.manifest.get("complete"):
            message = f"{self.root.name} is not a finished record"
            raise RuntimeError(message)
        self.max_pos = max_position(self.root)
        self.kv_record = kv_record_bytes(self.kv_cache)
        self.kv_bytes = kv_buffer_bytes(self.kv_cache, self.max_pos)
        self.session = Session()
        self.resident = _file_buffer(self.session, self.root / "resident.bin")
        resident_bytes = (self.root / "resident.bin").stat().st_size
        plane = 0 if self.kv_cache == "f32" else plane_bytes(self.max_pos)
        reserved = _reserved_bytes(resident_bytes, self.kv_bytes, plane)
        cap = parse_memory(max_memory)
        self.max_memory = available_bytes() if cap is None else cap
        self.atlas = ExpertAtlas(
            self.session,
            self.root,
            max_memory=self.max_memory,
            reserved=reserved,
        )
        self.tensors: dict[str, dict] = self.manifest["tensors"]
        self.hidden = self.session.zeros(HIDDEN * 4)
        self.normed = self.session.zeros(HIDDEN * 4)
        self.half = self.session.zeros(8192 * 2)
        self.wide = self.session.zeros(8192 * 4)
        self.z = self.session.zeros(4096 * 4)
        self.ab = self.session.zeros(32 * 4)
        self.ab2 = self.session.zeros(32 * 4)
        self.mixed = self.session.zeros(8192 * 4)
        self.gdn_y = self.session.zeros(4096 * 4)
        self.query = self.session.zeros(4096 * 4)
        self.gate = self.session.zeros(4096 * 4)
        self.kv = self.session.zeros(512 * 4)
        self.vv = self.session.zeros(512 * 4)
        self.attn = self.session.zeros(4096 * 4)
        self.expert = self.session.zeros(2048 * 4)
        self.mid = self.session.zeros(8 * 1024 * 4)
        self.gated = self.session.zeros(8 * 512 * 4)
        self.partial = self.session.zeros(8 * HIDDEN * 4)
        self.ids = self.session.zeros(8 * 4)
        order_path = self.root / "experts" / "order.bin"
        order = order_path.read_bytes()
        if len(order) != SLOT_MAP_BYTES:
            message = f"expert slot map is {len(order)} bytes"
            raise RuntimeError(message)
        self.expert_order = self.session.upload(order)
        self.mix_w = self.session.zeros(8 * 4)
        self.acc = self.session.zeros(HIDDEN * 4)
        self.router = self.session.zeros(256 * 4)
        self.coeff = self.session.zeros(4)
        self.logits = self.session.zeros(VOCAB * 4)
        self.history = self.session.zeros(30 * 3 * 8192 * 2)
        self.state = self.session.zeros(30 * 32 * 128 * 128 * 4)
        self.k_cache = [self.session.zeros(self.kv_bytes) for _ in range(10)]
        self.v_cache = [self.session.zeros(self.kv_bytes) for _ in range(10)]
        if plane:
            bind_plane(self)
        self.filled = 0
        self.prefix = open_session(self)
        self._prefill = None
        self._prefill_progress = None
        self._prefill_base = 0
        self._cached: list[int] = []
        self._cached_tokens = 0
        self._prefill_seconds = 0.0
        self._decode_seconds = 0.0
        self._output_ids: list[int] = []
        used = reserved + self.atlas.slots * self.atlas.file_bytes
        print(
            f"loaded {self.root.name} kv {self.kv_cache} context {self.max_pos} "
            f"atlas {self.atlas.slots}/40 memory {used}/{self.max_memory}",
            flush=True,
        )
        self._held = [
            self.resident,
            *self.atlas.buffers,
            self.hidden,
            self.history,
            self.state,
            *self.k_cache,
            *self.v_cache,
        ]

    def halt(self) -> None:
        """Stop at the next layer. The listener calls this on SIGINT and SIGTERM."""
        self._halted = True

    def _check_stop(self) -> None:
        if self._halted:
            raise RequestClosed()
        stop = self._stop
        if stop is not None and stop():
            raise RequestClosed()

    def close(self) -> None:
        prefix = getattr(self, "prefix", None)
        if prefix is not None and self.session is not None:
            try:
                prefix.flush()
            except OSError:
                pass
        self._prefill = None
        self._held.clear()
        self.atlas.close()
        self.resident = None
        self.session = None

    def expert_buffer(self, layer: int) -> Any:
        """The layer file in the atlas. A miss finishes the open command buffer first."""
        found = self.atlas.cached(layer)
        if found is not None:
            return found
        restart = self.session._recording
        if restart:
            self.session.end()
        else:
            self.session.wait_inflight()
        found = self.atlas.load(layer)
        if restart:
            self.session.begin()
        return found

    def reset(self) -> None:
        for buffer, nbytes in (
            (self.history, 30 * 3 * 8192 * 2),
            (self.state, 30 * 32 * 128 * 128 * 4),
        ):
            memoryview(buffer.contents().as_buffer(nbytes))[:] = b"\x00" * nbytes
        for cache in (*self.k_cache, *self.v_cache):
            _clear(cache, self.kv_bytes)
        self.filled = 0

    def _loc(self, name: str) -> tuple[Any, int, dict]:
        entry = self.tensors[name]
        return self.resident, int(entry["offset"]), entry

    def _scale(self, entry: dict, offset: int) -> int:
        return offset + int(entry["data_nbytes"]) + int(entry["scale_pad"])

    def _gemv_fp4(self, act: Any, act_off: int, name: str, dest: Any, dest_off: int = 0) -> None:
        buf, offset, entry = self._loc(name)
        rows, inner = entry["stored_shape"]
        self.session.encode(
            "gemv_fp4_vec",
            [(act, act_off), (buf, offset), (buf, self._scale(entry, offset)), (dest, dest_off)],
            [_u32(int(rows)), _u32(int(inner))],
            (int(rows), 1, 1),
            (32, 1, 1),
        )

    def _weight(self, name: str) -> tuple[tuple[Any, int], tuple[Any, int]]:
        buf, offset, entry = self._loc(name)
        return (buf, offset), (buf, self._scale(entry, offset))

    def _gemv_fp4_acc(self, act: Any, act_off: int, name: str, dest: Any) -> None:
        buf, offset, entry = self._loc(name)
        rows, inner = entry["stored_shape"]
        self.session.encode(
            "gemv_fp4_vec_acc",
            [(act, act_off), (buf, offset), (buf, self._scale(entry, offset)), (dest, 0)],
            [_u32(int(rows)), _u32(int(inner))],
            (int(rows), 1, 1),
            (32, 1, 1),
        )

    def _rms(
        self, src: Any, src_off: int, name: str, width: int, dest: Any, dest_off: int = 0
    ) -> None:
        buf, offset, _entry = self._loc(name)
        self.session.encode(
            "rmsnorm_f32",
            [(src, src_off), (buf, offset), (dest, dest_off)],
            [_u32(width), _f32(EPS)],
            (1, 1, 1),
            (256, 1, 1),
        )

    def _embed(self, token: int) -> None:
        buf, offset, entry = self._loc("model.language_model.embed_tokens.weight")
        _rows, inner = entry["stored_shape"]
        self.session.encode(
            "gather_fp4_row",
            [(buf, offset), (buf, self._scale(entry, offset)), (self.hidden, 0)],
            [_u32(token), _u32(int(inner))],
            (1, 1, 1),
            (32, 1, 1),
        )

    def _gdn(self, layer: int) -> None:
        base = f"model.language_model.layers.{layer}.linear_attn."
        qkv_w, qkv_s = self._weight(base + "in_proj_qkv.weight")
        z_w, z_s = self._weight(base + "in_proj_z.weight")
        a_w, a_s = self._weight(base + "in_proj_a.weight")
        b_w, b_s = self._weight(base + "in_proj_b.weight")
        self.session.encode(
            "gdn_in_proj",
            [
                (self.normed, 0),
                qkv_w,
                qkv_s,
                z_w,
                z_s,
                a_w,
                a_s,
                b_w,
                b_s,
                (self.wide, 0),
                (self.z, 0),
                (self.ab, 0),
                (self.ab2, 0),
            ],
            [],
            (12352, 1, 1),
            (32, 1, 1),
        )
        slot = layer - (layer // 4)
        hist = slot * 3 * 8192 * 2
        weight, woff, _entry = self._loc(base + "conv1d.weight")
        self.session.encode(
            "gdn_conv_shift",
            [(self.history, hist), (self.wide, 0), (weight, woff), (self.mixed, 0)],
            [_u32(8192)],
            (8192, 1, 1),
            (1, 1, 1),
        )
        along, aoff, _entry = self._loc(base + "A_log")
        bias, boff, _entry = self._loc(base + "dt_bias")
        norm, noff, _entry = self._loc(base + "norm.weight")
        state = slot * 32 * 128 * 128 * 4
        self.session.encode(
            "gdn_state",
            [
                (self.state, state),
                (self.mixed, 0),
                (self.ab, 0),
                (self.ab2, 0),
                (self.z, 0),
                (along, aoff),
                (bias, boff),
                (norm, noff),
                (self.gdn_y, 0),
            ],
            [_f32(EPS)],
            (32, 1, 1),
            (128, 1, 1),
        )
        self._gemv_fp4_acc(self.gdn_y, 0, base + "out_proj.weight", self.hidden)

    def _attend(self, layer: int, pos: int) -> None:
        base = f"model.language_model.layers.{layer}.self_attn."
        q_w, q_s = self._weight(base + "q_proj.weight")
        k_w, k_s = self._weight(base + "k_proj.weight")
        v_w, v_s = self._weight(base + "v_proj.weight")
        self.session.encode(
            "attn_qkv",
            [
                (self.normed, 0),
                q_w,
                q_s,
                k_w,
                k_s,
                v_w,
                v_s,
                (self.wide, 0),
                (self.kv, 0),
                (self.vv, 0),
            ],
            [],
            (9216, 1, 1),
            (32, 1, 1),
        )
        self.session.encode(
            "split_q_gate",
            [(self.wide, 0), (self.query, 0), (self.gate, 0)],
            [_u32(GATE_PER_HEAD)],
            (4096, 1, 1),
            (1, 1, 1),
        )
        q_buf, q_off, _entry = self._loc(base + "q_norm.weight")
        k_buf, k_off, _entry = self._loc(base + "k_norm.weight")
        positions = np.array([pos, pos, pos], dtype=np.uint32).tobytes()
        self.session.encode(
            "qk_norm_rope",
            [
                (self.query, 0),
                (self.kv, 0),
                (q_buf, q_off),
                (k_buf, k_off),
            ],
            [positions, _f32(1e7), _f32(EPS)],
            (Q_HEADS + KV, 1, 1),
            (HEAD, 1, 1),
        )
        slot = layer // 4
        if self.kv_cache == "f32":
            stride = self.max_pos * HEAD
            self.session.encode(
                "store_kv",
                [
                    (self.k_cache[slot], 0),
                    (self.v_cache[slot], 0),
                    (self.kv, 0),
                    (self.vv, 0),
                ],
                [_u32(pos), _u32(stride)],
                (KV, 1, 1),
                (HEAD, 1, 1),
            )
            self.session.encode(
                "attend_gqa_heads",
                [
                    (self.query, 0),
                    (self.k_cache[slot], 0),
                    (self.v_cache[slot], 0),
                    (self.gate, 0),
                    (self.attn, 0),
                ],
                [_u32(pos + 1), _u32(stride)],
                (Q_HEADS, 1, 1),
                (HEAD, 1, 1),
            )
        else:
            attend_plane(
                self,
                query=self.query,
                key=self.kv,
                value=self.vv,
                gate=self.gate,
                dest=self.attn,
                pos0=pos,
                count=1,
                slot=slot,
            )
        self._gemv_fp4_acc(self.attn, 0, base + "o_proj.weight", self.hidden)

    def _moe(self, layer: int, prefix: str) -> None:
        router, roff, _entry = self._loc(prefix + "gate.weight")
        coeff, coff, _entry = self._loc(prefix + "shared_expert_gate.weight")
        self.session.encode(
            "router_coeff",
            [
                (self.normed, 0),
                (router, roff),
                (coeff, coff),
                (self.router, 0),
                (self.coeff, 0),
            ],
            [],
            (257, 1, 1),
            (32, 1, 1),
        )
        self.session.encode(
            "router8",
            [(self.router, 0), (self.ids, 0), (self.mix_w, 0)],
            [],
            (1, 1, 1),
            (1, 1, 1),
        )
        if layer >= LANGUAGE_LAYERS:
            message = f"layer {layer} has no expert slot map"
            raise RuntimeError(message)
        self.session.encode(
            "remap_expert_ids",
            [(self.ids, 0), (self.expert_order, 0)],
            [_u32(8), _u32(layer)],
            (1, 1, 1),
            (32, 1, 1),
        )
        gate_w, gate_s = self._weight(prefix + "shared_expert.gate_up")
        down_w, down_s = self._weight(prefix + "shared_expert.down_proj.weight")
        experts = self.expert_buffer(layer)
        self.session.encode(
            "moe_gate_silu",
            [
                (self.normed, 0),
                (experts, 0),
                (self.ids, 0),
                gate_w,
                gate_s,
                (self.gated, 0),
                (self.expert, 0),
            ],
            [_u32(EXPERT_IMAGE)],
            (4608, 1, 1),
            (32, 1, 1),
        )
        self.session.encode(
            "moe_down_mix",
            [
                (self.gated, 0),
                (self.expert, 0),
                (experts, 0),
                (self.ids, 0),
                (self.mix_w, 0),
                down_w,
                down_s,
                (self.coeff, 0),
                (self.hidden, 0),
            ],
            [_u32(EXPERT_IMAGE)],
            (HIDDEN, 1, 1),
            (32, 1, 1),
        )

    def step(self, token: int, pos: int, *, logits: bool) -> np.ndarray | None:
        self._check_stop()
        if pos >= self.max_pos:
            message = f"position {pos} is past {self.max_pos}"
            raise RuntimeError(message)
        self.session.begin()
        self._embed(token)
        for layer in range(40):
            norm = f"model.language_model.layers.{layer}.input_layernorm.weight"
            self._rms(self.hidden, 0, norm, HIDDEN, self.normed)
            if layer % 4 == 3:
                self._attend(layer, pos)
            else:
                self._gdn(layer)
            prefix = f"model.language_model.layers.{layer}.mlp."
            self._rms(
                self.hidden,
                0,
                f"model.language_model.layers.{layer}.post_attention_layernorm.weight",
                HIDDEN,
                self.normed,
            )
            self._moe(layer, prefix)
        self._rms(self.hidden, 0, "model.language_model.norm.weight", HIDDEN, self.normed)
        if logits:
            self._gemv_fp4(self.normed, 0, "lm_head.weight", self.logits)
        self.session.end()
        self.filled = pos + 1
        if not logits:
            return None
        return self.session.download_f32(self.logits, VOCAB)

    def generate(
        self,
        prompt: list[int],
        max_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        presence: float,
        stop: set[int],
        forced: list[int] | None = None,
        stop_after_forced: bool = False,
        on_token: Any = None,
        think_close: list[int] | None = None,
        think_end: int | None = None,
        thinking_budget: int = 0,
    ) -> list[int]:
        self._hit_limit = False
        self._output_ids = []
        self._prefill_seconds = 0.0
        self._decode_seconds = 0.0
        if not prompt:
            self._cached_tokens = 0
            return []
        prompt = [int(token) for token in prompt]
        self._status(f"prompt tokens {len(prompt)}")
        hit = self.prefix.align(prompt)
        if self.prefix.loaded:
            self._status(f"prefix load {hit} tokens")
        self._status(f"cache hit {hit}" if hit else "cache miss")
        self._cached_tokens = hit
        suffix = prompt[hit:]
        prefill_row = None
        decode_row = None
        boundary = False
        out: list[int] = []
        try:
            prefill_row = self._begin_phase("prefill", len(suffix))
            self._prefill_progress = prefill_row
            started = time.perf_counter()
            if len(suffix) > 1:
                from qwen36_35b_a3b.prefill import prefill_ids

                self._prefill_base = 0
                prefill_ids(self, suffix[:-1], hit)
            logits = self.step(suffix[-1], len(prompt) - 1, logits=True)
            assert logits is not None
            self.prefix.note(prompt)
            self._store_prefix()
            boundary = True
            self._note_prefill(len(suffix))
            elapsed = max(time.perf_counter() - started, 1e-9)
            self._prefill_seconds = elapsed
            self._prefill_progress = None
            self._end_phase(
                prefill_row,
                f"prefill {len(suffix) / elapsed:.2f} tok/s ({len(suffix)} tok in {elapsed:.2f}s)",
            )
            prefill_row = None
            pending = list(forced or [])
            close_after = bool(stop_after_forced) and bool(pending)
            seen: set[int] = set()
            budget = None
            if think_close and think_end is not None and thinking_budget > 0:
                budget = ThinkBudget(think_close, think_end, thinking_budget)
            decode_row = self._begin_phase("decode", max_tokens, live=True)
            decode_started = time.perf_counter()
            for count in range(max_tokens):
                self._check_stop()
                if pending:
                    nxt = int(pending.pop(0))
                    finished_forced = close_after and not pending
                elif budget is not None and budget.forcing:
                    nxt = budget.forced()
                    finished_forced = False
                else:
                    banned = {budget.end} if budget is not None and budget.closed else None
                    nxt = _sample(
                        logits, temperature, top_p, top_k, presence, seen, banned, out
                    )
                    if budget is not None:
                        nxt = budget.take(nxt)
                    finished_forced = False
                if nxt in stop:
                    replaced = None if budget is None else budget.instead_of_stop(nxt, stop)
                    if replaced is None:
                        break
                    nxt = replaced
                out.append(nxt)
                seen.add(nxt)
                if decode_row is not None:
                    decode_row.done = len(out)
                if on_token is not None:
                    on_token(nxt)
                if finished_forced:
                    self.step(nxt, len(prompt) + count, logits=False)
                    break
                if count + 1 == max_tokens:
                    self._hit_limit = True
                    self.step(nxt, len(prompt) + count, logits=False)
                    break
                logits = self.step(nxt, len(prompt) + count, logits=True)
                assert logits is not None
            decode_elapsed = max(time.perf_counter() - decode_started, 1e-9)
            self._decode_seconds = decode_elapsed
            self._output_ids = list(out)
            self._end_phase(
                decode_row,
                f"decode {len(out) / decode_elapsed:.2f} tok/s "
                f"({len(out)} tok in {decode_elapsed:.2f}s)",
            )
            decode_row = None
            self.prefix.note((prompt + out)[: self.filled])
            self._store_prefix()
        except RequestClosed:
            if boundary:
                self.prefix.note((prompt + out)[: self.filled])
                self._store_prefix()
            else:
                self.prefix.tokens = []
                self.prefix.dirty = False
                self.reset()
            raise
        finally:
            self._prefill_progress = None
            self._end_phase(prefill_row, "")
            self._end_phase(decode_row, "")
        return out

    def _store_prefix(self) -> None:
        try:
            path = self.prefix.flush()
        except OSError as exc:
            self._status(f"prefix store failed {exc}")
            return
        if path is not None:
            self._status(f"prefix store {len(self.prefix.tokens)} tokens")

    def _status(self, line: str) -> None:
        console = self.console
        if console is None:
            print(line, flush=True)
            return
        console.log(line)

    def _begin_phase(self, phase: str, total: int, *, live: bool = False) -> Any:
        console = self.console
        if console is None:
            return None
        if live:
            if not getattr(console, "live_decode", False):
                return None
            # An open-ended count. A fraction of max_tokens stays empty.
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

    def _end_phase(self, progress: Any, summary: str) -> None:
        if progress is None:
            if summary:
                self._status(summary)
            return
        if progress.closed:
            return
        console = self.console
        if console is None:
            if summary:
                print(summary, flush=True)
            return
        console.end(progress, summary)

    def _note_prefill(self, done: int) -> None:
        progress = self._prefill_progress
        if progress is None:
            return
        capped = min(progress.total, done)
        if capped > progress.done:
            progress.done = capped


class ThinkBudget:
    """Close an open think block after ``budget`` completion tokens.

    The prompt already opened ``<think>``. Drawn tokens count until
    ``</think>``. At the budget, the next tokens are ``close``, one id at a
    time. After that sequence, ``end`` is removed from the logits.
    """

    def __init__(self, close: list[int], end: int, budget: int) -> None:
        self.close = [int(token) for token in close]
        self.end = int(end)
        self.budget = int(budget)
        self.counted = 0
        self.force_at: int | None = None
        self.closed = False

    @property
    def forcing(self) -> bool:
        return self.force_at is not None

    def take(self, sampled: int) -> int:
        if self.closed:
            return sampled
        if self.counted >= self.budget:
            return self._start()
        if sampled == self.end:
            self.closed = True
            return sampled
        self.counted += 1
        return sampled

    def forced(self) -> int:
        assert self.force_at is not None
        token = self.close[self.force_at]
        self.force_at += 1
        if self.force_at >= len(self.close):
            self.force_at = None
            self.closed = True
        return token

    def instead_of_stop(self, token: int, stop: set[int]) -> int | None:
        """The first close id when ``token`` would end an open think block."""
        if token not in stop or self.closed or self.forcing:
            return None
        return self._start()

    def _start(self) -> int:
        token = self.close[0]
        if len(self.close) == 1:
            self.closed = True
        else:
            self.force_at = 1
        return token


def _blocked_tokens(produced: list[int], span: int) -> set[int]:
    """Tokens that would repeat an n-gram already present in ``produced``.

    Paulus, Xiong, and Socher, "A Deep Reinforced Model for Abstractive
    Summarization", block a repeated n-gram by forbidding the token that
    would complete it.
    """
    if span < 2 or len(produced) < span:
        return set()
    prefix = tuple(produced[-(span - 1) :])
    blocked: set[int] = set()
    for start in range(len(produced) - span + 1):
        if tuple(produced[start : start + span - 1]) == prefix:
            blocked.add(produced[start + span - 1])
    return blocked


def _sample(
    logits: np.ndarray,
    temperature: float,
    top_p: float,
    top_k: int,
    presence: float,
    seen: set[int],
    banned: set[int] | None = None,
    produced: list[int] | None = None,
) -> int:
    scores = logits
    blocked = _blocked_tokens(produced or [], _REPEAT_SPAN)
    if (presence and seen) or banned or blocked:
        scores = np.array(logits, copy=True)
        if presence and seen:
            penalty = np.float32(presence)
            for token in seen:
                scores[token] -= penalty
        if banned:
            for token in banned:
                scores[token] = np.float32(-np.inf)
        for token in blocked:
            scores[token] = np.float32(-np.inf)
    if temperature == 0.0:
        return int(np.argmax(scores))
    if top_k > 0 and top_k < scores.size:
        keep = np.argpartition(scores, -top_k)[-top_k:]
        chosen = np.asarray(scores[keep], dtype=np.float64)
        return int(keep[_draw(chosen, temperature, top_p)])
    chosen = np.asarray(scores, dtype=np.float64)
    return _draw(chosen, temperature, top_p)


def _draw(chosen: np.ndarray, temperature: float, top_p: float) -> int:
    """Sample one index from ``chosen``. ``top_p`` drops the tail of that vector."""
    scaled = chosen / temperature
    scaled -= np.max(scaled)
    probs = np.exp(scaled)
    total = float(np.sum(probs))
    if not np.isfinite(total) or total <= 0.0:
        return int(np.argmax(chosen))
    probs /= total
    if 0.0 < top_p < 1.0:
        order = np.argsort(probs)[::-1]
        running = np.cumsum(probs[order])
        cut = int(np.searchsorted(running, top_p) + 1)
        order = order[:cut]
        probs = probs[order]
        probs /= float(np.sum(probs))
        return int(order[int(np.random.choice(order.size, p=probs))])
    return int(np.random.choice(chosen.size, p=probs))
