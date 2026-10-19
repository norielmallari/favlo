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
# While think is open, a five-token repeat starts the close.
# The same word with one token between the copies does the same.
# A markdown marker is not that word.
# A token whose piece has seven or more letters, drawn again, does the same,
# unless that token is already in the prompt.
# Masking that token lets the same clause continue with a synonym.
_REPEAT_SPAN = 8
_THINK_REPEAT_SPAN = 5
_LONG_LETTERS = 7


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


def limit_keeps_piece(open_think: bool, blank: bool) -> bool:
    """A blank piece is dropped when the token limit cuts an open think.

    The template trims reasoning, so that piece is absent from the next prompt.
    """
    return not open_think or not blank


_piece_decoder: tuple[Path, Any] | None = None


def piece_is_blank(root: Path, token: int) -> bool:
    """True when this token's text is only whitespace."""
    global _piece_decoder
    if _piece_decoder is None or _piece_decoder[0] != root:
        from qwen36_35b_a3b.chat import decode_text

        _piece_decoder = (root, decode_text)
    return _piece_decoder[1](root, [token]).strip() == ""


class _Pending:
    """The next token's layers, recorded before that token is known."""

    def __init__(self, session: Any, pos: int) -> None:
        self.pos = pos
        self.command = session.queue.commandBuffer()
        self.encoder = self.command.computeCommandEncoder()
        self.held: list[Any] = []

    def commit(self, session: Any) -> None:
        self.encoder.endEncoding()
        self.encoder = None
        self.command.commit()
        session._inflight.append((self.command, self.held))
        self.held = []

    def discard(self) -> None:
        if self.encoder is not None:
            self.encoder.endEncoding()
            self.encoder = None
        self.command = None
        self.held = []


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
        self._pending: _Pending | None = None
        self._prefill_progress = None
        self._prefill_base = 0
        self._cached: list[int] = []
        self._cached_tokens = 0
        self._prefill_seconds = 0.0
        self._decode_seconds = 0.0
        self._output_ids: list[int] = []
        used = reserved + self.atlas.bytes
        print(
            f"loaded {self.root.name} kv {self.kv_cache} context {self.max_pos} "
            f"atlas {self.atlas.slots} images memory {used}/{self.max_memory}",
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
        self._drop_pending()
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

    def _routed_ids(self, ids: Any, count: int) -> list[int]:
        """Finish the open buffer so the router ids can be read."""
        self.session.end()
        raw = np.frombuffer(self.session.download(ids, count * 4), dtype=np.uint32)
        return [int(value) for value in raw]

    def routed_layer(self, layer: int, ids: Any, count: int, offsets: Any) -> Any:
        """Load misses and write each file slot's byte offset. ``ids`` stay file slots."""
        file_slots = self._routed_ids(ids, count)
        placed = self.atlas.stage(layer, file_slots)
        table = np.zeros(256, dtype=np.uint64)
        image = np.uint64(EXPERT_IMAGE)
        for file_slot, atlas_slot in zip(file_slots, placed, strict=True):
            table[file_slot] = np.uint64(atlas_slot) * image
        raw = table.tobytes()
        memoryview(offsets.contents().as_buffer(len(raw)))[:] = raw
        self.session.begin()
        return self.atlas.buffer

    def _stage_decode(self, layer: int) -> Any:
        """Point the eight ids at atlas slots and return the atlas buffer."""
        file_slots = self._routed_ids(self.ids, 8)
        placed = np.asarray(self.atlas.stage(layer, file_slots), dtype=np.uint32)
        blob = placed.tobytes()
        memoryview(self.ids.contents().as_buffer(len(blob)))[:] = blob
        self.session.begin()
        return self.atlas.buffer

    def reset(self) -> None:
        for buffer, nbytes in (
            (self.history, 30 * 3 * 8192 * 2),
            (self.state, 30 * 32 * 128 * 128 * 4),
        ):
            memoryview(buffer.contents().as_buffer(nbytes))[:] = b"\x00" * nbytes
        for cache in (*self.k_cache, *self.v_cache):
            _clear(cache, self.kv_bytes)
        self._drop_pending()
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
        if self.atlas.complete:
            experts = self.atlas.layers[layer]
        else:
            experts = self._stage_decode(layer)
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
        pending = self._pending
        self._pending = None
        if pending is not None and pending.pos == pos:
            self.session.begin()
            self._embed(token)
            self.session.commit()
            pending.commit(self.session)
        else:
            if pending is not None:
                pending.discard()
            self.session.begin()
            self._embed(token)
            self._layers(pos, head=logits)
            self.session.commit()
        self._prepare_pending(pos + 1)
        self.session.wait_inflight()
        self.filled = pos + 1
        if not logits:
            return None
        return self.session.download_f32(self.logits, VOCAB)

    def _layers(self, pos: int, *, head: bool) -> None:
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
        if head:
            self._gemv_fp4(self.normed, 0, "lm_head.weight", self.logits)

    def _prepare_pending(self, pos: int) -> None:
        """Record the next token while this one runs, when every expert image is resident."""
        if not self.atlas.complete or pos >= self.max_pos or self.session._inflight == []:
            return
        pending = _Pending(self.session, pos)
        self.session._redirect = pending
        try:
            self._layers(pos, head=True)
        except BaseException:
            self.session._redirect = None
            pending.discard()
            raise
        self.session._redirect = None
        self._pending = pending

    def _drop_pending(self) -> None:
        pending = self._pending
        self._pending = None
        if pending is not None:
            pending.discard()

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
        thinking_budget: int | None = 0,
        tool_open: int | None = None,
        long_ids: set[int] | None = None,
        forbid: set[int] | None = None,
        word_ids: set[int] | None = None,
        frequency: float = 0.0,
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
            counts: dict[int, int] = {}
            budget = None
            if think_close and think_end is not None and thinking_budget != 0:
                budget = ThinkBudget(think_close, think_end, thinking_budget)
            decode_row = self._begin_phase("decode", max_tokens, live=True)
            decode_started = time.perf_counter()
            for count in range(max_tokens):
                self._check_stop()
                if budget is not None and budget.forcing:
                    nxt = budget.forced()
                    finished_forced = False
                elif pending:
                    nxt = int(pending.pop(0))
                    revised = revise_think_draw(
                        budget, out, nxt, tool_open, long_ids, set(prompt), word_ids
                    )
                    if revised is None:
                        break
                    nxt = revised
                    if budget is not None:
                        nxt = budget.take(nxt)
                    finished_forced = close_after and not pending
                else:
                    held = {budget.end} if budget is not None and budget.closed else set()
                    if forbid:
                        held |= forbid
                    nxt = 0
                    revised = None
                    for _attempt in range(8):
                        drawn = _sample(
                            logits,
                            temperature,
                            top_p,
                            top_k,
                            presence,
                            seen,
                            held,
                            out,
                            frequency,
                            counts,
                        )
                        revised = revise_think_draw(
                            budget, out, drawn, tool_open, long_ids, set(prompt), word_ids
                        )
                        if revised is not None:
                            nxt = revised
                            break
                        held.add(drawn)
                    if revised is None:
                        break
                    if budget is not None:
                        nxt = budget.take(nxt)
                    finished_forced = False
                if nxt in stop:
                    replaced = None if budget is None else budget.instead_of_stop(nxt, stop)
                    if replaced is None:
                        break
                    nxt = replaced
                open_think = (
                    budget is not None and not budget.closed and not budget.forcing
                )
                if count + 1 == max_tokens and not limit_keeps_piece(
                    open_think, piece_is_blank(self.root, nxt)
                ):
                    self._hit_limit = True
                    break
                out.append(nxt)
                seen.add(nxt)
                counts[nxt] = counts.get(nxt, 0) + 1
                if decode_row is not None:
                    decode_row.done = len(out)
                if on_token is not None:
                    on_token(nxt)
                repeat_done = (
                    budget is not None
                    and budget.stop_after
                    and budget.closed
                    and not budget.forcing
                )
                if finished_forced or repeat_done:
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

    def __init__(self, close: list[int], end: int, budget: int | None) -> None:
        self.close = [int(token) for token in close]
        self.end = int(end)
        self.budget = None if budget is None else int(budget)
        self.counted = 0
        self.force_at: int | None = None
        self.closed = False
        self.stop_after = False

    @property
    def forcing(self) -> bool:
        return self.force_at is not None

    def take(self, sampled: int) -> int:
        if self.closed:
            return sampled
        if self.budget is not None and self.counted >= self.budget:
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

    def _start(self, stop_after: bool = False) -> int:
        if stop_after:
            self.stop_after = True
        token = self.close[0]
        if len(self.close) == 1:
            self.closed = True
        else:
            self.force_at = 1
        return token


def _tool_call_started(produced: list[int], tool_open: int | None) -> bool:
    """True once ``tool_open`` has been drawn in this completion."""
    return tool_open is not None and tool_open in produced


def _stutter(
    produced: list[int], drawn: int, word_ids: set[int] | None = None
) -> bool:
    """True when ``drawn`` repeats the word two positions back.

    The recorded analysis drew `` messy``, a comma, and `` messy`` again.
    A markdown marker is not a word, so a bullet list stays.
    """
    if len(produced) < 2 or produced[-2] != drawn:
        return False
    return word_ids is None or drawn in word_ids


def _piece(text: str) -> str:
    """The stored piece without the GPT-2 space marker.

    That marker is U+0120, which ``isalpha`` counts as a letter. A run of
    markers is indentation, not a word.
    """
    return text.replace("Ġ", "")


def letter_token_ids(vocab: dict[str, int]) -> frozenset[int]:
    """Token ids whose stored piece contains a letter."""
    return frozenset(
        token
        for text, token in vocab.items()
        if any(character.isalpha() for character in _piece(text))
    )


def long_token_ids(vocab: dict[str, int]) -> frozenset[int]:
    """Token ids whose stored piece contains at least seven letters."""
    return frozenset(
        token
        for text, token in vocab.items()
        if sum(character.isalpha() for character in _piece(text)) >= _LONG_LETTERS
    )


def _long_repeat(
    produced: list[int],
    drawn: int,
    long_ids: set[int] | None,
    prompt_ids: set[int] | None = None,
) -> bool:
    """True when ``drawn`` is a long token already present in ``produced``.

    `` heavily`` is one token. The analysis drew it again at the start of the
    same clause. A token that is already in the prompt is the subject of the
    turn, so a second draw of that token stays.
    """
    if long_ids is None or drawn not in long_ids or drawn not in produced:
        return False
    return prompt_ids is None or drawn not in prompt_ids


def revise_think_draw(
    budget: ThinkBudget | None,
    produced: list[int],
    drawn: int,
    tool_open: int | None,
    long_ids: set[int] | None = None,
    prompt_ids: set[int] | None = None,
    word_ids: set[int] | None = None,
) -> int | None:
    """The token to emit, or None when ``drawn`` would continue a repeated clause.

    While ``<think>`` is open, a 5-token span already drawn starts the close.
    The same word with one token between the copies does the same, as does a
    second draw of a token whose piece has seven or more letters, unless that
    token is already in the prompt. After the close, any of those repeats is
    not drawn, and another token is sampled. A tool call is left as drawn,
    and a repeated span inside it stays.
    A second call in the same completion is not drawn. An 8-token block still
    applies to the draw itself.
    """
    if (
        budget is not None
        and tool_open is not None
        and drawn == tool_open
        and tool_open in produced
    ):
        return None
    if budget is None or budget.forcing or _tool_call_started(produced, tool_open):
        return drawn
    if tool_open is not None and drawn == tool_open:
        return drawn
    repeated = (
        _stutter(produced, drawn, word_ids)
        or _long_repeat(produced, drawn, long_ids, prompt_ids)
        or drawn in _blocked_tokens(produced, _THINK_REPEAT_SPAN)
    )
    if not budget.closed:
        if repeated:
            return budget._start()
        return drawn
    if repeated:
        return None
    return drawn


def assert_think_repeat_cases() -> None:
    """The clauses that looped, as token ids, do not get drawn again.

    The analysis repeated "many lines of code that are". A 7-token block
    missed it and an 8-token block missed it. "obfuscated or corrupted" did
    the same. The tool parameter repeats a shorter path from the think text,
    and that draw has to stay.
    """
    close = [20, 21, 22]
    clause = [1, 2, 3, 4, 5, 6]
    produced = [*clause, 8, *clause[:-1]]
    budget = ThinkBudget(close, 21, 64)
    got = revise_think_draw(budget, produced, clause[-1], None)
    if got != close[0] or budget.stop_after or got == clause[-1]:
        message = "a repeated think clause was drawn"
        raise AssertionError(message)
    if clause[-1] in _blocked_tokens(produced, 7):
        message = "the recorded clause is not a 7-token repeat"
        raise AssertionError(message)

    reply = ThinkBudget(close, 21, 64)
    reply.closed = True
    paraphrased = [*clause, 21, *clause[:-1]]
    if revise_think_draw(reply, paraphrased, clause[-1], 50) is not None:
        message = "the reply drew the think clause again"
        raise AssertionError(message)

    opening = ThinkBudget(close, 21, 64)
    if revise_think_draw(opening, [], 50, 50) != 50:
        message = "a tool call was replaced"
        raise AssertionError(message)
    if revise_think_draw(opening, [50, 7, 8], 50, 50) is not None:
        message = "a second tool call was drawn"
        raise AssertionError(message)

    path = [11, 12, 13, 14, 15, 16]
    tool = ThinkBudget(close, 21, 64)
    parameter = [*path, 50, *path[:-1]]
    if revise_think_draw(tool, parameter, path[-1], 50) != path[-1]:
        message = "the tool path was removed"
        raise AssertionError(message)

    # " contains many lines of code" is five tokens. A 6-token block missed it.
    five = [5435, 1599, 4965, 314, 1970]
    five_produced = [*five, 8, *five[:-1]]
    five_budget = ThinkBudget(close, 21, 64)
    if revise_think_draw(five_budget, five_produced, five[-1], None) != close[0]:
        message = "contains many lines of code was drawn again"
        raise AssertionError(message)
    if five[-1] in _blocked_tokens(five_produced, 6):
        message = "the recorded clause is a 6-token repeat"
        raise AssertionError(message)

    # " messy, messy" is that token, a comma, and the same token.
    stutter_budget = ThinkBudget(close, 21, 64)
    if revise_think_draw(stutter_budget, [760, 44318, 11], 44318, None) != close[0]:
        message = "messy, messy was drawn"
        raise AssertionError(message)
    distant = ThinkBudget(close, 21, 64)
    if revise_think_draw(distant, [44318, 1, 2, 3], 44318, None) != 44318:
        message = "a distant token was treated as a stutter"
        raise AssertionError(message)
    inside = ThinkBudget(close, 21, 64)
    if revise_think_draw(inside, [50, 44318, 11], 44318, 50) != 44318:
        message = "a stutter inside a tool call was removed"
        raise AssertionError(message)

    # " heavily" is one token. Drawing it again closes think. " file" stays.
    long_ids = {16421}
    word = ThinkBudget(close, 21, 64)
    if revise_think_draw(word, [16, 16421, 20], 16421, None, long_ids) != close[0]:
        message = "heavily was drawn again"
        raise AssertionError(message)
    short = ThinkBudget(close, 21, 64)
    if revise_think_draw(short, [999, 1, 2], 999, None, long_ids) != 999:
        message = "file was treated as a long repeat"
        raise AssertionError(message)
    kept = ThinkBudget(close, 21, 64)
    if revise_think_draw(kept, [50, 16421], 16421, 50, long_ids) != 16421:
        message = "a long token inside a tool call was removed"
        raise AssertionError(message)
    subject = ThinkBudget(close, 21, 64)
    if revise_think_draw(subject, [16, 59344], 59344, None, {59344}, {59344}) != 59344:
        message = "a prompt token was treated as a long repeat"
        raise AssertionError(message)
    bullet = ThinkBudget(close, 21, 64)
    if revise_think_draw(bullet, [348, 256], 348, None, None, None, {44318}) != 348:
        message = "a markdown bullet was treated as a stutter"
        raise AssertionError(message)
    spaces = long_token_ids({"ĠĠĠĠĠĠĠ": 285, "Ġheavily": 16421})
    if 285 in spaces or 16421 not in spaces:
        message = "a space run was treated as a long token"
        raise AssertionError(message)


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
    frequency: float = 0.0,
    counts: dict[int, int] | None = None,
) -> int:
    scores = logits
    blocked = _blocked_tokens(produced or [], _REPEAT_SPAN)
    counted = counts or {}
    if (presence and seen) or (frequency and counted) or banned or blocked:
        scores = np.array(logits, copy=True)
        if presence and seen:
            penalty = np.float32(presence)
            for token in seen:
                scores[token] -= penalty
        if frequency and counted:
            scale = np.float32(frequency)
            for token, count in counted.items():
                if count > 0:
                    scores[token] -= scale * count
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
