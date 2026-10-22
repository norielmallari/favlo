"""Text forward for one token at a time. The MTP block is packed and not drafted."""

from __future__ import annotations

import gc
import json
import struct
import time
from pathlib import Path
from typing import Any

import numpy as np

from qwen36_35b_a3b.atlas import (
    EXPERTS,
    LAYERS,
    MIN_IMAGES,
    ExpertAtlas,
    atlas_slots,
    available_bytes,
    read_expert_layout,
)
from qwen36_35b_a3b.attn import attend_plane, bind_plane, plane_bytes, resize_plane
from qwen36_35b_a3b.cache import (
    CONTEXT_STEP,
    DEFAULT_CONTEXT_LENGTH,
    fit_positions,
    kv_buffer_bytes,
    kv_mode,
    kv_record_bytes,
    max_position,
    parse_memory,
    repack_plane,
)
from qwen36_35b_a3b.convert import LANGUAGE_LAYERS, SLOT_MAP_BYTES
from qwen36_35b_a3b.firing import RouteScratch
from qwen36_35b_a3b.formats import EXPERT_FP8, EXPERT_IMAGE
from qwen36_35b_a3b.prefill import workspace_bytes
from qwen36_35b_a3b.prefix import open_session
from qwen36_35b_a3b.session import Session, _clear

HIDDEN = 2048
VOCAB = 248320
_DEVICE_BUFFERS = (
    "resident",
    "hidden",
    "normed",
    "half",
    "wide",
    "z",
    "ab",
    "ab2",
    "mixed",
    "gdn_y",
    "query",
    "gate",
    "kv",
    "vv",
    "attn",
    "expert",
    "mid",
    "gated",
    "partial",
    "ids",
    "route_ids",
    "expert_order",
    "mix_w",
    "acc",
    "router",
    "coeff",
    "logits",
    "history",
    "state",
    "k_cache",
    "v_cache",
    "plane_q",
    "plane_scores",
    "plane_probs",
    "plane_prob_half",
    "plane_row_scale",
    "plane_parts",
    "plane_partial",
    "plane_acc",
    "plane_max",
    "plane_sum",
    "plane_tail",
    "plane_chunk_max",
    "plane_chunk_sum",
    "expert_widths",
    "expert_offsets",
    "token_offsets",
)
HEAD = 256
Q_HEADS = 16
KV = 2
EPS = 1e-6
# Each head stores query then gate. The other split is the flat halves.
GATE_PER_HEAD = 1
def _scratch_bytes() -> int:
    """Activation scratch cleared while the record loads."""
    return (
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


def _reserved_bytes(resident: int, kv_one: int, plane: int) -> int:
    """Bytes allocated besides the expert atlas, including the chunk workspace."""
    return resident + _scratch_bytes() + workspace_bytes() + 20 * kv_one + plane


def _u32(value: int) -> bytes:
    return struct.pack("<I", value)


def _clock_line(phase: str, count: int, elapsed: float) -> str:
    """Phase tok/s. A zero duration writes a zero rate."""
    rate = 0.0 if elapsed <= 0.0 else count / elapsed
    return f"{phase} {rate:.2f} tok/s ({count} tok in {elapsed:.2f}s)"


def _f32(value: float) -> bytes:
    return struct.pack("<f", value)


def _file_buffer(session: Session, path: Path) -> Any:
    size = path.stat().st_size
    buffer = session.device.newBufferWithLength_options_(max(size, 16), session.shared)
    view = memoryview(buffer.contents().as_buffer(size))
    note = session._load_note
    with path.open("rb") as handle:
        offset = 0
        while offset < size:
            chunk = handle.read(8 << 20)
            if not chunk:
                break
            view[offset : offset + len(chunk)] = chunk
            offset += len(chunk)
            if note is not None:
                note(len(chunk))
    return buffer


class RequestClosed(Exception):
    """The client closed, or the listener asked the forward to stop."""


class ContextFull(Exception):
    """The next token does not fit, and the cache cannot grow any further."""


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
        self.routes = RouteScratch(LAYERS, EXPERTS)
        self._firing_open = False
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
        resident_bytes = (self.root / "resident.bin").stat().st_size
        cap = parse_memory(max_memory)
        self.max_memory = available_bytes() if cap is None else cap
        self._resident_bytes = resident_bytes
        card = max_position(self.root)
        fixed = _reserved_bytes(resident_bytes, 0, 0)
        self.expert_layout = read_expert_layout(self.root)
        image = EXPERT_FP8 if self.expert_layout.mixed else EXPERT_IMAGE
        # Eight images is one token. A larger atlas is what remains after the cache.

        def _plane(positions: int) -> int:
            if self.kv_cache == "f32":
                return 0
            return plane_bytes(positions)

        ceiling = fit_positions(
            self.max_memory, fixed, self.kv_cache, card, _plane, keep=MIN_IMAGES * image
        )
        if ceiling is None:
            message = "the key/value cache does not fit a 2048-token context"
            raise RuntimeError(message)
        self._context_limit = min(ceiling, DEFAULT_CONTEXT_LENGTH)
        self._plane_at = _plane
        self.max_pos = min(CONTEXT_STEP, ceiling)
        self.kv_record = kv_record_bytes(self.kv_cache)
        self.kv_bytes = kv_buffer_bytes(self.kv_cache, self.max_pos)
        plane = _plane(self.max_pos)
        reserved = _reserved_bytes(resident_bytes, self.kv_bytes, plane)
        images = atlas_slots(
            self.max_memory,
            image,
            reserved,
            layers=LAYERS * EXPERTS,
        )
        if self.expert_layout.mixed:
            room = self.max_memory - reserved
            total = self.expert_layout.total_bytes
            atlas_read = total if room >= total else 0
        else:
            atlas_read = LAYERS * EXPERTS * EXPERT_IMAGE if images >= LAYERS * EXPERTS else 0
        load_total = resident_bytes + atlas_read + _scratch_bytes() + 20 * self.kv_bytes + plane
        self._load_progress = self._begin_load(load_total)
        summary = ""
        try:
            self.session = Session()
            self.session._load_note = self._note_load
            self.resident = _file_buffer(self.session, self.root / "resident.bin")
            self.atlas = ExpertAtlas(
                self.session,
                self.root,
                max_memory=self.max_memory,
                reserved=reserved,
                layout=self.expert_layout,
            )
            if self.expert_layout.mixed:
                self.expert_widths = self.session.upload(self.expert_layout.widths)
                self.expert_offsets = self.session.upload(self.expert_layout.offsets)
                self.token_offsets = self.session.zeros(EXPERTS * 8)
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
            self.route_ids = self.session.zeros(LAYERS * 8 * 4)
            order_path = self.root / "experts" / "order.bin"
            self.expert_order = None
            if order_path.is_file():
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
            summary = (
                f"loaded {self.root.name} kv {self.kv_cache} context {self.max_pos} "
                f"atlas {self.atlas.slots} images memory {used}/{self.max_memory}"
            )
        finally:
            session = getattr(self, "session", None)
            if session is not None:
                session._load_note = None
            progress = self._load_progress
            self._load_progress = None
            if progress is not None and not progress.closed and self.console is not None:
                self.console.end(progress, summary)
            elif summary:
                print(summary, flush=True)
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
        """Finish device work and drop the buffers before another pipeline allocates."""
        if getattr(self, "_closed", False):
            return
        self._closed = True
        self._drop_pending()
        prefix = getattr(self, "prefix", None)
        session = getattr(self, "session", None)
        if prefix is not None and session is not None:
            try:
                prefix.flush()
            except OSError:
                pass
        if session is not None:
            try:
                session.end()
            except Exception:
                abort = getattr(session, "abort", None)
                if callable(abort):
                    abort()
        self._prefill = None
        held = getattr(self, "_held", None)
        if held is not None:
            held.clear()
        atlas = getattr(self, "atlas", None)
        if atlas is not None:
            atlas.close()
        for name in _DEVICE_BUFFERS:
            current = getattr(self, name, None)
            if isinstance(current, list):
                current.clear()
            else:
                setattr(self, name, None)
        self.atlas = None
        self.prefix = None
        self.session = None
        current = None
        session = None
        atlas = None
        prefix = None
        held = None
        gc.collect()

    def _note_routes(self) -> None:
        """Read the router ids copied before remap. One token, forty layers."""
        routes = self.routes
        route_ids = getattr(self, "route_ids", None)
        if routes is None or route_ids is None or self.session is None:
            return
        raw = self.session.download(route_ids, LAYERS * 8 * 4)
        ids = np.frombuffer(raw, dtype=np.uint32).reshape(LAYERS, 1, 8)
        routes.note_decode(ids)

    def _take_routes(self, *, prefill: bool) -> None:
        routes = self.routes
        if routes is None:
            return
        if prefill:
            routes.take_prefill_span()
        else:
            routes.take_decode(1)

    def _routed_ids(self, ids: Any, count: int) -> list[int]:
        """Finish the open buffer so the router ids can be read."""
        self.session.end()
        raw = np.frombuffer(self.session.download(ids, count * 4), dtype=np.uint32)
        return [int(value) for value in raw]

    def routed_layer(self, layer: int, ids: Any, count: int, offsets: Any) -> Any:
        """Load misses and write each file slot's byte offset. ``ids`` stay file slots."""
        file_slots = self._routed_ids(ids, count)
        table = np.zeros(EXPERTS, dtype=np.uint64)
        if self.atlas.layout.mixed:
            placed = self.atlas.stage_offsets(layer, file_slots)
            for file_slot, byte_off in zip(file_slots, placed, strict=True):
                table[file_slot] = np.uint64(byte_off)
        else:
            placed = self.atlas.stage(layer, file_slots)
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
        if self.atlas.layout.mixed:
            placed = self.atlas.stage_offsets(layer, file_slots)
            table = np.zeros(EXPERTS, dtype=np.uint64)
            for file_slot, byte_off in zip(file_slots, placed, strict=True):
                table[file_slot] = np.uint64(byte_off)
            raw = table.tobytes()
            memoryview(self.token_offsets.contents().as_buffer(len(raw)))[:] = raw
            self.session.begin()
            return self.atlas.buffer
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
        kernel = "gemv_fp8_vec" if entry.get("dtype") == "fp8" else "gemv_fp4_vec"
        self.session.encode(
            kernel,
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
        kernel = "gemv_fp8_vec_acc" if entry.get("dtype") == "fp8" else "gemv_fp4_vec_acc"
        self.session.encode(
            kernel,
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
        kernel = "gather_fp8_row" if entry.get("dtype") == "fp8" else "gather_fp4_row"
        self.session.encode(
            kernel,
            [(buf, offset), (buf, self._scale(entry, offset)), (self.hidden, 0)],
            [_u32(token), _u32(int(inner))],
            (1, 1, 1),
            (32, 1, 1),
        )

    def _gdn(self, layer: int) -> None:
        base = f"model.language_model.layers.{layer}.linear_attn."
        proj = (
            "gdn_in_proj_fp8"
            if self.tensors[base + "in_proj_qkv.weight"].get("dtype") == "fp8"
            else "gdn_in_proj"
        )
        qkv_w, qkv_s = self._weight(base + "in_proj_qkv.weight")
        z_w, z_s = self._weight(base + "in_proj_z.weight")
        a_w, a_s = self._weight(base + "in_proj_a.weight")
        b_w, b_s = self._weight(base + "in_proj_b.weight")
        self.session.encode(
            proj,
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
        proj = (
            "attn_qkv_fp8"
            if self.tensors[base + "q_proj.weight"].get("dtype") == "fp8"
            else "attn_qkv"
        )
        q_w, q_s = self._weight(base + "q_proj.weight")
        k_w, k_s = self._weight(base + "k_proj.weight")
        v_w, v_s = self._weight(base + "v_proj.weight")
        self.session.encode(
            proj,
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
        if self.routes is not None:
            self.session.encode(
                "copy_u32",
                [(self.ids, 0), (self.route_ids, layer * 8 * 4)],
                [_u32(8)],
                (1, 1, 1),
                (32, 1, 1),
            )
        if layer >= LANGUAGE_LAYERS:
            message = f"layer {layer} has no expert file"
            raise RuntimeError(message)
        if self.expert_order is not None:
            self.session.encode(
                "remap_expert_ids",
                [(self.ids, 0), (self.expert_order, 0)],
                [_u32(8), _u32(layer)],
                (1, 1, 1),
                (32, 1, 1),
            )
        gate_w, gate_s = self._weight(prefix + "shared_expert.gate_up")
        down_w, down_s = self._weight(prefix + "shared_expert.down_proj.weight")
        if self.atlas.layout.mixed:
            if self.atlas.complete:
                experts = self.atlas.layers[layer]
                offsets = (self.expert_offsets, layer * EXPERTS * 8)
            else:
                experts = self._stage_decode(layer)
                offsets = (self.token_offsets, 0)
            self._moe_mixed(layer, experts, offsets, gate_w, gate_s, down_w, down_s)
            return
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

    def _moe_mixed(
        self,
        layer: int,
        experts: Any,
        offsets: tuple[Any, int],
        gate_w: tuple[Any, int],
        gate_s: tuple[Any, int],
        down_w: tuple[Any, int],
        down_s: tuple[Any, int],
    ) -> None:
        """Shared expert is fp8. The fp8 routed kernel runs when the layer has one."""
        widths = (self.expert_widths, layer * EXPERTS)
        threads = (32, 1, 1)
        self.session.encode(
            "moe_gate_silu_w",
            [
                (self.normed, 0),
                (experts, 0),
                (self.ids, 0),
                gate_w,
                gate_s,
                (self.gated, 0),
                (self.expert, 0),
                offsets,
                widths,
            ],
            [],
            (4608, 1, 1),
            threads,
        )
        if self.atlas.layout.has(layer, 8):
            self.session.encode(
                "moe_gate_silu_fp8",
                [
                    (self.normed, 0),
                    (experts, 0),
                    (self.ids, 0),
                    (self.gated, 0),
                    offsets,
                    widths,
                ],
                [],
                (4096, 1, 1),
                threads,
            )
        self.session.encode(
            "moe_down_mix_w",
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
                offsets,
                widths,
            ],
            [],
            (HIDDEN, 1, 1),
            threads,
        )
        if self.atlas.layout.has(layer, 8):
            self.session.encode(
                "moe_down_mix_fp8",
                [
                    (self.gated, 0),
                    (experts, 0),
                    (self.ids, 0),
                    (self.mix_w, 0),
                    (self.hidden, 0),
                    offsets,
                    widths,
                ],
                [],
                (HIDDEN, 1, 1),
                threads,
            )

    def _grow(self, length: int) -> bool:
        """Widen the cache to cover ``length`` tokens and shrink the atlas to match."""
        if length <= self.max_pos:
            return True
        target = ((length + CONTEXT_STEP - 1) // CONTEXT_STEP) * CONTEXT_STEP
        target = min(target, self._context_limit)
        if target < length:
            return False
        new_bytes = kv_buffer_bytes(self.kv_cache, target)
        plane = self._plane_at(target)
        reserved = _reserved_bytes(self._resident_bytes, new_bytes, plane)
        try:
            image = EXPERT_FP8 if self.atlas.layout.mixed else EXPERT_IMAGE
            images = atlas_slots(
                self.max_memory,
                image,
                reserved,
                layers=LANGUAGE_LAYERS * EXPERTS,
            )
        except ValueError:
            return False
        if images < MIN_IMAGES:
            return False
        self._drop_pending()
        count = min(self.filled, self.max_pos)
        old_pos = self.max_pos
        keys = [self._plane_bytes(buf, old_pos) for buf in self.k_cache]
        values = [self._plane_bytes(buf, old_pos) for buf in self.v_cache]
        self._release_held([*self.k_cache, *self.v_cache, *self.atlas.buffers])
        self.k_cache = []
        self.v_cache = []
        self.atlas.retune(self.session, reserved)
        self.k_cache = [
            self._placed_plane(blob, old_pos, target, count, new_bytes, "key") for blob in keys
        ]
        self.v_cache = [
            self._placed_plane(blob, old_pos, target, count, new_bytes, "value") for blob in values
        ]
        self.max_pos = target
        self.kv_bytes = new_bytes
        if plane:
            resize_plane(self)
        self._remember()
        self._status(f"context {target} atlas {self.atlas.slots} images")
        return True

    def _plane_bytes(self, buf: Any, positions: int) -> np.ndarray:
        return np.frombuffer(
            self.session.download(buf, kv_buffer_bytes(self.kv_cache, positions)),
            dtype=np.uint8,
        ).copy()

    def _placed_plane(
        self,
        src: np.ndarray,
        old_pos: int,
        new_pos: int,
        count: int,
        new_bytes: int,
        kind: str,
    ) -> Any:
        made = self.session.zeros(new_bytes)
        if count < 1:
            return made
        packed = repack_plane(src, old_pos, new_pos, count, self.kv_cache, kind)
        memoryview(made.contents().as_buffer(new_bytes))[:] = packed.tobytes()
        return made

    def _release_held(self, buffers: list[Any]) -> None:
        drop = {id(buf) for buf in buffers}
        self._held = [buf for buf in self._held if id(buf) not in drop]

    def _remember(self) -> None:
        self._held = [
            self.resident,
            *self.atlas.buffers,
            self.hidden,
            self.history,
            self.state,
            *self.k_cache,
            *self.v_cache,
        ]

    def step(self, token: int, pos: int, *, logits: bool) -> np.ndarray | None:
        self._check_stop()
        if pos >= self.max_pos and not self._grow(pos + 1):
            raise ContextFull
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
        self._note_routes()
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
        forbid: set[int] | None = None,
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
        if not self._grow(len(prompt)):
            message = f"prompt length {len(prompt)} does not fit the memory budget"
            raise RuntimeError(message)
        # The caller passes the request cap, or the checkpoint ceiling when the
        # request sets none. The room is the context ceiling. Decode grows the
        # cache the same way the prompt does, and the atlas shrinks to fit.
        max_tokens = decode_steps(max_tokens, self._context_limit, len(prompt))
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
            self._take_routes(prefill=True)
            assert logits is not None
            self.prefix.note(prompt)
            self._note_prefill(len(suffix))
            # The forward is inside prefill. The first sample closes it.
            self._prefill_seconds = time.perf_counter() - started
            pending = list(forced or [])
            close_after = bool(stop_after_forced) and bool(pending)
            seen: set[int] = set()
            counts: dict[int, int] = {}
            budget = None
            if think_close and think_end is not None and thinking_budget != 0:
                budget = ThinkBudget(think_close, think_end, thinking_budget)
            decode_row = None
            prefill_open = True
            decode_started = started
            last_sample = started
            for count in range(max_tokens):
                self._check_stop()
                if budget is not None and budget.forcing:
                    nxt = budget.forced()
                    finished_forced = False
                elif pending:
                    nxt = int(pending.pop(0))
                    finished_forced = close_after and not pending
                    if budget is not None:
                        nxt = budget.take(nxt)
                else:
                    held = {budget.end} if budget is not None and budget.closed else set()
                    if forbid:
                        held |= forbid
                    nxt = _sample(
                        logits,
                        temperature,
                        top_p,
                        top_k,
                        presence,
                        seen,
                        held,
                        frequency,
                        counts,
                    )
                    if budget is not None:
                        nxt = budget.take(nxt)
                    finished_forced = False
                last_sample = time.perf_counter()
                if prefill_open:
                    elapsed = last_sample - started
                    self._prefill_seconds = elapsed
                    self._prefill_progress = None
                    self._end_phase(prefill_row, _clock_line("prefill", len(suffix), elapsed))
                    prefill_row = None
                    boundary = True
                    self._store_prefix()
                    decode_started = last_sample
                    decode_row = self._begin_phase("decode", max_tokens, live=True)
                    prefill_open = False
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
                    self._take_routes(prefill=False)
                    break
                if count + 1 == max_tokens:
                    self._hit_limit = True
                    self.step(nxt, len(prompt) + count, logits=False)
                    self._take_routes(prefill=False)
                    break
                logits = self.step(nxt, len(prompt) + count, logits=True)
                self._take_routes(prefill=False)
                assert logits is not None
            if prefill_open:
                elapsed = time.perf_counter() - started
                self._prefill_seconds = elapsed
                self._prefill_progress = None
                self._end_phase(prefill_row, _clock_line("prefill", len(suffix), elapsed))
                prefill_row = None
                boundary = True
                self._store_prefix()
                self._decode_seconds = 0.0
            else:
                decode_elapsed = last_sample - decode_started
                self._decode_seconds = decode_elapsed if decode_elapsed > 0.0 else 0.0
                self._end_phase(decode_row, _clock_line("decode", len(out), self._decode_seconds))
                decode_row = None
            self._output_ids = list(out)
            self.prefix.note((prompt + out)[: self.filled])
            self._store_prefix()
        except ContextFull:
            self._hit_limit = True
            if boundary:
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

    def _begin_load(self, total: int) -> Any:
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


def decode_steps(max_tokens: int, context_limit: int, prompt_tokens: int) -> int:
    """Completion steps that fit the context ceiling, not the prompt-sized cache."""
    return min(int(max_tokens), max(0, int(context_limit) - int(prompt_tokens)))


def _sample(
    logits: np.ndarray,
    temperature: float,
    top_p: float,
    top_k: int,
    presence: float,
    seen: set[int],
    banned: set[int] | None = None,
    frequency: float = 0.0,
    counts: dict[int, int] | None = None,
) -> int:
    scores = logits
    counted = counts or {}
    if (presence and seen) or (frequency and counted) or banned:
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
