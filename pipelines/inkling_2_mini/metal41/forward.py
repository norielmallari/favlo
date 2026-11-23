"""Inkling 2 Mini forward on Metal.

Prefill runs projections and experts through matmul2d tiles. The router,
the expert grouping, attention, and every norm and mix stay on the device.
A single token runs simdgroup GEMVs over float activations. One decode step
is one command buffer. The step that follows reads its token from the
device, so the host encodes step n + 1 while the device runs step n.

Each block of six layers sums its attention and expert outputs into its
block summary. A sublayer mixes the finished summaries with that partial.
Later core passes read the previous pass only through the rank-64 message
gates.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

from favlo.console import format_note, format_rate, publish_count
from favlo.memory import memory
from favlo.prefix import aim_session
from inkling_2_mini.metal41.atlas import CONTEXTS, ExpertAtlas, plan_opening
from inkling_2_mini.metal41.layout import (
    EPS,
    EXPERTS,
    HIDDEN,
    INTERMEDIATE,
    KV_DIM,
    LOGIT_SOFTCAP,
    Q_DIM,
    Q_L2,
    REL_L2,
    TOP_K,
    UNPADDED_VOCAB,
    VOCAB,
    WINDOW,
    stored_k,
)
from inkling_2_mini.metal41.prefix import VISIT_LAYERS, cache_slots, open_prefix
from inkling_2_mini.metal41.session import Session, f32, u32
from inkling_2_mini.metal41.weights import (
    ATTN_ROWS,
    HEAD_ROWS,
    Q_ROWS,
    ROW_GATE,
    ROW_K,
    ROW_REL,
    ROW_V,
    Checkpoint,
    Plane,
)
from inkling_2_mini.portable.closed import RequestClosed

CHUNK = 2048
GROUP_SPAN = 256
GROUP_SPLITS = 4
_PAIR = HIDDEN * 2
_FUSED = INTERMEDIATE * 2


def _pad64(count: int) -> int:
    return (int(count) + 63) // 64 * 64


def _max_rows(tokens: int) -> int:
    return tokens * TOP_K + EXPERTS * 31


def _max_jobs(tokens: int) -> int:
    return (tokens * TOP_K + 31) // 32 + EXPERTS


class InklingRuntime:
    """One open checkpoint. ``generate`` prefills then decodes a token list."""

    def __init__(
        self,
        path: Path,
        *,
        kv_cache: str = "half",
        console: Any = None,
        owner: str | None = None,
        calibrated: bool = True,
        context_limit: int | None = None,
        rows_limit: int | None = None,
    ) -> None:
        del kv_cache
        self.root = Path(path)
        self.packs = self.root
        self.capture: Any = None
        self.console = console
        self._owner = owner or f"inkling_2_mini:{id(self)}"
        self.kv_cache = "half"
        self._stop = None
        self._request_id = 0
        self._cached_tokens = 0
        self._output_ids: list[int] = []
        self._hit_limit = False
        self._ids: list[int] = []
        self._filled = 0
        self._prefill_seconds = 0.0
        self._decode_seconds = 0.0
        self._seed = 0x1234567
        self._snapshot_step = -1
        self._prefix_budget: int | None = None
        self._prefix_scratch: str | None = None
        self.prefix = None
        self.atlas = None
        self._states: list[Any] = []
        weight = self.root / "model-00001-of-00001.safetensors"
        if not weight.is_file():
            message = "Inkling checkpoint is not in the models folder"
            raise FileNotFoundError(message)
        self.max_memory = memory.budget()
        self._disk_done = 0
        self._alloc_done = 0
        self._prefill_progress = None
        self._prefill_done = 0
        self._load_row = self._begin_load(self._opening_bytes())
        self.session = None
        summary = ""
        try:
            self.session = Session(self._owner)
            self.weights = Checkpoint(
                self.root,
                self.session,
                progress=self,
                calibrated=calibrated,
                pack_root=self.packs,
            )
            room = max(0, memory.room() - (512 << 20))
            self.rows = CHUNK
            complete = False
            self.context = CONTEXTS[0]
            ceiling = CONTEXTS[0]
            if context_limit is not None:
                fitted = [ctx for ctx in CONTEXTS if ctx <= int(context_limit)]
                ceiling = fitted[0] if fitted else int(context_limit)
            row_choices = tuple(
                rows
                for rows in (CHUNK, 1024, 512, 256)
                if rows_limit is None or rows <= int(rows_limit)
            )
            if not row_choices:
                row_choices = (int(rows_limit or 256),)
            if ceiling == CONTEXTS[0] and rows_limit is None:
                for rows in row_choices:
                    if self.weights.expert_bytes + self._footprint(CONTEXTS[0], rows) <= room:
                        self.rows = rows
                        complete = True
                        break
                if not complete:
                    self.rows = 256
                    self.context, complete = plan_opening(
                        room, self.weights.expert_bytes, lambda ctx: self._footprint(ctx, self.rows)
                    )
            else:
                self.context = ceiling
                self.rows = row_choices[-1]
                for rows in row_choices:
                    if self.weights.expert_bytes + self._footprint(ceiling, rows) <= room:
                        self.rows = rows
                        complete = True
                        break
                if not complete:
                    from inkling_2_mini.metal41.atlas import partial_bytes

                    eight = partial_bytes()
                    floor = min(ceiling, 2048)
                    choices = [ctx for ctx in CONTEXTS if floor <= ctx <= ceiling] or [ceiling]
                    for ctx in choices:
                        need = self._footprint(ctx, self.rows)
                        if self.weights.expert_bytes + need <= room:
                            self.context = ctx
                            complete = True
                            break
                        if eight + need <= room:
                            self.context = ctx
                            break
                    else:
                        message = "Inkling does not fit at the calibration context"
                        raise MemoryError(message)
            self.atlas = ExpertAtlas(
                self.session, self.weights, owner=self._owner, complete=complete
            )
            self._allocate()
            self.reset()
            self.prefix = open_prefix(self)
            used = self.max_memory - memory.room()
            kind = "complete" if complete else "staged"
            summary = (
                f"loaded Inkling-2-Mini kv half context {self.context} rows {self.rows} "
                f"atlas {kind} dense {self.weights.dense_tag} memory {used}/{self.max_memory}"
            )
        except Exception:
            self.close()
            raise
        finally:
            row = self._load_row
            self._load_row = None
            if row is not None and self.console is not None:
                self.console.end(row, summary)
            elif summary:
                self._status(summary)

    def tick(self, done: int) -> None:
        """Disk bytes read so far. The expert pack is read after the checkpoint."""
        self._disk_done = int(done)
        self._publish_load()

    def _opening_bytes(self) -> int:
        """Checkpoint and expert pack plus the largest working set."""
        from inkling_2_mini.metal41.resident import EXPERT_PACK

        total = 0
        checkpoint = self.root / "model-00001-of-00001.safetensors"
        experts = getattr(self, "packs", self.root) / EXPERT_PACK
        if checkpoint.is_file():
            total += checkpoint.stat().st_size
        if experts.is_file():
            total += experts.stat().st_size
        total += self._footprint(CONTEXTS[0], CHUNK)
        return max(total, 1)

    def _note_alloc(self, step: int) -> None:
        if step <= 0:
            return
        self._alloc_done = int(getattr(self, "_alloc_done", 0)) + int(step)
        self._publish_load()

    def _publish_load(self) -> None:
        row = self._load_row
        if row is None:
            return
        done = int(getattr(self, "_disk_done", 0)) + int(getattr(self, "_alloc_done", 0))
        if done > int(row.total):
            row.total = done
        publish_count(self.console, row, done)

    def close(self) -> None:
        prefix = getattr(self, "prefix", None)
        if prefix is not None:
            self.prefix = None
            try:
                prefix.flush()
            except OSError:
                pass
        atlas = getattr(self, "atlas", None)
        if atlas is not None:
            atlas.close()
            self.atlas = None
        weights = getattr(self, "weights", None)
        if weights is not None:
            weights.close()
        session = getattr(self, "session", None)
        if session is not None:
            session.close()

    def reset(self) -> None:
        self._ids = []
        self._filled = 0
        if self.prefix is not None:
            self.prefix.tokens = []
            self.prefix.dirty = False
        state = getattr(self, "state_all", None)
        if state is not None:
            memory.zero(state, 0, self.state_bytes)

    # ------------------------------------------------------------------
    # Buffers

    def _footprint(self, ctx: int, rows: int) -> int:
        hidden = 24 * rows * HIDDEN * 4
        wide = rows * (ATTN_ROWS + Q_DIM + 64 * 4 + 128 + _PAIR) * 4
        half = rows * (_PAIR + 64) * 2
        moe = _max_rows(rows) * (HIDDEN * 4 + HIDDEN * 2 + stored_k(INTERMEDIATE) * 2)
        index = rows * TOP_K * 4 * 8
        slots = sum(cache_slots(layer, ctx) for layer in VISIT_LAYERS)
        kv = slots * 2 * KV_DIM * 2
        stage = (rows + min(WINDOW, ctx) + 128) * 2 * KV_DIM * 2
        small = 3 * 48 * (6 * HIDDEN + 6 * KV_DIM) * 4 + GROUP_SPLITS * 24 * 132 * 4 + VOCAB * 8
        return hidden + wide + half + moe + index + kv + stage + small

    def _allocate(self) -> None:
        rows = self.rows
        session_zeros = self.session.zeros

        def zeros(nbytes: int) -> Any:
            handle = session_zeros(nbytes)
            self._note_alloc(int(nbytes))
            return handle

        plane = rows * HIDDEN * 4

        def hidden(count: int = 1) -> Any:
            return zeros(count * plane)

        self.embed = hidden()
        self.prelude = hidden()
        self.pairs = [(hidden(), hidden()), (hidden(), hidden())]
        self.coda = hidden()
        self.final = hidden()
        self.scratch = hidden()
        self.stream3 = hidden(3)
        self.msg4 = hidden(4)
        self.msg48 = zeros(48 * HIDDEN * 4)
        table = [0, 1, 2]
        for index in range(12):
            table.extend((2 * index, 24 + 2 * index, 25 + 2 * index, 2 * index + 1))
        self.rows_table = self.session.upload(np.asarray(table, dtype=np.uint32))
        self.counters = zeros(64 * 4)
        self.merged3 = hidden(3)
        self.qkv_in = hidden(3)
        self.mlp_mix = hidden()
        self.mlp_ctl = hidden()
        self.moe_norm = hidden()
        self.moe_out = hidden()
        self.wo_out = hidden()
        self.proj = zeros(rows * ATTN_ROWS * 4)
        self.attn_out = zeros(rows * Q_DIM * 4)
        self.sink = zeros(rows * 64 * 4)
        self.k_taps = zeros(rows * 64 * 4)
        self.v_taps = zeros(rows * 64 * 4)
        self.router = zeros(rows * 128 * 4)
        self.h64 = zeros(rows * 64 * 4)
        self.fgout = zeros(rows * _PAIR * 4)
        self.half = zeros(rows * _PAIR * 2)
        self.h64_half = zeros(rows * 64 * 2)
        most = _max_rows(rows)
        self.pack = zeros(most * HIDDEN * 2)
        self.gated = zeros(most * stored_k(INTERMEDIATE) * 2)
        self.expert_out = zeros(most * HIDDEN * 4)
        self.ids = zeros(rows * TOP_K * 4)
        self.masked = zeros(rows * TOP_K * 4)
        self.route = zeros(rows * TOP_K * 4)
        self.csr_i = zeros(rows * TOP_K * 4)
        self.csr_lane = zeros(rows * TOP_K * 4)
        self.pack_row = zeros(rows * TOP_K * 4)
        self.expert_of = zeros(EXPERTS * 4)
        self.slot_n = zeros(EXPERTS * 4)
        self.slot_csr = zeros(EXPERTS * 4)
        self.row_base = zeros(EXPERTS * 4)
        self.jobs = zeros(_max_jobs(rows) * 2 * 4)
        self.info = zeros(16)
        self.tokens = zeros(max(rows, 16) * 4)
        self.partial = zeros(GROUP_SPLITS * 24 * 132 * 4)
        self.logits = zeros(HEAD_ROWS * 4)
        span_groups = (HEAD_ROWS + 255) // 256
        self.best_value = zeros(span_groups * 4)
        self.best_index = zeros(span_groups * 4)
        self.history = zeros((self.context + 8) * 4)
        self.k_cache = []
        self.v_cache = []
        for layer in VISIT_LAYERS:
            width = cache_slots(layer, self.context) * KV_DIM * 2
            self.k_cache.append(zeros(width))
            self.v_cache.append(zeros(width))
        staged = (rows + min(WINDOW, self.context) + 128) * KV_DIM * 2
        self.k_stage = zeros(staged)
        self.v_stage = zeros(staged)
        sizes = (3 * HIDDEN * 4, 3 * KV_DIM * 4, 3 * KV_DIM * 4, 3 * HIDDEN * 4)
        self.state_bytes = 48 * sum(sizes)
        self.state_all = zeros(self.state_bytes)
        self.state_snap = zeros(2 * self.state_bytes)
        self._states = []
        offset = 0
        for _visit in range(48):
            for size in sizes:
                self._states.append((self.state_all, offset))
                offset += size

    def _stopped(self) -> bool:
        stop = self._stop
        return bool(callable(stop) and stop())

    # ------------------------------------------------------------------
    # Small dispatch helpers

    def _zero(self, handle: Any, count: int) -> None:
        self.session.launch("zero_f32", [handle], [u32(count)], max(count, 1))

    def _rmsnorm(
        self,
        source: Any,
        gamma: Any,
        dest: Any,
        tokens: int,
        *,
        ny: int = 1,
        src_ys: int = 0,
        dst_ys: int = 0,
        gamma_ys: int = 0,
    ) -> None:
        self.session.dispatch(
            "rmsnorm",
            [source, dest if gamma is None else gamma, dest],
            [
                u32(tokens),
                u32(HIDDEN),
                u32(0 if gamma is None else 1),
                f32(EPS),
                u32(src_ys),
                u32(dst_ys),
                u32(gamma_ys),
            ],
            (tokens, ny, 1),
        )

    def _mix(
        self,
        query: Any,
        sources: list[Any],
        dest: Any,
        tokens: int,
        nq: int = 1,
        *,
        row: int = 0,
        dst_ys: int | None = None,
        gamma: Any = None,
        copies: int = 0,
        gamma_ys: int = 0,
    ) -> None:
        """Mix query rows table[row .. row + nq). With gamma, write RMS-normalized copies."""
        slots = list(sources[:6])
        while len(slots) < 6:
            slots.append(sources[0])
        self.session.dispatch(
            "mix_blocks",
            [query, *slots, dest, (self.rows_table, row * 4), dest if gamma is None else gamma],
            [
                u32(tokens),
                u32(HIDDEN),
                u32(len(sources)),
                f32(EPS),
                f32(HIDDEN**-0.5),
                u32(self.rows * HIDDEN if dst_ys is None else dst_ys),
                u32(copies if gamma is not None else 0),
                u32(gamma_ys),
            ],
            (tokens, nq, 1),
        )

    def _cast(self, source: Any, tokens: int, cols: int, dest: Any = None) -> int:
        padded = _pad64(tokens)
        self.session.launch(
            "cast_f16",
            [source, self.half if dest is None else dest],
            [u32(tokens), u32(padded), u32(cols)],
            padded * cols,
        )
        return padded

    def _matmul_f16(self, activation: Any, plane: Plane, dest: Any, padded: int) -> None:
        self.session.dispatch(
            "matmul_f16_m64",
            [activation, plane.handle, dest],
            [u32(padded), u32(plane.padded), u32(plane.inner)],
            (plane.padded // 64, padded // 64, 1),
        )

    def _matmul_f16_band(
        self,
        plane: Plane,
        row: int,
        cols: int,
        dest: Any,
        padded: int,
        stride: int,
    ) -> None:
        self.session.dispatch(
            "matmul_f16_band",
            [self.half, (plane.handle, row * plane.inner * 2), (dest, row * 4)],
            [u32(padded), u32(cols), u32(plane.inner), u32(stride)],
            (cols // 64, padded // 64, 1),
        )

    def _at(self, handle: Any, y: int) -> tuple[Any, int]:
        return handle, y * self.rows * HIDDEN * 4

    # ------------------------------------------------------------------
    # Message gates

    def _messages(self, layer_id: int, prev: tuple[Any, Any], tokens: int) -> tuple[Any, int]:
        """The four messages of one core layer. Returns (base, y-stride).

        A single token computes all 48 messages of a pass at the first core
        layer, since they read only the previous pass.
        """
        index = layer_id - 6
        if tokens == 1:
            if index == 0:
                self._mix(
                    self.weights.cross, [prev[0], prev[1]], self.msg48, 1, 48, row=3, dst_ys=HIDDEN
                )
            return (self.msg48, index * 4 * HIDDEN * 4), HIDDEN
        self._mix(self.weights.cross, [prev[0], prev[1]], self.msg4, tokens, 4, row=3 + 4 * index)
        return self.msg4, self.rows * HIDDEN

    def _control(
        self,
        layer: Any,
        gate0: int,
        count: int,
        stream: Any,
        s_ys: int,
        messages: tuple[Any, int],
        dest: Any,
        tokens: int,
    ) -> None:
        """Gates gate0 .. gate0 + count - 1 of one layer.

        The stream rows are s_ys floats apart; output rows are rows * hidden apart.
        """
        ys = self.rows * HIDDEN
        base, m_ys = messages
        if not isinstance(base, tuple):
            base = (base, 0)
        message = (base[0], base[1] + gate0 * m_ys * 4)
        if tokens == 1:
            self.session.dispatch(
                "ctl_down",
                [stream, message, layer.ctl_down, self.h64],
                [u32(HIDDEN), f32(EPS), u32(s_ys), u32(m_ys), u32(gate0)],
                (8, count, 1),
            )
            self.session.dispatch(
                "ctl_blend",
                [stream, message, layer.ctl_fg, self.h64, dest],
                [u32(HIDDEN), u32(s_ys), u32(m_ys), u32(gate0), u32(ys)],
                ((HIDDEN + 255) // 256, count, 1),
            )
            return
        padded = _pad64(tokens)
        for local in range(count):
            gate = gate0 + local
            s = stream if isinstance(stream, tuple) else (stream, 0)
            s_at = (s[0], s[1] + local * s_ys * 4)
            m_at = (message[0], message[1] + local * m_ys * 4)
            d_at = (
                (dest, local * ys * 4)
                if not isinstance(dest, tuple)
                else (dest[0], dest[1] + local * ys * 4)
            )
            self.session.dispatch(
                "ctl_pair_rows",
                [s_at, m_at, self.half],
                [u32(tokens), u32(HIDDEN), f32(EPS)],
                (padded, 1, 1),
            )
            self.session.dispatch(
                "matmul_f16_m64",
                [self.half, (layer.ctl_down, gate * 64 * _PAIR * 2), self.h64],
                [u32(padded), u32(64), u32(_PAIR)],
                (1, padded // 64, 1),
            )
            self._cast(self.h64, tokens, 64, self.h64_half)
            self.session.dispatch(
                "matmul_f16_m64",
                [self.h64_half, (layer.ctl_fg, gate * _PAIR * 64 * 2), self.fgout],
                [u32(padded), u32(_PAIR), u32(64)],
                (_PAIR // 64, padded // 64, 1),
            )
            self.session.launch(
                "ctl_blend_rows",
                [s_at, m_at, self.fgout, d_at],
                [u32(tokens * HIDDEN), u32(HIDDEN)],
                tokens * HIDDEN,
            )

    # ------------------------------------------------------------------
    # Attention

    def _window(self, mode: int, visit: int, origin: int, pos0: int, kend: int, slots: int) -> None:
        """Copy one local visit between its ring and the contiguous prefill span."""
        if mode == 0:
            rows = pos0 - origin if pos0 > origin else 0
        else:
            keep = kend - slots if kend > slots else 0
            if keep < origin:
                keep = origin
            rows = kend - keep if kend > keep else 0
        if rows <= 0:
            return
        self.session.dispatch(
            "kv_window",
            [self.k_stage, self.v_stage, self.k_cache[visit], self.v_cache[visit]],
            [u32(origin), u32(pos0), u32(kend), u32(slots), u32(mode)],
            ((rows + 255) // 256, 1, 1),
        )

    def _attention(
        self, layer: Any, visit: int, distinct: bool, pos0: int, tokens: int, dest: Any
    ) -> None:
        attn_state, k_state, v_state, _mlp = self._states[visit * 4 : visit * 4 + 4]
        plane = layer.attn
        q_in = self._at(self.qkv_in, 0)
        k_in = self._at(self.qkv_in, 1) if distinct else q_in
        v_in = self._at(self.qkv_in, 2) if distinct else q_in
        sink: Any = self.sink
        k_taps: Any = self.k_taps
        v_taps: Any = self.v_taps
        if tokens == 1:
            groups = ATTN_ROWS // 4 + 6 + (16 if layer.full else 0)
            self.session.dispatch(
                "gemv_attn",
                [
                    q_in,
                    k_in,
                    v_in,
                    plane.handle,
                    layer.sink.handle,
                    layer.k_proj.handle if layer.full else layer.sink.handle,
                    layer.v_proj.handle if layer.full else layer.sink.handle,
                    self.proj,
                ],
                [u32(ATTN_ROWS), u32(HIDDEN)],
                (groups, 1, 1),
            )
            sink = (self.proj, ATTN_ROWS * 4)
            k_taps = (self.proj, (ATTN_ROWS + 24) * 4)
            v_taps = (self.proj, (ATTN_ROWS + 56) * 4)
        else:
            if self.capture is not None:
                if distinct:
                    for stream, source in (("q", q_in), ("k", k_in), ("v", v_in)):
                        self.capture.add(layer.index, stream, self._gather(source, tokens, HIDDEN))
                else:
                    self.capture.add(layer.index, "shared", self._gather(q_in, tokens, HIDDEN))
            padded = self._cast(q_in, tokens, HIDDEN)
            if distinct:
                self._matmul_f16_band(plane, 0, Q_ROWS, self.proj, padded, ATTN_ROWS)
            else:
                self._matmul_f16(self.half, plane, self.proj, padded)
            self._matmul_f16(self.half, layer.sink, self.sink, padded)
            if distinct:
                self._cast(k_in, tokens, HIDDEN)
                self._matmul_f16_band(plane, ROW_K, KV_DIM, self.proj, padded, ATTN_ROWS)
            if layer.full:
                self._matmul_f16(self.half, layer.k_proj, self.k_taps, padded)
            if distinct:
                self._cast(v_in, tokens, HIDDEN)
                self._matmul_f16_band(plane, ROW_V, KV_DIM, self.proj, padded, ATTN_ROWS)
            if layer.full:
                self._matmul_f16(self.half, layer.v_proj, self.v_taps, padded)
        local = 0 if layer.full else cache_slots(layer.index, self.context)
        origin = 0
        kend = pos0 + tokens
        k_buf = self.k_cache[visit]
        v_buf = self.v_cache[visit]
        if local and tokens > 1:
            lo = max(0, pos0 + 1 - local)
            origin = lo & ~63
            if kend - origin > self.rows + min(WINDOW, self.context) + 128:
                message = "local attention span does not fit the staging buffer"
                raise RuntimeError(message)
            self._window(0, visit, origin, pos0, kend, local)
            k_buf = self.k_stage
            v_buf = self.v_stage
        slot_mask = (local - 1) if local and tokens == 1 else 0
        self.session.dispatch(
            "attn_prep",
            [
                self.proj,
                (self.proj, ROW_REL * 4),
                (self.proj, ROW_K * 4),
                (self.proj, ROW_V * 4),
                layer.q_scale,
                layer.rel_scale,
                layer.k_conv,
                layer.v_conv,
                k_taps,
                v_taps,
                k_state,
                v_state,
                k_buf,
                v_buf,
            ],
            [
                u32(tokens),
                u32(pos0),
                u32(1 if layer.full else 0),
                u32(ATTN_ROWS),
                u32(ATTN_ROWS),
                u32(ATTN_ROWS),
                u32(64),
                f32(EPS),
                f32(Q_L2),
                f32(REL_L2),
                u32(origin),
                u32(slot_mask),
            ],
            (tokens, 40, 1),
        )
        window = 0 if layer.full else local
        attend = [
            self.proj,
            k_buf,
            v_buf,
            (self.proj, ROW_REL * 4),
            layer.rel_table,
            sink,
            (self.proj, ROW_GATE * 4),
        ]
        if tokens == 1:
            lo = max(0, pos0 + 1 - window) if window else 0
            keys = pos0 + 1 - lo
            span = max(GROUP_SPAN, -(-keys // (GROUP_SPLITS * GROUP_SPAN)) * GROUP_SPAN)
            self.session.dispatch(
                "attend_group",
                [*attend, self.partial, self.counters, self.attn_out],
                [u32(pos0), u32(window), u32(span), u32(slot_mask)],
                (1, 8, -(-keys // span)),
            )
            self.session.dispatch(
                "gemv_f16_conv_add",
                [self.attn_out, layer.wo.handle, layer.attn_conv, attn_state, dest],
                [u32(HIDDEN), u32(Q_DIM)],
                (HIDDEN // 4, 1, 1),
            )
            return
        self.session.dispatch(
            "attend_flash",
            [*attend, self.attn_out],
            [
                u32(tokens),
                u32(pos0),
                u32(window),
                u32(ATTN_ROWS),
                u32(ATTN_ROWS),
                u32(64),
                u32(ATTN_ROWS),
                u32(origin),
            ],
            (24, (tokens + 31) // 32, 1),
        )
        if local:
            self._window(1, visit, origin, pos0, kend, local)
        self.session.dispatch(
            "conv_tail",
            [(self.proj, ROW_K * 4), k_state, (self.proj, ROW_V * 4), v_state],
            [u32(tokens), u32(KV_DIM), u32(ATTN_ROWS), u32(ATTN_ROWS)],
            ((KV_DIM + 255) // 256, 2, 1),
            (256, 1, 1),
        )
        if self.capture is not None:
            self.capture.add(layer.index, "wo", self._gather(self.attn_out, tokens, Q_DIM))
        padded = self._cast(self.attn_out, tokens, Q_DIM)
        self._matmul_f16(self.half, layer.wo, self.wo_out, padded)
        self._conv_add(self.wo_out, layer.attn_conv, attn_state, dest, tokens)

    def _conv_add(self, source: Any, weight: Any, state: Any, dest: Any, tokens: int) -> None:
        self.session.launch(
            "conv_add_rows",
            [source, weight, state, dest],
            [u32(tokens), u32(HIDDEN)],
            tokens * HIDDEN,
        )
        self.session.dispatch(
            "conv_tail",
            [source, state, source, state],
            [u32(tokens), u32(HIDDEN), u32(HIDDEN), u32(HIDDEN)],
            ((HIDDEN + 255) // 256, 1, 1),
            (256, 1, 1),
        )

    # ------------------------------------------------------------------
    # Experts

    def _moe(self, layer: Any, visit: int, tokens: int, dest: Any) -> None:
        mlp_state = self._states[visit * 4 + 3]
        complete = self.atlas.complete
        if tokens == 1:
            self.session.dispatch(
                "route_gemv",
                [
                    self.moe_norm,
                    layer.router.handle,
                    layer.router_bias,
                    self.ids,
                    self.route,
                    self.router,
                    (self.counters, 32 * 4),
                ],
                [u32(HIDDEN)],
                (EXPERTS // 4, 1, 1),
            )
        else:
            padded = self._cast(self.moe_norm, tokens, HIDDEN)
            self._matmul_f16(self.half, layer.router, self.router, padded)
            self.session.dispatch(
                "route_rows",
                [self.router, layer.router_bias, self.ids, self.route],
                [u32(tokens), u32(128)],
                (tokens, 1, 1),
            )
        if tokens == 1:
            ids = self.ids
            if not complete:
                self.session.end()
                picked = np.frombuffer(self.session.download(self.ids, TOP_K * 4), dtype=np.uint32)
                chosen = sorted({int(item) for item in picked})
                remap = {expert: slot for slot, expert in enumerate(chosen)}
                memory.store(
                    self.masked,
                    0,
                    np.asarray([remap[int(item)] for item in picked], dtype=np.uint32).tobytes(),
                )
                self.session.begin()
                self.atlas.stage(layer, chosen)
                ids = self.masked
            w13 = layer.w13
            w2 = layer.w2
            kind = "fp4" if w13.fp4 else "fp8"
            self.session.dispatch(
                f"moe_gate_{kind}",
                [self.moe_norm, w13.data, w13.scale, ids, self.expert_out],
                [u32(w13.data_stride), u32(w13.scale_stride), u32(w13.inner), u32(INTERMEDIATE)],
                (_FUSED // 4, TOP_K, 1),
            )
            self.session.dispatch(
                f"moe_down_{kind}",
                [
                    self.expert_out,
                    w2.data,
                    w2.scale,
                    ids,
                    self.route,
                    layer.mlp_conv,
                    mlp_state,
                    dest,
                ],
                [
                    u32(HIDDEN),
                    u32(w2.data_stride),
                    u32(w2.scale_stride),
                    u32(INTERMEDIATE),
                    u32(w2.inner),
                ],
                (HIDDEN // 4, 1, 1),
            )
            return
        if complete:
            self._grouped(layer, self.ids, tokens, accumulate=False)
        else:
            self.session.end()
            picked = np.frombuffer(
                self.session.download(self.ids, tokens * TOP_K * 4), dtype=np.uint32
            )
            picked = picked.reshape(tokens, TOP_K)
            experts = sorted({int(item) for item in picked.ravel()})
            for start in range(0, len(experts), TOP_K):
                subset = experts[start : start + TOP_K]
                lookup = np.full(EXPERTS, EXPERTS, dtype=np.uint32)
                lookup[np.asarray(subset, dtype=np.int64)] = np.arange(len(subset), dtype=np.uint32)
                memory.store(self.masked, 0, np.ascontiguousarray(lookup[picked]).tobytes())
                self.session.begin()
                self.atlas.stage(layer, subset)
                self._grouped(layer, self.masked, tokens, accumulate=start > 0)
                self.session.end()
            self.session.begin()
        self._conv_add(self.moe_out, layer.mlp_conv, mlp_state, dest, tokens)

    def _grouped(self, layer: Any, ids: Any, tokens: int, *, accumulate: bool) -> None:
        jobs = _max_jobs(tokens)
        self.session.dispatch(
            "moe_group",
            [
                ids,
                self.csr_i,
                self.expert_of,
                self.slot_n,
                self.slot_csr,
                self.row_base,
                self.jobs,
                self.info,
                self.csr_lane,
                self.pack_row,
            ],
            [u32(tokens)],
            (1, 1, 1),
        )
        self.session.dispatch(
            "gather_grouped",
            [
                self.moe_norm,
                self.csr_i,
                self.pack,
                self.slot_n,
                self.slot_csr,
                self.row_base,
                self.jobs,
                self.info,
            ],
            [u32(HIDDEN)],
            (jobs, 1, 1),
        )
        w13 = layer.w13
        w2 = layer.w2
        kind = "fp4" if w13.fp4 else "fp8"
        tile = f"expert_{kind}_tile"
        grouping = [self.jobs, self.expert_of, self.row_base, self.info]
        self.session.dispatch(
            tile,
            [self.pack, w13.data, w13.scale, self.expert_out, *grouping],
            [u32(w13.inner), u32(_FUSED), u32(w13.data_stride), u32(w13.scale_stride)],
            (_FUSED // 128, jobs, 1),
        )
        most = _max_rows(tokens)
        self.session.launch(
            "swiglu_grouped",
            [self.expert_out, self.gated, self.info],
            [u32(INTERMEDIATE), u32(w2.inner)],
            most * w2.inner,
        )
        self.session.dispatch(
            tile,
            [self.gated, w2.data, w2.scale, self.expert_out, *grouping],
            [u32(w2.inner), u32(HIDDEN), u32(w2.data_stride), u32(w2.scale_stride)],
            (HIDDEN // 128, jobs, 1),
        )
        self.session.dispatch(
            "scatter_grouped",
            [self.moe_out, self.expert_out, self.pack_row, self.route],
            [u32(HIDDEN), u32(tokens), u32(1 if accumulate else 0)],
            (tokens, 1, 1),
        )

    # ------------------------------------------------------------------
    # Blocks

    def _block(
        self,
        layers: range,
        completed: list[Any],
        prev: tuple[Any, Any] | None,
        pos0: int,
        tokens: int,
        dest: Any,
        visit0: int,
    ) -> None:
        ys = self.rows * HIDDEN
        self._zero(dest, tokens * HIDDEN)
        sources = [*completed, dest]
        for offset, layer_id in enumerate(layers):
            layer = self.weights.layers[layer_id]
            visit = visit0 + offset
            messaged = prev is not None and layer.core
            messages = self._messages(layer_id, prev, tokens) if messaged else None
            nq = layer.attn_queries
            gamma_ys = HIDDEN if layer.full else 0
            distinct = layer.full or nq == 3 or messaged
            if layer.attn_query is None:
                self._rmsnorm(completed[0], layer.norms, self.qkv_in, tokens)
            elif messaged:
                self._mix(layer.attn_query, sources, self.stream3, tokens, nq)
                self._control(
                    layer, 0, 3, self.stream3, ys if nq == 3 else 0, messages, self.merged3, tokens
                )
                self._rmsnorm(
                    self.merged3,
                    layer.norms,
                    self.qkv_in,
                    tokens,
                    ny=3,
                    src_ys=ys,
                    dst_ys=ys,
                    gamma_ys=gamma_ys,
                )
            else:
                copies = 3 if layer.full and nq == 1 else 1
                self._mix(
                    layer.attn_query,
                    sources,
                    self.qkv_in,
                    tokens,
                    nq,
                    gamma=layer.norms,
                    copies=copies,
                    gamma_ys=gamma_ys,
                )
            self._attention(layer, visit, distinct, pos0, tokens, dest)
            if messaged:
                self._mix(layer.mlp_query, sources, self.mlp_mix, tokens)
                self._control(layer, 3, 1, self.mlp_mix, 0, messages, self.mlp_ctl, tokens)
                self._rmsnorm(self.mlp_ctl, layer.mlp_norm, self.moe_norm, tokens)
            else:
                self._mix(
                    layer.mlp_query, sources, self.moe_norm, tokens, gamma=layer.mlp_norm, copies=1
                )
            self._moe(layer, visit, tokens, dest)
            if tokens > 1:
                self.session.flush()

    def _run(self, tokens: int, pos0: int, token_src: Any) -> Any:
        """Encode one forward. The caller owns begin and end."""
        self.session.launch(
            "embed_gather",
            [self.weights.embed, token_src, self.scratch],
            [u32(tokens), u32(HIDDEN)],
            tokens * HIDDEN,
        )
        self._scatter_vision(pos0, tokens)
        self._rmsnorm(self.scratch, self.weights.embed_norm, self.embed, tokens)
        self._block(range(0, 6), [self.embed], None, pos0, tokens, self.prelude, 0)
        prev: tuple[Any, Any] | None = None
        for pass_id in range(3):
            core_a, core_b = self.pairs[pass_id % 2]
            self._block(
                range(6, 12),
                [self.embed, self.prelude],
                prev,
                pos0,
                tokens,
                core_a,
                6 + pass_id * 12,
            )
            self._block(
                range(12, 18),
                [self.embed, self.prelude, core_a],
                prev,
                pos0,
                tokens,
                core_b,
                12 + pass_id * 12,
            )
            prev = (core_a, core_b)
        assert prev is not None
        self._block(
            range(18, 24),
            [self.embed, self.prelude, prev[0], prev[1]],
            None,
            pos0,
            tokens,
            self.coda,
            42,
        )
        self._mix(
            self.weights.output_query,
            [self.embed, self.prelude, prev[0], prev[1], self.coda],
            self.final,
            tokens,
            gamma=self.weights.final_norm,
            copies=1,
        )
        if self.capture is not None and tokens > 1:
            self.capture.add(-1, "head", self._gather(self.final, tokens, HIDDEN))
        return self.final

    def _gather(self, source: Any, tokens: int, width: int) -> np.ndarray:
        """Wait for the device and read ``tokens`` rows of one float buffer."""
        self.session.end()
        self.session.begin()
        handle, offset = source if isinstance(source, tuple) else (source, 0)
        return self.session.download_f32(handle, tokens * width, offset).reshape(tokens, width)

    def capture_prefill(self, tokens: list[int], capture: Any) -> None:
        """Prefill ``tokens`` from position 0 and add each dense plane input to ``capture``."""
        self.reset()
        self.capture = capture
        try:
            self._prefill([int(item) for item in tokens], 0, 0.0, False)
            capture.tokens += len(tokens)
        finally:
            self.capture = None
            self.reset()

    def _head(self, row: int) -> None:
        plane = self.weights.unembed
        self.session.dispatch(
            "gemv_f16_logits",
            [(self.final, row * HIDDEN * 4), plane.handle, self.logits],
            [u32(HEAD_ROWS), u32(HIDDEN), u32(UNPADDED_VOCAB), f32(LOGIT_SOFTCAP)],
            (HEAD_ROWS // 4, 1, 1),
        )

    def _pick(self, slot: int, temperature: float) -> None:
        """Argmax, or one Gumbel draw, of the head into history[slot]."""
        groups = (HEAD_ROWS + 255) // 256
        self._seed = (self._seed * 1103515245 + 12345) & 0xFFFFFFFF
        inv = 1.0 / temperature if temperature > 0.0 else 0.0
        self.session.dispatch(
            "argmax_span",
            [self.logits, self.best_value, self.best_index],
            [u32(HEAD_ROWS), f32(inv), u32(self._seed)],
            (groups, 1, 1),
        )
        self.session.dispatch(
            "argmax_index",
            [self.best_value, self.best_index, (self.history, slot * 4)],
            [u32(groups)],
            (1, 1, 1),
        )

    def _logits(self) -> np.ndarray:
        scores = np.full(VOCAB, -np.inf, dtype=np.float32)
        scores[:UNPADDED_VOCAB] = self.session.download_f32(self.logits, UNPADDED_VOCAB)
        return scores

    # ------------------------------------------------------------------
    # Sampling on the host, for top-p, top-k, penalties, and bans.

    def _sample(
        self,
        logits: np.ndarray,
        temperature: float,
        top_p: float,
        top_k: int,
        presence: float,
        frequency: float,
        seen: list[int],
        forbid: set[int] | None,
    ) -> int:
        scores = logits.copy()
        if forbid:
            for item in forbid:
                if 0 <= item < VOCAB:
                    scores[item] = -np.inf
        if (presence or frequency) and seen:
            ids, counts = np.unique(np.asarray(seen, dtype=np.int64), return_counts=True)
            keep = (ids >= 0) & (ids < VOCAB)
            ids = ids[keep]
            counts = counts[keep]
            scores[ids] -= presence + frequency * counts
        if temperature <= 0.0:
            return int(np.argmax(scores))
        scores = scores / temperature
        if 0 < top_k < VOCAB:
            chosen = np.argpartition(scores, -top_k)[-top_k:]
            masked = np.full(VOCAB, -np.inf, dtype=np.float32)
            masked[chosen] = scores[chosen]
            scores = masked
        if 0.0 < top_p < 1.0:
            order = np.argsort(scores)[::-1]
            shifted = scores[order] - np.max(scores)
            probs = np.exp(shifted)
            total = float(probs.sum())
            if total > 0.0:
                probs /= total
                cutoff = int(np.searchsorted(np.cumsum(probs), top_p)) + 1
                scores[order[cutoff:]] = -np.inf
        shifted = scores - np.max(scores)
        probs = np.exp(shifted)
        total = float(probs.sum())
        if not np.isfinite(total) or total <= 0.0:
            return int(np.argmax(logits))
        probs /= total
        return int(np.random.choice(VOCAB, p=probs))

    def _begin_load(self, total: int) -> Any:
        console = self.console
        if console is None or total <= 0 or not getattr(console, "shows_bars", False):
            return None
        detail = getattr(console, "load_detail", "") or self.root.name
        return console.begin("load", total=total, unit="bytes", detail=detail)

    def _begin_phase(self, phase: str, total: int, *, live: bool = False) -> Any:
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

    def _end_phase(self, progress: Any, summary: str) -> None:
        if progress is None:
            if summary:
                self._status(summary)
            return
        if getattr(progress, "closed", False):
            return
        console = self.console
        if console is None:
            if summary:
                self._status(summary)
            return
        console.end(progress, summary)

    def _note_prefill(self, done: int) -> None:
        progress = self._prefill_progress
        if progress is None:
            return
        capped = min(int(progress.total), int(done))
        if capped <= int(getattr(self, "_prefill_done", 0)):
            return
        self._prefill_done = capped
        publish_count(self.console, progress, capped)

    # ------------------------------------------------------------------
    # Generate

    def _prefill(self, fresh: list[int], pos: int, temperature: float, device_pick: bool) -> None:
        offset = 0
        last = 0
        while offset < len(fresh):
            piece = fresh[offset : offset + self.rows]
            memory.store(self.tokens, 0, np.asarray(piece, dtype=np.uint32).tobytes())
            self.session.begin()
            try:
                self._run(len(piece), pos + offset, self.tokens)
                last = len(piece)
                offset += last
                self._note_prefill(offset)
                if offset >= len(fresh):
                    self._head(last - 1)
                    if device_pick:
                        self._pick(0, temperature)
            finally:
                self.session.end()
            if self._stopped():
                raise RequestClosed()

    def generate(
        self,
        prompt: list[int],
        limit: int,
        stop: set[int],
        *,
        temperature: float = 0.0,
        top_p: float = 1.0,
        top_k: int = 0,
        presence: float = 0.0,
        frequency: float = 0.0,
        mtp: bool = False,
        mtp_depth: int | None = None,
        forced: list[int] | None = None,
        stop_after_forced: bool = False,
        tool_close: list[int] | None = None,
        fill: bool = False,
        on_token: Any = None,
        forbid: set[int] | None = None,
    ) -> list[int]:
        del mtp_depth, forced, stop_after_forced, tool_close
        if mtp:
            message = "this runtime does not draft"
            raise RuntimeError(message)
        if self._stopped():
            raise RequestClosed()
        if not callable(on_token):
            on_token = None
        tokens = [int(item) for item in prompt]
        dropped = 0
        if len(tokens) >= self.context:
            dropped = len(tokens) - (self.context - 1)
            tokens = tokens[dropped:]
        self._note("prompt", tokens=len(tokens))
        aim_session(self.prefix, budget=self._prefix_budget, scratch=self._prefix_scratch)
        pos = self.prefix.align(tokens)
        if self.prefix.loaded:
            self._note("prefix_load", tokens=pos)
        self._note("cache", hit=pos)
        self._cached_tokens = pos
        fresh = tokens[pos:]
        self._hit_limit = False
        self._output_ids = []
        device_pick = (
            not forbid and not presence and not frequency and top_k <= 0 and not (0.0 < top_p < 1.0)
        )
        from inkling_2_mini.metal41.vision import bind_vision, release_vision

        prefill_row = None
        decode_row = None
        try:
            bind_vision(self, tokens, pos, dropped)
            started = time.perf_counter()
            prefill_row = self._begin_phase("prefill", len(fresh))
            self._prefill_progress = prefill_row
            self._prefill_done = 0
            try:
                self._prefill(fresh, pos, temperature, device_pick)
            except RequestClosed:
                self.prefix.tokens = []
                self.prefix.dirty = False
                self.reset()
                raise
            release_vision(self)
            self._ids = list(tokens)
            self._filled = len(tokens)
            self._prefill_seconds = time.perf_counter() - started
            self._note_prefill(len(fresh))
            self._prefill_progress = None
            self._end_phase(prefill_row, format_rate("prefill", len(fresh), self._prefill_seconds))
            prefill_row = None
            self.prefix.note(tokens)
            self._store_prefix()
            limit = max(int(limit), 0)
            room = self.context - self._filled
            steps = min(limit, room)
            decode_row = self._begin_phase("decode", max(steps, 1), live=True)
            decode_started = time.perf_counter()
            try:
                if device_pick:
                    produced = self._decode_device(
                        steps, temperature, stop, fill, on_token, decode_row
                    )
                else:
                    produced = self._decode_host(
                        steps,
                        temperature,
                        top_p,
                        top_k,
                        presence,
                        frequency,
                        forbid,
                        stop,
                        fill,
                        on_token,
                        decode_row,
                        tokens,
                    )
            except RequestClosed:
                self.prefix.note(self._ids[: self._filled])
                self._store_prefix()
                raise
            self._hit_limit = steps > 0 and len(produced) >= steps
            self._decode_seconds = time.perf_counter() - decode_started
            self._settle(len(tokens), produced)
            self.prefix.note(self._ids[: self._filled])
            self._store_prefix()
            self._end_phase(decode_row, format_rate("decode", len(produced), self._decode_seconds))
            decode_row = None
            self._output_ids = produced
            return produced
        finally:
            self._end_phase(prefill_row, "")
            self._end_phase(decode_row, "")
            self._prefill_progress = None
            release_vision(self)

    def _settle(self, prompt: int, produced: list[int]) -> None:
        """Leave the device holding exactly the prompt and every produced token.

        Pipelined decode may have applied the stop token after the last produced
        one: the state snapshot taken before that step goes back. A decode that
        ended at its length limit has not applied its last token: one more step
        applies it.
        """
        want = prompt + len(produced)
        if self._filled == want + 1 and self._snapshot_step == len(produced):
            slot = (len(produced) % 2) * self.state_bytes
            self._copy_states((self.state_snap, slot), self.state_all)
            self._filled = want
            del self._ids[want:]
        elif self._filled == want - 1 and want <= self.context:
            memory.store(self.tokens, 0, np.asarray([produced[-1]], dtype=np.uint32).tobytes())
            self.session.begin()
            try:
                self._run(1, self._filled, self.tokens)
            finally:
                self.session.end()
            self._ids.append(int(produced[-1]))
            self._filled = want

    def _copy_states(self, source: Any, dest: Any) -> None:
        count = self.state_bytes // 16
        self.session.launch("copy_words", [source, dest], [u32(count)], count)

    def _scatter_vision(self, pos0: int, tokens: int) -> None:
        bank = getattr(self, "_vision_bank", None)
        slots = getattr(self, "_vision_slots", None)
        if bank is None or slots is None:
            return
        self.session.launch(
            "vision_scatter",
            [self.scratch, bank, slots],
            [u32(pos0), u32(tokens), u32(HIDDEN)],
            tokens * HIDDEN,
        )

    def _status(self, line: str) -> None:
        console = self.console
        if console is None:
            print(line, flush=True)
            return
        console.log(line)

    def _note(self, topic: str, **fields: Any) -> None:
        note = getattr(self.console, "note", None)
        if callable(note):
            note(topic, **fields)
            return
        self._status(format_note(topic, **fields))

    def _store_prefix(self) -> None:
        try:
            path = self.prefix.flush()
        except OSError as exc:
            self._note("prefix_store_failed", error=exc)
            return
        if path is not None:
            self._note("prefix_store", tokens=len(self.prefix.tokens))

    def _emit(self, token: int, produced: list[int], on_token: Any, decode_row: Any) -> None:
        produced.append(token)
        if on_token is not None:
            on_token(token)
        publish_count(self.console, decode_row, len(produced))

    def _decode_device(
        self,
        steps: int,
        temperature: float,
        stop: set[int],
        fill: bool,
        on_token: Any,
        decode_row: Any,
    ) -> list[int]:
        """Each step reads its token from history[i] and writes history[i + 1].

        Unless fill is set, each step first copies the short-convolution states
        into snapshot slot i % 2, so a step that applied the stop token can be
        taken back.
        """
        produced: list[int] = []
        self._snapshot_step = -1
        if steps < 1:
            return produced
        first = int(np.frombuffer(self.session.download(self.history, 4, 0), dtype=np.uint32)[0])
        if not fill and first in stop:
            return produced
        self._emit(first, produced, on_token, decode_row)
        encoded = 0
        session = self.session

        def encode(step: int) -> None:
            session.begin()
            if not fill:
                self._copy_states(self.state_all, (self.state_snap, (step % 2) * self.state_bytes))
            self._run(1, self._filled + step, (self.history, step * 4))
            self._head(0)
            self._pick(step + 1, temperature)
            session.commit()

        try:
            if steps > 1:
                encode(0)
                encoded = 1
            for step in range(steps - 1):
                if self._stopped():
                    raise RequestClosed()
                if encoded < steps - 1:
                    encode(encoded)
                    encoded += 1
                session.wait_oldest()
                token = int(
                    np.frombuffer(
                        session.download(self.history, 4, (step + 1) * 4), dtype=np.uint32
                    )[0]
                )
                if not fill and token in stop:
                    break
                self._emit(token, produced, on_token, decode_row)
        finally:
            session.drain()
            # Every encoded step ran, so its token is in the cache.
            if encoded:
                fed = np.frombuffer(session.download(self.history, encoded * 4, 0), dtype=np.uint32)
                self._ids.extend(int(item) for item in fed)
                self._filled += encoded
                if not fill:
                    self._snapshot_step = encoded - 1
        return produced

    def _decode_host(
        self,
        steps: int,
        temperature: float,
        top_p: float,
        top_k: int,
        presence: float,
        frequency: float,
        forbid: Any,
        stop: set[int],
        fill: bool,
        on_token: Any,
        decode_row: Any,
        prompt: list[int],
    ) -> list[int]:
        produced: list[int] = []
        self._snapshot_step = -1
        seen = list(prompt)
        logits = self._logits()
        for step in range(steps):
            if self._stopped():
                raise RequestClosed()
            token = self._sample(
                logits, temperature, top_p, top_k, presence, frequency, seen, forbid
            )
            if not fill and token in stop:
                break
            self._emit(token, produced, on_token, decode_row)
            seen.append(token)
            if step + 1 >= steps:
                break
            memory.store(self.tokens, 0, np.asarray([token], dtype=np.uint32).tobytes())
            self.session.begin()
            try:
                self._run(1, self._filled, self.tokens)
                self._head(0)
            finally:
                self.session.end()
            self._ids.append(token)
            self._filled += 1
            logits = self._logits()
        return produced
