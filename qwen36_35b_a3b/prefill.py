"""Chunked prompt prefill. Projections are fp4 matmul2d tiles of 64 rows.

Decode stays a GEMV. A chunk is at most 2048 tokens so a routed expert's
group is large enough for that tile. The groups are built on device, and the
layer stays one command buffer. The Gated DeltaNet scan is chunks of 64,
with the state update and the outputs dispatched across the chunk and the 32
heads. The short convolution is one thread per token and channel. Float
attention is one dispatch per layer. fp8 and fp4 attention
scores and value mixes are ``matmul2d`` tiles. Expert groups are gathered,
multiplied, and scattered back onto the token rows. A widths file
dispatches the fp8 expert tile when that expert's width byte is 8.
"""

from __future__ import annotations

import struct
from typing import Any

import numpy as np

from qwen36_35b_a3b.attn import attend_plane
from qwen36_35b_a3b.convert import GATE_UP_BYTES
from qwen36_35b_a3b.formats import EXPERT_IMAGE, FP8_DOWN_DATA, FP8_DOWN_SCALE, FP8_GATE_SCALE

CHUNK = 2048
HIDDEN = 2048
HEAD = 256
EPS = 1e-6
_GDN_CHUNK = 64
_GATE_SCALE = 1_048_576
_DOWN_SCALE = 524_288
_EXPERTS = 256
# Pad each expert to 64 rows. The bound is every route in its own expert.
_MAX_ROWS = CHUNK * 8 + 63 * _EXPERTS
_JOBS = (_MAX_ROWS + 63) // 64
_SLACK = 1024


def _u32(value: int) -> bytes:
    return struct.pack("<I", value)


def _f32(value: float) -> bytes:
    return struct.pack("<f", value)


def _write(buffer: Any, blob: bytes) -> None:
    if not blob:
        return
    memoryview(buffer.contents().as_buffer(len(blob)))[: len(blob)] = blob


class _Workspace:
    def __init__(self, session: Any) -> None:
        cap = CHUNK
        zeros = session.zeros
        self.hidden = zeros(cap * HIDDEN * 4)
        self.normed = zeros(cap * HIDDEN * 4)
        self.half = zeros(cap * 4096 * 2)
        self.wide = zeros(cap * 8192 * 4)
        self.mixed = zeros(cap * 8192 * 4)
        self.z = zeros(cap * 4096 * 4)
        self.ab = zeros(cap * 32 * 4)
        self.ab2 = zeros(cap * 32 * 4)
        self.y = zeros(cap * 4096 * 4)
        self.query = zeros(cap * 4096 * 4)
        self.gate = zeros(cap * 4096 * 4)
        self.kv = zeros(cap * 512 * 4)
        self.vv = zeros(cap * 512 * 4)
        self.attn = zeros(cap * 4096 * 4)
        self.router = zeros(cap * 256 * 4)
        self.coeff = zeros(cap * 4)
        self.ids = zeros(cap * 8 * 4)
        self.route_ids = zeros(40 * cap * 8 * 4)
        self.mix = zeros(cap * 8 * 4)
        self.gate_up = zeros(cap * 1024 * 4)
        self.gated = zeros(cap * 512 * 4)
        self.down = zeros(cap * HIDDEN * 4)
        self.pack = zeros(cap * HIDDEN * 2)
        self.expert_pack = zeros(_MAX_ROWS * HIDDEN * 2 + _SLACK)
        self.expert_out = zeros(_MAX_ROWS * 1024 * 4 + _SLACK)
        self.expert_down = zeros(_MAX_ROWS * HIDDEN * 4 + _SLACK)
        self.expert_gated = zeros(_MAX_ROWS * 512 * 4 + _SLACK)
        self.jobs = zeros(_JOBS * 2 * 4)
        self.expert_of = zeros(_EXPERTS * 4)
        self.slot_n = zeros(_EXPERTS * 4)
        self.slot_csr = zeros(_EXPERTS * 4)
        self.row_base = zeros(_EXPERTS * 4)
        self.group = zeros(8 * 16)
        self.pack_row = zeros(cap * 8 * 4)
        self.tokens = zeros(cap * 4)
        self.rope = zeros(cap * 3 * 4)
        self.offsets = zeros(_EXPERTS * 8)
        # Alignment padding between experts can add up to three slots each.
        self.csr_i = zeros((cap * 8 + 1024) * 4)
        self.csr_w = zeros((cap * 8 + 1024) * 4)
        self.csr_lane = zeros((cap * 8 + 1024) * 4)
        self.iota = zeros(cap * 4)
        _write(self.iota, np.arange(cap, dtype=np.uint32).tobytes())
        self.gdn_kkt = zeros(32 * _GDN_CHUNK * _GDN_CHUNK * 4)
        self.gdn_w = zeros(32 * _GDN_CHUNK * 128 * 4)
        self.gdn_u = zeros(32 * _GDN_CHUNK * 128 * 4)
        self.gdn_vbar = zeros(32 * _GDN_CHUNK * 128 * 4)
        self.gdn_logg = zeros(32 * _GDN_CHUNK * 4)


def workspace_bytes() -> int:
    """Bytes of the chunk workspace, reserved before the atlas is sized."""
    cap = CHUNK
    return (
        cap * HIDDEN * 4
        + cap * HIDDEN * 4
        + cap * 4096 * 2
        + cap * 8192 * 4
        + cap * 8192 * 4
        + cap * 4096 * 4
        + cap * 32 * 4
        + cap * 32 * 4
        + cap * 4096 * 4
        + cap * 4096 * 4
        + cap * 4096 * 4
        + cap * 512 * 4
        + cap * 512 * 4
        + cap * 4096 * 4
        + cap * 256 * 4
        + cap * 4
        + cap * 8 * 4
        + 40 * cap * 8 * 4
        + cap * 8 * 4
        + cap * 1024 * 4
        + cap * 512 * 4
        + cap * HIDDEN * 4
        + cap * HIDDEN * 2
        + _MAX_ROWS * HIDDEN * 2
        + _SLACK
        + _MAX_ROWS * 1024 * 4
        + _SLACK
        + _MAX_ROWS * HIDDEN * 4
        + _SLACK
        + _MAX_ROWS * 512 * 4
        + _SLACK
        + _JOBS * 2 * 4
        + _EXPERTS * 4
        + _EXPERTS * 4
        + _EXPERTS * 4
        + _EXPERTS * 4
        + _EXPERTS * 8
        + 8 * 16
        + cap * 8 * 4
        + cap * 4
        + cap * 3 * 4
        + (cap * 8 + 1024) * 4
        + (cap * 8 + 1024) * 4
        + (cap * 8 + 1024) * 4
        + cap * 4
        + _gdn_scratch_bytes()
    )


def _gdn_scratch_bytes() -> int:
    return 32 * _GDN_CHUNK * (_GDN_CHUNK + 128 * 3) * 4 + 32 * _GDN_CHUNK * 4


def prefill_ids(runtime: Any, ids: list[int], pos0: int = 0) -> None:
    """Advance state and the KV cache through ``ids``. The last prompt token is separate."""
    if not ids:
        return
    if pos0 + len(ids) >= runtime.max_pos:
        message = f"position {pos0 + len(ids)} is past {runtime.max_pos}"
        raise RuntimeError(message)
    workspace = getattr(runtime, "_prefill", None)
    if workspace is None:
        workspace = _Workspace(runtime.session)
        runtime._prefill = workspace
    cursor = pos0
    remaining = ids
    while remaining:
        take = min(CHUNK, len(remaining))
        runtime._prefill_base = cursor - pos0
        _chunk(runtime, workspace, remaining[:take], cursor)
        cursor += take
        remaining = remaining[take:]
        note = getattr(runtime, "_note_prefill", None)
        if note is not None:
            note(cursor - pos0)


def _note_chunk(runtime: Any, finished: int, count: int) -> None:
    note = getattr(runtime, "_note_prefill", None)
    if note is None:
        return
    base = int(getattr(runtime, "_prefill_base", 0))
    note(base + finished * count // 40)


def _chunk(runtime: Any, work: _Workspace, ids: list[int], pos0: int) -> None:
    count = len(ids)
    rows = (count + 63) // 64 * 64
    _write(work.tokens, np.asarray(ids, dtype=np.uint32).tobytes())
    runtime.session.begin()
    _zero(runtime, work.hidden, rows * HIDDEN)
    _embed(runtime, work, count)
    runtime.session.end()
    _splice_vision(runtime, work, ids, pos0)
    _write_rope(runtime, work, count, pos0)
    finished = 0
    try:
        for layer in range(40):
            check = getattr(runtime, "_check_stop", None)
            if check is not None:
                check()
            runtime.session.begin()
            _zero(runtime, work.normed, rows * HIDDEN)
            _rms(runtime, work.hidden, _input_norm(layer), work.normed, count)
            if layer % 4 == 3:
                _attend(runtime, work, layer, count, rows, pos0)
            else:
                _gdn(runtime, work, layer, count, rows)
            _zero(runtime, work.normed, rows * HIDDEN)
            _rms(runtime, work.hidden, _post_norm(layer), work.normed, count)
            _route(runtime, work, layer, count, rows)
            _experts(runtime, work, layer, count, rows)
            runtime.session.commit()
            if runtime.session.inflight() > 1:
                runtime.session.wait_oldest()
                finished += 1
                _note_chunk(runtime, finished, count)
        runtime.session.wait_inflight()
        _note_chunk(runtime, 40, count)
        _collect_routes(runtime, work, count)
    finally:
        if runtime.session._recording:
            runtime.session.abort()
        runtime.session.wait_inflight()


def _splice_vision(runtime: Any, work: _Workspace, ids: list[int], pos0: int) -> None:
    table = getattr(runtime, "_vision_at", None)
    if not table:
        return
    count = len(ids)
    slots = [index for index in range(count) if pos0 + index in table]
    if not slots:
        return
    raw = runtime.session.download(work.hidden, count * HIDDEN * 4)
    hidden = np.frombuffer(raw, dtype=np.float32).reshape(count, HIDDEN).copy()
    for index in slots:
        hidden[index] = table[pos0 + index]
    _write(work.hidden, np.ascontiguousarray(hidden, dtype=np.float32).tobytes())


def _write_rope(runtime: Any, work: _Workspace, count: int, pos0: int) -> None:
    table = getattr(runtime, "_rope_positions", None)
    if table is None:
        base = np.arange(pos0, pos0 + count, dtype=np.uint32)
        packed = np.stack((base, base, base), axis=1)
    else:
        packed = np.ascontiguousarray(table[:, pos0 : pos0 + count].T, dtype=np.uint32)
    _write(work.rope, packed.tobytes())


def _embed(runtime: Any, work: _Workspace, count: int) -> None:
    buf, offset, entry = runtime._loc("model.language_model.embed_tokens.weight")
    _rows, inner = entry["stored_shape"]
    kernel = "embed_fp8_rows" if entry.get("dtype") == "fp8" else "embed_fp4_rows"
    runtime.session.encode(
        kernel,
        [
            (buf, offset),
            (buf, runtime._scale(entry, offset)),
            (work.tokens, 0),
            (work.hidden, 0),
        ],
        [_u32(count), _u32(int(inner))],
        (count, 1, 1),
        (32, 1, 1),
    )


def _gdn(runtime: Any, work: _Workspace, layer: int, count: int, rows: int) -> None:
    base = f"model.language_model.layers.{layer}.linear_attn."
    _cast(runtime, work.normed, work.half, rows * HIDDEN)
    _matmul(runtime, work.half, base + "in_proj_qkv.weight", work.wide, rows)
    _matmul(runtime, work.half, base + "in_proj_z.weight", work.z, rows)
    _matmul(runtime, work.half, base + "in_proj_a.weight", work.ab, rows)
    _matmul(runtime, work.half, base + "in_proj_b.weight", work.ab2, rows)
    slot = layer - (layer // 4)
    hist = slot * 3 * 8192 * 2
    weight, woff, _entry = runtime._loc(base + "conv1d.weight")
    runtime.session.encode(
        "gdn_conv_rows",
        [(runtime.history, hist), (work.wide, 0), (weight, woff), (work.mixed, 0)],
        [_u32(count), _u32(8192)],
        (count, 32, 1),
        (256, 1, 1),
    )
    runtime.session.encode(
        "gdn_conv_keep",
        [(runtime.history, hist), (work.wide, 0)],
        [_u32(count), _u32(8192)],
        (32, 1, 1),
        (256, 1, 1),
    )
    along, aoff, _entry = runtime._loc(base + "A_log")
    bias, boff, _entry = runtime._loc(base + "dt_bias")
    norm, noff, _entry = runtime._loc(base + "norm.weight")
    state = slot * 32 * 128 * 128 * 4
    _zero(runtime, work.y, rows * 4096)
    runtime.session.encode(
        "gdn_l2_rows",
        [(work.mixed, 0)],
        [_f32(EPS), _u32(count)],
        (count, 16, 1),
        (128, 1, 1),
    )
    cursor = 0
    while cursor < count:
        clen = min(_GDN_CHUNK, count - cursor)
        _gdn_chunk(runtime, work, state, along, aoff, bias, boff, cursor, clen)
        cursor += clen
    runtime.session.encode(
        "gdn_y_norm",
        [(work.y, 0), (work.z, 0), (norm, noff)],
        [_f32(EPS), _u32(count)],
        (count, 32, 1),
        (128, 1, 1),
    )
    _cast(runtime, work.y, work.half, rows * 4096)
    _matmul(runtime, work.half, base + "out_proj.weight", work.down, rows)
    _add(runtime, work, count)


def _gdn_chunk(
    runtime: Any,
    work: _Workspace,
    state: int,
    along: Any,
    aoff: int,
    bias: Any,
    boff: int,
    c0: int,
    clen: int,
) -> None:
    runtime.session.encode(
        "gdn_chunk_solve",
        [
            (runtime.state, state),
            (work.mixed, 0),
            (work.ab, 0),
            (work.ab2, 0),
            (along, aoff),
            (bias, boff),
            (work.gdn_kkt, 0),
            (work.gdn_w, 0),
            (work.gdn_u, 0),
            (work.gdn_vbar, 0),
            (work.gdn_logg, 0),
        ],
        [_u32(c0), _u32(clen)],
        (32, 1, 1),
        (128, 1, 1),
    )
    runtime.session.encode(
        "gdn_chunk_out",
        [
            (runtime.state, state),
            (work.mixed, 0),
            (work.gdn_vbar, 0),
            (work.gdn_logg, 0),
            (work.y, 0),
        ],
        [_u32(c0), _u32(clen)],
        (32, clen, 1),
        (128, 1, 1),
    )
    runtime.session.encode(
        "gdn_chunk_state",
        [
            (runtime.state, state),
            (work.mixed, 0),
            (work.gdn_vbar, 0),
            (work.gdn_logg, 0),
        ],
        [_u32(c0), _u32(clen)],
        (32, 128, 1),
        (128, 1, 1),
    )


def _attend(
    runtime: Any, work: _Workspace, layer: int, count: int, rows: int, pos0: int
) -> None:
    base = f"model.language_model.layers.{layer}.self_attn."
    _cast(runtime, work.normed, work.half, rows * HIDDEN)
    _matmul(runtime, work.half, base + "q_proj.weight", work.wide, rows)
    _matmul(runtime, work.half, base + "k_proj.weight", work.kv, rows)
    _matmul(runtime, work.half, base + "v_proj.weight", work.vv, rows)
    runtime.session.encode(
        "split_q_gate_rows",
        [(work.wide, 0), (work.query, 0), (work.gate, 0)],
        [_u32(count)],
        (count * 16, 1, 1),
        (256, 1, 1),
    )
    q_buf, q_off, _entry = runtime._loc(base + "q_norm.weight")
    k_buf, k_off, _entry = runtime._loc(base + "k_norm.weight")
    runtime.session.encode(
        "qk_norm_rope_rows",
        [
            (work.query, 0),
            (work.kv, 0),
            (q_buf, q_off),
            (k_buf, k_off),
            (work.rope, 0),
        ],
        [_f32(1e7), _f32(EPS)],
        (count, 18, 1),
        (HEAD, 1, 1),
    )
    slot = layer // 4
    _zero(runtime, work.attn, rows * 4096)
    if runtime.kv_cache == "f32":
        stride = runtime.max_pos * HEAD
        runtime.session.encode(
            "store_kv_rows",
            [
                (runtime.k_cache[slot], 0),
                (runtime.v_cache[slot], 0),
                (work.kv, 0),
                (work.vv, 0),
            ],
            [_u32(pos0), _u32(stride)],
            (count, 2, 1),
            (HEAD, 1, 1),
        )
        runtime.session.encode(
            "attend_prefill2",
            [
                (work.query, 0),
                (runtime.k_cache[slot], 0),
                (runtime.v_cache[slot], 0),
                (work.gate, 0),
                (work.attn, 0),
            ],
            [_u32(count), _u32(pos0), _u32(stride)],
            ((count + 1) // 2, 16, 1),
            (32, 1, 1),
        )
    else:
        attend_plane(
            runtime,
            query=work.query,
            key=work.kv,
            value=work.vv,
            gate=work.gate,
            dest=work.attn,
            pos0=pos0,
            count=count,
            slot=slot,
        )
    _cast(runtime, work.attn, work.half, rows * 4096)
    _matmul(runtime, work.half, base + "o_proj.weight", work.down, rows)
    _add(runtime, work, count)


def _route(runtime: Any, work: _Workspace, layer: int, count: int, rows: int) -> None:
    prefix = f"model.language_model.layers.{layer}.mlp."
    router, roff, entry = runtime._loc(prefix + "gate.weight")
    _rows, inner = entry["shape"]
    _cast(runtime, work.normed, work.half, rows * HIDDEN)
    runtime.session.encode(
        "prefill_f16",
        [(work.half, 0), (router, roff), (work.router, 0)],
        [_u32(rows), _u32(int(_rows)), _u32(int(inner))],
        (int(_rows) // 32, rows // 64, 1),
        _tile_threads(runtime),
    )
    coeff, coff, centry = runtime._loc(prefix + "shared_expert_gate.weight")
    _crows, cinner = centry["shape"]
    runtime.session.encode(
        "coeff_rows",
        [(work.normed, 0), (coeff, coff), (work.coeff, 0)],
        [_u32(count), _u32(int(cinner))],
        (count, 1, 1),
        (32, 1, 1),
    )
    runtime.session.encode(
        "router8_rows",
        [(work.router, 0), (work.ids, 0), (work.mix, 0)],
        [_u32(count)],
        ((count + 255) // 256, 1, 1),
        (256, 1, 1),
    )
    assignments = count * 8
    if getattr(runtime, "routes", None) is not None:
        runtime.session.encode(
            "copy_u32",
            [(work.ids, 0), (work.route_ids, layer * assignments * 4)],
            [_u32(assignments)],
            ((assignments + 255) // 256, 1, 1),
            (256, 1, 1),
        )
    if getattr(runtime, "expert_order", None) is not None:
        runtime.session.encode(
            "remap_expert_ids",
            [(work.ids, 0), (runtime.expert_order, 0)],
            [_u32(assignments), _u32(layer)],
            ((assignments + 255) // 256, 1, 1),
            (256, 1, 1),
        )


_ABSENT = np.uint32(0xFFFFFFFF)


def masked_expert_passes(ids: np.ndarray, slots: int) -> list[np.ndarray] | None:
    """One masked id list per atlas-sized group, or None when one pass fits.

    An id of 0xFFFFFFFF is absent from that pass. The grouped kernels skip it.
    """
    if slots < 1:
        message = "expert atlas has no images"
        raise ValueError(message)
    seen: list[int] = []
    have: set[int] = set()
    for expert in ids.tolist():
        value = int(expert)
        if value in have:
            continue
        have.add(value)
        seen.append(value)
    if len(seen) <= slots:
        return None
    passes: list[np.ndarray] = []
    for start in range(0, len(seen), slots):
        wave = set(seen[start : start + slots])
        masked = np.array(ids, dtype=np.uint32, copy=True)
        for index, expert in enumerate(masked.tolist()):
            if int(expert) not in wave:
                masked[index] = _ABSENT
        passes.append(masked)
    return passes


def _collect_routes(runtime: Any, work: _Workspace, count: int) -> None:
    """Bin this chunk's router ids. The copy ran before each layer's remap."""
    routes = getattr(runtime, "routes", None)
    if routes is None or count <= 0:
        return
    raw = runtime.session.download(work.route_ids, 40 * count * 8 * 4)
    ids = np.frombuffer(raw, dtype=np.uint32).reshape(40, count, 8)
    for layer in range(40):
        routes.add_prefill(layer, ids[layer])


def _experts(runtime: Any, work: _Workspace, layer: int, count: int, rows: int) -> None:
    """Shared expert, then routed experts. Groups are built on device."""
    prefix = f"model.language_model.layers.{layer}.mlp."
    if runtime.atlas.complete:
        offsets = runtime.expert_offsets if runtime.atlas.layout.mixed else None
        _expert_pass(
            runtime,
            work,
            layer,
            count,
            rows,
            prefix,
            runtime.atlas.layers[layer],
            offsets,
            True,
        )
        return
    runtime.session.end()
    raw = np.frombuffer(
        runtime.session.download(work.ids, count * 8 * 4), dtype=np.uint32
    ).copy()
    passes = masked_expert_passes(raw, runtime.atlas.slots)
    if passes is None:
        file = runtime.routed_layer(layer, work.ids, count * 8, work.offsets)
        _expert_pass(runtime, work, layer, count, rows, prefix, file, work.offsets, True)
        return
    for index, masked in enumerate(passes):
        _write(work.ids, masked.tobytes())
        file = _stage_present(runtime, layer, masked, work.offsets)
        runtime.session.begin()
        _expert_pass(
            runtime,
            work,
            layer,
            count,
            rows,
            prefix,
            file,
            work.offsets,
            index == 0,
        )
        if index + 1 < len(passes):
            runtime.session.end()


def _stage_present(runtime: Any, layer: int, ids: np.ndarray, offsets: Any) -> Any:
    """Load the experts this pass keeps and write their byte offsets."""
    present = [
        int(value)
        for value in dict.fromkeys(int(item) for item in ids.tolist())
        if int(value) < 256
    ]
    table = np.zeros(_EXPERTS, dtype=np.uint64)
    if runtime.atlas.layout.mixed:
        placed = runtime.atlas.stage_offsets(layer, present)
        for file_slot, byte_off in zip(present, placed, strict=True):
            table[file_slot] = np.uint64(byte_off)
    else:
        placed = runtime.atlas.stage(layer, present)
        image = np.uint64(EXPERT_IMAGE)
        for file_slot, atlas_slot in zip(present, placed, strict=True):
            table[file_slot] = np.uint64(atlas_slot) * image
    raw = table.tobytes()
    memoryview(offsets.contents().as_buffer(len(raw)))[:] = raw
    return runtime.atlas.buffer


def _expert_pass(
    runtime: Any,
    work: _Workspace,
    layer: int,
    count: int,
    rows: int,
    prefix: str,
    file: Any,
    offsets: Any,
    shared: bool,
) -> None:
    """One grouped expert pass. ``shared`` runs the shared expert on the first pass."""
    if shared:
        _cast(runtime, work.normed, work.half, rows * HIDDEN)
        _matmul(runtime, work.half, prefix + "shared_expert.gate_up", work.gate_up, rows)
        runtime.session.encode(
            "silu_mul_rows",
            [(work.gate_up, 0), (work.gated, 0)],
            [_u32(rows)],
            (rows * 2, 1, 1),
            (256, 1, 1),
        )
        _cast(runtime, work.gated, work.half, rows * 512)
        _matmul(runtime, work.half, prefix + "shared_expert.down_proj.weight", work.down, rows)
        runtime.session.encode(
            "scatter_add_rows",
            [(work.hidden, 0), (work.down, 0), (work.iota, 0), (work.coeff, 0)],
            [_u32(count), _u32(HIDDEN)],
            (count, 1, 1),
            (256, 1, 1),
        )
    runtime.session.encode(
        "moe_group",
        [
            (work.ids, 0),
            (work.mix, 0),
            (work.csr_i, 0),
            (work.csr_w, 0),
            (work.expert_of, 0),
            (work.slot_n, 0),
            (work.slot_csr, 0),
            (work.row_base, 0),
            (work.jobs, 0),
            (work.group, 0),
            (work.csr_lane, 0),
            (work.pack_row, 0),
        ],
        [_u32(count)],
        (1, 1, 1),
        (256, 1, 1),
    )
    runtime.session.split()
    threads = (256, 1, 1)
    _indirect(
        runtime,
        work.group,
        "zero_f32",
        [(work.expert_pack, 0)],
        [],
        1,
        threads,
    )
    _indirect(
        runtime,
        work.group,
        "gather_half_grouped",
        [
            (work.normed, 0),
            (work.csr_i, 0),
            (work.expert_pack, 0),
            (work.slot_n, 0),
            (work.slot_csr, 0),
            (work.row_base, 0),
            (work.jobs, 0),
            (work.group, 0),
        ],
        [_u32(HIDDEN)],
        0,
        threads,
    )
    _expert_tiles(
        runtime,
        work,
        layer,
        file,
        offsets,
        work.expert_out,
        1024,
        HIDDEN,
        0,
        _GATE_SCALE,
        0,
        FP8_GATE_SCALE,
        2,
    )
    _indirect(
        runtime,
        work.group,
        "silu_mul_grouped",
        [(work.expert_out, 0), (work.expert_gated, 0), (work.group, 0)],
        [],
        3,
        threads,
    )
    _indirect(
        runtime,
        work.group,
        "cast_f32_f16_n",
        [(work.expert_gated, 0), (work.expert_pack, 0), (work.group, 0)],
        [],
        4,
        threads,
    )
    _expert_tiles(
        runtime,
        work,
        layer,
        file,
        offsets,
        work.expert_down,
        HIDDEN,
        512,
        GATE_UP_BYTES,
        GATE_UP_BYTES + _DOWN_SCALE,
        FP8_DOWN_DATA,
        FP8_DOWN_SCALE,
        5,
    )
    runtime.session.encode(
        "scatter_add_grouped",
        [
            (work.hidden, 0),
            (work.expert_down, 0),
            (work.pack_row, 0),
            (work.mix, 0),
        ],
        [_u32(HIDDEN), _u32(count)],
        (count, 1, 1),
        threads,
    )


def _matmul(runtime: Any, half: Any, name: str, dest: Any, rows: int) -> None:
    buf, offset, entry = runtime._loc(name)
    cols, inner = (int(value) for value in entry["stored_shape"])
    kind = "fp8" if entry.get("dtype") == "fp8" else "fp4"
    _quant_rows(
        runtime, kind, half, buf, offset, runtime._scale(entry, offset), dest, rows, cols, inner
    )


def _fp4(
    runtime: Any,
    half: Any,
    buf: Any,
    data_off: int,
    scale_off: int,
    dest: Any,
    rows: int,
    cols: int,
    inner: int,
) -> None:
    _quant_rows(runtime, "fp4", half, buf, data_off, scale_off, dest, rows, cols, inner)


def _quant_rows(
    runtime: Any,
    kind: str,
    half: Any,
    buf: Any,
    data_off: int,
    scale_off: int,
    dest: Any,
    rows: int,
    cols: int,
    inner: int,
) -> None:
    wide = cols % 64 == 0
    runtime.session.encode(
        f"prefill_{kind}_n64" if wide else f"prefill_{kind}_e8m0",
        [(half, 0), (buf, data_off), (buf, scale_off), (dest, 0)],
        [_u32(rows), _u32(cols), _u32(inner)],
        (cols // (64 if wide else 32), rows // 64, 1),
        _tile_threads(runtime),
    )


def _indirect(
    runtime: Any,
    group: Any,
    name: str,
    bindings: list[tuple[Any, int]],
    constants: list[bytes],
    record: int,
    threads: tuple[int, int, int],
) -> None:
    runtime.session.encode_indirect(
        name,
        bindings,
        constants,
        group,
        16 + record * 16,
        threads,
    )


def _expert_tiles(
    runtime: Any,
    work: _Workspace,
    layer: int,
    file: Any,
    offsets: Any,
    dest: Any,
    cols: int,
    inner: int,
    fp4_data: int,
    fp4_scale: int,
    fp8_data: int,
    fp8_scale: int,
    record: int,
) -> None:
    """Launch the fp4 tile, and the fp8 tile when this layer stores an fp8 expert."""
    layout = runtime.atlas.layout
    if not layout.mixed:
        _grouped_tiles(
            runtime, work, file, offsets, dest, cols, inner, fp4_data, fp4_scale, record
        )
        return
    if offsets is None:
        message = "a mixed expert file has no byte offsets"
        raise RuntimeError(message)
    table = getattr(runtime, "expert_offsets", None)
    off_at = layer * _EXPERTS * 8 if offsets is table else 0
    widths = runtime.expert_widths
    width_at = layer * _EXPERTS
    if layout.has(layer, 4):
        _width_tiles(
            runtime,
            work,
            "prefill_fp4_expert_tile_w",
            file,
            offsets,
            off_at,
            widths,
            width_at,
            dest,
            cols,
            inner,
            fp4_data,
            fp4_scale,
            record,
        )
    if layout.has(layer, 8):
        _width_tiles(
            runtime,
            work,
            "prefill_fp8_expert_tile_at",
            file,
            offsets,
            off_at,
            widths,
            width_at,
            dest,
            cols,
            inner,
            fp8_data,
            fp8_scale,
            record,
        )


def _width_tiles(
    runtime: Any,
    work: _Workspace,
    name: str,
    file: Any,
    offsets: Any,
    off_at: int,
    widths: Any,
    width_at: int,
    dest: Any,
    cols: int,
    inner: int,
    image_data: int,
    image_scale: int,
    record: int,
) -> None:
    _indirect(
        runtime,
        work.group,
        name,
        [
            (work.expert_pack, 0),
            (file, 0),
            (dest, 0),
            (work.jobs, 0),
            (work.expert_of, 0),
            (work.row_base, 0),
            (work.group, 0),
            (offsets, off_at),
            (widths, width_at),
        ],
        [_u32(inner), _u32(cols), _u32(image_data), _u32(image_scale)],
        record,
        _tile_threads(runtime),
    )


def _grouped_tiles(
    runtime: Any,
    work: _Workspace,
    file: Any,
    offsets: Any,
    dest: Any,
    cols: int,
    inner: int,
    image_data: int,
    image_scale: int,
    record: int,
) -> None:
    bindings = [
        (work.expert_pack, 0),
        (file, 0),
        (dest, 0),
        (work.jobs, 0),
        (work.expert_of, 0),
        (work.row_base, 0),
        (work.group, 0),
    ]
    name = "prefill_fp4_expert_tile"
    if offsets is not None:
        bindings.append((offsets, 0))
        name = "prefill_fp4_expert_tile_at"
    _indirect(
        runtime,
        work.group,
        name,
        bindings,
        [
            _u32(inner),
            _u32(cols),
            _u32(image_data),
            _u32(image_scale),
            _u32(EXPERT_IMAGE),
        ],
        record,
        _tile_threads(runtime),
    )


def _zero_at(runtime: Any, buf: Any, byte_off: int, n_floats: int) -> None:
    runtime.session.encode(
        "zero_f32",
        [(buf, byte_off)],
        [],
        (n_floats // 256, 1, 1),
        (256, 1, 1),
    )


def _cast_at(
    runtime: Any, src: Any, src_off: int, dst: Any, dst_off: int, n: int
) -> None:
    runtime.session.encode(
        "cast_f32_f16",
        [(src, src_off), (dst, dst_off)],
        [],
        (n // 256, 1, 1),
        (256, 1, 1),
    )


def _tile_threads(runtime: Any) -> tuple[int, int, int]:
    width = int(runtime.session.pipes["prefill_fp4_e8m0"].threadExecutionWidth())
    return (width * 4, 1, 1)


def _cast(runtime: Any, src: Any, dst: Any, n: int) -> None:
    runtime.session.encode(
        "cast_f32_f16",
        [(src, 0), (dst, 0)],
        [],
        (n // 256, 1, 1),
        (256, 1, 1),
    )


def _zero(runtime: Any, buf: Any, n: int) -> None:
    runtime.session.encode(
        "zero_f32",
        [(buf, 0)],
        [],
        (n // 256, 1, 1),
        (256, 1, 1),
    )


def _rms(runtime: Any, src: Any, name: str, dest: Any, rows: int) -> None:
    buf, offset, _entry = runtime._loc(name)
    runtime.session.encode(
        "rmsnorm_rows",
        [(src, 0), (buf, offset), (dest, 0)],
        [_u32(HIDDEN), _f32(EPS), _u32(rows)],
        (rows, 1, 1),
        (256, 1, 1),
    )


def _add(runtime: Any, work: _Workspace, count: int) -> None:
    runtime.session.encode(
        "residual_rows",
        [(work.hidden, 0), (work.down, 0), (work.hidden, 0)],
        [_u32(count * HIDDEN)],
        (count * 8, 1, 1),
        (256, 1, 1),
    )


def _input_norm(layer: int) -> str:
    return f"model.language_model.layers.{layer}.input_layernorm.weight"


def _post_norm(layer: int) -> str:
    return f"model.language_model.layers.{layer}.post_attention_layernorm.weight"
