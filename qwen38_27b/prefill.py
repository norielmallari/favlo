"""Chunked prompt prefill. Projections are fp4 ``matmul2d`` tiles of 64 rows.

The feed-forward is uniform fp4: SiLU on the gate, multiply by the up
projection, then the down projection. Gated DeltaNet and full attention run
in the same command buffer as that layer's feed-forward. Decode stays a GEMV.
"""

from __future__ import annotations

import struct
from typing import Any

from qwen38_27b.attn import attend_plane
from qwen38_27b.layout import (
    AB,
    ATTN_OUT,
    CHUNK,
    EPS,
    HIDDEN,
    INTERMEDIATE,
    KV_PROJ,
    MIX,
    Q_HEADS,
    Q_PROJ,
    THETA,
    Z_WIDTH,
)


def workspace_bytes() -> int:
    """Bytes of the chunk workspace, reserved before the cache is sized."""
    cap = CHUNK
    return (
        cap * HIDDEN * 4
        + cap * HIDDEN * 4
        + cap * INTERMEDIATE * 2
        + cap * INTERMEDIATE * 4
        + cap * INTERMEDIATE * 4
        + cap * INTERMEDIATE * 4
        + cap * ATTN_OUT * 4
        + cap * KV_PROJ * 4
        + cap * KV_PROJ * 4
        + cap * AB * 4
        + cap * AB * 4
        + cap * 4
    )


class _Workspace:
    def __init__(self, session: Any) -> None:
        cap = CHUNK
        zeros = session.zeros
        self.hidden = zeros(cap * HIDDEN * 4)
        self.normed = zeros(cap * HIDDEN * 4)
        self.half = zeros(cap * INTERMEDIATE * 2)
        self.big0 = zeros(cap * INTERMEDIATE * 4)
        self.big1 = zeros(cap * INTERMEDIATE * 4)
        self.big2 = zeros(cap * INTERMEDIATE * 4)
        self.gate = zeros(cap * ATTN_OUT * 4)
        self.kv = zeros(cap * KV_PROJ * 4)
        self.vv = zeros(cap * KV_PROJ * 4)
        self.ab = zeros(cap * AB * 4)
        self.ab2 = zeros(cap * AB * 4)
        self.tokens = zeros(cap * 4)


def prefill_ids(runtime: Any, ids: list[int], pos0: int = 0) -> None:
    """Advance state and the key/value cache through ``ids``."""
    if not ids:
        return
    if pos0 + len(ids) > runtime.max_pos:
        message = f"position {pos0 + len(ids)} is past {runtime.max_pos}"
        raise RuntimeError(message)
    workspace = getattr(runtime, "_prefill", None)
    if workspace is None:
        workspace = _Workspace(runtime.session)
        runtime._prefill = workspace
    cursor = pos0
    remaining = ids
    while remaining:
        check = getattr(runtime, "_check_stop", None)
        if check is not None:
            check()
        take = min(CHUNK, len(remaining))
        _chunk(runtime, workspace, remaining[:take], cursor)
        cursor += take
        remaining = remaining[take:]
        note = getattr(runtime, "_note_prefill", None)
        if note is not None:
            note(cursor - pos0)


def _chunk(runtime: Any, work: _Workspace, ids: list[int], pos0: int) -> None:
    count = len(ids)
    rows = (count + 63) // 64 * 64
    _write(work.tokens, _pack_u32(ids))
    runtime.session.begin()
    _zero(runtime, work.hidden, rows * HIDDEN)
    _embed(runtime, work, count)
    for layer, kind in enumerate(runtime.layer_types):
        _zero(runtime, work.normed, rows * HIDDEN)
        _rms(runtime, work.hidden, runtime.weight(layer, "input_layernorm.weight"), work.normed, count)
        if kind == "full_attention":
            _attend(runtime, work, layer, count, rows, pos0)
        else:
            _gdn(runtime, work, layer, count, rows)
        _zero(runtime, work.normed, rows * HIDDEN)
        _rms(
            runtime,
            work.hidden,
            runtime.weight(layer, "post_attention_layernorm.weight"),
            work.normed,
            count,
        )
        _ffn(runtime, work, layer, count, rows)
    runtime.session.end()


def _embed(runtime: Any, work: _Workspace, count: int) -> None:
    buf, offset, entry = runtime.locate("model.language_model.embed_tokens.weight", "model.embed_tokens.weight")
    _rows, inner = entry["stored_shape"]
    runtime.session.encode(
        "embed_fp4_rows",
        [
            (buf, offset),
            (buf, runtime.scale_offset(entry, offset)),
            (work.tokens, 0),
            (work.hidden, 0),
        ],
        [_u32(count), _u32(int(inner))],
        (count, 1, 1),
        (32, 1, 1),
    )


def _gdn(runtime: Any, work: _Workspace, layer: int, count: int, rows: int) -> None:
    _cast_half(runtime, work, work.normed, count * HIDDEN, rows, HIDDEN)
    _matmul(runtime, work.half, runtime.weight(layer, "linear_attn.in_proj_qkv.weight"), work.big0, rows)
    _matmul(runtime, work.half, runtime.weight(layer, "linear_attn.in_proj_z.weight"), work.big1, rows)
    _matmul(runtime, work.half, runtime.weight(layer, "linear_attn.in_proj_a.weight"), work.ab, rows)
    _matmul(runtime, work.half, runtime.weight(layer, "linear_attn.in_proj_b.weight"), work.ab2, rows)
    slot = runtime.linear_slot[layer]
    hist = slot * 3 * MIX * 2
    weight, woff, _entry = runtime.locate_weight(layer, "linear_attn.conv1d.weight")
    runtime.session.encode(
        "gdn_conv_rows",
        [(runtime.history, hist), (work.big0, 0), (weight, woff), (work.big0, 0)],
        [_u32(count), _u32(MIX)],
        (MIX // 256, 1, 1),
        (256, 1, 1),
    )
    along, aoff, _entry = runtime.locate_weight(layer, "linear_attn.A_log")
    bias, boff, _entry = runtime.locate_weight(layer, "linear_attn.dt_bias")
    norm, noff, _entry = runtime.locate_weight(layer, "linear_attn.norm.weight")
    state = slot * 48 * 128 * 128 * 4
    runtime.session.encode(
        "gdn_state_rows",
        [
            (runtime.state, state),
            (work.big0, 0),
            (work.ab, 0),
            (work.ab2, 0),
            (work.big1, 0),
            (along, aoff),
            (bias, boff),
            (norm, noff),
            (work.big2, 0),
        ],
        [_f32(EPS), _u32(count)],
        (48, 1, 1),
        (128, 1, 1),
    )
    _cast_half(runtime, work, work.big2, count * Z_WIDTH, rows, Z_WIDTH)
    _matmul(runtime, work.half, runtime.weight(layer, "linear_attn.out_proj.weight"), work.big0, rows)
    _add(runtime, work, count, work.big0)


def _attend(
    runtime: Any, work: _Workspace, layer: int, count: int, rows: int, pos0: int
) -> None:
    _cast_half(runtime, work, work.normed, count * HIDDEN, rows, HIDDEN)
    _matmul(runtime, work.half, runtime.weight(layer, "self_attn.q_proj.weight"), work.big0, rows)
    _matmul(runtime, work.half, runtime.weight(layer, "self_attn.k_proj.weight"), work.kv, rows)
    _matmul(runtime, work.half, runtime.weight(layer, "self_attn.v_proj.weight"), work.vv, rows)
    runtime.session.encode(
        "split_q_gate_rows",
        [(work.big0, 0), (work.big2, 0), (work.gate, 0)],
        [_u32(count)],
        (count * Q_HEADS, 1, 1),
        (256, 1, 1),
    )
    q_buf, q_off, _entry = runtime.locate_weight(layer, "self_attn.q_norm.weight")
    k_buf, k_off, _entry = runtime.locate_weight(layer, "self_attn.k_norm.weight")
    runtime.session.encode(
        "qk_norm_rope_rows",
        [(work.big2, 0), (work.kv, 0), (q_buf, q_off), (k_buf, k_off)],
        [_f32(THETA), _f32(EPS), _u32(pos0)],
        (count, Q_HEADS + 4, 1),
        (256, 1, 1),
    )
    attend_plane(
        runtime,
        query=work.big2,
        key=work.kv,
        value=work.vv,
        gate=work.gate,
        dest=work.big0,
        pos0=pos0,
        count=count,
        slot=runtime.full_slot[layer],
    )
    _cast_half(runtime, work, work.big0, count * ATTN_OUT, rows, ATTN_OUT)
    _matmul(runtime, work.half, runtime.weight(layer, "self_attn.o_proj.weight"), work.big1, rows)
    _add(runtime, work, count, work.big1)


def _ffn(runtime: Any, work: _Workspace, layer: int, count: int, rows: int) -> None:
    _cast_half(runtime, work, work.normed, count * HIDDEN, rows, HIDDEN)
    _matmul(runtime, work.half, runtime.weight(layer, "mlp.gate_proj.weight"), work.big0, rows)
    _matmul(runtime, work.half, runtime.weight(layer, "mlp.up_proj.weight"), work.big1, rows)
    runtime.session.encode(
        "swiglu_f32",
        [(work.big0, 0), (work.big1, 0), (work.big2, 0)],
        [],
        (rows * INTERMEDIATE // 256, 1, 1),
        (256, 1, 1),
    )
    _cast_half(runtime, work, work.big2, rows * INTERMEDIATE, rows, INTERMEDIATE)
    _matmul(runtime, work.half, runtime.weight(layer, "mlp.down_proj.weight"), work.big0, rows)
    _add(runtime, work, count, work.big0)


def _matmul(runtime: Any, half: Any, name: str, dest: Any, rows: int) -> None:
    buf, offset, entry = runtime.locate(name)
    cols, inner = (int(value) for value in entry["stored_shape"])
    if cols % 64 or inner % 32 or rows % 64:
        message = f"{name} tile {rows}x{cols}x{inner} is not a 64-row fp4 tile"
        raise RuntimeError(message)
    width = int(runtime.session.pipes["prefill_fp4_n64"].threadExecutionWidth())
    runtime.session.encode(
        "prefill_fp4_n64",
        [
            (half, 0),
            (buf, offset),
            (buf, runtime.scale_offset(entry, offset)),
            (dest, 0),
        ],
        [_u32(rows), _u32(cols), _u32(inner)],
        (cols // 64, rows // 64, 1),
        (width * 4, 1, 1),
    )


def _cast_half(runtime: Any, work: _Workspace, src: Any, n: int, rows: int, inner: int) -> None:
    span = rows * inner
    if span % 2:
        message = "half tile is an odd number of elements"
        raise RuntimeError(message)
    nfloat = span // 2
    runtime.session.encode(
        "zero_f32",
        [(work.half, 0)],
        [],
        (nfloat // 256, 1, 1),
        (256, 1, 1),
    )
    if n % 256:
        message = f"cast length {n} is not a multiple of 256"
        raise RuntimeError(message)
    runtime.session.encode(
        "cast_f32_f16",
        [(src, 0), (work.half, 0)],
        [],
        (n // 256, 1, 1),
        (256, 1, 1),
    )


def _zero(runtime: Any, buf: Any, n: int) -> None:
    if n % 256:
        message = f"zero length {n} is not a multiple of 256"
        raise RuntimeError(message)
    runtime.session.encode("zero_f32", [(buf, 0)], [], (n // 256, 1, 1), (256, 1, 1))


def _rms(runtime: Any, src: Any, name: str, dest: Any, rows: int) -> None:
    buf, offset, _entry = runtime.locate(name)
    runtime.session.encode(
        "rmsnorm_rows",
        [(src, 0), (buf, offset), (dest, 0)],
        [_u32(HIDDEN), _f32(EPS), _u32(rows)],
        (rows, 1, 1),
        (256, 1, 1),
    )


def _add(runtime: Any, work: _Workspace, count: int, src: Any) -> None:
    n = count * HIDDEN
    runtime.session.encode(
        "residual_rows",
        [(work.hidden, 0), (src, 0), (work.hidden, 0)],
        [_u32(n)],
        (n // 256, 1, 1),
        (256, 1, 1),
    )


def _write(buffer: Any, blob: bytes) -> None:
    if blob:
        memoryview(buffer.contents().as_buffer(len(blob)))[: len(blob)] = blob


def _pack_u32(values: list[int]) -> bytes:
    return b"".join(struct.pack("<I", int(value)) for value in values)


def _u32(value: int) -> bytes:
    return struct.pack("<I", value)


def _f32(value: float) -> bytes:
    return struct.pack("<f", value)
