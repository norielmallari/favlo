"""Chunked prompt prefill. Projections are ``matmul2d`` tiles of 64 rows.

The down projection uses a 32×128 tile. Gate, up, and the attention projections stay 64×64.

SiLU runs on the gate, then multiplies by the up projection, then the down
projection. A promoted feed-forward matrix is fp8. The others are fp4.
Gated DeltaNet and full attention run in the same command buffer as that
layer's feed-forward. Decode stays a GEMV.
"""

from __future__ import annotations

import struct
from typing import Any

import numpy as np

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
    MTP_SLOT,
    Q_HEADS,
    Q_PROJ,
    THETA,
    Z_WIDTH,
)


def workspace_bytes(*, drafts: bool = False) -> int:
    """Bytes of the chunk workspace, reserved before the cache is sized."""
    cap = CHUNK
    total = (
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
    if drafts:
        total += cap * MIX * 4
    return total


class _Workspace:
    def __init__(self, session: Any, *, drafts: bool = False) -> None:
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
        self.mix = zeros(cap * MIX * 4) if drafts else None


def prefill_ids(
    runtime: Any,
    ids: list[int],
    pos0: int = 0,
    *,
    follow: int | None = None,
    mtp: bool = True,
    progress: bool = True,
) -> None:
    """Advance state and the key/value cache through ``ids``.

    ``follow`` is the token after the last id. The MTP layer at each position
    reads that position's residual and the embedding of the next token.
    """
    if not ids:
        return
    if pos0 + len(ids) > runtime.max_pos:
        message = f"position {pos0 + len(ids)} is past {runtime.max_pos}"
        raise RuntimeError(message)
    workspace = getattr(runtime, "_prefill", None)
    if workspace is None:
        workspace = _Workspace(runtime.session, drafts=bool(getattr(runtime, "drafts", False)))
        runtime._prefill = workspace
    cursor = pos0
    remaining = ids
    while remaining:
        check = getattr(runtime, "_check_stop", None)
        if check is not None:
            check()
        take = min(CHUNK, len(remaining))
        nxt = remaining[take] if len(remaining) > take else follow
        _chunk(runtime, workspace, remaining[:take], cursor, nxt, mtp)
        cursor += take
        remaining = remaining[take:]
        note = getattr(runtime, "_note_prefill", None)
        if progress and note is not None:
            note(cursor - pos0)


def _chunk(
    runtime: Any,
    work: _Workspace,
    ids: list[int],
    pos0: int,
    follow: int | None = None,
    mtp: bool = True,
) -> None:
    count = len(ids)
    rows = (count + 63) // 64 * 64
    _write(work.tokens, _pack_u32(ids))
    runtime.session.begin()
    try:
        _zero(runtime, work.hidden, rows * HIDDEN)
        _embed(runtime, work, count)
        from qwen38_27b.tower import scatter

        scatter(runtime, work.hidden, pos0, count)
        for layer, kind in enumerate(runtime.layer_types):
            if layer > 0 and layer % 4 == 0:
                runtime.session.submit()
            _rms(runtime, work.hidden, runtime.weight(layer, "input_layernorm.weight"), work.normed, count)
            if kind == "full_attention":
                _attend(runtime, work, layer, count, rows, pos0)
            else:
                _gdn(runtime, work, layer, count, rows)
            _rms(
                runtime,
                work.hidden,
                runtime.weight(layer, "post_attention_layernorm.weight"),
                work.normed,
                count,
            )
            _ffn(runtime, work, layer, count, rows)
        if mtp and getattr(runtime, "drafts", False):
            _mtp_rows(runtime, work, ids, pos0, follow, rows)
        runtime.session.end()
    except Exception:
        runtime.session.finish()
        raise


def _mtp_rows(
    runtime: Any,
    work: _Workspace,
    ids: list[int],
    pos0: int,
    follow: int | None,
    rows: int,
) -> None:
    """One MTP layer. Each row mixes that residual with the next token's embedding."""
    shifted = [int(token) for token in ids[1:]]
    if follow is not None:
        shifted.append(int(follow))
    count = len(shifted)
    if count < 1 or work.mix is None:
        return
    _write(work.tokens, _pack_u32(shifted))
    _zero(runtime, work.mix, rows * MIX)
    _embed(runtime, work, count, work.big0)
    _rms(runtime, work.hidden, "mtp.pre_fc_norm_hidden.weight", work.normed, count)
    _rms(runtime, work.big0, "mtp.pre_fc_norm_embedding.weight", work.big1, count)
    runtime.session.encode(
        "concat_pair",
        [(work.big1, 0), (work.normed, 0), (work.mix, 0)],
        [_u32(HIDDEN), _u32(count)],
        (count, 1, 1),
        (256, 1, 1),
    )
    _cast_half(runtime, work, work.mix, rows * MIX, rows, MIX)
    _matmul(runtime, work.half, "mtp.fc.weight", work.hidden, rows)
    _rms(runtime, work.hidden, "mtp.layers.0.input_layernorm.weight", work.normed, count)
    _attend(runtime, work, None, count, rows, pos0, prefix="mtp.layers.0", slot=MTP_SLOT)
    _rms(runtime, work.hidden, "mtp.layers.0.post_attention_layernorm.weight", work.normed, count)
    _ffn(runtime, work, None, count, rows, prefix="mtp.layers.0")


def _embed(runtime: Any, work: _Workspace, count: int, dest: Any = None) -> None:
    buf, offset, entry = runtime.locate("model.language_model.embed_tokens.weight", "model.embed_tokens.weight")
    _rows, inner = entry["stored_shape"]
    runtime.session.encode(
        "embed_fp4_rows",
        [
            (buf, offset),
            (buf, runtime.scale_offset(entry, offset)),
            (work.tokens, 0),
            (work.hidden if dest is None else dest, 0),
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
    runtime: Any,
    work: _Workspace,
    layer: int | None,
    count: int,
    rows: int,
    pos0: int,
    *,
    prefix: str | None = None,
    slot: int | None = None,
) -> None:
    def weight(rest: str) -> str:
        if prefix is None:
            return runtime.weight(layer, rest)
        return f"{prefix}.{rest}"

    if slot is None:
        slot = runtime.full_slot[layer]
    _cast_half(runtime, work, work.normed, count * HIDDEN, rows, HIDDEN)
    _matmul(runtime, work.half, weight("self_attn.q_proj.weight"), work.big0, rows)
    _matmul(runtime, work.half, weight("self_attn.k_proj.weight"), work.kv, rows)
    _matmul(runtime, work.half, weight("self_attn.v_proj.weight"), work.vv, rows)
    runtime.session.encode(
        "split_q_gate_rows",
        [(work.big0, 0), (work.big2, 0), (work.gate, 0)],
        [_u32(count)],
        (count * Q_HEADS, 1, 1),
        (256, 1, 1),
    )
    q_buf, q_off, _entry = runtime.locate(weight("self_attn.q_norm.weight"))
    k_buf, k_off, _entry = runtime.locate(weight("self_attn.k_norm.weight"))
    rope = getattr(runtime, "_rope_ids", None)
    if rope is None:
        runtime.session.encode(
            "qk_norm_rope_rows",
            [(work.big2, 0), (work.kv, 0), (q_buf, q_off), (k_buf, k_off)],
            [_f32(THETA), _f32(EPS), _u32(pos0)],
            (count, Q_HEADS + 4, 1),
            (256, 1, 1),
        )
    else:
        runtime.session.encode(
            "qk_norm_mrope_rows",
            [
                (work.big2, 0),
                (work.kv, 0),
                (q_buf, q_off),
                (k_buf, k_off),
                (runtime.session.upload(np.ascontiguousarray(rope[pos0 : pos0 + count])), 0),
            ],
            [_f32(THETA), _f32(EPS)],
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
        slot=slot,
    )
    _cast_half(runtime, work, work.big0, count * ATTN_OUT, rows, ATTN_OUT)
    _matmul(runtime, work.half, weight("self_attn.o_proj.weight"), work.big1, rows)
    _add(runtime, work, count, work.big1)


def _ffn(
    runtime: Any,
    work: _Workspace,
    layer: int | None,
    count: int,
    rows: int,
    *,
    prefix: str | None = None,
) -> None:
    def weight(rest: str) -> str:
        if prefix is None:
            return runtime.weight(layer, rest)
        return f"{prefix}.{rest}"

    if layer is not None:
        runtime.prepare_ffn(layer)
    _cast_half(runtime, work, work.normed, count * HIDDEN, rows, HIDDEN)
    _matmul(runtime, work.half, weight("mlp.gate_proj.weight"), work.big0, rows)
    _matmul(runtime, work.half, weight("mlp.up_proj.weight"), work.big1, rows)
    runtime.session.encode(
        "swiglu_f32",
        [(work.big0, 0), (work.big1, 0), (work.big2, 0)],
        [],
        (rows * INTERMEDIATE // 256, 1, 1),
        (256, 1, 1),
    )
    _cast_half(runtime, work, work.big2, rows * INTERMEDIATE, rows, INTERMEDIATE)
    _matmul(runtime, work.half, weight("mlp.down_proj.weight"), work.big0, rows)
    _add(runtime, work, count, work.big0)


def _matmul(runtime: Any, half: Any, name: str, dest: Any, rows: int) -> None:
    buf, offset, entry = runtime.locate(name)
    cols, inner = (int(value) for value in entry["stored_shape"])
    kind = str(entry.get("dtype", "fp4"))
    if kind not in {"fp4", "fp8"} or cols % 64 or inner % 32 or rows % 64:
        message = f"{name} tile {rows}x{cols}x{inner} is not a 64-row {kind} tile"
        raise RuntimeError(message)
    wide = name.endswith("mlp.down_proj.weight") and cols % 128 == 0 and rows % 32 == 0
    if wide:
        kernel = "prefill_fp8_m32" if kind == "fp8" else "prefill_fp4_m32"
        groups = (cols // 128, rows // 32, 1)
    else:
        kernel = "prefill_fp8_n64" if kind == "fp8" else "prefill_fp4_n64"
        groups = (cols // 64, rows // 64, 1)
    width = int(runtime.session.pipes[kernel].threadExecutionWidth())
    runtime.session.encode(
        kernel,
        [
            (half, 0),
            (buf, offset),
            (buf, runtime.scale_offset(entry, offset)),
            (dest, 0),
        ],
        [_u32(rows), _u32(cols), _u32(inner)],
        groups,
        (width * 4, 1, 1),
    )


def _cast_half(runtime: Any, work: _Workspace, src: Any, n: int, rows: int, inner: int) -> None:
    span = rows * inner
    if span % 2 or n % 2:
        message = "half tile is an odd number of elements"
        raise RuntimeError(message)
    if n < span:
        # The cast writes the live rows. Only the padding rows still need a zero.
        _zero(runtime, work.half, (span - n) // 2, offset=n * 2)
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


def _zero(runtime: Any, buf: Any, n: int, offset: int = 0) -> None:
    if n == 0:
        return
    if n % 256:
        message = f"zero length {n} is not a multiple of 256"
        raise RuntimeError(message)
    runtime.session.encode("zero_f32", [(buf, offset)], [], (n // 256, 1, 1), (256, 1, 1))


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
