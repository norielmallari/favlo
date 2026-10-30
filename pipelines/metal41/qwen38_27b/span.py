"""Two-token language step. Each fp4 weight row is read once.

The second residual is already known at the start of a layer, so the projections
share one read. The Gated DeltaNet recurrence and the attention mix still run
in token order. A third draft token stays on the serial step.
"""

from __future__ import annotations

import struct
from typing import Any

from qwen38_27b.layout import (
    AB,
    ATTN_OUT,
    EPS,
    HIDDEN,
    INTERMEDIATE,
    KV_PROJ,
    LINEAR_V,
    MIX,
    Q_HEADS,
    Q_PROJ,
    THETA,
    Z_WIDTH,
)


class _Span:
    def __init__(self, session: Any) -> None:
        zeros = session.zeros
        self.hidden = zeros(2 * HIDDEN * 4)
        self.normed = zeros(2 * HIDDEN * 4)
        self.qkv = zeros(2 * MIX * 4)
        self.z = zeros(2 * max(Z_WIDTH, INTERMEDIATE) * 4)
        self.a = zeros(2 * AB * 4)
        self.b = zeros(2 * AB * 4)
        self.y = zeros(2 * max(Z_WIDTH, ATTN_OUT, INTERMEDIATE) * 4)
        self.query = zeros(2 * Q_PROJ * 4)
        self.q = zeros(2 * ATTN_OUT * 4)
        self.kv = zeros(2 * KV_PROJ * 4)
        self.vv = zeros(2 * KV_PROJ * 4)
        self.gate = zeros(2 * ATTN_OUT * 4)
        self.out = zeros(2 * HIDDEN * 4)


def span_pair(runtime: Any, token0: int, token1: int, pos: int) -> None:
    """Advance two tokens from ``pos``. Both residuals stay in ``runtime._span``."""
    span = _buffers(runtime)
    runtime.session.begin()
    try:
        runtime._embed(int(token0))
        _copy(runtime, runtime.hidden, 0, span.hidden, 0, HIDDEN)
        runtime._embed(int(token1))
        _copy(runtime, runtime.hidden, 0, span.hidden, HIDDEN * 4, HIDDEN)
        for layer, kind in enumerate(runtime.layer_types):
            if layer > 0 and layer % 4 == 0:
                runtime.session.submit()
            _rms(runtime, span.hidden, runtime.weight(layer, "input_layernorm.weight"), span.normed)
            if kind == "full_attention":
                _attend(runtime, span, layer, pos)
            else:
                _gdn(runtime, span, layer)
            _rms(runtime, span.hidden, runtime.weight(layer, "post_attention_layernorm.weight"), span.normed)
            _ffn(runtime, span, layer)
        runtime.session.end()
    except Exception:
        runtime.session.finish()
        raise
    runtime.filled = int(pos) + 2


def load_row(runtime: Any, index: int) -> None:
    """Copy one span residual into ``runtime.hidden``."""
    span = _buffers(runtime)
    runtime.session.begin()
    try:
        _copy(runtime, span.hidden, int(index) * HIDDEN * 4, runtime.hidden, 0, HIDDEN)
        runtime.session.end()
    except Exception:
        runtime.session.finish()
        raise


def _buffers(runtime: Any) -> _Span:
    span = getattr(runtime, "_span", None)
    if span is None:
        span = _Span(runtime.session)
        runtime._span = span
    return span


def _gdn(runtime: Any, span: _Span, layer: int) -> None:
    qkv_w, qkv_s = runtime._pair(runtime.weight(layer, "linear_attn.in_proj_qkv.weight"))
    z_w, z_s = runtime._pair(runtime.weight(layer, "linear_attn.in_proj_z.weight"))
    a_w, a_s = runtime._pair(runtime.weight(layer, "linear_attn.in_proj_a.weight"))
    b_w, b_s = runtime._pair(runtime.weight(layer, "linear_attn.in_proj_b.weight"))
    runtime.session.encode(
        "gdn_in_proj_span2",
        [
            (span.normed, 0),
            (span.normed, HIDDEN * 4),
            qkv_w,
            qkv_s,
            z_w,
            z_s,
            a_w,
            a_s,
            b_w,
            b_s,
            (span.qkv, 0),
            (span.z, 0),
            (span.a, 0),
            (span.b, 0),
        ],
        [],
        (MIX + Z_WIDTH + AB + AB, 1, 1),
        (32, 1, 1),
    )
    slot = runtime.linear_slot[layer]
    weight, woff, _entry = runtime.locate_weight(layer, "linear_attn.conv1d.weight")
    runtime.session.encode(
        "gdn_conv_rows",
        [(runtime.history, slot * 3 * MIX * 2), (span.qkv, 0), (weight, woff), (span.qkv, 0)],
        [_u32(2), _u32(MIX)],
        (MIX // 256, 1, 1),
        (256, 1, 1),
    )
    along, aoff, _entry = runtime.locate_weight(layer, "linear_attn.A_log")
    bias, boff, _entry = runtime.locate_weight(layer, "linear_attn.dt_bias")
    norm, noff, _entry = runtime.locate_weight(layer, "linear_attn.norm.weight")
    runtime.session.encode(
        "gdn_state_rows",
        [
            (runtime.state, slot * LINEAR_V * 128 * 128 * 4),
            (span.qkv, 0),
            (span.a, 0),
            (span.b, 0),
            (span.z, 0),
            (along, aoff),
            (bias, boff),
            (norm, noff),
            (span.y, 0),
        ],
        [_f32(EPS), _u32(2)],
        (LINEAR_V, 1, 1),
        (128, 1, 1),
    )
    _project(runtime, span.y, Z_WIDTH, runtime.weight(layer, "linear_attn.out_proj.weight"), span.out)
    _residual(runtime, span)


def _attend(runtime: Any, span: _Span, layer: int, pos: int) -> None:
    from qwen38_27b.attn import attend_plane

    q_name = runtime.weight(layer, "self_attn.q_proj.weight")
    k_name = runtime.weight(layer, "self_attn.k_proj.weight")
    v_name = runtime.weight(layer, "self_attn.v_proj.weight")
    if runtime._dtype(q_name) == "fp4":
        q_w, q_s = runtime._pair(q_name)
        k_w, k_s = runtime._pair(k_name)
        v_w, v_s = runtime._pair(v_name)
        runtime.session.encode(
            "attn_qkv_span2",
            [
                (span.normed, 0),
                (span.normed, HIDDEN * 4),
                q_w,
                q_s,
                k_w,
                k_s,
                v_w,
                v_s,
                (span.query, 0),
                (span.kv, 0),
                (span.vv, 0),
            ],
            [],
            (Q_PROJ + KV_PROJ + KV_PROJ, 1, 1),
            (32, 1, 1),
        )
    else:
        _gemv_rows(runtime, span.normed, HIDDEN, q_name, span.query, Q_PROJ)
        _gemv_rows(runtime, span.normed, HIDDEN, k_name, span.kv, KV_PROJ)
        _gemv_rows(runtime, span.normed, HIDDEN, v_name, span.vv, KV_PROJ)
    runtime.session.encode(
        "split_q_gate_rows",
        [(span.query, 0), (span.q, 0), (span.gate, 0)],
        [_u32(2)],
        (2 * Q_HEADS, 1, 1),
        (256, 1, 1),
    )
    q_buf, q_off, _entry = runtime.locate(runtime.weight(layer, "self_attn.q_norm.weight"))
    k_buf, k_off, _entry = runtime.locate(runtime.weight(layer, "self_attn.k_norm.weight"))
    rope = getattr(runtime, "_rope_ids", None)
    slot = runtime.full_slot[layer]
    for token in (0, 1):
        if rope is None:
            angles = (pos + token, pos + token, pos + token)
        else:
            row = rope[pos + token]
            angles = (int(row[0]), int(row[1]), int(row[2]))
        runtime.session.encode(
            "qk_norm_rope",
            [
                (span.q, token * ATTN_OUT * 4),
                (span.kv, token * KV_PROJ * 4),
                (q_buf, q_off),
                (k_buf, k_off),
            ],
            [_f32(THETA), _f32(EPS), _u32(angles[0]), _u32(angles[1]), _u32(angles[2])],
            (Q_HEADS + 4, 1, 1),
            (256, 1, 1),
        )
    attend_plane(
        runtime,
        query=span.q,
        key=span.kv,
        value=span.vv,
        gate=span.gate,
        dest=span.y,
        pos0=pos,
        count=1,
        slot=slot,
    )
    _copy(runtime, span.q, ATTN_OUT * 4, span.q, 0, ATTN_OUT)
    _copy(runtime, span.kv, KV_PROJ * 4, span.kv, 0, KV_PROJ)
    _copy(runtime, span.vv, KV_PROJ * 4, span.vv, 0, KV_PROJ)
    _copy(runtime, span.gate, ATTN_OUT * 4, span.gate, 0, ATTN_OUT)
    attend_plane(
        runtime,
        query=span.q,
        key=span.kv,
        value=span.vv,
        gate=span.gate,
        dest=span.out,
        pos0=pos + 1,
        count=1,
        slot=slot,
    )
    _copy(runtime, span.out, 0, span.y, ATTN_OUT * 4, ATTN_OUT)
    _project(runtime, span.y, ATTN_OUT, runtime.weight(layer, "self_attn.o_proj.weight"), span.out)
    _residual(runtime, span)


def _ffn(runtime: Any, span: _Span, layer: int) -> None:
    runtime.prepare_ffn(layer)
    gate = runtime.weight(layer, "mlp.gate_proj.weight")
    up = runtime.weight(layer, "mlp.up_proj.weight")
    down = runtime.weight(layer, "mlp.down_proj.weight")
    if runtime._dtype(gate) == "fp4" and runtime._dtype(up) == "fp4":
        gate_w, gate_s = runtime._pair(gate)
        up_w, up_s = runtime._pair(up)
        runtime.session.encode(
            "ffn_gate_up_span2",
            [
                (span.normed, 0),
                (span.normed, HIDDEN * 4),
                gate_w,
                gate_s,
                up_w,
                up_s,
                (span.y, 0),
                (span.z, 0),
            ],
            [_u32(INTERMEDIATE), _u32(HIDDEN)],
            (INTERMEDIATE // 4 * 2, 1, 1),
            (32, 1, 1),
        )
    else:
        _gemv_rows(runtime, span.normed, HIDDEN, gate, span.y, INTERMEDIATE)
        _gemv_rows(runtime, span.normed, HIDDEN, up, span.z, INTERMEDIATE)
    for token in (0, 1):
        runtime.session.encode(
            "swiglu_f32",
            [
                (span.y, token * INTERMEDIATE * 4),
                (span.z, token * INTERMEDIATE * 4),
                (span.y, token * INTERMEDIATE * 4),
            ],
            [],
            (INTERMEDIATE // 256, 1, 1),
            (256, 1, 1),
        )
    _project(runtime, span.y, INTERMEDIATE, down, span.out)
    for token in (0, 1):
        runtime.session.encode(
            "residual_rows",
            [
                (span.hidden, token * HIDDEN * 4),
                (span.out, token * HIDDEN * 4),
                (span.hidden, token * HIDDEN * 4),
            ],
            [_u32(HIDDEN)],
            (HIDDEN // 256, 1, 1),
            (256, 1, 1),
        )


def _project(runtime: Any, src: Any, width: int, name: str, dest: Any) -> None:
    buf, offset, entry = runtime.locate(name)
    rows, inner = (int(value) for value in entry["stored_shape"])
    if int(width) != int(inner):
        message = f"{name} span width {width} is not inner {inner}"
        raise RuntimeError(message)
    if runtime._dtype(name) != "fp4":
        _gemv_rows(runtime, src, inner, name, dest, rows)
        return
    scale = runtime.scale_offset(entry, offset)
    runtime.session.encode(
        "gemv_fp4_span2",
        [
            (src, 0),
            (src, inner * 4),
            (buf, offset),
            (buf, scale),
            (dest, 0),
            (dest, rows * 4),
        ],
        [_u32(rows), _u32(inner)],
        (rows // 4, 1, 1),
        (32, 1, 1),
    )


def _gemv_rows(
    runtime: Any, src: Any, src_width: int, name: str, dest: Any, dest_width: int
) -> None:
    for token in (0, 1):
        _gemv_at(
            runtime,
            src,
            token * src_width * 4,
            name,
            dest,
            token * dest_width * 4,
        )


def _gemv_at(runtime: Any, act: Any, act_off: int, name: str, dest: Any, dest_off: int) -> None:
    buf, offset, entry = runtime.locate(name)
    rows, inner = (int(value) for value in entry["stored_shape"])
    scale = runtime.scale_offset(entry, offset)
    kind = str(entry.get("dtype", "fp4"))
    bindings = [(act, act_off), (buf, offset), (buf, scale), (dest, dest_off)]
    if kind == "fp8":
        threads = 160 if (inner >> 4) % 160 == 0 else 256
        runtime.session.encode(
            "gemv_fp8_x4",
            bindings,
            [_u32(rows), _u32(inner)],
            (rows // 4, 1, 1),
            (threads, 1, 1),
        )
        return
    runtime.session.encode(
        "gemv_fp4_x4",
        bindings,
        [_u32(rows), _u32(inner)],
        (rows // 4, 1, 1),
        (32, 1, 1),
    )


def _residual(runtime: Any, span: _Span) -> None:
    for token in (0, 1):
        runtime.session.encode(
            "residual_rows",
            [
                (span.hidden, token * HIDDEN * 4),
                (span.out, token * HIDDEN * 4),
                (span.hidden, token * HIDDEN * 4),
            ],
            [_u32(HIDDEN)],
            (HIDDEN // 256, 1, 1),
            (256, 1, 1),
        )


def _rms(runtime: Any, src: Any, name: str, dest: Any) -> None:
    buf, offset, _entry = runtime.locate(name)
    runtime.session.encode(
        "rmsnorm_rows",
        [(src, 0), (buf, offset), (dest, 0)],
        [_u32(HIDDEN), _f32(EPS), _u32(2)],
        (2, 1, 1),
        (256, 1, 1),
    )


def _copy(runtime: Any, src: Any, src_off: int, dest: Any, dest_off: int, count: int) -> None:
    runtime.session.encode(
        "copy_f32",
        [(src, src_off), (dest, dest_off)],
        [_u32(count)],
        ((count + 255) // 256, 1, 1),
        (256, 1, 1),
    )


def _u32(value: int) -> bytes:
    return struct.pack("<I", int(value))


def _f32(value: float) -> bytes:
    return struct.pack("<f", float(value))
