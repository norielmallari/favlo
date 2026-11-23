"""MMDiT denoise step. References sit in front of the target. The velocity is the tail."""

from __future__ import annotations

import numpy as np

from klein_4b.cuda.constants import (
    DIT_DIM,
    DIT_DOUBLE,
    DIT_HEAD_DIM,
    DIT_HEADS,
    DIT_SINGLE,
    NORM_EPS,
)
from klein_4b.cuda.ops import Ops, Plane
from klein_4b.cuda.store import Store


def step(
    ops: Ops,
    store: Store,
    image: Plane,
    context: Plane,
    sigma: float,
    text_cos: int,
    text_sin: int,
    image_cos: int,
    image_sin: int,
    joint_cos: int,
    joint_sin: int,
    text_len: int,
) -> Plane:
    cond = _conditioning(ops, store, sigma)
    img = ops.linear(image, store.get("dit/x_embedder.weight"))
    txt = ops.linear(context, store.get("dit/context_embedder.weight"))
    for index in range(DIT_DOUBLE):
        img, txt = _double(ops, store, img, txt, cond, index, text_cos, text_sin, image_cos, image_sin)
    seq = ops.concat_rows([txt, img])
    ops.free(txt, img)
    for index in range(DIT_SINGLE):
        seq = _single(ops, store, seq, cond, index, joint_cos, joint_sin)
    img = ops.slice_rows(seq, text_len, seq.rows - text_len)
    ops.free(seq)
    velocity = _readout(ops, store, img, cond)
    ops.free(img, cond)
    return velocity


def _conditioning(ops: Ops, store: Store, sigma: float) -> Plane:
    feat = ops.upload_f16(_sinusoid(sigma).reshape(1, 256), 1, 256)
    hidden = ops.linear(feat, store.get("dit/time_guidance_embed.timestep_embedder.linear_1.weight"))
    ops.free(feat)
    hidden = _swap(ops, hidden, ops.silu(hidden))
    hidden = _swap(ops, hidden, ops.linear(hidden, store.get("dit/time_guidance_embed.timestep_embedder.linear_2.weight")))
    return _swap(ops, hidden, ops.silu(hidden))


def _sinusoid(sigma: float) -> np.ndarray:
    half = 128
    freqs = np.exp(-np.log(10000.0) * np.arange(half, dtype=np.float64) / half)
    args = (sigma * 1000.0) * freqs
    values = np.empty(256, dtype=np.float32)
    values[:half] = np.cos(args)
    values[half:] = np.sin(args)
    return values.astype(np.float16)


def _double(
    ops: Ops,
    store: Store,
    img: Plane,
    txt: Plane,
    cond: Plane,
    index: int,
    text_cos: int,
    text_sin: int,
    image_cos: int,
    image_sin: int,
) -> tuple[Plane, Plane]:
    root = f"dit/transformer_blocks.{index}"
    img_mod = _chunks(ops, cond, store.get("dit/double_stream_modulation_img.linear.weight"), 6)
    txt_mod = _chunks(ops, cond, store.get("dit/double_stream_modulation_txt.linear.weight"), 6)
    img_q, img_k, img_v = _qkv(
        ops,
        store,
        img,
        img_mod[0],
        img_mod[1],
        f"{root}.attn.to_q.weight",
        f"{root}.attn.to_k.weight",
        f"{root}.attn.to_v.weight",
        f"{root}.attn.norm_q.weight",
        f"{root}.attn.norm_k.weight",
        image_cos,
        image_sin,
    )
    txt_q, txt_k, txt_v = _qkv(
        ops,
        store,
        txt,
        txt_mod[0],
        txt_mod[1],
        f"{root}.attn.add_q_proj.weight",
        f"{root}.attn.add_k_proj.weight",
        f"{root}.attn.add_v_proj.weight",
        f"{root}.attn.norm_added_q.weight",
        f"{root}.attn.norm_added_k.weight",
        text_cos,
        text_sin,
    )
    query = ops.concat_rows([txt_q, img_q])
    key = ops.concat_rows([txt_k, img_k])
    value = ops.concat_rows([txt_v, img_v])
    ops.free(txt_q, img_q, txt_k, img_k, txt_v, img_v)
    mixed = ops.attend(query, key, value, query.rows, DIT_HEADS, DIT_HEADS, DIT_HEAD_DIM, 0, query.rows)
    ops.free(query, key, value)
    txt_attn = ops.slice_rows(mixed, 0, txt.rows)
    img_attn = ops.slice_rows(mixed, txt.rows, img.rows)
    ops.free(mixed)
    img_attn = _swap(ops, img_attn, ops.linear(img_attn, _weight(store, f"{root}.attn.to_out.0.weight", f"{root}.attn.to_out.weight")))
    txt_attn = _swap(ops, txt_attn, ops.linear(txt_attn, _weight(store, f"{root}.attn.to_add_out.weight", f"{root}.attn.to_add_out.0.weight")))
    img = _swap(ops, img, ops.gated(img, img_mod[2], img_attn))
    txt = _swap(ops, txt, ops.gated(txt, txt_mod[2], txt_attn))
    ops.free(img_attn, txt_attn)
    img = _swap(ops, img, _ff(ops, store, img, img_mod[3], img_mod[4], img_mod[5], f"{root}.ff"))
    txt = _swap(ops, txt, _ff(ops, store, txt, txt_mod[3], txt_mod[4], txt_mod[5], f"{root}.ff_context"))
    ops.free(*img_mod, *txt_mod)
    return img, txt


def _single(ops: Ops, store: Store, seq: Plane, cond: Plane, index: int, cos: int, sin: int) -> Plane:
    root = f"dit/single_transformer_blocks.{index}.attn"
    shift, scale, gate = _chunks(ops, cond, store.get("dit/single_stream_modulation.linear.weight"), 3)
    normed = ops.layernorm(seq, NORM_EPS)
    normed = _swap(ops, normed, ops.modulate(normed, shift, scale))
    projected = ops.linear(normed, store.get(f"{root}.to_qkv_mlp_proj.weight"))
    ops.free(normed)
    qkv = ops.slice_cols(projected, 0, DIT_DIM * 3)
    mlp = ops.slice_cols(projected, DIT_DIM * 3, DIT_DIM * 6)
    ops.free(projected)
    query = ops.slice_cols(qkv, 0, DIT_DIM)
    key = ops.slice_cols(qkv, DIT_DIM, DIT_DIM)
    value = ops.slice_cols(qkv, DIT_DIM * 2, DIT_DIM)
    ops.free(qkv)
    query = _swap(ops, query, ops.rmsnorm_heads(query, store.get(f"{root}.norm_q.weight"), DIT_HEADS, DIT_HEAD_DIM, NORM_EPS))
    key = _swap(ops, key, ops.rmsnorm_heads(key, store.get(f"{root}.norm_k.weight"), DIT_HEADS, DIT_HEAD_DIM, NORM_EPS))
    query = _swap(ops, query, ops.rope(query, seq.rows, DIT_HEADS, DIT_HEAD_DIM, cos, sin, text=False))
    key = _swap(ops, key, ops.rope(key, seq.rows, DIT_HEADS, DIT_HEAD_DIM, cos, sin, text=False))
    mixed = ops.attend(query, key, value, seq.rows, DIT_HEADS, DIT_HEADS, DIT_HEAD_DIM, 0, seq.rows)
    ops.free(query, key, value)
    activated = ops.swiglu(mlp)
    ops.free(mlp)
    fused = ops.concat_cols(mixed, activated)
    ops.free(mixed, activated)
    branch = ops.linear(fused, store.get(f"{root}.to_out.weight"))
    ops.free(fused)
    out = ops.gated(seq, gate, branch)
    ops.free(seq, branch, shift, scale, gate)
    return out


def _readout(ops: Ops, store: Store, img: Plane, cond: Plane) -> Plane:
    scale, shift = _chunks(ops, cond, store.get("dit/norm_out.linear.weight"), 2)
    normed = ops.layernorm(img, NORM_EPS)
    normed = _swap(ops, normed, ops.modulate(normed, shift, scale))
    ops.free(scale, shift)
    out = ops.linear(normed, store.get("dit/proj_out.weight"))
    ops.free(normed)
    return out


def _qkv(
    ops: Ops,
    store: Store,
    hidden: Plane,
    shift: Plane,
    scale: Plane,
    q_name: str,
    k_name: str,
    v_name: str,
    qn: str,
    kn: str,
    cos: int,
    sin: int,
) -> tuple[Plane, Plane, Plane]:
    normed = ops.layernorm(hidden, NORM_EPS)
    normed = _swap(ops, normed, ops.modulate(normed, shift, scale))
    query = ops.linear(normed, store.get(q_name))
    key = ops.linear(normed, store.get(k_name))
    value = ops.linear(normed, store.get(v_name))
    ops.free(normed)
    query = _swap(ops, query, ops.rmsnorm_heads(query, store.get(qn), DIT_HEADS, DIT_HEAD_DIM, NORM_EPS))
    key = _swap(ops, key, ops.rmsnorm_heads(key, store.get(kn), DIT_HEADS, DIT_HEAD_DIM, NORM_EPS))
    query = _swap(ops, query, ops.rope(query, hidden.rows, DIT_HEADS, DIT_HEAD_DIM, cos, sin, text=False))
    key = _swap(ops, key, ops.rope(key, hidden.rows, DIT_HEADS, DIT_HEAD_DIM, cos, sin, text=False))
    return query, key, value


def _ff(ops: Ops, store: Store, hidden: Plane, shift: Plane, scale: Plane, gate: Plane, root: str) -> Plane:
    normed = ops.layernorm(hidden, NORM_EPS)
    normed = _swap(ops, normed, ops.modulate(normed, shift, scale))
    inner = ops.linear(normed, store.get(f"{root}.linear_in.weight"))
    ops.free(normed)
    inner = _swap(ops, inner, ops.swiglu(inner))
    branch = ops.linear(inner, store.get(f"{root}.linear_out.weight"))
    ops.free(inner)
    out = ops.gated(hidden, gate, branch)
    ops.free(branch)
    return out


def _chunks(ops: Ops, cond: Plane, weight, count: int) -> list[Plane]:
    raw = ops.linear(cond, weight)
    parts = [ops.slice_cols(raw, index * DIT_DIM, DIT_DIM) for index in range(count)]
    ops.free(raw)
    return parts


def _weight(store: Store, *names: str):
    for name in names:
        if store.has(name):
            return store.get(name)
    return store.get(names[0])


def _swap(ops: Ops, old: Plane, new: Plane) -> Plane:
    ops.free(old)
    return new
