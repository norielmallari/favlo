"""Twenty-seven Qwen3 layers. Hidden states 9, 18, and 27 are the caption."""

from __future__ import annotations

import numpy as np

from klein_4b.cuda.constants import (
    TEXT_CAPTURE,
    TEXT_EPS,
    TEXT_HEAD_DIM,
    TEXT_HEADS,
    TEXT_KV_HEADS,
    TEXT_LAYERS,
)
from klein_4b.cuda.ops import Ops, Plane
from klein_4b.cuda.rope import text_table
from klein_4b.cuda.store import Store


def caption(ops: Ops, store: Store, ids: np.ndarray, valid: int) -> Plane:
    cos_host, sin_host = text_table(ids.shape[0])
    cos = ops.upload_f32(cos_host)
    sin = ops.upload_f32(sin_host)
    hidden = ops.embed(ids, store.get("text/model.embed_tokens.weight"))
    kept: list[Plane] = []
    for layer in range(TEXT_LAYERS):
        hidden = _layer(ops, store, hidden, layer, valid, cos, sin)
        if layer in TEXT_CAPTURE:
            kept.append(ops.slice_rows(hidden, 0, hidden.rows))
    ops.free(hidden, cos, sin)
    joined = kept[0]
    for part in kept[1:]:
        nxt = ops.concat_cols(joined, part)
        ops.free(joined, part)
        joined = nxt
    store.free_prefix("text/")
    return joined


def _layer(ops: Ops, store: Store, hidden: Plane, layer: int, valid: int, cos: int, sin: int) -> Plane:
    root = f"text/model.layers.{layer}"
    normed = ops.rmsnorm(hidden, store.get(f"{root}.input_layernorm.weight"), TEXT_EPS)
    query = ops.linear(normed, store.get(f"{root}.self_attn.q_proj.weight"))
    key = ops.linear(normed, store.get(f"{root}.self_attn.k_proj.weight"))
    value = ops.linear(normed, store.get(f"{root}.self_attn.v_proj.weight"))
    ops.free(normed)
    query = _swap(ops, query, ops.rmsnorm_heads(query, store.get(f"{root}.self_attn.q_norm.weight"), TEXT_HEADS, TEXT_HEAD_DIM, TEXT_EPS))
    key = _swap(ops, key, ops.rmsnorm_heads(key, store.get(f"{root}.self_attn.k_norm.weight"), TEXT_KV_HEADS, TEXT_HEAD_DIM, TEXT_EPS))
    query = _swap(ops, query, ops.rope(query, hidden.rows, TEXT_HEADS, TEXT_HEAD_DIM, cos, sin, text=True))
    key = _swap(ops, key, ops.rope(key, hidden.rows, TEXT_KV_HEADS, TEXT_HEAD_DIM, cos, sin, text=True))
    mixed = ops.attend(query, key, value, hidden.rows, TEXT_HEADS, TEXT_KV_HEADS, TEXT_HEAD_DIM, 1, valid)
    ops.free(query, key, value)
    mixed = _swap(ops, mixed, ops.linear(mixed, store.get(f"{root}.self_attn.o_proj.weight")))
    hidden = _swap(ops, hidden, ops.gated(hidden, None, mixed))
    ops.free(mixed)
    normed = ops.rmsnorm(hidden, store.get(f"{root}.post_attention_layernorm.weight"), TEXT_EPS)
    gate = ops.linear(normed, store.get(f"{root}.mlp.gate_proj.weight"))
    up = ops.linear(normed, store.get(f"{root}.mlp.up_proj.weight"))
    ops.free(normed)
    mid = ops.silu_mul(gate, up)
    ops.free(gate, up)
    down = ops.linear(mid, store.get(f"{root}.mlp.down_proj.weight"))
    ops.free(mid)
    out = ops.gated(hidden, None, down)
    ops.free(hidden, down)
    return out


def _swap(ops: Ops, old: Plane, new: Plane) -> Plane:
    ops.free(old)
    return new
