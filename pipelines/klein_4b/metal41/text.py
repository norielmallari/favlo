"""Qwen3-4B, layers 0..26. Hidden states 9, 18, and 27 are concatenated."""

from __future__ import annotations

import numpy as np

from klein_4b.portable.constants import (
    PROMPT_LENGTH,
    TEXT_CAPTURE,
    TEXT_HEAD_DIM,
    TEXT_HEADS,
    TEXT_HIDDEN,
    TEXT_KV_HEADS,
    TEXT_LAYERS,
)
from klein_4b.portable.prompt import encode
from klein_4b.portable.rope import text_table
from klein_4b.portable.store import Store

from .metal import Gpu, View


def embed(store: Store, prompt: str) -> tuple[np.ndarray, int]:
    """Token embeddings for the wrapped prompt, and the unpadded length."""
    ids, valid = encode(prompt, store.path / "tokenizer" / "tokenizer.json")
    table = store.matrix("text/model.embed_tokens.weight")
    if int(ids.max()) >= table.n:
        message = "prompt token is outside the embedding table"
        raise ValueError(message)
    chosen = np.asarray(ids, dtype=np.int64)
    rows = np.asarray(table.codes[chosen, :TEXT_HIDDEN], dtype=np.float32)
    scales = np.asarray(table.scales[chosen], dtype=np.float32)
    return np.ascontiguousarray(rows * scales[:, None], dtype=np.float16), valid


def encode_prompt(gpu: Gpu, store: Store, prompt: str) -> View:
    """``[512, 7680]`` float16 on device. Pad keys are outside the causal window."""
    hidden_host, valid = embed(store, prompt)
    hidden = gpu.copy_in(hidden_host)
    cos_host, sin_host = text_table(PROMPT_LENGTH)
    cos = gpu.copy_in(cos_host)
    sin = gpu.copy_in(sin_host)
    captured: list[View] = []
    for layer in range(TEXT_LAYERS):
        hidden = _layer(gpu, hidden, layer, cos, sin, valid)
        if layer in TEXT_CAPTURE:
            captured.append(gpu.clone(hidden))
    gpu.drop(hidden, cos, sin)
    joined = gpu.concat_cols(captured)
    gpu.drop(*captured)
    return joined


def _layer(gpu: Gpu, hidden: View, layer: int, cos: View, sin: View, valid: int) -> View:
    prefix = f"text/model.layers.{layer}"
    normed = gpu.rmsnorm(hidden, gpu.vector(f"{prefix}.input_layernorm.weight"), store="f16")
    attended = _attention(gpu, normed, prefix, cos, sin, valid)
    gpu.drop(normed)
    residual = gpu.gate(hidden, None, attended)
    gpu.drop(hidden, attended)
    normed = gpu.rmsnorm(residual, gpu.vector(f"{prefix}.post_attention_layernorm.weight"), store="f16")
    gate = gpu.linear_view(normed, f"{prefix}.mlp.gate_proj.weight")
    up = gpu.linear_view(normed, f"{prefix}.mlp.up_proj.weight")
    gpu.drop(normed)
    mixed = gpu.silu_mul(gate, up)
    gpu.drop(gate, up)
    down = gpu.linear_view(mixed, f"{prefix}.mlp.down_proj.weight")
    gpu.drop(mixed)
    out = gpu.gate(residual, None, down)
    gpu.drop(residual, down)
    return out


def _attention(gpu: Gpu, hidden: View, prefix: str, cos: View, sin: View, valid: int) -> View:
    query = gpu.linear_view(hidden, f"{prefix}.self_attn.q_proj.weight")
    key = gpu.linear_view(hidden, f"{prefix}.self_attn.k_proj.weight")
    value = gpu.linear_view(hidden, f"{prefix}.self_attn.v_proj.weight")
    seq = int(hidden.shape[0])
    gpu.reshape(query, (seq, TEXT_HEADS, TEXT_HEAD_DIM))
    gpu.reshape(key, (seq, TEXT_KV_HEADS, TEXT_HEAD_DIM))
    gpu.reshape(value, (seq, TEXT_KV_HEADS, TEXT_HEAD_DIM))
    query_n = gpu.rmsnorm(query, gpu.vector(f"{prefix}.self_attn.q_norm.weight"), store="f32")
    key_n = gpu.rmsnorm(key, gpu.vector(f"{prefix}.self_attn.k_norm.weight"), store="f32")
    gpu.drop(query, key)
    query_r = gpu.rope_text(query_n, cos, sin)
    key_r = gpu.rope_text(key_n, cos, sin)
    gpu.drop(query_n, key_n)
    repeat = TEXT_HEADS // TEXT_KV_HEADS
    key_r = _replace(gpu, key_r, gpu.repeat_heads(key_r, repeat))
    value_f = gpu.cast(value, "f32")
    gpu.drop(value)
    value_r = gpu.repeat_heads(value_f, repeat)
    gpu.drop(value_f)
    mixed = gpu.attend_view(query_r, key_r, value_r, causal=True, valid=valid)
    gpu.drop(query_r, key_r, value_r)
    flat = gpu.cast(mixed, "f16")
    gpu.drop(mixed)
    gpu.reshape(flat, (seq, TEXT_HEADS * TEXT_HEAD_DIM))
    out = gpu.linear_view(flat, f"{prefix}.self_attn.o_proj.weight")
    gpu.drop(flat)
    return out


def _replace(gpu: Gpu, previous: View, current: View) -> View:
    gpu.drop(previous)
    return current
