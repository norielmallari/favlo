"""FLUX.2 MMDiT. Reference tokens, when present, stay in front of the target."""

from __future__ import annotations

import numpy as np

from .constants import DIT_DIM, DIT_DOUBLE, DIT_HEAD_DIM, DIT_HEADS, DIT_SINGLE, PROMPT_LENGTH
from .metal import Gpu, View


def velocity(
    gpu: Gpu,
    image: View,
    text: View,
    cos: View,
    sin: View,
    sigma: float,
    target_tokens: int,
) -> View:
    """Velocity for the target tail. ``image`` is consumed. ``text`` stays."""
    timed = _time(gpu, sigma)
    ready = gpu.silu(timed)
    gpu.drop(timed)
    img_mod = gpu.linear_view(ready, "dit/double_stream_modulation_img.linear.weight")
    txt_mod = gpu.linear_view(ready, "dit/double_stream_modulation_txt.linear.weight")
    single_mod = gpu.linear_view(ready, "dit/single_stream_modulation.linear.weight")
    hidden = gpu.linear_view(image, "dit/x_embedder.weight")
    gpu.drop(image)
    context = gpu.linear_view(text, "dit/context_embedder.weight")
    image_rows = int(hidden.shape[0])
    text_cos = gpu.slice_rows(cos, 0, PROMPT_LENGTH)
    image_cos = gpu.slice_rows(cos, PROMPT_LENGTH, image_rows)
    text_sin = gpu.slice_rows(sin, 0, PROMPT_LENGTH)
    image_sin = gpu.slice_rows(sin, PROMPT_LENGTH, image_rows)
    img_parts = _spans(gpu, img_mod)
    txt_parts = _spans(gpu, txt_mod)
    for block in range(DIT_DOUBLE):
        context, hidden = _double(
            gpu, hidden, context, img_parts, txt_parts, text_cos, text_sin, image_cos, image_sin, block
        )
    stream = gpu.concat_rows([context, hidden])
    gpu.drop(context, hidden)
    single_parts = _spans(gpu, single_mod, 3)
    for block in range(DIT_SINGLE):
        stream = _single(gpu, stream, single_parts, cos, sin, block)
    image_out = gpu.clone(gpu.slice_rows(stream, PROMPT_LENGTH, int(stream.shape[0]) - PROMPT_LENGTH))
    gpu.drop(stream)
    scale_shift = gpu.linear_view(ready, "dit/norm_out.linear.weight")
    gpu.drop(ready, img_mod, txt_mod, single_mod)
    shift = gpu.span(scale_shift, DIT_DIM, DIT_DIM)
    scale = gpu.span(scale_shift, 0, DIT_DIM)
    readout = gpu.modulate(image_out, shift, scale)
    gpu.drop(image_out, scale_shift)
    projected = gpu.linear_view(readout, "dit/proj_out.weight")
    gpu.drop(readout)
    tail = gpu.clone(gpu.slice_rows(projected, int(projected.shape[0]) - int(target_tokens), int(target_tokens)))
    gpu.drop(projected)
    return tail


def _time(gpu: Gpu, sigma: float) -> View:
    features = gpu.copy_in(_sinusoid(sigma * 1000.0).astype(np.float16)[None, :])
    hidden = gpu.linear_view(features, "dit/time_guidance_embed.timestep_embedder.linear_1.weight")
    gpu.drop(features)
    activated = gpu.silu(hidden)
    gpu.drop(hidden)
    timed = gpu.linear_view(activated, "dit/time_guidance_embed.timestep_embedder.linear_2.weight")
    gpu.drop(activated)
    return timed


def _sinusoid(timestep: float) -> np.ndarray:
    half = 128
    freqs = np.exp(-np.log(10000.0) * np.arange(half, dtype=np.float64) / half)
    args = np.float64(timestep) * freqs
    return np.concatenate([np.cos(args), np.sin(args)]).astype(np.float32)


def _double(
    gpu: Gpu,
    image: View,
    text: View,
    img_parts: list[View],
    txt_parts: list[View],
    text_cos: View,
    text_sin: View,
    image_cos: View,
    image_sin: View,
    block: int,
) -> tuple[View, View]:
    prefix = f"dit/transformer_blocks.{block}"
    img_shift, img_scale, img_gate, img_mlp_shift, img_mlp_scale, img_mlp_gate = img_parts
    txt_shift, txt_scale, txt_gate, txt_mlp_shift, txt_mlp_scale, txt_mlp_gate = txt_parts
    img_norm = gpu.modulate(image, img_shift, img_scale)
    txt_norm = gpu.modulate(text, txt_shift, txt_scale)
    img_out, txt_out = _joint(gpu, img_norm, txt_norm, text_cos, text_sin, image_cos, image_sin, prefix)
    gpu.drop(img_norm, txt_norm)
    image = _gated(gpu, image, img_gate, img_out)
    text = _gated(gpu, text, txt_gate, txt_out)
    image = _gated(gpu, image, img_mlp_gate, _ff(gpu, gpu.modulate(image, img_mlp_shift, img_mlp_scale), f"{prefix}.ff"))
    text = _gated(
        gpu,
        text,
        txt_mlp_gate,
        _ff(gpu, gpu.modulate(text, txt_mlp_shift, txt_mlp_scale), f"{prefix}.ff_context"),
    )
    return text, image


def _joint(
    gpu: Gpu,
    image: View,
    text: View,
    text_cos: View,
    text_sin: View,
    image_cos: View,
    image_sin: View,
    prefix: str,
) -> tuple[View, View]:
    img_q, img_k, img_v = _qkv(gpu, image, prefix, added=False, cos=image_cos, sin=image_sin)
    txt_q, txt_k, txt_v = _qkv(gpu, text, prefix, added=True, cos=text_cos, sin=text_sin)
    query = gpu.concat_rows([txt_q, img_q])
    key = gpu.concat_rows([txt_k, img_k])
    value = gpu.concat_rows([txt_v, img_v])
    gpu.drop(txt_q, img_q, txt_k, img_k, txt_v, img_v)
    mixed = gpu.attend_view(query, key, value, causal=False, valid=int(query.shape[0]))
    gpu.drop(query, key, value)
    text_len = int(text.shape[0])
    image_len = int(image.shape[0])
    text_mix = gpu.cast(gpu.slice_rows(mixed, 0, text_len), "f16")
    image_mix = gpu.cast(gpu.slice_rows(mixed, text_len, image_len), "f16")
    gpu.drop(mixed)
    gpu.reshape(text_mix, (text_len, DIT_DIM))
    gpu.reshape(image_mix, (image_len, DIT_DIM))
    text_out = gpu.linear_view(text_mix, f"{prefix}.attn.to_add_out.weight")
    image_out = gpu.linear_view(image_mix, f"{prefix}.attn.to_out.0.weight")
    gpu.drop(text_mix, image_mix)
    return image_out, text_out


def _qkv(
    gpu: Gpu,
    hidden: View,
    prefix: str,
    *,
    added: bool,
    cos: View,
    sin: View,
) -> tuple[View, View, View]:
    if added:
        q_key = f"{prefix}.attn.add_q_proj.weight"
        k_key = f"{prefix}.attn.add_k_proj.weight"
        v_key = f"{prefix}.attn.add_v_proj.weight"
        q_norm = f"{prefix}.attn.norm_added_q.weight"
        k_norm = f"{prefix}.attn.norm_added_k.weight"
    else:
        q_key = f"{prefix}.attn.to_q.weight"
        k_key = f"{prefix}.attn.to_k.weight"
        v_key = f"{prefix}.attn.to_v.weight"
        q_norm = f"{prefix}.attn.norm_q.weight"
        k_norm = f"{prefix}.attn.norm_k.weight"
    query = gpu.linear_view(hidden, q_key)
    key = gpu.linear_view(hidden, k_key)
    value = gpu.linear_view(hidden, v_key)
    seq = int(hidden.shape[0])
    gpu.reshape(query, (seq, DIT_HEADS, DIT_HEAD_DIM))
    gpu.reshape(key, (seq, DIT_HEADS, DIT_HEAD_DIM))
    gpu.reshape(value, (seq, DIT_HEADS, DIT_HEAD_DIM))
    query_n = gpu.rmsnorm(query, gpu.vector(q_norm), store="f32")
    key_n = gpu.rmsnorm(key, gpu.vector(k_norm), store="f32")
    gpu.drop(query, key)
    query_r = gpu.rope_dit(query_n, cos, sin)
    key_r = gpu.rope_dit(key_n, cos, sin)
    gpu.drop(query_n, key_n)
    value_f = gpu.cast(value, "f32")
    gpu.drop(value)
    return query_r, key_r, value_f


def _ff(gpu: Gpu, hidden: View, prefix: str) -> View:
    projected = gpu.linear_view(hidden, f"{prefix}.linear_in.weight")
    gpu.drop(hidden)
    mixed = gpu.swiglu(projected)
    gpu.drop(projected)
    out = gpu.linear_view(mixed, f"{prefix}.linear_out.weight")
    gpu.drop(mixed)
    return out


def _single(gpu: Gpu, stream: View, parts: list[View], cos: View, sin: View, block: int) -> View:
    shift, scale, gate = parts
    prefix = f"dit/single_transformer_blocks.{block}.attn"
    normed = gpu.modulate(stream, shift, scale)
    projected = gpu.linear_view(normed, f"{prefix}.to_qkv_mlp_proj.weight")
    gpu.drop(normed)
    qkv = gpu.slice_cols(projected, 0, 3 * DIT_DIM)
    mlp = gpu.slice_cols(projected, 3 * DIT_DIM, 6 * DIT_DIM)
    gpu.drop(projected)
    seq = int(stream.shape[0])
    query = gpu.slice_cols(qkv, 0, DIT_DIM)
    key = gpu.slice_cols(qkv, DIT_DIM, DIT_DIM)
    value = gpu.slice_cols(qkv, 2 * DIT_DIM, DIT_DIM)
    gpu.drop(qkv)
    gpu.reshape(query, (seq, DIT_HEADS, DIT_HEAD_DIM))
    gpu.reshape(key, (seq, DIT_HEADS, DIT_HEAD_DIM))
    gpu.reshape(value, (seq, DIT_HEADS, DIT_HEAD_DIM))
    query_n = gpu.rmsnorm(query, gpu.vector(f"{prefix}.norm_q.weight"), store="f32")
    key_n = gpu.rmsnorm(key, gpu.vector(f"{prefix}.norm_k.weight"), store="f32")
    gpu.drop(query, key)
    query_r = gpu.rope_dit(query_n, cos, sin)
    key_r = gpu.rope_dit(key_n, cos, sin)
    gpu.drop(query_n, key_n)
    value_f = gpu.cast(value, "f32")
    gpu.drop(value)
    mixed = gpu.attend_view(query_r, key_r, value_f, causal=False, valid=seq)
    gpu.drop(query_r, key_r, value_f)
    activated = gpu.swiglu(mlp)
    gpu.drop(mlp)
    fused = gpu.pack_attn_mlp(mixed, activated)
    gpu.drop(mixed, activated)
    out = gpu.linear_view(fused, f"{prefix}.to_out.weight")
    gpu.drop(fused)
    return _gated(gpu, stream, gate, out)


def _gated(gpu: Gpu, residual: View, gate: View, branch: View) -> View:
    out = gpu.gate(residual, gate, branch)
    gpu.drop(residual, branch)
    return out


def _spans(gpu: Gpu, view: View, parts: int = 6) -> list[View]:
    return [gpu.span(view, index * DIT_DIM, DIT_DIM) for index in range(parts)]
