"""S3-DiT velocity at one noise fraction.

Caption tokens are refined without AdaLN. Image tokens are refined with AdaLN.
The main stack sees image tokens, then caption tokens. RoPE puts the caption
index on the temporal axis and the patch coordinate on the two spatial axes.
DiT rotation pairs adjacent elements. Gates are the hyperbolic tangent of the
AdaLN slice. The final readout is a parameter-free LayerNorm, then the scale.
``w1`` is the SwiGLU gate and ``w3`` is the up projection. AdaLN slices are
scale and gate for attention, then scale and gate for the feed-forward.
The modulation is ``x · (1 + scale)``, and the gate multiplies after the
second RMSNorm.
"""

from __future__ import annotations

import numpy as np

from z_image_turbo.formats import TILE_M, pad_to
from z_image_turbo.metal import Gpu
from z_image_turbo.ops import project
from z_image_turbo.record import Plane
from z_image_turbo.schedule import timestep
from z_image_turbo.scratch import DIT_FFN, DIT_HIDDEN, Scratch

SEQ = 32

HEADS = 30
HEAD_DIM = 128
EPS = 1e-5
FINAL_EPS = 1e-6
THETA = 256.0
AXES = (32, 48, 48)
LAYERS = 30


def velocity(
    gpu: Gpu,
    scratch: Scratch,
    weights: dict[str, Plane],
    tokens: np.ndarray,
    caption: np.ndarray,
    sigma: float,
    ones: object,
    patch_h: int,
    patch_w: int,
) -> np.ndarray:
    """``tokens`` is ``[image, 64]``. ``caption`` is ``[length, 3840]`` float32."""
    image_count = int(tokens.shape[0])
    cap_len = int(caption.shape[0])
    image_attend = _align(image_count)
    cap_attend = _align(cap_len)
    image_rows = pad_to(image_count, TILE_M)
    cap_rows = pad_to(cap_len, TILE_M)
    total = image_attend + cap_attend
    seq_rows = pad_to(total, TILE_M)
    t0 = cap_attend + 1
    _embed_image(gpu, scratch, weights, tokens, image_rows)
    _fill_pad(gpu, scratch.image, weights, "x_pad_token", image_count, image_attend, image_rows)
    _refine(
        gpu,
        scratch,
        weights,
        scratch.image,
        scratch.image2,
        image_rows,
        image_attend,
        _image_positions(patch_h, patch_w, image_rows, t0),
        sigma,
        ones,
        "noise_refiner",
        True,
    )
    _store_caption(gpu, scratch, caption, cap_rows)
    _fill_pad(gpu, scratch.seq, weights, "cap_pad_token", cap_len, cap_attend, cap_rows)
    _refine(
        gpu,
        scratch,
        weights,
        scratch.seq,
        scratch.seq2,
        cap_rows,
        cap_attend,
        _text_positions(cap_len, cap_rows),
        sigma,
        ones,
        "context_refiner",
        False,
    )
    _join(
        gpu,
        scratch,
        cap_attend,
        cap_rows,
        image_attend,
        image_rows,
        seq_rows,
    )
    _refine(
        gpu,
        scratch,
        weights,
        scratch.seq,
        scratch.seq2,
        seq_rows,
        total,
        _joined_positions(
            cap_len,
            cap_attend,
            image_count,
            image_attend,
            patch_h,
            patch_w,
            seq_rows,
        ),
        sigma,
        ones,
        "layers",
        True,
        LAYERS,
    )
    return _final(gpu, scratch, weights, sigma, ones, image_count, seq_rows)


def _embed_image(
    gpu: Gpu,
    scratch: Scratch,
    weights: dict[str, Plane],
    tokens: np.ndarray,
    rows: int,
) -> None:
    act = np.zeros((rows, 256), dtype=np.float32)
    act[: tokens.shape[0], :64] = tokens
    gpu.store_f32(scratch.tokens, act)
    project(
        gpu,
        scratch.tokens,
        weights["all_x_embedder.2-1.weight"],
        weights["all_x_embedder.2-1.bias"],
        scratch.out,
        scratch.image,
        rows,
        256,
    )


def _align(count: int) -> int:
    return (count + SEQ - 1) // SEQ * SEQ


def _fill_pad(
    gpu: Gpu,
    buf: object,
    weights: dict[str, Plane],
    name: str,
    start: int,
    stop: int,
    rows: int,
) -> None:
    if stop <= start:
        return
    plane = weights[name]
    token = gpu.read_f16(plane.data, int(np.prod(plane.shape))).astype(np.float32).reshape(-1)
    act = gpu.read_f32(buf, rows * DIT_HIDDEN).reshape(rows, DIT_HIDDEN)
    act[start:stop] = token[:DIT_HIDDEN]
    gpu.store_f32(buf, act)


def _store_caption(gpu: Gpu, scratch: Scratch, caption: np.ndarray, rows: int) -> None:
    act = np.zeros((rows, DIT_HIDDEN), dtype=np.float32)
    act[: caption.shape[0]] = caption
    gpu.store_f32(scratch.seq, act)


def _join(
    gpu: Gpu,
    scratch: Scratch,
    cap_len: int,
    cap_rows: int,
    image_count: int,
    image_rows: int,
    seq_rows: int,
) -> None:
    cap = gpu.read_f32(scratch.seq, cap_rows * DIT_HIDDEN).reshape(cap_rows, DIT_HIDDEN)
    image = gpu.read_f32(scratch.image, image_rows * DIT_HIDDEN).reshape(image_rows, DIT_HIDDEN)
    joined = np.zeros((seq_rows, DIT_HIDDEN), dtype=np.float32)
    joined[:image_count] = image[:image_count]
    joined[image_count : image_count + cap_len] = cap[:cap_len]
    gpu.store_f32(scratch.seq, joined)


def _refine(
    gpu: Gpu,
    scratch: Scratch,
    weights: dict[str, Plane],
    src: object,
    dst: object,
    rows: int,
    limit: int,
    pos: np.ndarray,
    sigma: float,
    ones: object,
    prefix: str,
    adaln: bool,
    count: int = 2,
) -> None:
    positions = gpu.upload(pos)
    cond = _time(gpu, scratch, weights, sigma, ones) if adaln else None
    left, right = src, dst
    for index in range(count):
        mod = None
        if adaln and cond is not None:
            mod = _modulate(gpu, scratch, weights, cond, f"{prefix}.{index}.adaLN_modulation.0")
        _block(
            gpu,
            scratch,
            weights,
            left,
            right,
            rows,
            limit,
            positions,
            f"{prefix}.{index}",
            mod,
        )
        left, right = right, left


def _final(
    gpu: Gpu,
    scratch: Scratch,
    weights: dict[str, Plane],
    sigma: float,
    ones: object,
    image_count: int,
    seq_rows: int,
) -> np.ndarray:
    cond = _time(gpu, scratch, weights, sigma, ones)
    gpu.silu_f32(cond, cond, 256)
    mod = _modulate(gpu, scratch, weights, cond, "all_final_layer.2-1.adaLN_modulation.1")
    gpu.layernorm(scratch.seq, scratch.seq2, seq_rows, DIT_HIDDEN, FINAL_EPS)
    gpu.channel_scale(scratch.seq2, mod, scratch.seq, seq_rows, DIT_HIDDEN, 0)
    project(
        gpu,
        scratch.seq,
        weights["all_final_layer.2-1.linear.weight"],
        weights["all_final_layer.2-1.linear.bias"],
        scratch.out,
        scratch.ffn,
        seq_rows,
        DIT_HIDDEN,
    )
    flat = gpu.read_f32(scratch.ffn, seq_rows * 64).reshape(seq_rows, 64)
    return flat[:image_count].astype(np.float32)


def _time(
    gpu: Gpu,
    scratch: Scratch,
    weights: dict[str, Plane],
    sigma: float,
    ones: object,
) -> object:
    freq = _sinusoidal(timestep(sigma), 256)
    act = np.zeros((64, 256), dtype=np.float32)
    act[0] = freq
    gpu.store_f32(scratch.time, act)
    project(
        gpu,
        scratch.time,
        weights["t_embedder.mlp.0.weight"],
        weights["t_embedder.mlp.0.bias"],
        scratch.wide,
        scratch.cond,
        64,
        256,
    )
    gpu.silu_mul(scratch.cond, ones, scratch.cond, 1024)
    project(
        gpu,
        scratch.cond,
        weights["t_embedder.mlp.2.weight"],
        weights["t_embedder.mlp.2.bias"],
        scratch.wide,
        scratch.time,
        64,
        1024,
    )
    return scratch.time


def _modulate(
    gpu: Gpu, scratch: Scratch, weights: dict[str, Plane], cond: object, name: str
) -> object:
    project(
        gpu,
        cond,
        weights[f"{name}.weight"],
        weights[f"{name}.bias"],
        scratch.wide,
        scratch.mod,
        64,
        256,
    )
    return scratch.mod


def _block(
    gpu: Gpu,
    scratch: Scratch,
    weights: dict[str, Plane],
    src: object,
    dst: object,
    rows: int,
    limit: int,
    positions: object,
    prefix: str,
    mod: object | None,
) -> None:
    gpu.rmsnorm(
        src,
        weights[f"{prefix}.attention_norm1.weight"].data,
        scratch.ffn,
        rows,
        DIT_HIDDEN,
        EPS,
    )
    hidden = scratch.ffn
    if mod is not None:
        gpu.channel_scale(scratch.ffn, mod, scratch.ffn2, rows, DIT_HIDDEN, 0)
        hidden = scratch.ffn2
    _attention(gpu, scratch, weights, hidden, rows, limit, positions, prefix)
    gpu.rmsnorm(
        scratch.attn,
        weights[f"{prefix}.attention_norm2.weight"].data,
        scratch.q,
        rows,
        DIT_HIDDEN,
        EPS,
    )
    if mod is None:
        gpu.add_rows(src, scratch.q, dst, rows * DIT_HIDDEN)
    else:
        gpu.channel_gate(src, scratch.q, mod, dst, rows, DIT_HIDDEN, DIT_HIDDEN * 4)
    gpu.rmsnorm(dst, weights[f"{prefix}.ffn_norm1.weight"].data, scratch.q, rows, DIT_HIDDEN, EPS)
    ffn_in = scratch.q
    if mod is not None:
        gpu.channel_scale(scratch.q, mod, scratch.k, rows, DIT_HIDDEN, DIT_HIDDEN * 8)
        ffn_in = scratch.k
    project(
        gpu,
        ffn_in,
        weights[f"{prefix}.feed_forward.w1.weight"],
        None,
        scratch.out,
        scratch.ffn,
        rows,
        DIT_HIDDEN,
    )
    project(
        gpu,
        ffn_in,
        weights[f"{prefix}.feed_forward.w3.weight"],
        None,
        scratch.out,
        scratch.ffn2,
        rows,
        DIT_HIDDEN,
    )
    gpu.silu_mul(scratch.ffn, scratch.ffn2, scratch.ffn, rows * DIT_FFN)
    project(
        gpu,
        scratch.ffn,
        weights[f"{prefix}.feed_forward.w2.weight"],
        None,
        scratch.out,
        scratch.q,
        rows,
        DIT_FFN,
    )
    gpu.rmsnorm(
        scratch.q,
        weights[f"{prefix}.ffn_norm2.weight"].data,
        scratch.k,
        rows,
        DIT_HIDDEN,
        EPS,
    )
    if mod is None:
        gpu.add_rows(dst, scratch.k, dst, rows * DIT_HIDDEN)
    else:
        gpu.channel_gate(dst, scratch.k, mod, dst, rows, DIT_HIDDEN, DIT_HIDDEN * 12)


def _attention(
    gpu: Gpu,
    scratch: Scratch,
    weights: dict[str, Plane],
    hidden: object,
    rows: int,
    limit: int,
    positions: object,
    prefix: str,
) -> None:
    project(
        gpu,
        hidden,
        weights[f"{prefix}.attention.to_q.weight"],
        None,
        scratch.out,
        scratch.q,
        rows,
        DIT_HIDDEN,
    )
    project(
        gpu,
        hidden,
        weights[f"{prefix}.attention.to_k.weight"],
        None,
        scratch.out,
        scratch.k,
        rows,
        DIT_HIDDEN,
    )
    project(
        gpu,
        hidden,
        weights[f"{prefix}.attention.to_v.weight"],
        None,
        scratch.out,
        scratch.v,
        rows,
        DIT_HIDDEN,
    )
    packed = rows * HEADS
    gpu.rmsnorm(
        scratch.q,
        weights[f"{prefix}.attention.norm_q.weight"].data,
        scratch.attn,
        packed,
        HEAD_DIM,
        EPS,
    )
    gpu.rmsnorm(
        scratch.k,
        weights[f"{prefix}.attention.norm_k.weight"].data,
        scratch.q,
        packed,
        HEAD_DIM,
        EPS,
    )
    gpu.rope(scratch.attn, positions, rows, HEADS, HEAD_DIM, THETA, AXES, True)
    gpu.rope(scratch.q, positions, rows, HEADS, HEAD_DIM, THETA, AXES, True)
    gpu.attend(
        scratch.attn,
        scratch.q,
        scratch.v,
        scratch.k,
        rows,
        limit,
        HEADS,
        HEADS,
        HEAD_DIM,
        False,
    )
    project(
        gpu,
        scratch.k,
        weights[f"{prefix}.attention.to_out.0.weight"],
        None,
        scratch.out,
        scratch.attn,
        rows,
        DIT_HIDDEN,
    )


def _sinusoidal(value: float, dim: int) -> np.ndarray:
    half = dim // 2
    freqs = np.exp(-np.log(10000.0) * np.arange(half, dtype=np.float64) / half)
    angles = float(value) * freqs
    return np.concatenate([np.cos(angles), np.sin(angles)]).astype(np.float32)


def _text_positions(limit: int, rows: int) -> np.ndarray:
    pos = np.zeros((rows, 3), dtype=np.int32)
    pos[:limit, 0] = np.arange(1, limit + 1, dtype=np.int32)
    return pos


def _image_positions(patch_h: int, patch_w: int, rows: int, origin: int) -> np.ndarray:
    pos = np.zeros((rows, 3), dtype=np.int32)
    index = 0
    for y in range(patch_h):
        for x in range(patch_w):
            pos[index, 0] = origin
            pos[index, 1] = y
            pos[index, 2] = x
            index += 1
    return pos


def _joined_positions(
    cap_len: int,
    cap_attend: int,
    image_count: int,
    image_attend: int,
    patch_h: int,
    patch_w: int,
    rows: int,
) -> np.ndarray:
    pos = np.zeros((rows, 3), dtype=np.int32)
    origin = cap_attend + 1
    index = 0
    for y in range(patch_h):
        for x in range(patch_w):
            pos[index, 0] = origin
            pos[index, 1] = y
            pos[index, 2] = x
            index += 1
    cap_at = image_attend
    pos[cap_at : cap_at + cap_len, 0] = np.arange(1, cap_len + 1, dtype=np.int32)
    if index != image_count:
        message = f"image tokens {index} do not match {image_count}"
        raise ValueError(message)
    return pos
