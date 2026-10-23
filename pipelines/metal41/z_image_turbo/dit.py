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
    caption_len: int,
    sigma: float,
    ones: object,
    patch_h: int,
    patch_w: int,
) -> None:
    """Image tokens are already in ``scratch.tokens``. The caption is in ``scratch.cap``.

    The predicted velocity is written into ``scratch.vel`` as ``[16, H, W]``.
    """
    image_count = patch_h * patch_w
    cap_len = int(caption_len)
    image_attend = _align(image_count)
    cap_attend = _align(cap_len)
    image_rows = pad_to(image_count, TILE_M)
    cap_rows = pad_to(cap_len, TILE_M)
    total = image_attend + cap_attend
    seq_rows = pad_to(total, TILE_M)
    t0 = cap_attend + 1
    gpu.image_positions(scratch.pos_noise, image_rows, patch_h, patch_w, t0)
    gpu.text_positions(scratch.pos_text, cap_rows, cap_len, 1)
    gpu.joined_positions(
        scratch.pos_main,
        seq_rows,
        image_count,
        image_attend,
        patch_w,
        t0,
        cap_len,
    )
    _embed_image(gpu, scratch, weights, image_rows)
    _fill_pad(gpu, scratch.image, weights, "x_pad_token", image_count, image_attend)
    _refine(
        gpu,
        scratch,
        weights,
        scratch.image,
        scratch.image2,
        image_rows,
        image_attend,
        scratch.pos_noise,
        sigma,
        ones,
        "noise_refiner",
        True,
    )
    gpu.zero(scratch.seq, cap_rows * DIT_HIDDEN * 4)
    gpu.copy_rows(scratch.cap, scratch.seq, DIT_HIDDEN, DIT_HIDDEN, DIT_HIDDEN, 0, 0, cap_len)
    _fill_pad(gpu, scratch.seq, weights, "cap_pad_token", cap_len, cap_attend)
    _refine(
        gpu,
        scratch,
        weights,
        scratch.seq,
        scratch.seq2,
        cap_rows,
        cap_attend,
        scratch.pos_text,
        sigma,
        ones,
        "context_refiner",
        False,
    )
    _join(gpu, scratch, cap_attend, image_attend, seq_rows)
    _refine(
        gpu,
        scratch,
        weights,
        scratch.seq,
        scratch.seq2,
        seq_rows,
        total,
        scratch.pos_main,
        sigma,
        ones,
        "layers",
        True,
        LAYERS,
    )
    _final(gpu, scratch, weights, sigma, ones, patch_h, patch_w, seq_rows)


def _embed_image(
    gpu: Gpu,
    scratch: Scratch,
    weights: dict[str, Plane],
    rows: int,
) -> None:
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
) -> None:
    gpu.fill_half_rows(weights[name].data, buf, DIT_HIDDEN, start, stop)


def _join(
    gpu: Gpu,
    scratch: Scratch,
    cap_len: int,
    image_count: int,
    seq_rows: int,
) -> None:
    # Park the refined caption. ``scratch.cap`` stays the original embedding.
    gpu.copy_rows(scratch.seq, scratch.out, DIT_HIDDEN, DIT_HIDDEN, DIT_HIDDEN, 0, 0, cap_len)
    gpu.zero(scratch.seq, seq_rows * DIT_HIDDEN * 4)
    gpu.copy_rows(scratch.image, scratch.seq, DIT_HIDDEN, DIT_HIDDEN, DIT_HIDDEN, 0, 0, image_count)
    gpu.copy_rows(
        scratch.out,
        scratch.seq,
        DIT_HIDDEN,
        DIT_HIDDEN,
        DIT_HIDDEN,
        0,
        image_count,
        cap_len,
    )


def _refine(
    gpu: Gpu,
    scratch: Scratch,
    weights: dict[str, Plane],
    src: object,
    dst: object,
    rows: int,
    limit: int,
    positions: object,
    sigma: float,
    ones: object,
    prefix: str,
    adaln: bool,
    count: int = 2,
) -> None:
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
    patch_h: int,
    patch_w: int,
    seq_rows: int,
) -> None:
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
    gpu.unpatchify(scratch.ffn, scratch.vel, 16, patch_h * 2, patch_w * 2, 64)


def _time(
    gpu: Gpu,
    scratch: Scratch,
    weights: dict[str, Plane],
    sigma: float,
    ones: object,
) -> object:
    gpu.zero(scratch.time, 64 * 256 * 4)
    gpu.sinusoid(scratch.time, float(timestep(sigma)), 256)
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
