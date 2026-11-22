"""AutoencoderKLFlux2. The mean is packed 2x2, then whitened with the stored BatchNorm."""

from __future__ import annotations

import numpy as np

from klein_4b.portable.store import Store

from .metal import Gpu, View

_SAME = (1, 1, 1, 1)
_NONE = (0, 0, 0, 0)
_DOWN = (0, 1, 0, 1)


def encode(gpu: Gpu, store: Store, plane: np.ndarray) -> tuple[View, int, int]:
    """Float16 tokens ``[grid, 128]`` on device, and the packed grid."""
    hidden = _encoder(gpu, store, gpu.copy_in(plane))
    mean = gpu.prefix_channels(hidden, 32)
    tokens = gpu.patch_pack(mean, gpu.vector("vae/bn.running_mean"), gpu.vector("vae/bn.running_var"))
    grid_h = int(mean.shape[1]) // 2
    grid_w = int(mean.shape[2]) // 2
    gpu.drop(hidden)
    return tokens, grid_h, grid_w


def decode(gpu: Gpu, store: Store, tokens: View, grid_h: int, grid_w: int) -> np.ndarray:
    """Float CHW in model range, from the target tokens."""
    latent = gpu.unpack_latent(tokens, gpu.vector("vae/bn.running_mean"), gpu.vector("vae/bn.running_var"), grid_h, grid_w)
    hidden = _replace(gpu, latent, gpu.conv(latent, "vae/post_quant_conv.weight", stride=1, pads=_NONE))
    hidden = _replace(gpu, hidden, gpu.conv(hidden, "vae/decoder.conv_in.weight", stride=1, pads=_SAME))
    _settle(gpu, store, "vae/post_quant_conv", "vae/decoder.conv_in")
    hidden = _mid(gpu, store, hidden, "vae/decoder.mid_block")
    _settle(gpu, store, "vae/decoder.mid_block")
    count = _count(store, "vae/decoder.up_blocks")
    for index in range(count):
        prefix = f"vae/decoder.up_blocks.{index}"
        hidden = _resnets(gpu, store, hidden, f"{prefix}.resnets")
        if store.has(f"{prefix}.upsamplers.0.conv.weight"):
            hidden = _upsample(gpu, hidden, f"{prefix}.upsamplers.0.conv.weight")
        _settle(gpu, store, prefix)
    normed = gpu.groupnorm(
        hidden,
        gpu.vector("vae/decoder.conv_norm_out.weight"),
        gpu.vector("vae/decoder.conv_norm_out.bias"),
        silu=True,
    )
    gpu.drop(hidden)
    hidden = gpu.conv(normed, "vae/decoder.conv_out.weight", stride=1, pads=_SAME)
    gpu.drop(normed)
    plane = gpu.copy_out(hidden)
    gpu.drop(hidden)
    return plane


def _encoder(gpu: Gpu, store: Store, hidden: View) -> View:
    hidden = _replace(gpu, hidden, gpu.conv(hidden, "vae/encoder.conv_in.weight", stride=1, pads=_SAME))
    count = _count(store, "vae/encoder.down_blocks")
    for index in range(count):
        prefix = f"vae/encoder.down_blocks.{index}"
        hidden = _resnets(gpu, store, hidden, f"{prefix}.resnets")
        if store.has(f"{prefix}.downsamplers.0.conv.weight"):
            hidden = _downsample(gpu, hidden, f"{prefix}.downsamplers.0.conv.weight")
    hidden = _mid(gpu, store, hidden, "vae/encoder.mid_block")
    normed = gpu.groupnorm(
        hidden,
        gpu.vector("vae/encoder.conv_norm_out.weight"),
        gpu.vector("vae/encoder.conv_norm_out.bias"),
        silu=True,
    )
    gpu.drop(hidden)
    hidden = gpu.conv(normed, "vae/encoder.conv_out.weight", stride=1, pads=_SAME)
    gpu.drop(normed)
    quantized = gpu.conv(hidden, "vae/quant_conv.weight", stride=1, pads=_NONE)
    gpu.drop(hidden)
    return quantized


def _mid(gpu: Gpu, store: Store, hidden: View, prefix: str) -> View:
    hidden = _resnet(gpu, store, hidden, f"{prefix}.resnets.0")
    if store.has(f"{prefix}.attentions.0.to_q.weight"):
        hidden = _attend(gpu, hidden, f"{prefix}.attentions.0")
    index = 1
    while store.has(f"{prefix}.resnets.{index}.conv1.weight"):
        hidden = _resnet(gpu, store, hidden, f"{prefix}.resnets.{index}")
        index += 1
    return hidden


def _resnets(gpu: Gpu, store: Store, hidden: View, prefix: str) -> View:
    index = 0
    while store.has(f"{prefix}.{index}.conv1.weight"):
        hidden = _resnet(gpu, store, hidden, f"{prefix}.{index}")
        index += 1
    if index == 0:
        message = f"{prefix} has no resnets"
        raise RuntimeError(message)
    return hidden


def _resnet(gpu: Gpu, store: Store, hidden: View, prefix: str) -> View:
    first = _activated_conv(gpu, hidden, prefix, "norm1", "conv1")
    branch = _activated_conv(gpu, first, prefix, "norm2", "conv2")
    gpu.drop(first)
    if store.has(f"{prefix}.conv_shortcut.weight"):
        hidden = _replace(gpu, hidden, gpu.conv(hidden, f"{prefix}.conv_shortcut.weight", stride=1, pads=_NONE))
    out = gpu.add(hidden, branch)
    gpu.drop(hidden, branch)
    return out


def _activated_conv(gpu: Gpu, hidden: View, prefix: str, norm: str, conv_name: str) -> View:
    normed = gpu.groupnorm(
        hidden,
        gpu.vector(f"{prefix}.{norm}.weight"),
        gpu.vector(f"{prefix}.{norm}.bias"),
        silu=True,
    )
    out = gpu.conv(normed, f"{prefix}.{conv_name}.weight", stride=1, pads=_SAME)
    gpu.drop(normed)
    return out


def _attend(gpu: Gpu, hidden: View, prefix: str) -> View:
    channels, height, width = (int(hidden.shape[0]), int(hidden.shape[1]), int(hidden.shape[2]))
    normed = gpu.groupnorm(
        hidden,
        gpu.vector(f"{prefix}.group_norm.weight"),
        gpu.vector(f"{prefix}.group_norm.bias"),
        silu=False,
    )
    flat = gpu.seq_from_chw(normed)
    gpu.drop(normed)
    wide = flat.kind == "f32"
    kept = "f32" if wide else "f16"
    query = gpu.linear_view(flat, f"{prefix}.to_q.weight", store=kept)
    key = gpu.linear_view(flat, f"{prefix}.to_k.weight", store=kept)
    value = gpu.linear_view(flat, f"{prefix}.to_v.weight", store=kept)
    gpu.drop(flat)
    seq = height * width
    gpu.reshape(query, (seq, 1, channels))
    gpu.reshape(key, (seq, 1, channels))
    gpu.reshape(value, (seq, 1, channels))
    if wide:
        query_f, key_f, value_f = query, key, value
    else:
        query_f = gpu.cast(query, "f32")
        key_f = gpu.cast(key, "f32")
        value_f = gpu.cast(value, "f32")
        gpu.drop(query, key, value)
    mixed = gpu.attend_view(query_f, key_f, value_f, causal=False, valid=seq)
    gpu.drop(query_f, key_f, value_f)
    flat = mixed if wide else gpu.cast(mixed, "f16")
    if not wide:
        gpu.drop(mixed)
    gpu.reshape(flat, (seq, channels))
    out = gpu.linear_view(flat, f"{prefix}.to_out.0.weight", store=kept)
    gpu.drop(flat)
    volume = gpu.chw_from_seq(out, height, width)
    gpu.drop(out)
    added = gpu.add(hidden, volume)
    gpu.drop(hidden, volume)
    return added


def _downsample(gpu: Gpu, hidden: View, key: str) -> View:
    """Stride-2 conv. One zero is added on the bottom and the right."""
    return _replace(gpu, hidden, gpu.conv(hidden, key, stride=2, pads=_DOWN))


def _upsample(gpu: Gpu, hidden: View, key: str) -> View:
    doubled = gpu.nearest2(hidden)
    gpu.drop(hidden)
    out = gpu.conv(doubled, key, stride=1, pads=_SAME)
    gpu.drop(doubled)
    return out


def _settle(gpu: Gpu, store: Store, *prefixes: str) -> None:
    """Wait for this block, then drop its weights and its pooled gathers."""
    gpu.release_pool()
    for prefix in prefixes:
        gpu.drop_prefix(prefix)
        store.close_prefix(prefix)


def _replace(gpu: Gpu, previous: View, current: View) -> View:
    gpu.drop(previous)
    return current


def _count(store: Store, prefix: str) -> int:
    index = 0
    while store.has(f"{prefix}.{index}.resnets.0.conv1.weight"):
        index += 1
    return index
