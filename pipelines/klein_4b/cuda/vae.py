"""Flux2 VAE. The encoder volume is fp16. The decoder volume is fp32."""

from __future__ import annotations

import numpy as np

from klein_4b.cuda.constants import BN_EPS, VAE_CHANNELS, VAE_EPS, VAE_GROUPS
from klein_4b.cuda.ops import Ops, Plane, Volume
from klein_4b.cuda.store import Store, Tensor


def encode(ops: Ops, store: Store, pixels: np.ndarray) -> Plane:
    """``pixels`` is float CHW in ``[-1, 1]``. Tokens are ``[grid, 128]``."""
    volume = ops.upload_volume(pixels.astype(np.float16), fp32=False)
    volume = _swap_volume(ops, volume, ops.conv(volume, store.get("vae/encoder.conv_in.weight"), stride=1, pad=1))
    current = VAE_CHANNELS[0]
    for index, out_ch in enumerate(VAE_CHANNELS):
        for resnet_index in range(2):
            prefix = f"vae/encoder.down_blocks.{index}.resnets.{resnet_index}"
            volume = _swap_volume(ops, volume, _resnet(ops, store, volume, prefix, current, out_ch))
            current = out_ch
        if index != len(VAE_CHANNELS) - 1:
            volume = _swap_volume(ops, volume, _downsample(ops, store, volume, f"vae/encoder.down_blocks.{index}.downsamplers.0.conv.weight"))
    volume = _swap_volume(ops, volume, _resnet(ops, store, volume, "vae/encoder.mid_block.resnets.0", current, current))
    volume = _swap_volume(ops, volume, _attention(ops, store, volume, "vae/encoder.mid_block.attentions.0"))
    volume = _swap_volume(ops, volume, _resnet(ops, store, volume, "vae/encoder.mid_block.resnets.1", current, current))
    volume = _swap_volume(ops, volume, ops.groupnorm(volume, store.get("vae/encoder.conv_norm_out.weight"), groups=VAE_GROUPS, eps=VAE_EPS, silu=True))
    volume = _swap_volume(ops, volume, ops.conv(volume, store.get("vae/encoder.conv_out.weight"), stride=1, pad=1))
    volume = _swap_volume(ops, volume, ops.conv(volume, store.get("vae/quant_conv.weight"), stride=1, pad=0))
    mean = ops.prefix_channels(volume, 32)
    ops.free(volume)
    grid_h = mean.height // 2
    grid_w = mean.width // 2
    tokens = ops.pack_latent(mean, _stat(store, "running_mean"), _stat(store, "running_var"), grid_h, grid_w, BN_EPS)
    ops.free(mean)
    return tokens


def decode(ops: Ops, store: Store, tokens: Plane, grid_h: int, grid_w: int) -> np.ndarray:
    volume = ops.unpack_latent(tokens, _stat(store, "running_mean"), _stat(store, "running_var"), grid_h, grid_w, BN_EPS)
    volume = _swap_volume(ops, volume, ops.conv(volume, store.get("vae/post_quant_conv.weight"), stride=1, pad=0))
    volume = _swap_volume(ops, volume, ops.conv(volume, store.get("vae/decoder.conv_in.weight"), stride=1, pad=1))
    current = VAE_CHANNELS[-1]
    reversed_channels = list(reversed(VAE_CHANNELS))
    for index, out_ch in enumerate(reversed_channels):
        for resnet_index in range(3):
            cin = current if resnet_index == 0 else out_ch
            prefix = f"vae/decoder.up_blocks.{index}.resnets.{resnet_index}"
            volume = _swap_volume(ops, volume, _resnet(ops, store, volume, prefix, cin, out_ch))
            current = out_ch
        if index != len(reversed_channels) - 1:
            volume = _swap_volume(ops, volume, ops.nearest2(volume))
            volume = _swap_volume(
                ops,
                volume,
                ops.conv(volume, store.get(f"vae/decoder.up_blocks.{index}.upsamplers.0.conv.weight"), stride=1, pad=1),
            )
    volume = _swap_volume(ops, volume, ops.groupnorm(volume, store.get("vae/decoder.conv_norm_out.weight"), groups=VAE_GROUPS, eps=VAE_EPS, silu=True))
    volume = _swap_volume(ops, volume, ops.conv(volume, store.get("vae/decoder.conv_out.weight"), stride=1, pad=1))
    pixels = ops.download_volume(volume).astype(np.float32)
    ops.free(volume)
    store.free_prefix("vae/")
    return pixels


def _resnet(ops: Ops, store: Store, volume: Volume, prefix: str, cin: int, cout: int) -> Volume:
    hidden = ops.groupnorm(volume, store.get(f"{prefix}.norm1.weight"), groups=VAE_GROUPS, eps=VAE_EPS, silu=True)
    hidden = _swap_volume(ops, hidden, ops.conv(hidden, store.get(f"{prefix}.conv1.weight"), stride=1, pad=1))
    hidden = _swap_volume(ops, hidden, ops.groupnorm(hidden, store.get(f"{prefix}.norm2.weight"), groups=VAE_GROUPS, eps=VAE_EPS, silu=True))
    hidden = _swap_volume(ops, hidden, ops.conv(hidden, store.get(f"{prefix}.conv2.weight"), stride=1, pad=1))
    if cin != cout:
        skip = ops.conv(volume, store.get(f"{prefix}.conv_shortcut.weight"), stride=1, pad=0)
    else:
        skip = volume
    out = ops.add_volume(skip, hidden)
    ops.free(hidden)
    if skip is not volume:
        ops.free(skip)
    return out


def _downsample(ops: Ops, store: Store, volume: Volume, name: str) -> Volume:
    padded = ops.pad_edge(volume)
    out = ops.conv(padded, store.get(name), stride=2, pad=0)
    ops.free(padded)
    return out


def _attention(ops: Ops, store: Store, volume: Volume, prefix: str) -> Volume:
    norm = f"{prefix}.group_norm.weight"
    if not store.has(norm):
        norm = f"{prefix}.norm.weight"
    hidden = ops.groupnorm(volume, store.get(norm), groups=VAE_GROUPS, eps=VAE_EPS, silu=False)
    query = ops.conv(hidden, store.get(f"{prefix}.to_q.weight"), stride=1, pad=0)
    key = ops.conv(hidden, store.get(f"{prefix}.to_k.weight"), stride=1, pad=0)
    value = ops.conv(hidden, store.get(f"{prefix}.to_v.weight"), stride=1, pad=0)
    ops.free(hidden)
    q_tok = ops.to_tokens(query)
    k_tok = ops.to_tokens(key)
    v_tok = ops.to_tokens(value)
    ops.free(query, key, value)
    mixed = ops.attend(q_tok, k_tok, v_tok, q_tok.rows, 1, 1, q_tok.cols, 0, q_tok.rows)
    ops.free(q_tok, k_tok, v_tok)
    spatial = ops.to_volume(mixed, volume.height, volume.width, fp32=volume.fp32)
    ops.free(mixed)
    out_name = f"{prefix}.to_out.0.weight"
    if not store.has(out_name):
        out_name = f"{prefix}.to_out.weight"
    projected = ops.conv(spatial, store.get(out_name), stride=1, pad=0)
    ops.free(spatial)
    out = ops.add_volume(volume, projected)
    ops.free(projected)
    return out


def _stat(store: Store, name: str) -> Tensor:
    for key in (f"vae/bn.{name}", f"vae/batch_norm.{name}"):
        if store.has(key):
            return store.get(key)
    return store.get(f"vae/bn.{name}")


def _swap_volume(ops: Ops, old: Volume, new: Volume) -> Volume:
    ops.free(old)
    return new
