"""FLUX VAE decoder.

The latent is ``z / 0.3611 + 0.1159`` on device. GroupNorm uses 32 groups and
epsilon 1e-5. The mid-block attention is one head across the 512 channels.
The last read is uint8 HWC, ``rint((clip(x, -1, 1) + 1) * 127.5)``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from z_image_turbo.formats import INT8_TILE_N, TILE_M, TILE_N, pad_to, stored_k
from z_image_turbo.metal import Gpu
from z_image_turbo.ops import project
from z_image_turbo.record import Plane

SCALE = 0.3611
SHIFT = 0.1159
GROUPS = 32
EPS = 1e-5
CHUNK = 4096


class _Vol:
    def __init__(self, buf: Any, cap: int, channels: int, height: int, width: int) -> None:
        self.buf = buf
        self.cap = cap
        self.channels = channels
        self.height = height
        self.width = width

    @property
    def count(self) -> int:
        return self.channels * self.height * self.width


def decode(
    gpu: Gpu,
    weights: dict[str, Plane],
    latent: Any,
    height: int,
    width: int,
    *,
    on_layer: Callable[[], None] | None = None,
) -> np.ndarray:
    """``latent`` is device ``[16, h, w]``. The return is uint8 ``[H, W, 3]``."""

    def mark() -> None:
        if on_layer is not None:
            on_layer()

    gpu.affine(latent, SCALE, SHIFT, 16 * height * width)
    bank: list[_Vol] = []
    ws: dict[str, tuple[Any, int]] = {}
    current = _Vol(latent, 16 * height * width, 16, height, width)
    produced = _conv(gpu, weights, bank, ws, current, "decoder.conv_in", 3, 1)
    _release(bank, current)
    mark()
    current = _resnet(gpu, weights, bank, ws, produced, "decoder.mid_block.resnets.0")
    mark()
    current = _attend(gpu, weights, bank, current, "decoder.mid_block.attentions.0")
    mark()
    current = _resnet(gpu, weights, bank, ws, current, "decoder.mid_block.resnets.1")
    mark()
    for index, upsample in ((0, True), (1, True), (2, True), (3, False)):
        for resnet in range(3):
            current = _resnet(
                gpu, weights, bank, ws, current, f"decoder.up_blocks.{index}.resnets.{resnet}"
            )
            mark()
        if upsample:
            bigger = _upsample(gpu, bank, current)
            _release(bank, current)
            mark()
            current = _conv(
                gpu,
                weights,
                bank,
                ws,
                bigger,
                f"decoder.up_blocks.{index}.upsamplers.0.conv",
                3,
                1,
            )
            _release(bank, bigger)
            mark()
    normed = _norm_silu(gpu, weights, bank, current, "decoder.conv_norm_out")
    _release(bank, current)
    mark()
    current = _conv(gpu, weights, bank, ws, normed, "decoder.conv_out", 3, 1)
    _release(bank, normed)
    mark()
    pixels = gpu.empty(current.height * current.width * 3)
    gpu.clip_u8(current.buf, pixels, current.height, current.width)
    mark()
    raw = gpu.read_u8(pixels, current.height * current.width * 3)
    return raw.reshape(current.height, current.width, 3)


def _conv(
    gpu: Gpu,
    weights: dict[str, Plane],
    bank: list[_Vol],
    ws: dict[str, tuple[Any, int]],
    source: _Vol,
    name: str,
    kernel: int,
    pad: int,
) -> _Vol:
    weight = weights[f"{name}.weight"]
    bias = weights[f"{name}.bias"]
    out_ch = int(weight.shape[0])
    dest = _alloc(gpu, bank, out_ch, source.height, source.width)
    logical = int(np.prod(weight.shape[1:]))
    sk = weight.stored_k or stored_k(logical)
    tile_n = INT8_TILE_N if weight.plane == "int8" else TILE_N
    cols = weight.padded_n if weight.plane == "int8" else pad_to(out_ch, tile_n)
    act = _ws(gpu, ws, "act", CHUNK * sk * 2)
    out = _ws(gpu, ws, "out", CHUNK * cols * 4)
    pixels = source.height * source.width
    for origin in range(0, pixels, CHUNK):
        count = min(CHUNK, pixels - origin)
        rows = pad_to(count, TILE_M)
        gpu.zero(act, rows * sk * 2)
        gpu.im2col(
            source.buf,
            act,
            source.channels,
            source.height,
            source.width,
            kernel,
            pad,
            sk,
            origin,
            count,
        )
        if weight.plane == "int8":
            gpu.project_i8(
                act,
                "f16",
                weight.data,
                weight.scale,
                None,
                out,
                rows,
                out_ch,
                logical,
                sk,
                cols,
                sk,
                dest_stride=cols,
                acc=out,
            )
        else:
            gpu.matmul(act, weight.data, weight.scale, out, rows, cols, sk)
        gpu.scatter_bias(
            out,
            bias.data,
            dest.buf,
            count,
            out_ch,
            cols,
            dest.height,
            dest.width,
            origin,
        )
    return dest


def _resnet(
    gpu: Gpu,
    weights: dict[str, Plane],
    bank: list[_Vol],
    ws: dict[str, tuple[Any, int]],
    source: _Vol,
    prefix: str,
) -> _Vol:
    norm1 = _norm_silu(gpu, weights, bank, source, f"{prefix}.norm1")
    conv1 = _conv(gpu, weights, bank, ws, norm1, f"{prefix}.conv1", 3, 1)
    _release(bank, norm1)
    norm2 = _norm_silu(gpu, weights, bank, conv1, f"{prefix}.norm2")
    _release(bank, conv1)
    conv2 = _conv(gpu, weights, bank, ws, norm2, f"{prefix}.conv2", 3, 1)
    _release(bank, norm2)
    shortcut = f"{prefix}.conv_shortcut.weight"
    if shortcut in weights:
        skip = _conv(gpu, weights, bank, ws, source, f"{prefix}.conv_shortcut", 1, 0)
    else:
        skip = source
    dest = _alloc(gpu, bank, conv2.channels, conv2.height, conv2.width)
    gpu.add_nchw(skip.buf, conv2.buf, dest.buf, dest.count)
    _release(bank, conv2)
    _release(bank, skip)
    if skip is not source:
        _release(bank, source)
    return dest


def _norm_silu(
    gpu: Gpu, weights: dict[str, Plane], bank: list[_Vol], source: _Vol, name: str
) -> _Vol:
    dest = _alloc(gpu, bank, source.channels, source.height, source.width)
    gpu.group_norm(
        source.buf,
        weights[f"{name}.weight"].data,
        weights[f"{name}.bias"].data,
        dest.buf,
        source.channels,
        source.height,
        source.width,
        GROUPS,
        EPS,
    )
    gpu.silu_f32(dest.buf, dest.buf, dest.count)
    return dest


def _upsample(gpu: Gpu, bank: list[_Vol], source: _Vol) -> _Vol:
    dest = _alloc(gpu, bank, source.channels, source.height * 2, source.width * 2)
    gpu.nearest_up(source.buf, dest.buf, source.channels, source.height, source.width)
    return dest


def _attend(
    gpu: Gpu, weights: dict[str, Plane], bank: list[_Vol], source: _Vol, prefix: str
) -> _Vol:
    normed = _alloc(gpu, bank, source.channels, source.height, source.width)
    gpu.group_norm(
        source.buf,
        weights[f"{prefix}.group_norm.weight"].data,
        weights[f"{prefix}.group_norm.bias"].data,
        normed.buf,
        source.channels,
        source.height,
        source.width,
        GROUPS,
        EPS,
    )
    pixels = source.height * source.width
    rows = pad_to(pixels, TILE_M)
    width = source.channels
    flat = gpu.empty(rows * width * 4)
    out = gpu.empty(rows * pad_to(width, TILE_N) * 4)
    q = gpu.empty(rows * width * 4)
    k = gpu.empty(rows * width * 4)
    v = gpu.empty(rows * width * 4)
    attn = gpu.empty(rows * width * 4)
    gpu.nchw_to_f32(normed.buf, flat, width, source.height, source.width, rows)
    for kind, dest in (("to_q", q), ("to_k", k), ("to_v", v)):
        project(
            gpu,
            flat,
            weights[f"{prefix}.{kind}.weight"],
            weights[f"{prefix}.{kind}.bias"],
            out,
            dest,
            rows,
            width,
        )
    gpu.attend(q, k, v, attn, rows, pixels, 1, 1, width, False)
    project(
        gpu,
        attn,
        weights[f"{prefix}.to_out.0.weight"],
        weights[f"{prefix}.to_out.0.bias"],
        out,
        q,
        rows,
        width,
    )
    dest = _alloc(gpu, bank, source.channels, source.height, source.width)
    gpu.rows_add_nchw(q, source.buf, dest.buf, width, source.height, source.width)
    _release(bank, normed)
    _release(bank, source)
    return dest


def _ws(gpu: Gpu, ws: dict[str, tuple[Any, int]], key: str, nbytes: int) -> Any:
    slot = ws.get(key)
    if slot is None or slot[1] < nbytes:
        buf = gpu.empty(nbytes)
        ws[key] = (buf, nbytes)
        return buf
    return slot[0]


def _alloc(gpu: Gpu, bank: list[_Vol], channels: int, height: int, width: int) -> _Vol:
    need = channels * height * width
    for index, vol in enumerate(bank):
        if vol.cap >= need:
            found = bank.pop(index)
            found.channels = channels
            found.height = height
            found.width = width
            return found
    buf = gpu.empty(need * 4)
    return _Vol(buf, need, channels, height, width)


def _release(bank: list[_Vol], vol: _Vol) -> None:
    bank.append(vol)
