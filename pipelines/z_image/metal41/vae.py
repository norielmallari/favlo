"""FLUX VAE decoder and encoder.

The decoder latent is ``z / 0.3611 + 0.1159`` on device. GroupNorm uses 32
groups. The decoder epsilon is 1e-5 and the encoder epsilon is 1e-6. The
mid-block attention is one head across the 512 channels. The last decoder read
is uint8 HWC, ``rint((clip(x, -1, 1) + 1) * 127.5)``. The encoder returns the
scaled sample ``(z − 0.1159) · 0.3611``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from z_image.metal41.formats import INT8_TILE_N, TILE_M, TILE_N, pad_to, stored_k
from z_image.metal41.metal import Gpu
from z_image.metal41.ops import project
from z_image.metal41.record import Plane

SCALE = 0.3611
SHIFT = 0.1159
GROUPS = 32
EPS = 1e-5
ENCODER_EPS = 1e-6
CHUNK = 4096
_LOGVAR_MIN = -30.0
_LOGVAR_MAX = 20.0


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


def encode(
    gpu: Gpu,
    weights: dict[str, Plane],
    image: np.ndarray,
    generator: np.random.Generator,
    *,
    on_layer: Callable[[], None] | None = None,
) -> np.ndarray:
    """RGB uint8 HWC in, diffusion latent ``[16, h/8, w/8]`` out.

    The network writes 32 channels. The first 16 are the mean and the next 16
    are the log-variance, clamped to ``[-30, 20]``. The sample uses
    ``generator``, then ``(sample − shift) · scale``. A downsample pads one
    pixel on the right and the bottom, then a stride-2 convolution with no
    padding.
    """

    def mark() -> None:
        if on_layer is not None:
            on_layer()

    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        message = f"encoder expects uint8 HWC RGB, got {image.shape} {image.dtype}"
        raise ValueError(message)
    height, width, _ = image.shape
    if height % 8 or width % 8:
        message = f"encoder edges must be multiples of 8, got {width}x{height}"
        raise ValueError(message)
    flat = np.transpose(image.astype(np.float32) / np.float32(127.5) - np.float32(1.0), (2, 0, 1))
    flat = np.ascontiguousarray(flat)
    source = _Vol(gpu.empty(flat.size * 4), flat.size, 3, height, width)
    gpu.store_f32(source.buf, flat)
    bank: list[_Vol] = []
    ws: dict[str, tuple[Any, int]] = {}
    current = _conv(gpu, weights, bank, ws, source, "encoder.conv_in", 3, 1)
    _release(bank, source)
    mark()
    for index, down in ((0, True), (1, True), (2, True), (3, False)):
        for resnet in range(2):
            current = _resnet(
                gpu,
                weights,
                bank,
                ws,
                current,
                f"encoder.down_blocks.{index}.resnets.{resnet}",
                eps=ENCODER_EPS,
            )
            mark()
        if down:
            current = _downsample(
                gpu, weights, bank, ws, current, f"encoder.down_blocks.{index}.downsamplers.0.conv"
            )
            mark()
    current = _resnet(
        gpu, weights, bank, ws, current, "encoder.mid_block.resnets.0", eps=ENCODER_EPS
    )
    mark()
    current = _attend(
        gpu, weights, bank, current, "encoder.mid_block.attentions.0", eps=ENCODER_EPS
    )
    mark()
    current = _resnet(
        gpu, weights, bank, ws, current, "encoder.mid_block.resnets.1", eps=ENCODER_EPS
    )
    mark()
    normed = _norm_silu(gpu, weights, bank, current, "encoder.conv_norm_out", eps=ENCODER_EPS)
    _release(bank, current)
    mark()
    current = _conv(gpu, weights, bank, ws, normed, "encoder.conv_out", 3, 1)
    _release(bank, normed)
    mark()
    if current.channels != 32:
        message = f"encoder conv_out wrote {current.channels} channels"
        raise RuntimeError(message)
    moments = gpu.read_f32(current.buf, current.count).reshape(32, current.height, current.width)
    mean = moments[:16]
    logvar = np.clip(moments[16:], _LOGVAR_MIN, _LOGVAR_MAX)
    sample = mean + np.exp(np.float32(0.5) * logvar) * generator.standard_normal(mean.shape).astype(
        np.float32
    )
    scaled = (sample - np.float32(SHIFT)) * np.float32(SCALE)
    return np.ascontiguousarray(scaled, dtype=np.float32)


def _conv(
    gpu: Gpu,
    weights: dict[str, Plane],
    bank: list[_Vol],
    ws: dict[str, tuple[Any, int]],
    source: _Vol,
    name: str,
    kernel: int,
    pad: int,
    stride: int = 1,
    end_pad: int = 0,
) -> _Vol:
    weight = weights[f"{name}.weight"]
    bias = weights[f"{name}.bias"]
    out_ch = int(weight.shape[0])
    out_h = (source.height + end_pad + 2 * pad - kernel) // stride + 1
    out_w = (source.width + end_pad + 2 * pad - kernel) // stride + 1
    dest = _alloc(gpu, bank, out_ch, out_h, out_w)
    logical = int(np.prod(weight.shape[1:]))
    sk = weight.stored_k or stored_k(logical)
    tile_n = INT8_TILE_N if weight.plane == "int8" else TILE_N
    cols = weight.padded_n if weight.plane == "int8" else pad_to(out_ch, tile_n)
    act = _ws(gpu, ws, "act", CHUNK * sk * 2)
    out = _ws(gpu, ws, "out", CHUNK * cols * 4)
    pixels = out_h * out_w
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
            stride,
            out_w,
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


def _downsample(
    gpu: Gpu,
    weights: dict[str, Plane],
    bank: list[_Vol],
    ws: dict[str, tuple[Any, int]],
    source: _Vol,
    name: str,
) -> _Vol:
    """Stride-2 convolution. One zero column and one zero row sit on the end."""
    dest = _conv(gpu, weights, bank, ws, source, name, 3, 0, stride=2, end_pad=1)
    _release(bank, source)
    return dest


def _resnet(
    gpu: Gpu,
    weights: dict[str, Plane],
    bank: list[_Vol],
    ws: dict[str, tuple[Any, int]],
    source: _Vol,
    prefix: str,
    eps: float = EPS,
) -> _Vol:
    norm1 = _norm_silu(gpu, weights, bank, source, f"{prefix}.norm1", eps=eps)
    conv1 = _conv(gpu, weights, bank, ws, norm1, f"{prefix}.conv1", 3, 1)
    _release(bank, norm1)
    norm2 = _norm_silu(gpu, weights, bank, conv1, f"{prefix}.norm2", eps=eps)
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
    gpu: Gpu,
    weights: dict[str, Plane],
    bank: list[_Vol],
    source: _Vol,
    name: str,
    eps: float = EPS,
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
        eps,
    )
    gpu.silu_f32(dest.buf, dest.buf, dest.count)
    return dest


def _upsample(gpu: Gpu, bank: list[_Vol], source: _Vol) -> _Vol:
    dest = _alloc(gpu, bank, source.channels, source.height * 2, source.width * 2)
    gpu.nearest_up(source.buf, dest.buf, source.channels, source.height, source.width)
    return dest


def _attend(
    gpu: Gpu,
    weights: dict[str, Plane],
    bank: list[_Vol],
    source: _Vol,
    prefix: str,
    eps: float = EPS,
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
        eps,
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
