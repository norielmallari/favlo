"""Wan2.2 causal VAE. The checkpoint stays float32.

A convolution of at least 32 outputs, 1024 spatial sites, and 64 taps is
im2col plus ``matmul_f16``. Smaller convolutions use float ``conv3d``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from wan22_ti2v_5b.checkpoint import Archive, load_state
from wan22_ti2v_5b.gpu import Gpu

_MEAN = np.array(
    [
        -0.2289, -0.0052, -0.1323, -0.2339, -0.2799, 0.0174, 0.1838, 0.1557,
        -0.1382, 0.0542, 0.2813, 0.0891, 0.157, -0.0098, 0.0375, -0.1825,
        -0.2246, -0.1207, -0.0698, 0.5109, 0.2665, -0.2108, -0.2158, 0.2502,
        -0.2055, -0.0322, 0.1109, 0.1567, -0.0729, 0.0899, -0.2799, -0.123,
        -0.0313, -0.1649, 0.0117, 0.0723, -0.2839, -0.2083, -0.052, 0.3748,
        0.0152, 0.1957, 0.1433, -0.2944, 0.3573, -0.0548, -0.1681, -0.0667,
    ],
    dtype=np.float32,
)
_STD = np.array(
    [
        0.4765, 1.0364, 0.4514, 1.1677, 0.5313, 0.499, 0.4818, 0.5013,
        0.8158, 1.0344, 0.5894, 1.0901, 0.6885, 0.6165, 0.8454, 0.4978,
        0.5759, 0.3523, 0.7135, 0.6804, 0.5833, 1.4146, 0.8986, 0.5659,
        0.7069, 0.5338, 0.4889, 0.4917, 0.4069, 0.4999, 0.6866, 0.4093,
        0.5709, 0.6065, 0.6415, 0.4944, 0.5726, 1.2042, 0.5458, 1.6887,
        0.3971, 1.06, 0.3943, 0.5537, 0.5444, 0.4089, 0.7468, 0.7744,
    ],
    dtype=np.float32,
)


class WanVae:
    def __init__(self, path: Path, gpu: Gpu) -> None:
        self.gpu = gpu
        self._archive: Archive | None = None
        self.path = path

    def close(self) -> None:
        if self._archive is not None:
            self._archive.close()
            self._archive = None

    def encode(self, frame: np.ndarray) -> np.ndarray:
        """One RGB frame in ``[-1, 1]``, HWC, to a normalized latent ``[48, 1, h, w]``."""
        archive = self._open()
        pixels = _patch(frame)
        cache = _Cache()
        hidden = self._encode_chunk(archive, pixels, cache)
        moments = self._conv(archive, hidden, "conv1", causal=True)
        mean = moments[:48]
        return (mean - _MEAN.reshape(48, 1, 1, 1)) / _STD.reshape(48, 1, 1, 1)

    def decode(self, latent: np.ndarray) -> np.ndarray:
        """Normalized latent ``[48, T, H, W]`` to RGB ``[T_pix, H*16, W*16, 3]`` in ``[-1, 1]``."""
        archive = self._open()
        hidden = latent * _STD.reshape(48, 1, 1, 1) + _MEAN.reshape(48, 1, 1, 1)
        hidden = self._conv(archive, hidden, "conv2", causal=True)
        cache = _Cache()
        pieces = []
        for index in range(hidden.shape[1]):
            cache.rewind()
            pieces.append(
                self._decode_chunk(archive, hidden[:, index : index + 1], cache, first=index == 0)
            )
        folded = np.concatenate(pieces, axis=1)
        return np.clip(_unpatch(folded), -1.0, 1.0)

    def _open(self) -> Archive:
        if self._archive is None:
            if not self.path.is_file():
                message = f"VAE checkpoint is missing: {self.path}"
                raise FileNotFoundError(message)
            self._archive = load_state(self.path)
        return self._archive

    def _encode_chunk(self, archive: Archive, hidden: np.ndarray, cache: _Cache) -> np.ndarray:
        cache.rewind()
        hidden = self._cached(archive, hidden, "encoder.conv1", cache)
        stages = (
            (160, False, True),
            (320, True, True),
            (640, True, True),
            (640, False, False),
        )
        for index, (out_channels, time, space) in enumerate(stages):
            entered = hidden
            stem = f"encoder.downsamples.{index}.downsamples"
            for block in range(2):
                hidden = self._residual(archive, hidden, f"{stem}.{block}", cache)
            if space:
                hidden = self._down(archive, hidden, f"{stem}.2", time=time, cache=cache)
            factor_t = 2 if time else 1
            factor_s = 2 if space else 1
            hidden = hidden + _avg_down(entered, out_channels, factor_t, factor_s)
        hidden = self._residual(archive, hidden, "encoder.middle.0", cache)
        hidden = self._attend(archive, hidden, "encoder.middle.1")
        hidden = self._residual(archive, hidden, "encoder.middle.2", cache)
        hidden = _silu(_rms(hidden, archive.tensor("encoder.head.0.gamma")))
        return self._cached(archive, hidden, "encoder.head.2", cache)

    def _decode_chunk(
        self,
        archive: Archive,
        hidden: np.ndarray,
        cache: _Cache,
        *,
        first: bool,
    ) -> np.ndarray:
        cache.rewind()
        hidden = self._cached(archive, hidden, "decoder.conv1", cache)
        hidden = self._residual(archive, hidden, "decoder.middle.0", cache)
        hidden = self._attend(archive, hidden, "decoder.middle.1")
        hidden = self._residual(archive, hidden, "decoder.middle.2", cache)
        stages = (
            (1024, True),
            (1024, True),
            (512, False),
            (256, False),
        )
        for index, (out_channels, time) in enumerate(stages):
            entered = hidden
            stem = f"decoder.upsamples.{index}.upsamples"
            for block in range(3):
                hidden = self._residual(archive, hidden, f"{stem}.{block}", cache)
            if index < 3:
                hidden = self._up(archive, hidden, f"{stem}.3", time=time, cache=cache)
                hidden = hidden + _dup_up(entered, out_channels, 2 if time else 1, 2, first=first)
        hidden = _silu(_rms(hidden, archive.tensor("decoder.head.0.gamma")))
        return self._cached(archive, hidden, "decoder.head.2", cache)

    def _residual(
        self, archive: Archive, hidden: np.ndarray, prefix: str, cache: _Cache
    ) -> np.ndarray:
        updated = _silu(_rms(hidden, archive.tensor(f"{prefix}.residual.0.gamma")))
        updated = self._cached(archive, updated, f"{prefix}.residual.2", cache)
        updated = _silu(_rms(updated, archive.tensor(f"{prefix}.residual.3.gamma")))
        updated = self._cached(archive, updated, f"{prefix}.residual.6", cache)
        shortcut = f"{prefix}.shortcut.weight"
        if shortcut in archive.keys():
            hidden = self._conv(archive, hidden, f"{prefix}.shortcut", causal=True)
        return updated + hidden

    def _cached(self, archive: Archive, hidden: np.ndarray, name: str, cache: _Cache) -> np.ndarray:
        slot = cache.slot()
        previous = _remember(cache, slot, hidden)
        weight = archive.tensor(f"{name}.weight")
        pad_t = int(weight.shape[2]) - 1
        data = hidden
        if isinstance(previous, np.ndarray) and pad_t > 0:
            data = np.concatenate((previous, hidden), axis=1)
            pad_t -= int(previous.shape[1])
        return self._conv(
            archive,
            data,
            name,
            causal=True,
            padding=(pad_t, int(weight.shape[3]) // 2, int(weight.shape[4]) // 2),
            out_size=(int(hidden.shape[1]), int(hidden.shape[2]), int(hidden.shape[3])),
        )

    def _down(
        self,
        archive: Archive,
        hidden: np.ndarray,
        prefix: str,
        *,
        time: bool,
        cache: _Cache,
    ) -> np.ndarray:
        padded = np.pad(hidden, ((0, 0), (0, 0), (0, 1), (0, 1)))
        weight = archive.tensor(f"{prefix}.resample.1.weight")
        bias = archive.tensor(f"{prefix}.resample.1.bias")
        hidden = self.gpu.conv3d(
            padded,
            weight,
            bias,
            stride=(1, 2, 2),
            padding=(0, 0, 0),
            out_size=(hidden.shape[1], hidden.shape[2] // 2, hidden.shape[3] // 2),
        )
        if not time:
            return hidden
        slot = cache.slot()
        previous = cache.slots[slot]
        if previous is None:
            cache.slots[slot] = np.ascontiguousarray(hidden)
            return hidden
        cache.slots[slot] = np.ascontiguousarray(hidden[:, -1:])
        joined = np.concatenate((previous[:, -1:], hidden), axis=1)
        out_t = (joined.shape[1] - 3) // 2 + 1
        return self._conv(
            archive,
            joined,
            f"{prefix}.time_conv",
            causal=False,
            stride=(2, 1, 1),
            padding=(0, 0, 0),
            out_size=(out_t, hidden.shape[2], hidden.shape[3]),
        )

    def _up(
        self,
        archive: Archive,
        hidden: np.ndarray,
        prefix: str,
        *,
        time: bool,
        cache: _Cache,
    ) -> np.ndarray:
        if time:
            hidden = self._time_up(archive, hidden, prefix, cache)
        _channels, frames, height, width = hidden.shape
        spatial = np.repeat(np.repeat(hidden, 2, axis=2), 2, axis=3)
        weight = archive.tensor(f"{prefix}.resample.1.weight")
        bias = archive.tensor(f"{prefix}.resample.1.bias")
        return self.gpu.conv3d(
            spatial,
            weight,
            bias,
            stride=(1, 1, 1),
            padding=(0, 1, 1),
            out_size=(frames, height * 2, width * 2),
        )

    def _time_up(
        self, archive: Archive, hidden: np.ndarray, prefix: str, cache: _Cache
    ) -> np.ndarray:
        slot = cache.slot()
        previous = cache.slots[slot]
        if previous is None:
            cache.slots[slot] = "Rep"
            return hidden
        tail = hidden[:, -_CACHE_T:]
        if tail.shape[1] < _CACHE_T and isinstance(previous, str):
            tail = np.concatenate((np.zeros_like(tail), tail), axis=1)
        elif tail.shape[1] < _CACHE_T and isinstance(previous, np.ndarray):
            tail = np.concatenate((previous[:, -1:], tail), axis=1)
        data = hidden
        pad_t = 2
        if isinstance(previous, np.ndarray):
            data = np.concatenate((previous, hidden), axis=1)
            pad_t -= int(previous.shape[1])
        doubled = self._conv(
            archive,
            data,
            f"{prefix}.time_conv",
            causal=True,
            padding=(pad_t, 0, 0),
            out_size=(int(hidden.shape[1]), int(hidden.shape[2]), int(hidden.shape[3])),
        )
        cache.slots[slot] = np.ascontiguousarray(tail)
        channels = int(doubled.shape[0]) // 2
        left, right = doubled[:channels], doubled[channels:]
        stacked = np.stack((left, right), axis=2)
        return stacked.reshape(channels, hidden.shape[1] * 2, hidden.shape[2], hidden.shape[3])

    def _attend(self, archive: Archive, hidden: np.ndarray, prefix: str) -> np.ndarray:
        channels, frames, height, width = hidden.shape
        gamma = archive.tensor(f"{prefix}.norm.gamma")
        qkv_w = archive.tensor(f"{prefix}.to_qkv.weight").reshape(channels * 3, channels)
        qkv_b = archive.tensor(f"{prefix}.to_qkv.bias")
        proj_w = archive.tensor(f"{prefix}.proj.weight").reshape(channels, channels)
        proj_b = archive.tensor(f"{prefix}.proj.bias")
        pieces = []
        for frame in range(frames):
            plane = hidden[:, frame]
            normed = _rms(plane, gamma.reshape(channels, 1, 1))
            flat = normed.reshape(channels, -1).T.astype(np.float16)
            mixed = self.gpu.linear(flat, qkv_w.astype(np.float16), None, kind="f16", bias=qkv_b)
            query, key, value = np.split(mixed, 3, axis=1)
            scale = channels**-0.5
            scores = self.gpu.gemm_half(
                query.astype(np.float16),
                key.astype(np.float16),
                transpose_right=True,
            )
            scores = scores * np.float32(scale)
            weights = _softmax(scores).astype(np.float16)
            mixed = self.gpu.gemm_half(weights, value.astype(np.float16), transpose_right=False)
            projected = self.gpu.linear(
                mixed.astype(np.float16),
                proj_w.astype(np.float16),
                None,
                kind="f16",
                bias=proj_b,
            )
            pieces.append(plane + projected.T.reshape(channels, height, width))
        return np.stack(pieces, axis=1)

    def _conv(
        self,
        archive: Archive,
        hidden: np.ndarray,
        name: str,
        *,
        causal: bool,
        stride: tuple[int, int, int] = (1, 1, 1),
        padding: tuple[int, int, int] | None = None,
        out_size: tuple[int, int, int] | None = None,
    ) -> np.ndarray:
        weight = archive.tensor(f"{name}.weight")
        bias = archive.tensor(f"{name}.bias")
        if weight.ndim == 5 and padding is None:
            pad_t = int(weight.shape[2]) - 1 if causal else int(weight.shape[2]) // 2
            pad_h = int(weight.shape[3]) // 2
            pad_w = int(weight.shape[4]) // 2
            padding = (pad_t, pad_h, pad_w)
            if causal and out_size is None and stride == (1, 1, 1):
                out_size = (hidden.shape[1], hidden.shape[2], hidden.shape[3])
        if padding is None:
            padding = (0, 0, 0)
        return self.gpu.conv3d(
            hidden, weight, bias, stride=stride, padding=padding, out_size=out_size
        )


def _patch(frame: np.ndarray) -> np.ndarray:
    """Group each 2×2. Channel order is color, then the width offset, then the height offset."""
    height, width, _channels = frame.shape
    image = frame.astype(np.float32).reshape(height // 2, 2, width // 2, 2, 3)
    image = np.transpose(image, (4, 3, 1, 0, 2))
    folded = image.reshape(12, height // 2, width // 2)
    return folded[:, None, :, :]


def _unpatch(folded: np.ndarray) -> np.ndarray:
    _channels, frames, height, width = folded.shape
    image = folded.reshape(3, 2, 2, frames, height, width)
    image = np.transpose(image, (3, 4, 2, 5, 1, 0))
    return image.reshape(frames, height * 2, width * 2, 3)


def _rms(values: np.ndarray, gamma: np.ndarray) -> np.ndarray:
    scale = gamma.reshape((values.shape[0],) + (1,) * (values.ndim - 1))
    square = values.astype(np.float32) ** 2
    inv = np.reciprocal(np.sqrt(np.mean(square, axis=0, keepdims=True) + np.float32(1e-6)))
    return values.astype(np.float32) * inv * scale


def _silu(values: np.ndarray) -> np.ndarray:
    data = values.astype(np.float32)
    return data / (1.0 + np.exp(-data))


_CACHE_T = 2


class _Cache:
    def __init__(self) -> None:
        self.slots: list = []
        self.index = 0

    def rewind(self) -> None:
        self.index = 0

    def slot(self) -> int:
        index = self.index
        if index == len(self.slots):
            self.slots.append(None)
        self.index += 1
        return index


def _remember(cache: _Cache, slot: int, hidden: np.ndarray) -> object:
    previous = cache.slots[slot]
    tail = hidden[:, -_CACHE_T:]
    if tail.shape[1] < _CACHE_T and isinstance(previous, np.ndarray):
        tail = np.concatenate((previous[:, -1:], tail), axis=1)
    cache.slots[slot] = np.ascontiguousarray(tail)
    return previous


def _avg_down(hidden: np.ndarray, out_channels: int, factor_t: int, factor_s: int) -> np.ndarray:
    """Group the block input and average. Time is padded on the past side."""
    channels, frames, height, width = hidden.shape
    pad_t = (factor_t - frames % factor_t) % factor_t
    if pad_t:
        hidden = np.pad(hidden, ((0, 0), (pad_t, 0), (0, 0), (0, 0)))
        frames = int(hidden.shape[1])
    grouped = hidden.reshape(
        channels,
        frames // factor_t,
        factor_t,
        height // factor_s,
        factor_s,
        width // factor_s,
        factor_s,
    )
    grouped = np.transpose(grouped, (0, 2, 4, 6, 1, 3, 5))
    factor = factor_t * factor_s * factor_s
    flat = grouped.reshape(
        channels * factor, frames // factor_t, height // factor_s, width // factor_s
    )
    group = channels * factor // out_channels
    shaped = flat.reshape(out_channels, group, flat.shape[1], flat.shape[2], flat.shape[3])
    return shaped.mean(axis=1)


def _dup_up(
    hidden: np.ndarray,
    out_channels: int,
    factor_t: int,
    factor_s: int,
    *,
    first: bool,
) -> np.ndarray:
    """Repeat the block input into the upsampled grid."""
    in_channels = int(hidden.shape[0])
    repeats = out_channels * factor_t * factor_s * factor_s // in_channels
    repeated = np.repeat(hidden, repeats, axis=0)
    frames, height, width = (int(v) for v in repeated.shape[1:])
    viewed = repeated.reshape(out_channels, factor_t, factor_s, factor_s, frames, height, width)
    viewed = np.transpose(viewed, (0, 4, 1, 5, 2, 6, 3))
    expanded = viewed.reshape(
        out_channels,
        frames * factor_t,
        height * factor_s,
        width * factor_s,
    )
    if first:
        return expanded[:, factor_t - 1 :]
    return expanded


def _softmax(scores: np.ndarray) -> np.ndarray:
    peak = np.max(scores, axis=1, keepdims=True)
    weights = np.exp(scores - peak)
    return weights / np.sum(weights, axis=1, keepdims=True)
