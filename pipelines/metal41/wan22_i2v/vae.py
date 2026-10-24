"""Wan 2.1 causal VAE.

The encoder walks the first pixel frame, then groups of four. The first chunk
skips each temporal convolution and only caches the feature. Later chunks
concatenate that cached frame and stride by 2. A kernel of width 3 keeps two
past frames and does not pad the future. The decoder walks one latent frame
at a time. Its first temporal upsample only marks the cache. The next chunk
upsamples without reading that frame. Later chunks do. The first latent frame
is expanded in space only. Later frames are expanded by four. The latent
affine is the published 16-channel mean and standard deviation.
"""

from __future__ import annotations

import io
import pickle
import time
import zipfile
from functools import lru_cache
from collections import OrderedDict
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from wan22_i2v.metal import Gpu
from wan22_i2v.schedule import LATENT_MEAN, LATENT_STD, geometry
from wan22_i2v.source import VAE_FILE, vae_dir

_EPS = np.float32(1e-12)
ENCODER_BLOCKS = 16
DECODER_BLOCKS = 20


class _FloatStorage:
    pass


def vae_path(root: Path | None = None) -> Path:
    return vae_dir(root) / VAE_FILE


@lru_cache(maxsize=2)
def vae_device_bytes(path: str) -> int:
    """Float32 bytes if every checkpoint tensor and the latent affine are on device."""
    weights = load_vae(Path(path))
    total = sum(int(array.nbytes) for array in weights.values())
    total += int(LATENT_MEAN.nbytes) + int(LATENT_STD.nbytes)
    return total


def encode_peak_bytes(height: int, width: int, lat_t: int, lat_h: int, lat_w: int) -> int:
    """Encoder output plus one RGB chunk while weights are resident."""
    latent = 16 * lat_t * lat_h * lat_w * 4
    chunk = 3 * 4 * height * width * 4
    return latent + chunk


def load_vae(path: Path) -> dict[str, np.ndarray]:
    """Read the float checkpoint. The unpickler accepts only tensor rebuilds."""
    archive = zipfile.ZipFile(path)
    storages: dict[str, np.ndarray] = {}

    class _Unpickler(pickle.Unpickler):
        def find_class(self, module: str, name: str) -> type:
            if name == "OrderedDict":
                return OrderedDict
            if name == "FloatStorage":
                return _FloatStorage
            if name == "_rebuild_tensor_v2":
                return _rebuild
            message = f"unexpected pickle type {module}.{name}"
            raise RuntimeError(message)

        def persistent_load(self, pid: tuple) -> np.ndarray:
            key = str(pid[2])
            count = int(pid[4])
            if key not in storages:
                raw = archive.read(f"wanx_vae/data/{key}")
                storages[key] = np.frombuffer(raw, dtype="<f4", count=count).copy()
            return storages[key]

    def _rebuild(
        storage: np.ndarray,
        offset: int,
        size: tuple[int, ...],
        stride: tuple[int, ...],
        requires_grad: bool,
        hooks: object,
        metadata: object = None,
    ) -> np.ndarray:
        del requires_grad, hooks, metadata
        view = np.lib.stride_tricks.as_strided(
            storage[int(offset) :],
            shape=tuple(int(v) for v in size),
            strides=tuple(int(v) * 4 for v in stride),
        )
        return np.array(view, dtype=np.float32, copy=True)

    try:
        weights = _Unpickler(io.BytesIO(archive.read("wanx_vae/data.pkl"))).load()
    finally:
        archive.close()
    if not isinstance(weights, dict):
        message = "VAE checkpoint is not a tensor map"
        raise RuntimeError(message)
    return {str(name): array for name, array in weights.items()}


class _Vol:
    """One device volume ``[C, T, H, W]``."""

    def __init__(self, buf: Any, shape: tuple[int, int, int, int]) -> None:
        self.buf = buf
        self.shape = shape

    def count(self) -> int:
        channels, time, height, width = self.shape
        return channels * time * height * width

    def plane(self) -> int:
        _, time, height, width = self.shape
        return time * height * width


class WanVAE:
    """Encode one still into the DiT latent and decode a latent back to pixels."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.weights: dict[str, np.ndarray] | None = None
        self.cache: dict[str, _Vol | str] = {}
        self.gpu: Gpu | None = None
        self.on_block: Callable[[], None] | None = None
        self._params: dict[str, tuple[Any, tuple[int, ...]]] = {}
        self._mean: Any = None
        self._std: Any = None
        self._own = False

    def _weights(self) -> dict[str, np.ndarray]:
        if self.weights is None:
            started = time.perf_counter()
            self.weights = load_vae(self.path)
            elapsed = time.perf_counter() - started
            print(f"vae load {elapsed:.2f}s", flush=True)
        return self.weights

    def encode(
        self,
        image: np.ndarray,
        frames: int,
        on_pipeline: Callable[[], None] | None = None,
        on_block: Callable[[], None] | None = None,
        gpu: Gpu | None = None,
    ) -> Any:
        """uint8 HWC image to a kept device latent ``[16, T, H, W]``."""
        height, width = int(image.shape[0]), int(image.shape[1])
        lat_t, lat_h, lat_w, _ = geometry(frames, height, width)
        started = time.perf_counter()
        self._open(gpu, on_pipeline, on_block)
        try:
            pieces: list[_Vol] = []
            starts = [0, *range(1, frames, 4)]
            for start in starts:
                count = 1 if start == 0 else min(4, frames - start)
                chunk = self._chunk(image, height, width, start, count)
                encoded = self._encoder(chunk)
                self._free(chunk)
                stats = self._project(encoded, "conv1.weight", "conv1.bias")
                self._free(encoded)
                pieces.append(self._mu(stats))
                self._free(stats)
            latent = pieces[0]
            for extra in pieces[1:]:
                joined = self._concat(latent, extra)
                self._free(latent)
                self._free(extra)
                latent = joined
            self.gpu.keep(latent.buf)
            self.gpu.end()
        finally:
            self._clear_cache()
            if not self._own:
                self.release_device()
            self._close()
        if latent.shape != (16, lat_t, lat_h, lat_w):
            message = f"encoder emitted {latent.shape}, expected {(16, lat_t, lat_h, lat_w)}"
            raise RuntimeError(message)
        elapsed = time.perf_counter() - started
        print(f"vae encode {elapsed:.2f}s", flush=True)
        return latent.buf

    def decode(
        self,
        latent: Any,
        shape: tuple[int, int, int, int],
        on_pipeline: Callable[[], None] | None = None,
        on_block: Callable[[], None] | None = None,
        gpu: Gpu | None = None,
        resident_frames: int | None = None,
    ) -> np.ndarray:
        """Device latent ``[16, T, H, W]`` to uint8 ``[frames, height, width, 3]``."""
        expected = 1 + (int(shape[1]) - 1) * 4
        started = time.perf_counter()
        self._open(gpu, on_pipeline, on_block)
        try:
            restored = self._inv(_Vol(latent, shape))
            z = self._project(restored, "conv2.weight", "conv2.bias")
            self._free(restored)
            pieces: list[_Vol] = []
            host: list[np.ndarray] = []
            window = shape[1] if resident_frames is None else max(1, min(resident_frames, shape[1]))
            for index in range(shape[1]):
                frame = self._slice(z, index, 1)
                pieces.append(self._decoder(frame))
                self._free(frame)
                if len(pieces) < window and index + 1 < shape[1]:
                    continue
                video = pieces[0]
                for extra in pieces[1:]:
                    joined = self._concat(video, extra)
                    self._free(video)
                    self._free(extra)
                    video = joined
                host.append(self._u8(video))
                self._free(video)
                pieces = []
            self._free(z)
            pixels = host[0] if len(host) == 1 else np.concatenate(host, axis=0)
            if pixels.shape[0] != expected:
                message = f"decoder emitted {pixels.shape[0]} frames, expected {expected}"
                raise RuntimeError(message)
        finally:
            self._clear_cache()
            if not self._own:
                self.release_device()
            self._close()
        elapsed = time.perf_counter() - started
        print(f"vae decode {elapsed:.2f}s", flush=True)
        return pixels

    def release_device(self) -> None:
        """Drop GPU copies of VAE weights. Host ``weights`` stay for a later stage."""
        gpu = self.gpu
        if gpu is None:
            return
        for buf, _ in list(self._params.values()):
            gpu.drop(buf)
        self._params.clear()
        if self._mean is not None:
            gpu.drop(self._mean)
            self._mean = None
        if self._std is not None:
            gpu.drop(self._std)
            self._std = None
        gpu.reclaim()

    def release_host(self) -> None:
        """Drop host checkpoint arrays after decode."""
        self.weights = {}

    def _open(
        self,
        gpu: Gpu | None,
        on_pipeline: Callable[[], None] | None,
        on_block: Callable[[], None] | None,
    ) -> None:
        self.on_block = on_block
        self.cache = {}
        if gpu is None:
            self.gpu = Gpu(on_pipeline=on_pipeline)
            self._own = True
        else:
            self.gpu = gpu
            self._own = False
        if self._mean is None:
            mean = np.ascontiguousarray(LATENT_MEAN, dtype=np.float32)
            std = np.ascontiguousarray(LATENT_STD, dtype=np.float32)
            self._mean = self.gpu.keep(self.gpu.upload(mean))
            self._std = self.gpu.keep(self.gpu.upload(std))

    def _close(self) -> None:
        self.on_block = None
        if self._own and self.gpu is not None:
            self.gpu.close()
            self._params.clear()
            self._mean = None
            self._std = None
        self.gpu = None
        self._own = False

    def _chunk(
        self, image: np.ndarray, height: int, width: int, start: int, count: int
    ) -> _Vol:
        video = np.zeros((3, count, height, width), dtype=np.float32)
        if start == 0:
            pixels = image.astype(np.float32) / np.float32(127.5) - np.float32(1.0)
            video[:, 0] = np.transpose(pixels, (2, 0, 1))
        return self._host(video)

    def _host(self, array: np.ndarray) -> _Vol:
        raw = np.ascontiguousarray(array, dtype=np.float32)
        gpu = self._gpu()
        return _Vol(gpu.upload(raw), tuple(int(v) for v in raw.shape))

    def _free(self, vol: _Vol | None) -> None:
        if vol is None or vol.buf is None:
            return
        self._gpu().release(vol.buf)
        vol.buf = None

    def _clear_cache(self) -> None:
        for value in self.cache.values():
            if isinstance(value, _Vol):
                self._free(value)
        self.cache = {}

    def _mark(self, vol: _Vol) -> _Vol:
        self._gpu().commit()
        mark = self.on_block
        if mark is not None:
            mark()
        return vol

    def _encoder(self, x: _Vol) -> _Vol:
        x = self._mark(self._causal("encoder.conv1", x))
        x = self._mark(self._residual(x, "encoder.downsamples.0"))
        x = self._mark(self._residual(x, "encoder.downsamples.1"))
        x = self._mark(self._down_spatial(x, "encoder.downsamples.2"))
        x = self._mark(self._residual(x, "encoder.downsamples.3"))
        x = self._mark(self._residual(x, "encoder.downsamples.4"))
        x = self._mark(self._down3d(x, "encoder.downsamples.5"))
        x = self._mark(self._residual(x, "encoder.downsamples.6"))
        x = self._mark(self._residual(x, "encoder.downsamples.7"))
        x = self._mark(self._down3d(x, "encoder.downsamples.8"))
        x = self._mark(self._residual(x, "encoder.downsamples.9"))
        x = self._mark(self._residual(x, "encoder.downsamples.10"))
        x = self._mark(self._residual(x, "encoder.middle.0"))
        x = self._mark(self._attend(x, "encoder.middle.1"))
        x = self._mark(self._residual(x, "encoder.middle.2"))
        return self._mark(self._head(x, "encoder.head"))

    def _decoder(self, x: _Vol) -> _Vol:
        x = self._mark(self._causal("decoder.conv1", x))
        x = self._mark(self._residual(x, "decoder.middle.0"))
        x = self._mark(self._attend(x, "decoder.middle.1"))
        x = self._mark(self._residual(x, "decoder.middle.2"))
        for index in range(3):
            x = self._mark(self._residual(x, f"decoder.upsamples.{index}"))
        x = self._mark(self._up3d(x, "decoder.upsamples.3"))
        for index in range(4, 7):
            x = self._mark(self._residual(x, f"decoder.upsamples.{index}"))
        x = self._mark(self._up3d(x, "decoder.upsamples.7"))
        for index in range(8, 11):
            x = self._mark(self._residual(x, f"decoder.upsamples.{index}"))
        x = self._mark(self._up_spatial(x, "decoder.upsamples.11"))
        for index in range(12, 15):
            x = self._mark(self._residual(x, f"decoder.upsamples.{index}"))
        return self._mark(self._head(x, "decoder.head"))

    def _residual(self, x: _Vol, prefix: str) -> _Vol:
        shortcut = f"{prefix}.shortcut.weight"
        skip = None
        if shortcut in self._weights():
            skip = self._project(x, shortcut, f"{prefix}.shortcut.bias")
        hidden = self._rms(x, f"{prefix}.residual.0.gamma")
        hidden = self._causal(f"{prefix}.residual.2", self._take(hidden, self._silu(hidden)))
        hidden = self._take(hidden, self._rms(hidden, f"{prefix}.residual.3.gamma"))
        hidden = self._take(hidden, self._silu(hidden))
        hidden = self._causal(f"{prefix}.residual.6", hidden)
        base = x if skip is None else skip
        out = self._add(base, hidden)
        self._free(hidden)
        self._free(x)
        if skip is not None:
            self._free(skip)
        return out

    def _head(self, x: _Vol, prefix: str) -> _Vol:
        normed = self._rms(x, f"{prefix}.0.gamma")
        hidden = self._silu(normed)
        self._free(normed)
        out = self._causal(f"{prefix}.2", hidden)
        self._free(x)
        return out

    def _attend(self, x: _Vol, prefix: str) -> _Vol:
        normed = self._rms(x, f"{prefix}.norm.gamma")
        qkv = self._project(normed, f"{prefix}.to_qkv.weight", f"{prefix}.to_qkv.bias")
        self._free(normed)
        mixed = self._spatial(qkv)
        self._free(qkv)
        projected = self._project(mixed, f"{prefix}.proj.weight", f"{prefix}.proj.bias")
        self._free(mixed)
        out = self._add(x, projected)
        self._free(projected)
        self._free(x)
        return out

    def _down_spatial(self, x: _Vol, prefix: str) -> _Vol:
        padded = self._pad(x)
        out = self._conv(f"{prefix}.resample.1", padded, (1, 2, 2), (0, 0, 0), time=True)
        self._free(padded)
        self._free(x)
        return out

    def _up_spatial(self, x: _Vol, prefix: str) -> _Vol:
        nearest = self._nearest(x)
        out = self._conv(f"{prefix}.resample.1", nearest, (1, 1, 1), (0, 1, 1), time=True)
        self._free(nearest)
        self._free(x)
        return out

    def _down3d(self, x: _Vol, prefix: str) -> _Vol:
        x = self._down_spatial(x, prefix)
        name = f"{prefix}.time_conv"
        if name not in self.cache:
            self.cache[name] = self._slice(x, 0, x.shape[1])
            return x
        cached = self.cache[name]
        if not isinstance(cached, _Vol):
            message = f"{name} cache is not a volume"
            raise RuntimeError(message)
        context = self._slice(cached, cached.shape[1] - 1, 1)
        self._free(cached)
        tail = self._slice(x, x.shape[1] - 1, 1)
        self.cache[name] = tail
        joined = self._concat(context, x)
        self._free(context)
        out = self._conv(name, joined, (2, 1, 1), (0, 0, 0))
        self._free(joined)
        self._free(x)
        return out

    def _up3d(self, x: _Vol, prefix: str) -> _Vol:
        name = f"{prefix}.time_conv"
        if name not in self.cache:
            self.cache[name] = "Rep"
            return self._up_spatial(x, prefix)
        cached = self.cache[name]
        repeat = not isinstance(cached, _Vol)
        tail = self._tail(x, cached if isinstance(cached, _Vol) else None, repeat)
        if repeat:
            expanded = self._conv(name, x, (1, 1, 1), (2, 0, 0))
        else:
            assert isinstance(cached, _Vol)
            joined = self._concat(cached, x)
            pad_t = 2 - int(cached.shape[1])
            expanded = self._conv(name, joined, (1, 1, 1), (pad_t, 0, 0))
            self._free(joined)
        if isinstance(cached, _Vol):
            self._free(cached)
        self.cache[name] = tail
        stacked = self._pairs(expanded)
        self._free(expanded)
        self._free(x)
        return self._up_spatial(stacked, prefix)

    def _causal(self, name: str, x: _Vol) -> _Vol:
        kt = int(self._weights()[f"{name}.weight"].shape[2])
        previous = self.cache.get(name)
        cached = previous if isinstance(previous, _Vol) else None
        if kt > 1:
            self.cache[name] = self._causal_tail(x, cached)
        if cached is not None:
            joined = self._concat(cached, x)
            pad_t = (kt - 1) - int(cached.shape[1])
            out = self._conv(name, joined, (1, 1, 1), (pad_t, kt // 2, kt // 2))
            self._free(joined)
        else:
            out = self._conv(name, x, (1, 1, 1), (kt - 1, kt // 2, kt // 2))
        if isinstance(previous, _Vol) and previous is not self.cache.get(name):
            self._free(previous)
        self._free(x)
        return out

    def _causal_tail(self, x: _Vol, previous: _Vol | None) -> _Vol:
        count = min(2, x.shape[1])
        tail = self._slice(x, x.shape[1] - count, count)
        if tail.shape[1] >= 2 or previous is None:
            return tail
        prior = self._slice(previous, previous.shape[1] - 1, 1)
        joined = self._concat(prior, tail)
        self._free(prior)
        self._free(tail)
        return joined

    def _tail(self, x: _Vol, previous: _Vol | None, repeat: bool) -> _Vol:
        count = min(2, x.shape[1])
        tail = self._slice(x, x.shape[1] - count, count)
        if tail.shape[1] >= 2:
            return tail
        if repeat or previous is None:
            zeros = self._zeros_like_time(tail, 1)
            joined = self._concat(zeros, tail)
            self._free(zeros)
            self._free(tail)
            return joined
        prior = self._slice(previous, previous.shape[1] - 1, 1)
        joined = self._concat(prior, tail)
        self._free(prior)
        self._free(tail)
        return joined

    def _zeros_like_time(self, vol: _Vol, count: int) -> _Vol:
        channels, _, height, width = vol.shape
        raw = np.zeros((channels, count, height, width), dtype=np.float32)
        return self._host(raw)

    def _conv(
        self,
        name: str,
        vol: _Vol,
        stride: tuple[int, int, int],
        pad: tuple[int, int, int],
        *,
        time: bool = False,
    ) -> _Vol:
        weight, shape = self._kernel(f"{name}.weight", time=time)
        bias = self._vector(f"{name}.bias")
        dest, out = self._gpu().conv_into(vol.buf, vol.shape, weight, shape, bias, stride, pad)
        return _Vol(dest, out)

    def _project(self, vol: _Vol, weight: str, bias: str) -> _Vol:
        kernel, shape = self._kernel(weight)
        bias_buf = self._vector(bias)
        dest, out = self._gpu().conv_into(
            vol.buf, vol.shape, kernel, shape, bias_buf, (1, 1, 1), (0, 0, 0)
        )
        return _Vol(dest, out)

    def _rms(self, vol: _Vol, name: str) -> _Vol:
        gamma = self._vector(name)
        dest = self._gpu().alloc_f32(vol.count())
        self._gpu().channel_rms(vol.buf, gamma, dest, vol.shape[0], vol.plane(), float(_EPS))
        return _Vol(dest, vol.shape)

    def _silu(self, vol: _Vol) -> _Vol:
        dest = self._gpu().alloc_f32(vol.count())
        self._gpu().silu_clip(vol.buf, dest, vol.count())
        return _Vol(dest, vol.shape)

    def _add(self, left: _Vol, right: _Vol) -> _Vol:
        dest = self._gpu().alloc_f32(left.count())
        self._gpu().copy_f32(left.buf, 0, dest, left.count())
        self._gpu().add_f32(dest, right.buf, left.count())
        return _Vol(dest, left.shape)

    def _spatial(self, qkv: _Vol) -> _Vol:
        channels = qkv.shape[0] // 3
        _, time, height, width = qkv.shape
        dest = self._gpu().alloc_f32(channels * time * height * width)
        self._gpu().spatial_attend(qkv.buf, dest, channels, time, height, width)
        return _Vol(dest, (channels, time, height, width))

    def _mu(self, vol: _Vol) -> _Vol:
        plane = vol.plane()
        dest = self._gpu().alloc_f32(16 * plane)
        self._gpu().affine_mu(vol.buf, self._mean, self._std, dest, plane)
        return _Vol(dest, (16, vol.shape[1], vol.shape[2], vol.shape[3]))

    def _inv(self, vol: _Vol) -> _Vol:
        plane = vol.plane()
        dest = self._gpu().alloc_f32(16 * plane)
        self._gpu().affine_inv(vol.buf, self._mean, self._std, dest, plane)
        return _Vol(dest, vol.shape)

    def _pad(self, vol: _Vol) -> _Vol:
        channels, time, height, width = vol.shape
        dest = self._gpu().alloc_f32(channels * time * (height + 1) * (width + 1))
        self._gpu().end_pad(vol.buf, dest, vol.shape)
        return _Vol(dest, (channels, time, height + 1, width + 1))

    def _nearest(self, vol: _Vol) -> _Vol:
        channels, time, height, width = vol.shape
        dest = self._gpu().alloc_f32(channels * time * height * 2 * width * 2)
        self._gpu().nearest2(vol.buf, dest, vol.shape)
        return _Vol(dest, (channels, time, height * 2, width * 2))

    def _concat(self, left: _Vol, right: _Vol) -> _Vol:
        channels, left_time, height, width = left.shape
        right_time = right.shape[1]
        dest = self._gpu().alloc_f32(channels * (left_time + right_time) * height * width)
        self._gpu().concat_time(
            left.buf,
            right.buf,
            dest,
            channels,
            left_time,
            right_time,
            height,
            width,
        )
        return _Vol(dest, (channels, left_time + right_time, height, width))

    def _slice(self, vol: _Vol, start: int, count: int) -> _Vol:
        channels, _, height, width = vol.shape
        dest = self._gpu().alloc_f32(channels * count * height * width)
        self._gpu().slice_time(vol.buf, dest, vol.shape, start, count)
        return _Vol(dest, (channels, count, height, width))

    def _pairs(self, vol: _Vol) -> _Vol:
        channels, time, height, width = vol.shape
        half = channels // 2
        dest = self._gpu().alloc_f32(half * time * 2 * height * width)
        self._gpu().stack_pairs(vol.buf, dest, (half, time, height, width))
        return _Vol(dest, (half, time * 2, height, width))

    def _u8(self, vol: _Vol) -> np.ndarray:
        _, frames, height, width = vol.shape
        count = frames * height * width * 3
        dest = self._gpu()._alloc(count)
        self._gpu().pixels_u8(vol.buf, dest, frames, height, width)
        raw = self._gpu().read_u8(dest, count)
        self._gpu().drop(dest)
        return raw.reshape(frames, height, width, 3)

    def _take(self, old: _Vol, new: _Vol) -> _Vol:
        if old.buf is not new.buf:
            self._free(old)
        return new

    def _kernel(
        self, name: str, *, time: bool = False
    ) -> tuple[Any, tuple[int, int, int, int, int]]:
        key = f"{name}#t" if time else name
        found = self._params.get(key)
        if found is not None:
            buf, shape = found
            return buf, (shape[0], shape[1], shape[2], shape[3], shape[4])
        array = np.ascontiguousarray(self._weights()[name], dtype=np.float32)
        if time:
            array = array[:, :, None]
        if array.ndim == 2:
            array = array.reshape(array.shape[0], array.shape[1], 1, 1, 1)
        elif array.ndim == 4:
            array = array.reshape(
                array.shape[0], array.shape[1], 1, array.shape[2], array.shape[3]
            )
        array = np.ascontiguousarray(array)
        buf = self._gpu().keep(self._gpu().upload(array))
        shape = tuple(int(v) for v in array.shape)
        self._params[key] = (buf, shape)
        return buf, (shape[0], shape[1], shape[2], shape[3], shape[4])

    def _vector(self, name: str) -> Any:
        found = self._params.get(name)
        if found is not None:
            return found[0]
        flat = np.ascontiguousarray(self._weights()[name].reshape(-1), dtype=np.float32)
        buf = self._gpu().keep(self._gpu().upload(flat))
        self._params[name] = (buf, tuple(int(v) for v in flat.shape))
        return buf

    def _gpu(self) -> Gpu:
        if self.gpu is None:
            message = "VAE convolution has no device"
            raise RuntimeError(message)
        return self.gpu

