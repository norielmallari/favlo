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
from collections import OrderedDict
from pathlib import Path

import numpy as np

from wan22_i2v.metal import Gpu
from wan22_i2v.schedule import LATENT_MEAN, LATENT_STD, geometry
from wan22_i2v.source import source_dir

_EPS = np.float32(1e-12)


class _FloatStorage:
    pass


def vae_path(root: Path | None = None) -> Path:
    return source_dir(root) / "Wan2.1_VAE.pth"


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


class WanVAE:
    """Encode one still into the DiT latent and decode a latent back to pixels."""

    def __init__(self, path: Path) -> None:
        started = time.perf_counter()
        self.weights = load_vae(path)
        self.cache: dict[str, np.ndarray] = {}
        self.gpu: Gpu | None = None
        elapsed = time.perf_counter() - started
        print(f"vae load {elapsed:.2f}s", flush=True)

    def encode(self, image: np.ndarray, frames: int) -> np.ndarray:
        """uint8 HWC image to the normalized latent ``[16, T, H, W]``."""
        height, width = int(image.shape[0]), int(image.shape[1])
        lat_t, lat_h, lat_w, _ = geometry(frames, height, width)
        pixels = image.astype(np.float32) / np.float32(127.5) - np.float32(1.0)
        video = np.zeros((3, frames, height, width), dtype=np.float32)
        video[:, 0] = np.transpose(pixels, (2, 0, 1))
        started = time.perf_counter()
        self.cache = {}
        gpu = Gpu()
        try:
            self.gpu = gpu
            pieces = [self._encoder(video[:, :1])]
            for start in range(1, frames, 4):
                pieces.append(self._encoder(video[:, start : start + 4]))
            encoded = np.concatenate(pieces, axis=1)
            stats = _project(encoded, self.weights["conv1.weight"], self.weights["conv1.bias"])
        finally:
            gpu.close()
            self.gpu = None
        mu = stats[:16]
        latent = (mu - LATENT_MEAN[:, None, None, None]) / LATENT_STD[:, None, None, None]
        latent = np.ascontiguousarray(latent, dtype=np.float32)
        if latent.shape != (16, lat_t, lat_h, lat_w):
            message = f"encoder emitted {latent.shape}, expected {(16, lat_t, lat_h, lat_w)}"
            raise RuntimeError(message)
        elapsed = time.perf_counter() - started
        print(f"vae encode {elapsed:.2f}s", flush=True)
        return latent

    def decode(self, latent: np.ndarray) -> np.ndarray:
        """Normalized latent to uint8 ``[frames, height, width, 3]``."""
        restored = latent * LATENT_STD[:, None, None, None] + LATENT_MEAN[:, None, None, None]
        expected = 1 + (int(latent.shape[1]) - 1) * 4
        started = time.perf_counter()
        self.cache = {}
        gpu = Gpu()
        try:
            self.gpu = gpu
            z = _project(restored, self.weights["conv2.weight"], self.weights["conv2.bias"])
            pieces = [self._decoder(z[:, index : index + 1]) for index in range(z.shape[1])]
            video = np.concatenate(pieces, axis=1)
        finally:
            gpu.close()
            self.gpu = None
        if video.shape[1] != expected:
            message = f"decoder emitted {video.shape[1]} frames, expected {expected}"
            raise RuntimeError(message)
        clipped = np.clip(video, -1.0, 1.0)
        uint8 = np.rint((clipped + 1.0) * 127.5).astype(np.uint8)
        elapsed = time.perf_counter() - started
        print(f"vae decode {elapsed:.2f}s", flush=True)
        return np.transpose(uint8, (1, 2, 3, 0))

    def _encoder(self, x: np.ndarray) -> np.ndarray:
        x = self._causal("encoder.conv1", x)
        x = self._residual(x, "encoder.downsamples.0")
        x = self._residual(x, "encoder.downsamples.1")
        x = self._down_spatial(x, "encoder.downsamples.2")
        x = self._residual(x, "encoder.downsamples.3")
        x = self._residual(x, "encoder.downsamples.4")
        x = self._down3d(x, "encoder.downsamples.5")
        x = self._residual(x, "encoder.downsamples.6")
        x = self._residual(x, "encoder.downsamples.7")
        x = self._down3d(x, "encoder.downsamples.8")
        x = self._residual(x, "encoder.downsamples.9")
        x = self._residual(x, "encoder.downsamples.10")
        x = self._residual(x, "encoder.middle.0")
        x = self._attend(x, "encoder.middle.1")
        x = self._residual(x, "encoder.middle.2")
        return self._head(x, "encoder.head")

    def _decoder(self, x: np.ndarray) -> np.ndarray:
        x = self._causal("decoder.conv1", x)
        x = self._residual(x, "decoder.middle.0")
        x = self._attend(x, "decoder.middle.1")
        x = self._residual(x, "decoder.middle.2")
        for index in range(3):
            x = self._residual(x, f"decoder.upsamples.{index}")
        x = self._up3d(x, "decoder.upsamples.3")
        for index in range(4, 7):
            x = self._residual(x, f"decoder.upsamples.{index}")
        x = self._up3d(x, "decoder.upsamples.7")
        for index in range(8, 11):
            x = self._residual(x, f"decoder.upsamples.{index}")
        x = self._up_spatial(x, "decoder.upsamples.11")
        for index in range(12, 15):
            x = self._residual(x, f"decoder.upsamples.{index}")
        return self._head(x, "decoder.head")

    def _residual(self, x: np.ndarray, prefix: str) -> np.ndarray:
        shortcut = f"{prefix}.shortcut.weight"
        if shortcut in self.weights:
            skip = _project(x, self.weights[shortcut], self.weights[f"{prefix}.shortcut.bias"])
        else:
            skip = x
        hidden = _silu(_rms(x, self.weights[f"{prefix}.residual.0.gamma"]))
        hidden = self._causal(f"{prefix}.residual.2", hidden)
        hidden = _silu(_rms(hidden, self.weights[f"{prefix}.residual.3.gamma"]))
        hidden = self._causal(f"{prefix}.residual.6", hidden)
        return skip + hidden

    def _head(self, x: np.ndarray, prefix: str) -> np.ndarray:
        hidden = _silu(_rms(x, self.weights[f"{prefix}.0.gamma"]))
        return self._causal(f"{prefix}.2", hidden)

    def _down_spatial(self, x: np.ndarray, prefix: str) -> np.ndarray:
        padded = np.pad(x, ((0, 0), (0, 0), (0, 1), (0, 1)))
        weight = self.weights[f"{prefix}.resample.1.weight"][:, :, None, :, :]
        bias = self.weights[f"{prefix}.resample.1.bias"]
        gpu = self._gpu()
        return gpu.conv3d(np.ascontiguousarray(padded), weight, bias, (1, 2, 2), (0, 0, 0))

    def _up_spatial(self, x: np.ndarray, prefix: str) -> np.ndarray:
        nearest = np.repeat(np.repeat(x, 2, axis=2), 2, axis=3)
        weight = self.weights[f"{prefix}.resample.1.weight"][:, :, None, :, :]
        bias = self.weights[f"{prefix}.resample.1.bias"]
        return self._gpu().conv3d(np.ascontiguousarray(nearest), weight, bias, (1, 1, 1), (0, 1, 1))

    def _down3d(self, x: np.ndarray, prefix: str) -> np.ndarray:
        x = self._down_spatial(x, prefix)
        name = f"{prefix}.time_conv"
        if name not in self.cache:
            self.cache[name] = np.ascontiguousarray(x)
            return x
        context = self.cache[name][:, -1:]
        self.cache[name] = np.ascontiguousarray(x[:, -1:])
        return self._conv(name, np.concatenate((context, x), axis=1), (2, 1, 1), (0, 0, 0))

    def _up3d(self, x: np.ndarray, prefix: str) -> np.ndarray:
        name = f"{prefix}.time_conv"
        if name not in self.cache:
            self.cache[name] = "Rep"
            return self._up_spatial(x, prefix)
        cached = self.cache[name]
        repeat = not isinstance(cached, np.ndarray)
        tail = x[:, -min(2, x.shape[1]) :]
        if tail.shape[1] < 2 and repeat:
            tail = np.concatenate((np.zeros_like(tail), tail), axis=1)
        elif tail.shape[1] < 2:
            tail = np.concatenate((cached[:, -1:], tail), axis=1)
        if repeat:
            expanded = self._conv(name, x, (1, 1, 1), (2, 0, 0))
        else:
            expanded = self._conv(
                name,
                np.concatenate((cached, x), axis=1),
                (1, 1, 1),
                (2 - int(cached.shape[1]), 0, 0),
            )
        self.cache[name] = np.ascontiguousarray(tail)
        channels = expanded.shape[0] // 2
        pair = expanded.reshape(2, channels, *expanded.shape[1:])
        stacked = np.stack((pair[0], pair[1]), axis=2)
        frames = stacked.reshape(channels, -1, *expanded.shape[2:])
        return self._up_spatial(np.ascontiguousarray(frames), prefix)

    def _attend(self, x: np.ndarray, prefix: str) -> np.ndarray:
        channels, time, height, width = x.shape
        qkv_w = self.weights[f"{prefix}.to_qkv.weight"]
        qkv_b = self.weights[f"{prefix}.to_qkv.bias"]
        proj_w = self.weights[f"{prefix}.proj.weight"]
        proj_b = self.weights[f"{prefix}.proj.bias"]
        gamma = self.weights[f"{prefix}.norm.gamma"]
        frames: list[np.ndarray] = []
        scale = np.float32(channels) ** np.float32(-0.5)
        for index in range(time):
            frame = x[:, index]
            normed = _rms(frame, gamma)
            qkv = _project(normed, qkv_w, qkv_b).reshape(3, channels, height * width)
            scores = np.matmul(qkv[0].T, qkv[1]) * scale
            scores -= np.max(scores, axis=1, keepdims=True)
            weights = np.exp(scores)
            weights /= np.sum(weights, axis=1, keepdims=True)
            mixed = np.matmul(weights, qkv[2].T).T.reshape(channels, height, width)
            frames.append(frame + _project(mixed, proj_w, proj_b))
        return np.stack(frames, axis=1).astype(np.float32)

    def _causal(self, name: str, x: np.ndarray) -> np.ndarray:
        weight = self.weights[f"{name}.weight"]
        kt = int(weight.shape[2])
        previous = self.cache.get(name)
        tail = x[:, -min(2, x.shape[1]) :]
        if tail.shape[1] < 2 and isinstance(previous, np.ndarray):
            tail = np.concatenate((previous[:, -1:], tail), axis=1)
        if kt > 1:
            self.cache[name] = np.ascontiguousarray(tail)
        if isinstance(previous, np.ndarray):
            joined = np.concatenate((previous, x), axis=1)
            pad_t = (kt - 1) - int(previous.shape[1])
        else:
            joined = x
            pad_t = kt - 1
        return self._conv(name, joined, (1, 1, 1), (pad_t, kt // 2, kt // 2))

    def _conv(
        self,
        name: str,
        source: np.ndarray,
        stride: tuple[int, int, int],
        pad: tuple[int, int, int],
    ) -> np.ndarray:
        return self._gpu().conv3d(
            np.ascontiguousarray(source),
            self.weights[f"{name}.weight"],
            self.weights[f"{name}.bias"],
            stride,
            pad,
        )

    def _gpu(self) -> Gpu:
        if self.gpu is None:
            message = "VAE convolution has no device"
            raise RuntimeError(message)
        return self.gpu


def _rms(x: np.ndarray, gamma: np.ndarray) -> np.ndarray:
    squares = np.sum(x * x, axis=0, keepdims=True)
    norm = np.maximum(np.sqrt(squares), _EPS)
    scale = np.sqrt(np.float32(x.shape[0]))
    return x / norm * scale * gamma


def _silu(x: np.ndarray) -> np.ndarray:
    return x * (np.float32(1.0) / (np.float32(1.0) + np.exp(-np.clip(x, -60.0, 60.0))))


def _project(x: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
    out_c = int(weight.shape[0])
    in_c = int(weight.shape[1])
    matrix = weight.reshape(out_c, in_c)
    flat = x.reshape(in_c, -1)
    mixed = matrix @ flat
    mixed += bias.reshape(out_c, 1)
    return mixed.reshape(out_c, *x.shape[1:]).astype(np.float32)
