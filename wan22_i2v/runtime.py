"""Load one packed record and write an AVI."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

from wan22_i2v.dit import Forward, drop_expert, upload_expert
from wan22_i2v.formats import TILE_M, pad_to
from wan22_i2v.latent import condition_volume, stride_decode, stride_encode
from wan22_i2v.metal import Gpu
from wan22_i2v.record import load_embed, load_expert, load_manifest
from wan22_i2v.schedule import expert_name, geometry, sigmas, timesteps
from wan22_i2v.text import encode_prompt, encoder_path, tokenizer_path
from wan22_i2v.vae import WanVAE, vae_path
from wan22_i2v.video import write_avi


class Runtime:
    """One packed Wan record. One expert is on device at a time."""

    def __init__(self, path: Path, console: Any = None) -> None:
        del console
        self.path = Path(path)
        self.manifest = load_manifest(self.path)
        self.config = self.manifest["config"]
        self.embed = load_embed(self.path)
        self.gpu: Gpu | None = None
        self._resident: Any = None

    def close(self) -> None:
        if self.gpu is None:
            return
        if self._resident is not None:
            drop_expert(self.gpu, self._resident)
            self._resident = None
        self.gpu.close()
        self.gpu = None

    def generate(
        self,
        prompt: str,
        image: np.ndarray,
        frames: int,
        width: int,
        height: int,
        seed: int,
    ) -> tuple[bytes, dict[str, Any]]:
        if image.shape[0] != height or image.shape[1] != width:
            message = f"image is {image.shape[1]}x{image.shape[0]}, expected {width}x{height}"
            raise ValueError(message)
        lat_t, lat_h, lat_w, tokens = geometry(frames, height, width)
        weights = vae_path()
        vae = WanVAE(weights) if weights.is_file() else None
        latent = vae.encode(image, frames) if vae is not None else stride_encode(image, frames)
        noise = np.random.default_rng(seed).standard_normal((16, lat_t, lat_h, lat_w))
        noise = noise.astype(np.float32)
        table = _context(
            prompt, self.embed, int(self.config["text_len"]), int(self.config["text_dim"])
        )
        gpu = self._open()
        forward = Forward(gpu, self.config, tokens, int(np.prod((36, lat_t, lat_h, lat_w))))
        knots = sigmas()
        times = timesteps()
        step_times: list[float] = []
        loaded: str | None = None
        try:
            for index, timestep in enumerate(times[:-1]):
                name = expert_name(float(timestep), float(self.config["boundary"]))
                if loaded != name:
                    self._use(name)
                    loaded = name
                started = time.perf_counter()
                volume = condition_volume(noise, latent, frames)
                velocity = forward.velocity(
                    self._resident, volume, table, float(timestep), (lat_t, lat_h, lat_w)
                )
                noise = noise + float(knots[index + 1] - knots[index]) * velocity
                elapsed = time.perf_counter() - started
                step_times.append(elapsed)
                print(f"step {index} expert {name} {elapsed:.2f}s", flush=True)
        finally:
            if self._resident is not None and self.gpu is not None:
                drop_expert(self.gpu, self._resident)
                self._resident = None
        if not np.isfinite(noise).all():
            message = "latent is not finite"
            raise RuntimeError(message)
        self.close()
        pixels = vae.decode(noise) if vae is not None else stride_decode(noise, frames)
        avi = write_avi(pixels, fps=int(self.config["fps"]))
        return avi, {"steps": step_times}

    def _open(self) -> Gpu:
        if self.gpu is None:
            self.gpu = Gpu()
        return self.gpu

    def _use(self, name: str) -> None:
        gpu = self._open()
        if self._resident is not None:
            drop_expert(gpu, self._resident)
            self._resident = None
        expert = load_expert(self.path / name, int(self.config["layers"]))
        self._resident = upload_expert(gpu, expert)


def open_runtime(
    path: Path,
    *,
    kv_cache: str = "e4m3",
    console: Any = None,
    resident_percent: float | None = None,
    max_memory: str | int | None = None,
) -> Runtime:
    del kv_cache, resident_percent, max_memory
    return Runtime(Path(path), console)


def _context(prompt: str, embed: np.ndarray, length: int, width: int) -> np.ndarray:
    rows = pad_to(length, TILE_M)
    table = np.zeros((rows, width), dtype=np.float32)
    if width == 4096 and encoder_path().is_file() and tokenizer_path().is_file():
        vectors = encode_prompt(prompt)
        count = min(int(vectors.shape[0]), length, rows)
        if count:
            table[:count] = vectors[:count]
        return table
    raw = prompt.encode("utf-8")
    count = min(len(raw), length)
    if count:
        ids = np.frombuffer(raw[:count], dtype=np.uint8).astype(np.int32)
        table[:count] = embed[ids].astype(np.float32)
    return table
