"""Load one packed record and write an AVI."""

from __future__ import annotations

import gc
import time
from pathlib import Path
from typing import Any

import numpy as np
import objc

from wan22_i2v.dit import Forward, drop_expert, upload_expert
from wan22_i2v.formats import TILE_M, pad_to
from wan22_i2v.latent import stride_decode, stride_encode
from wan22_i2v.metal import _NAMES, Gpu
from wan22_i2v.png import resize_rgb
from wan22_i2v.progress import ClipProgress, clip_label, sample_total
from wan22_i2v.record import half_names, linear_names, load_embed, load_expert, load_manifest
from wan22_i2v.schedule import expert_name, geometry, mask_latent, sigmas, timesteps
from wan22_i2v.text import _LAYERS, encode_prompt, encoder_path, tokenizer_path
from wan22_i2v.vae import DECODER_BLOCKS, ENCODER_BLOCKS, WanVAE, vae_path
from wan22_i2v.video import write_avi


class Runtime:
    """One packed Wan record. One expert is on device at a time."""

    def __init__(self, path: Path, console: Any = None) -> None:
        self.console = console
        self.path = Path(path)
        self.manifest = load_manifest(self.path)
        self.config = self.manifest["config"]
        self.embed = load_embed(self.path)
        self.gpu: Gpu | None = None
        self._resident: Any = None
        self._forward: Forward | None = None

    def close(self) -> None:
        """Drop the expert and the device buffers. The process has no run loop."""
        forward = self._forward
        resident = self._resident
        gpu = self.gpu
        if forward is None and resident is None and gpu is None:
            return
        self._forward = None
        self._resident = None
        self.gpu = None
        with objc.autorelease_pool():
            if forward is not None:
                forward.release()
            forward = None
            if resident is not None and gpu is not None:
                drop_expert(gpu, resident)
            elif resident is not None:
                _clear_resident(resident)
            resident = None
            if gpu is not None:
                gpu.close()
            gpu = None
            gc.collect()

    def generate(
        self,
        prompt: str,
        image: np.ndarray,
        frames: int,
        width: int,
        height: int,
        seed: int,
        progress: ClipProgress | None = None,
    ) -> tuple[bytes, dict[str, Any]]:
        image = resize_rgb(image, height, width)
        lat_t, lat_h, lat_w, tokens = geometry(frames, height, width)
        times = timesteps()
        layers = int(self.config["layers"])
        boundary = float(self.config["boundary"])
        text_layers = _LAYERS if _text_encoder(int(self.config["text_dim"])) else 0
        vae_file = vae_path().is_file()
        own = progress is None
        if progress is None:
            progress = ClipProgress(
                console=self.console,
                clip=clip_label(width, height, frames, int(self.config["fps"])),
            )
        progress.open(
            sample_total(
                steps=len(times) - 1,
                dit_layers=layers,
                loads=_loads(times, boundary),
                weights=2 * (len(linear_names(layers)) + len(half_names(layers))),
                text_layers=text_layers,
                vae_blocks=lat_t * (ENCODER_BLOCKS + DECODER_BLOCKS) if vae_file else 0,
                pipelines=len(_NAMES),
                gpus=1,
            )
        )
        try:
            progress.stage("compile")
            gpu = self._open(progress.advance)
            progress.stage("text")
            table = _context(
                prompt,
                self.embed,
                int(self.config["text_len"]),
                int(self.config["text_dim"]),
                progress.advance if text_layers else None,
                gpu,
            )
            progress.stage("encode")
            weights = vae_path()
            vae = WanVAE(weights) if weights.is_file() else None
            if vae is not None:
                cond = vae.encode(image, frames, on_block=progress.advance, gpu=gpu)
            else:
                cond = gpu.keep(gpu.upload(stride_encode(image, frames)))
            noise_np = np.random.default_rng(seed).standard_normal((16, lat_t, lat_h, lat_w))
            noise = gpu.keep(gpu.upload(noise_np.astype(np.float32)))
            mask = gpu.keep(gpu.upload(np.ascontiguousarray(mask_latent(frames), dtype=np.float32)))
            knots = sigmas()
            step_times: list[float] = []
            loaded: str | None = None
            with objc.autorelease_pool():
                volume_size = int(np.prod((36, lat_t, lat_h, lat_w)))
                self._forward = Forward(gpu, self.config, tokens, volume_size)
            count = 16 * lat_t * lat_h * lat_w
            for index, timestep in enumerate(times[:-1]):
                name = expert_name(float(timestep), boundary)
                if loaded != name:
                    progress.stage("load")
                    self._use(name, progress.advance)
                    loaded = name
                progress.stage(name)
                started = time.perf_counter()
                with objc.autorelease_pool():
                    forward = self._forward
                    if forward is None:
                        message = "the DiT forward is closed"
                        raise RuntimeError(message)
                    forward.prepare(noise, mask, cond, lat_t, lat_h, lat_w)
                    forward.velocity(
                        self._resident,
                        table,
                        float(timestep),
                        (lat_t, lat_h, lat_w),
                        on_layer=progress.advance,
                    )
                    gpu.axpy(noise, forward.vel, float(knots[index + 1] - knots[index]), count)
                elapsed = time.perf_counter() - started
                step_times.append(elapsed)
                progress.note(f"step {index} expert {name} {elapsed:.2f}s")
            latent = gpu.read_f32(noise, count).reshape(16, lat_t, lat_h, lat_w)
            if not np.isfinite(latent).all():
                message = "latent is not finite"
                raise RuntimeError(message)
            self._drop_forward()
            gpu.drop(cond)
            gpu.drop(mask)
            progress.stage("decode")
            if vae is not None:
                pixels = vae.decode(
                    noise,
                    (16, lat_t, lat_h, lat_w),
                    on_block=progress.advance,
                    gpu=gpu,
                )
            else:
                pixels = stride_decode(latent, frames)
            progress.stage("pack")
            avi = write_avi(pixels, fps=int(self.config["fps"]))
            progress.advance()
        except Exception:
            if own:
                progress.close(progress.summary(failed=True))
            raise
        finally:
            self.close()
        if own:
            progress.close(progress.summary())
        return avi, {"steps": step_times}

    def _drop_forward(self) -> None:
        forward = self._forward
        resident = self._resident
        gpu = self.gpu
        self._forward = None
        self._resident = None
        if forward is not None and gpu is not None:
            forward.detach(gpu)
        elif forward is not None:
            forward.release()
        if resident is not None and gpu is not None:
            drop_expert(gpu, resident)
        elif resident is not None:
            _clear_resident(resident)

    def _open(self, on_pipeline: Any = None) -> Gpu:
        if self.gpu is None:
            self.gpu = Gpu(on_pipeline=on_pipeline)
        return self.gpu

    def _use(self, name: str, on_file: Any) -> None:
        gpu = self._open()
        if self._resident is not None:
            drop_expert(gpu, self._resident)
            self._resident = None
            gc.collect()
        layers = int(self.config["layers"])
        expert = load_expert(self.path / name, layers, on_file=on_file)
        with objc.autorelease_pool():
            self._resident = upload_expert(gpu, expert, on_file=on_file)


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


def _loads(times: np.ndarray, boundary: float) -> int:
    loads = 0
    previous: str | None = None
    for timestep in times[:-1]:
        name = expert_name(float(timestep), boundary)
        if name != previous:
            loads += 1
            previous = name
    return loads


def _text_encoder(width: int) -> bool:
    return width == 4096 and encoder_path().is_file() and tokenizer_path().is_file()


def _context(
    prompt: str,
    embed: np.ndarray,
    length: int,
    width: int,
    on_layer: Any = None,
    gpu: Any = None,
) -> np.ndarray:
    rows = pad_to(length, TILE_M)
    table = np.zeros((rows, width), dtype=np.float32)
    if _text_encoder(width):
        vectors = encode_prompt(prompt, on_layer=on_layer, gpu=gpu)
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


def _clear_resident(resident: Any) -> None:
    resident.owned.clear()
    resident.weights.clear()
    resident.halves.clear()
    resident.host.clear()
    resident.packs.clear()
