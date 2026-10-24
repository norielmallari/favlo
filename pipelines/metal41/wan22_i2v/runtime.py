"""Load one packed record and write an AVI."""

from __future__ import annotations

import gc
import time
from pathlib import Path
from typing import Any

import numpy as np
import objc

from favlo.memory import memory
from wan22_i2v.dit import Forward, drop_expert, forward_bytes, upload_expert
from wan22_i2v.formats import TILE_M, pad_to
from wan22_i2v.hold import ClipHold, ClipPaused, interrupt, reload_quietly
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

    def __init__(
        self,
        path: Path,
        console: Any = None,
        *,
        owner: str | None = None,
        max_memory: str | int | None = None,
    ) -> None:
        self.console = console
        self._owner = owner or f"wan22-i2v:{id(self)}"
        self._memory_cap = _parse_memory(max_memory)
        self.path = Path(path)
        self.manifest = load_manifest(self.path)
        self.config = self.manifest["config"]
        self.embed = load_embed(self.path)
        self.gpu: Gpu | None = None
        self._resident: Any = None
        self._forward: Forward | None = None

    def close(self) -> None:
        """Drop the expert and the device buffers. The process has no run loop."""
        try:
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
        finally:
            self._sweep_owner()

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
        hold = getattr(progress, "hold", None)
        if hold is not None and not isinstance(hold, ClipHold):
            hold = None
        if hold is not None:
            progress.done = hold.done
            if progress.row is not None:
                progress.row.done = progress.done
        step_times = list(hold.step_times) if hold is not None else []
        noise_shape = (16, lat_t, lat_h, lat_w)
        mask_shape = mask_latent(frames).shape
        try:
            avi = hold.avi if hold is not None and hold.avi is not None else None
            pixels = hold.pixels if hold is not None and hold.pixels is not None else None
            if avi is None and pixels is None:
                self._reserve(frames, width, height, lat_t, lat_h, lat_w, tokens)
                progress.stage("compile")
                if hold is not None and hold.reached("compile"):
                    progress.counting(False)
                gpu = self._open(progress.advance)
                progress.counting(True)
                self._pause(progress, ClipHold(stage="compile", step_times=step_times))
                if hold is not None and hold.table is not None:
                    table = hold.table
                else:
                    progress.stage("text")
                    table = _context(
                        prompt,
                        self.embed,
                        int(self.config["text_len"]),
                        int(self.config["text_dim"]),
                        progress.advance if text_layers else None,
                        gpu,
                    )
                    self._pause(
                        progress,
                        ClipHold(stage="text", table=table, step_times=step_times),
                    )
                weights = vae_path()
                vae = WanVAE(weights) if weights.is_file() else None
                if hold is not None and hold.noise is not None:
                    if hold.cond is None or hold.mask is None:
                        message = "the paused encode is missing a latent"
                        raise RuntimeError(message)
                    noise = gpu.keep(gpu.upload(hold.noise))
                    cond = gpu.keep(gpu.upload(hold.cond))
                    mask = gpu.keep(gpu.upload(hold.mask))
                else:
                    progress.stage("encode")
                    if vae is not None:
                        cond = vae.encode(image, frames, on_block=progress.advance, gpu=gpu)
                    else:
                        cond = gpu.keep(gpu.upload(stride_encode(image, frames)))
                    noise_np = np.random.default_rng(seed).standard_normal(noise_shape)
                    noise = gpu.keep(gpu.upload(noise_np.astype(np.float32)))
                    mask = gpu.keep(
                        gpu.upload(np.ascontiguousarray(mask_latent(frames), dtype=np.float32))
                    )
                    self._pause(
                        progress,
                        ClipHold(
                            stage="encode",
                            table=table,
                            noise=_volume(gpu, noise, noise_shape),
                            cond=_volume(gpu, cond, noise_shape),
                            mask=_volume(gpu, mask, mask_shape),
                            step_times=step_times,
                        ),
                    )
                knots = sigmas()
                loaded: str | None = None
                start = hold.step if hold is not None and hold.stage == "step" else 0
                with objc.autorelease_pool():
                    volume_size = int(np.prod((36, lat_t, lat_h, lat_w)))
                    self._forward = Forward(gpu, self.config, tokens, volume_size)
                count = 16 * lat_t * lat_h * lat_w
                for index, timestep in enumerate(times[:-1]):
                    if index < start:
                        continue
                    name = expert_name(float(timestep), boundary)
                    if loaded != name:
                        progress.stage("load")
                        if reload_quietly(hold, name, index):
                            progress.counting(False)
                        self._use(
                            name,
                            progress.advance,
                            self._load_stop(
                                progress,
                                index,
                                name,
                                table,
                                noise,
                                cond,
                                mask,
                                noise_shape,
                                mask_shape,
                                step_times,
                                gpu,
                                quiet=reload_quietly(hold, name, index),
                            ),
                        )
                        progress.counting(True)
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
                        gpu.axpy(
                            noise,
                            forward.vel,
                            float(knots[index + 1] - knots[index]),
                            count,
                        )
                    elapsed = time.perf_counter() - started
                    step_times.append(elapsed)
                    progress.note(f"step {index} expert {name} {elapsed:.2f}s")
                    self._pause(
                        progress,
                        ClipHold(
                            stage="step",
                            step=index + 1,
                            expert=name,
                            table=table,
                            noise=_volume(gpu, noise, noise_shape),
                            cond=_volume(gpu, cond, noise_shape),
                            mask=_volume(gpu, mask, mask_shape),
                            step_times=list(step_times),
                        ),
                    )
                latent = gpu.read_f32(noise, count).reshape(noise_shape)
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
                        noise_shape,
                        on_block=progress.advance,
                        gpu=gpu,
                        resident_frames=resident_latent_frames(
                            self._room(), frames, height, width, lat_t
                        ),
                    )
                else:
                    pixels = stride_decode(latent, frames)
                self._pause(
                    progress,
                    ClipHold(stage="decode", pixels=pixels, step_times=list(step_times)),
                )
            if avi is None:
                progress.stage("pack")
                avi = write_avi(pixels, fps=int(self.config["fps"]))
                progress.advance()
                self._pause(
                    progress,
                    ClipHold(stage="pack", avi=avi, step_times=list(step_times)),
                )
        except ClipPaused:
            raise
        except Exception:
            if own:
                progress.close(progress.summary(failed=True))
            raise
        finally:
            self.close()
        if own:
            progress.close(progress.summary())
        return avi, {"steps": step_times}

    def _pause(self, progress: ClipProgress, hold: ClipHold) -> None:
        reason = interrupt(progress)
        if reason is None:
            return
        hold.done = progress.done
        raise ClipPaused(reason, hold)

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

    def _reserve(
        self,
        frames: int,
        width: int,
        height: int,
        lat_t: int,
        lat_h: int,
        lat_w: int,
        tokens: int,
    ) -> None:
        """Require one expert and the scratch for every frame of this clip."""
        room = self._room()
        weights = expert_bytes(self.path)
        scratch = forward_bytes(self.config, tokens, 36 * lat_t * lat_h * lat_w)
        if weights + scratch > room:
            raise RuntimeError(f"scratch for {frames} frames at {width}x{height} does not fit")
        resident_latent_frames(room, frames, height, width, lat_t)

    def _room(self) -> int:
        ceiling = memory.budget()
        if self._memory_cap is not None:
            ceiling = min(ceiling, self._memory_cap)
        return max(0, ceiling - memory.live - memory.pending)

    def _sweep_owner(self) -> None:
        owner = getattr(self, "_owner", None)
        if isinstance(owner, str):
            memory.sweep(owner)

    def _open(self, on_pipeline: Any = None) -> Gpu:
        if self.gpu is None:
            self.gpu = Gpu(on_pipeline=on_pipeline, owner=self._owner)
        return self.gpu

    def _load_stop(
        self,
        progress: ClipProgress,
        index: int,
        name: str,
        table: np.ndarray,
        noise: Any,
        cond: Any,
        mask: Any,
        noise_shape: tuple[int, ...],
        mask_shape: tuple[int, ...],
        step_times: list[float],
        gpu: Gpu,
        *,
        quiet: bool,
    ) -> Any:
        """Stop between weight files when a chat needs the device."""

        def stop(after_file: bool = False) -> None:
            del after_file
            reason = interrupt(progress)
            if reason is None:
                return
            self._pause(
                progress,
                ClipHold(
                    stage="step",
                    step=index,
                    expert=name if quiet else None,
                    table=table,
                    noise=_volume(gpu, noise, noise_shape),
                    cond=_volume(gpu, cond, noise_shape),
                    mask=_volume(gpu, mask, mask_shape),
                    step_times=list(step_times),
                ),
            )

        return stop

    def _use(self, name: str, on_file: Any, stop: Any = None) -> None:
        def tick() -> None:
            if on_file is not None:
                on_file()
            if stop is not None:
                stop(True)

        if stop is not None:
            stop(False)
        gpu = self._open()
        if self._resident is not None:
            drop_expert(gpu, self._resident)
            self._resident = None
            gc.collect()
        layers = int(self.config["layers"])
        expert = load_expert(self.path / name, layers, on_file=tick)
        with objc.autorelease_pool():
            self._resident = upload_expert(gpu, expert, on_file=tick)


def open_runtime(
    path: Path,
    *,
    kv_cache: str = "e4m3",
    console: Any = None,
    resident_percent: float | None = None,
    max_memory: str | int | None = None,
    owner: str | None = None,
) -> Runtime:
    del kv_cache, resident_percent
    return Runtime(Path(path), console, owner=owner, max_memory=max_memory)


def expert_bytes(path: Path) -> int:
    """Largest on-disk expert. One expert is resident at a time."""
    largest = 0
    for name in ("high", "low"):
        folder = path / name
        if not folder.is_dir():
            continue
        total = sum(item.stat().st_size for item in folder.iterdir() if item.is_file())
        largest = max(largest, total)
    return largest


def resident_latent_frames(room: int, frames: int, height: int, width: int, lat_t: int) -> int:
    """How many decoded latent frames stay on device. The clip length stays the request."""
    if _pixel_bytes(frames, height, width) <= room:
        return lat_t
    if _pixel_bytes(min(4, frames), height, width) <= room:
        return 1
    raise RuntimeError(f"scratch for {frames} frames at {width}x{height} does not fit")


def _pixel_bytes(frames: int, height: int, width: int) -> int:
    return 3 * frames * height * width * 4


def _parse_memory(text: str | int | None) -> int | None:
    """``32GiB``, ``512MiB``, or a byte count. Empty leaves the ledger uncapped."""
    if text is None or text == "":
        return None
    if isinstance(text, int):
        if text < 1:
            message = f"maximum memory {text} is not positive"
            raise ValueError(message)
        return text
    raw = str(text).strip().replace("_", "")
    suffix = ""
    digits = raw
    for ending, name in (
        ("gib", "GiB"),
        ("mib", "MiB"),
        ("gb", "GiB"),
        ("mb", "MiB"),
        ("g", "GiB"),
        ("m", "MiB"),
    ):
        if raw.lower().endswith(ending):
            suffix = name
            digits = raw[: -len(ending)]
            break
    try:
        number = int(digits)
    except ValueError as exc:
        message = f"maximum memory {text} is not a byte count"
        raise ValueError(message) from exc
    if number < 1:
        message = f"maximum memory {text} is not positive"
        raise ValueError(message)
    if suffix == "GiB":
        return number << 30
    if suffix == "MiB":
        return number << 20
    return number


def _volume(gpu: Gpu, buf: Any, shape: tuple[int, ...]) -> np.ndarray:
    count = int(np.prod(shape))
    return gpu.read_f32(buf, count).reshape(shape)


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
