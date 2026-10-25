"""Four-knot sampler. The last knot decodes x0 and does not step to 0."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from favlo.videos import encode_mp4, reported_quality, store_source
from wan22_ti2v_5b.avi import encode_avi
from wan22_ti2v_5b.dit import DiT
from wan22_ti2v_5b.request import VideoRequest
from wan22_ti2v_5b.schedule import (
    blend,
    image_mask,
    latent_grid,
    renoise,
    sigmas,
    text_mask,
    timesteps,
    token_grid,
    token_timesteps,
    x0_from_velocity,
)
from wan22_ti2v_5b.spec import LATENT_CHANNELS
from wan22_ti2v_5b.text import TextEncoder
from wan22_ti2v_5b.vae import WanVae


@dataclass
class ClipHold:
    """Pickleable pause between knots."""

    request: VideoRequest
    video_id: str
    latent: np.ndarray
    image: np.ndarray | None
    mask: np.ndarray
    text: np.ndarray
    text_valid: np.ndarray
    rng_state: dict
    step: int


class Clip:
    def __init__(self, dit: DiT, text: TextEncoder, vae: WanVae) -> None:
        self.dit = dit
        self.text = text
        self.vae = vae

    def sample(
        self,
        request: VideoRequest,
        *,
        video_id: str,
        home: Path | None,
        videos: dict | None,
        hold: ClipHold | None = None,
        stop=None,
    ) -> tuple[dict, ClipHold | None]:
        if hold is None:
            print("encode text", flush=True)
            text, text_valid = self.text.encode(request.prompt)
            grid = latent_grid(request.frames, request.height, request.width)
            rng = np.random.default_rng(request.seed)
            latent = rng.standard_normal((LATENT_CHANNELS, *grid)).astype(np.float32)
            image = None
            if request.image is None:
                mask = text_mask(*grid)
            else:
                image = self.vae.encode(request.image)
                mask = image_mask(*grid)
                latent = blend(latent, image, mask)
            state = ClipHold(
                request=request,
                video_id=video_id,
                latent=latent,
                image=image,
                mask=mask,
                text=text,
                text_valid=text_valid,
                rng_state=rng.bit_generator.state,
                step=0,
            )
        else:
            state = hold
        knots = timesteps()
        noise = sigmas()
        grid = latent_grid(state.request.frames, state.request.height, state.request.width)
        tokens = token_grid(state.request.frames, state.request.height, state.request.width)
        rng = np.random.default_rng()
        rng.bit_generator.state = state.rng_state
        while state.step < len(knots):
            if stop is not None and stop():
                return {"id": state.video_id, "status": "cancelled"}, None
            sigma = float(noise[state.step])
            print(f"knot {state.step} {knots[state.step]:.6f}", flush=True)
            state.latent = blend(state.latent, _image(state), state.mask)
            clock = token_timesteps(state.mask, float(int(knots[state.step])))
            velocity = self.dit.velocity(state.latent, clock, state.text, state.text_valid, tokens)
            clean = x0_from_velocity(state.latent, sigma, velocity)
            if state.step + 1 < len(knots):
                fresh = rng.standard_normal(state.latent.shape).astype(np.float32)
                state.latent = renoise(clean, float(noise[state.step + 1]), fresh)
            else:
                state.latent = clean
            state.latent = blend(state.latent, _image(state), state.mask)
            state.step += 1
            if state.step == len(knots):
                break
            state.rng_state = rng.bit_generator.state
            if stop is not None and stop(pause=True):
                return {"id": state.video_id, "status": "in_progress"}, state
        print("decode", flush=True)
        frames = self.vae.decode(state.latent.astype(np.float32))
        payload = _store(state, frames, home, videos)
        return payload, None


def _image(state: ClipHold) -> np.ndarray:
    if state.image is not None:
        return state.image
    return np.zeros_like(state.latent)


def _store(state: ClipHold, frames: np.ndarray, home: Path | None, videos: dict | None) -> dict:
    rgb = np.clip((frames + 1.0) * 127.5, 0.0, 255.0).astype(np.uint8)
    quality = reported_quality(state.request.quality)
    source = store_source(state.video_id, encode_avi(rgb), home=home)
    encode_mp4(source, quality)
    job = {
        "id": state.video_id,
        "status": "completed",
        "size": state.request.size,
        "seconds": state.request.seconds,
        "progress": 1,
    }
    if isinstance(videos, dict):
        videos[state.video_id] = job
    return job
