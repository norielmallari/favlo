"""One video request. Frames win when seconds is also set."""

from __future__ import annotations

import base64
from dataclasses import dataclass

import numpy as np

from favlo.media_pipeline import MediaRequestError
from wan22_ti2v_5b.png import decode_png
from wan22_ti2v_5b.schedule import frames_from_duration, snap_frames
from wan22_ti2v_5b.spec import FPS

_DEFAULT_SIZE = (1280, 704)
_DEFAULT_FRAMES = 121


class VideoRequestError(MediaRequestError):
    """The body cannot be sampled."""

    def __init__(self, message: str, param: str | None = None) -> None:
        super().__init__(message, param)


@dataclass
class VideoRequest:
    prompt: str
    negative_prompt: str
    width: int
    height: int
    frames: int
    seed: int
    quality: str
    image: np.ndarray | None

    @property
    def seconds(self) -> str:
        return f"{self.frames / FPS:.3f}"

    @property
    def size(self) -> str:
        return f"{self.width}x{self.height}"


def video_modes() -> tuple[str, ...]:
    """Text-to-video and image-to-video share this pipeline."""
    return ("i2v", "t2v")


def list_resolutions() -> list[dict[str, str]]:
    """The two served frames. Landscape is wider."""
    return [
        {"name": "1280x704", "orientation": "landscape", "size": "1280x704"},
        {"name": "704x1280", "orientation": "portrait", "size": "704x1280"},
    ]


def parse_request(body: dict) -> VideoRequest:
    prompt = body.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        message = "prompt is required"
        raise VideoRequestError(message)
    negative = body.get("negative_prompt") or ""
    if not isinstance(negative, str):
        message = "negative_prompt must be text"
        raise VideoRequestError(message)
    width, height = _size(body.get("size"))
    frames = _frames(body)
    seed = body.get("seed") or 0
    if not isinstance(seed, int) or isinstance(seed, bool):
        message = "seed must be an integer"
        raise VideoRequestError(message)
    quality = body.get("quality") or "high"
    if not isinstance(quality, str):
        message = "quality must be text"
        raise VideoRequestError(message)
    _steps(body.get("steps", 0))
    _guidance(body.get("guidance_scale"))
    return VideoRequest(
        prompt=prompt,
        negative_prompt=negative,
        width=width,
        height=height,
        frames=frames,
        seed=seed,
        quality=quality,
        image=_image(body.get("image"), width, height),
    )


def _size(value: object) -> tuple[int, int]:
    text = _DEFAULT_SIZE if value in (None, "") else value
    if not isinstance(text, str) or "x" not in text:
        if text == _DEFAULT_SIZE:
            return _DEFAULT_SIZE
        message = "size must be WIDTHxHEIGHT"
        raise VideoRequestError(message)
    raw_w, raw_h = text.lower().split("x", 1)
    try:
        width, height = int(raw_w), int(raw_h)
    except ValueError as exc:
        message = "size must be WIDTHxHEIGHT"
        raise VideoRequestError(message) from exc
    if width % 32 != 0 or height % 32 != 0 or width < 32 or height < 32:
        message = "edges must be multiples of 32"
        raise VideoRequestError(message)
    return width, height


def _frames(body: dict) -> int:
    duration = body.get("duration")
    if duration is not None and duration != "":
        if isinstance(duration, str):
            try:
                duration = float(duration)
            except ValueError as exc:
                message = "duration must be a number"
                raise VideoRequestError(message) from exc
        if isinstance(duration, bool) or not isinstance(duration, (int, float)):
            message = "duration must be a number"
            raise VideoRequestError(message)
        try:
            return frames_from_duration(float(duration))
        except ValueError as exc:
            raise VideoRequestError(str(exc)) from exc
    frames = body.get("frames") or 0
    seconds = body.get("seconds") or ""
    if frames:
        if not isinstance(frames, int) or isinstance(frames, bool):
            message = "frames must be an integer"
            raise VideoRequestError(message)
        if frames < 5 or (frames - 1) % 4 != 0:
            message = "frames must be 4n+1 and at least 5"
            raise VideoRequestError(message)
        return frames
    if isinstance(seconds, str) and seconds.strip():
        try:
            duration = float(seconds)
        except ValueError as exc:
            message = "seconds must be a number"
            raise VideoRequestError(message) from exc
        return snap_frames(duration)
    if isinstance(seconds, int | float) and not isinstance(seconds, bool):
        return snap_frames(float(seconds))
    return _DEFAULT_FRAMES


def _steps(value: object) -> int:
    if value in (None, 0, 4):
        return 4
    message = "steps must be 0 or 4"
    raise VideoRequestError(message)


def _guidance(value: object) -> None:
    if value is None or value == 1 or value == 1.0:
        return
    message = "guidance_scale must be omitted or 1"
    raise VideoRequestError(message)


def _image(value: object, width: int, height: int) -> np.ndarray | None:
    if value is None or value == "":
        return None
    if not isinstance(value, str):
        message = "image must be base64 png"
        raise VideoRequestError(message)
    try:
        raw = base64.b64decode(value, validate=False)
        image = decode_png(raw)
    except (ValueError, TypeError) as exc:
        message = "image must be a png"
        raise VideoRequestError(message) from exc
    if image.shape[1] != width or image.shape[0] != height:
        image = _resize(image, width, height)
    return image.astype(np.float32) / np.float32(127.5) - np.float32(1.0)


def _resize(image: np.ndarray, width: int, height: int) -> np.ndarray:
    y_pos = np.linspace(0, image.shape[0] - 1, height)
    x_pos = np.linspace(0, image.shape[1] - 1, width)
    y_idx = np.clip(np.rint(y_pos).astype(np.int32), 0, image.shape[0] - 1)
    x_idx = np.clip(np.rint(x_pos).astype(np.int32), 0, image.shape[1] - 1)
    return image[y_idx][:, x_idx]
