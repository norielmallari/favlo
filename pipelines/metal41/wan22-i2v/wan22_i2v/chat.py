"""Chat is refused. Video is one MP4 job. The listener turns the reply into HTTP."""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np

from favlo.media_pipeline import MediaCall, MediaPipeline, MediaRequestError
from favlo.videos import reported_quality, store_source, sweep_videos
from wan22_i2v.hold import ClipHold, ClipPaused
from wan22_i2v.png import decode_png, resize_rgb
from wan22_i2v.progress import ClipProgress, clip_label
from wan22_i2v.schedule import (
    frames_from_duration,
    FIVE_SEC_FRAMES,
    FPS,
    P320_HEIGHT,
    P320_WIDTH,
    STEPS,
    require_geometry,
    require_guidance,
    require_steps,
)
from wan22_i2v.video import write_mp4


class VideoRequestError(MediaRequestError):
    def __init__(self, message: str, param: str | None) -> None:
        super().__init__(message, param)


class VideoFields(NamedTuple):
    prompt: str
    image: np.ndarray
    frames: int
    width: int
    height: int
    seed: int
    quality: str
    negative_prompt: str
    guidance: tuple[float, float]
    steps: int


def handle_chat(
    reply: Any,
    runtime: Any,
    body: dict[str, Any],
    selection: Any,
    args: Any,
    state: dict[str, Any],
) -> None:
    del runtime, body, selection, args, state
    _reject(reply, "wan22-i2v generates video at POST /v1/videos", None)


class Pipeline(MediaPipeline):
    """Wan I2V's listener contract. The DiT stays in the runtime."""

    kind = "video"
    id_prefix = "video_"
    object_name = "video"
    table_name = "videos"

    def parse(self, body: dict[str, Any]) -> VideoFields:
        return video_fields(body)

    def describe(self, request: VideoFields) -> dict[str, Any]:
        return {
            "prompt": request.prompt,
            "seconds": _seconds_label(request.frames),
            "size": f"{request.width}x{request.height}",
            "frames": request.frames,
            "quality": request.quality,
            "negative_prompt": request.negative_prompt,
            "guidance_scale": [request.guidance[0], request.guidance[1]],
            "steps": request.steps,
        }

    def identity(self, request: VideoFields) -> tuple[Any, ...]:
        return (
            request.seed,
            request.prompt,
            request.frames,
            request.quality,
            f"{request.width}x{request.height}",
            request.negative_prompt,
            (request.guidance[0], request.guidance[1]),
            request.steps,
        )

    def matches(self, request: VideoFields, identity: Any) -> bool:
        return identity == self.identity(request)

    def generate(
        self,
        runtime: Any,
        request: VideoFields,
        hold: Any,
        job: dict[str, Any],
        stop: Any,
        call: MediaCall,
    ) -> bytes:
        progress = ClipProgress(
            job=job,
            console=getattr(runtime, "console", None),
            clip=clip_label(request.width, request.height, request.frames, FPS),
        )
        progress.hold = hold if isinstance(hold, ClipHold) else None
        progress.interrupt = stop
        call.progress = progress
        if isinstance(hold, ClipHold) and hold.avi is not None:
            avi = hold.avi
        else:
            avi, _timing = runtime.generate(
                request.prompt,
                request.image,
                request.frames,
                request.width,
                request.height,
                request.seed,
                progress,
                negative_prompt=request.negative_prompt,
                guidance=request.guidance,
                steps=request.steps,
            )
        reason = stop()
        if reason:
            raise ClipPaused(reason, ClipHold(stage="pack", avi=avi, done=progress.done))
        return avi

    def pack(
        self,
        runtime: Any,
        request: VideoFields,
        result: bytes,
        job: dict[str, Any],
        home: Path | None,
        table: dict[str, Any],
        call: MediaCall,
    ) -> None:
        del runtime
        progress = call.progress
        if progress is not None:
            progress.stage("mp4")
        write_mp4(store_source(job["id"], result, home=home), request.quality)
        sweep_videos(table, home=home)
        return None

    def on_pause(self, runtime: Any, reason: str, call: MediaCall) -> None:
        del runtime
        progress = call.progress
        if progress is None:
            return
        progress.close("sample paused" if reason == "preempt" else "sample cancelled")

    def on_fail(self, runtime: Any, exc: BaseException, call: MediaCall) -> None:
        del runtime, exc
        progress = call.progress
        if progress is None:
            return
        progress.close(progress.summary(failed=True))

    def on_complete(self, runtime: Any, call: MediaCall) -> None:
        del runtime
        progress = call.progress
        if progress is None:
            return
        progress.complete()
        progress.close(progress.summary())


handle_video = Pipeline()


def video_fields(body: dict[str, Any]) -> VideoFields:
    """One clip request. Omitted size is 320p. Omitted length is 81 frames."""
    prompt = body.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise VideoRequestError("prompt is required", "prompt")
    try:
        image = decode_png(_image_bytes(body))
    except VideoRequestError:
        raise
    except (ValueError, TypeError) as exc:
        raise VideoRequestError("input_reference must be a PNG", "input_reference") from exc
    width, height = _size(body.get("size"))
    frames = _frames(body)
    seed = body.get("seed", 0)
    if isinstance(seed, str) and seed.lstrip("-").isdigit():
        seed = int(seed)
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise VideoRequestError("seed must be an integer", "seed")
    try:
        require_geometry(frames, height, width)
    except ValueError as exc:
        raise VideoRequestError(str(exc), "frames") from exc
    resized = resize_rgb(image, height, width)
    return VideoFields(
        prompt.strip(),
        resized,
        frames,
        width,
        height,
        seed,
        _quality(body),
        _negative(body),
        _guidance(body),
        _steps(body),
    )


def _quality(body: dict[str, Any]) -> str:
    value = body.get("quality")
    if value is None:
        value = "auto"
    if not isinstance(value, str):
        raise VideoRequestError(
            "quality must be auto, low, medium, high, xhigh, max, standard, or hd",
            "quality",
        )
    try:
        return reported_quality(value)
    except ValueError as exc:
        raise VideoRequestError(str(exc), "quality") from exc


def _image_bytes(body: dict[str, Any]) -> bytes:
    ref = body.get("input_reference", body.get("image"))
    if isinstance(ref, dict):
        url = ref.get("image_url")
        if isinstance(url, dict):
            url = url.get("url")
        ref = url
    if not isinstance(ref, str) or not ref:
        raise VideoRequestError("input_reference is required", "input_reference")
    if ref.startswith("data:"):
        _header, _sep, ref = ref.partition(",")
    try:
        return base64.b64decode(ref)
    except (ValueError, TypeError) as exc:
        raise VideoRequestError("input_reference must be a PNG", "input_reference") from exc


def _frames(body: dict[str, Any]) -> int:
    if "duration" in body and body.get("duration") is not None:
        duration = body.get("duration")
        if isinstance(duration, str):
            try:
                duration = float(duration)
            except ValueError as exc:
                raise VideoRequestError("duration must be a number", "duration") from exc
        if isinstance(duration, bool) or not isinstance(duration, (int, float)):
            raise VideoRequestError("duration must be a number", "duration")
        try:
            return frames_from_duration(float(duration))
        except ValueError as exc:
            raise VideoRequestError(str(exc), "duration") from exc
    if "frames" in body and body.get("frames") is not None:
        return _whole(body.get("frames"), "frames")
    if "seconds" in body and body.get("seconds") is not None:
        seconds = body.get("seconds")
        if isinstance(seconds, str):
            try:
                seconds = float(seconds)
            except ValueError as exc:
                raise VideoRequestError("seconds must be a number", "seconds") from exc
        if isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or seconds <= 0:
            raise VideoRequestError("seconds must be a number", "seconds")
        counted = int(round(float(seconds) * FPS))
        return 1 + 4 * int(round((counted - 1) / 4))
    return FIVE_SEC_FRAMES


def _negative(body: dict[str, Any]) -> str:
    value = body.get("negative_prompt", "")
    if value is None:
        value = ""
    if not isinstance(value, str):
        raise VideoRequestError("negative_prompt must be text", "negative_prompt")
    return value.strip()


def _guidance(body: dict[str, Any]) -> tuple[float, float]:
    value = body.get("guidance_scale")
    if value is None:
        return (1.0, 1.0)
    if isinstance(value, (list, tuple)):
        if len(value) == 1:
            low = high = _scale(value[0])
        elif len(value) == 2:
            low, high = _scale(value[0]), _scale(value[1])
        else:
            raise VideoRequestError(
                "guidance_scale must be a number or a low, high pair",
                "guidance_scale",
            )
    else:
        low = high = _scale(value)
    try:
        return require_guidance(low, high)
    except ValueError as exc:
        raise VideoRequestError(str(exc), "guidance_scale") from exc


def _scale(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise VideoRequestError(
            "guidance_scale must be a number or a low, high pair",
            "guidance_scale",
        )
    return float(value)


def _steps(body: dict[str, Any]) -> int:
    if "steps" not in body or body.get("steps") is None:
        return STEPS
    try:
        return require_steps(_whole(body.get("steps"), "steps"))
    except ValueError as exc:
        raise VideoRequestError(str(exc), "steps") from exc


def _whole(value: Any, name: str) -> int:
    if isinstance(value, str) and value.isdigit():
        value = int(value)
    if isinstance(value, bool) or not isinstance(value, int):
        raise VideoRequestError(f"{name} must be an integer", name)
    return value


def _seconds_label(frames: int) -> str:
    text = f"{frames / FPS:.4f}".rstrip("0").rstrip(".")
    return text or "0"


def _size(value: Any) -> tuple[int, int]:
    if value is None:
        return P320_WIDTH, P320_HEIGHT
    if not isinstance(value, str) or "x" not in value:
        raise VideoRequestError("size must be WIDTHxHEIGHT", "size")
    left, _, right = value.partition("x")
    if not left.isdigit() or not right.isdigit():
        raise VideoRequestError("size must be WIDTHxHEIGHT", "size")
    return int(left), int(right)


def _reject(reply: Any, message: str, param: str | None) -> None:
    reply.json(
        400,
        {
            "error": {
                "message": message,
                "type": "invalid_request_error",
                "param": param,
                "code": None,
            }
        },
    )
