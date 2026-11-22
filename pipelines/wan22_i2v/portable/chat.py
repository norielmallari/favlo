"""Chat is refused. Video is one MP4 job. The listener turns the reply into HTTP."""

from __future__ import annotations

import base64
import hashlib
from collections.abc import Sequence
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np

from favlo.bulk_media_pipeline import BulkMediaPipeline
from favlo.media_pipeline import MediaCall, MediaRequestError, MediaTool, checkpoint
from favlo.videos import reported_quality, store_source, sweep_videos
from wan22_i2v.portable.progress import ClipProgress, clip_label
from wan22_i2v.portable.schedule import (
    FIVE_SEC_FRAMES,
    FPS,
    P320_HEIGHT,
    P320_WIDTH,
    STEPS,
    apply_split,
    frames_from_duration,
    require_geometry,
    require_guidance,
    require_steps,
)


def _device_packages() -> frozenset[str]:
    """Device directories shipped beside this entry. The host picks one."""
    root = Path(__file__).resolve().parent.parent
    return frozenset(
        name for name in ("metal41", "cuda") if (root / name / "__init__.py").is_file()
    )


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
    last_image: np.ndarray | None = None
    split: int | None = None
    overlap: int | None = None
    social: bool = False
    mp4: bool = True


def handle_chat(
    reply: Any,
    runtime: Any,
    body: dict[str, Any],
    selection: Any,
    args: Any,
    state: dict[str, Any],
) -> None:
    del runtime, body, selection, args, state
    _reject(reply, "wan22_i2v generates video at POST /v1/videos", None)


class Pipeline(BulkMediaPipeline):
    """Wan I2V's listener contract. The DiT stays in the runtime."""

    kind = "video"
    id_prefix = "video_"
    object_name = "video"
    table_name = "videos"

    def parse(self, body: dict[str, Any]) -> VideoFields:
        return video_fields(body)

    def describe(self, request: VideoFields) -> dict[str, Any]:
        fields: dict[str, Any] = {
            "prompt": request.prompt,
            "seconds": _seconds_label(request.frames),
            "size": f"{request.width}x{request.height}",
            "frames": request.frames,
            "quality": request.quality,
            "negative_prompt": request.negative_prompt,
            "guidance_scale": [request.guidance[0], request.guidance[1]],
            "steps": request.steps,
        }
        if request.last_image is not None:
            fields["last_image"] = True
        if request.split is not None:
            fields["split"] = request.split
            fields["overlap"] = request.overlap
        if request.social:
            fields["social"] = True
        if not request.mp4:
            fields["mp4"] = False
        return fields

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
            _still_mark(request.last_image),
            request.split,
            request.overlap,
            request.social,
            request.mp4,
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
        from wan22_i2v.metal41.hold import ClipHold

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
                last_image=request.last_image,
                split=request.split,
                overlap=request.overlap,
            )
        checkpoint(stop, ClipHold(stage="pack", avi=avi, done=progress.done))
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
        from wan22_i2v.metal41.video import write_mp4

        progress = call.progress
        source = store_source(job["id"], result, home=home)
        if request.mp4:
            if progress is not None:
                progress.stage("mp4")
            write_mp4(source, request.quality, social=request.social)
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



    def main(self, argv: Sequence[str]) -> int:
        from wan22_i2v.metal41.commands import main as cli_main

        return int(cli_main(argv))

    def open_runtime(self, path: Path, **kwargs: Any) -> Any:
        from favlo.host import select_device
        from favlo.pipelines import bind_open_runtime

        chosen = select_device(_device_packages())
        if chosen != "metal41":
            message = "this runtime cannot run here"
            raise RuntimeError(message)
        from wan22_i2v.metal41.runtime import open_device

        return bind_open_runtime(open_device, path, **kwargs)

    def mcp_tools(self) -> Sequence[MediaTool]:
        return [
            MediaTool(
                name="generate_video",
                description="MP4 from a still when the user asked for video or motion, not a still.",
                parameters={
                    "type": "object",
                    "properties": {
                        "prompt": {"type": "string", "description": "Motion prompt; still is frame 0."},
                        "size": {"type": "string", "description": "Required WIDTHxHEIGHT."},
                        "image": {"description": "image_ id, PNG, file://, or name."},
                        "duration": {"type": "number", "description": "Length in seconds."},
                        "seed": {"type": "integer", "description": "Omit for random."},
                        "quality": {"type": "string", "description": "auto, low, medium, high, xhigh, max."},
                        "negative_prompt": {"type": "string", "description": "Empty uses the sample negative."},
                        "guidance_scale": {"description": "Number or [low, high]; default 1."},
                        "steps": {"type": "integer", "description": "Euler 1-40; 0 uses 4."},
                        "last_image": {"description": "Optional end still, same forms as image."},
                        "split": {"type": "integer", "description": "How many full denoises. Omit for one."},
                        "overlap": {
                            "type": "integer",
                            "description": "Shared latent frames. 0 abuts the slices and averages the doubled boundary frame. Default 1.",
                        },
                        "social": {
                            "type": "boolean",
                            "description": "Pad to 1080p, 30 fps minterpolate, H.264 High for short-form upload.",
                        },
                        "mp4": {
                            "type": "boolean",
                            "description": "False stores the AVI and skips the MP4.",
                        },
                    },
                    "required": ["prompt", "size", "image"],
                },
                handler="handle_video",
                result="video",
                task=True,
            )
        ]

pipeline = Pipeline()
handle_video = pipeline
def video_fields(body: dict[str, Any]) -> VideoFields:
    """One clip request. Omitted size is 320p. Omitted length is 81 frames."""
    from wan22_i2v.metal41.png import decode_png, resize_rgb

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
    split, overlap = _split(body)
    share = 1 if overlap is None else overlap
    try:
        frames, _context = apply_split(frames, split, share)
    except ValueError as exc:
        param = "overlap" if str(exc).startswith("overlap") else "split"
        raise VideoRequestError(str(exc), param) from exc
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
    last = _last_image(body, height, width, frames)
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
        last,
        split,
        overlap,
        _social(body),
        _mp4(body),
    )


def _mp4(body: dict[str, Any]) -> bool:
    """True encodes the served MP4. False leaves the uncompressed AVI."""
    value = body.get("mp4")
    if value is None:
        return True
    if isinstance(value, bool):
        kept = value
    elif isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in ("1", "true", "yes", "on"):
            kept = True
        elif lowered in ("0", "false", "no", "off", ""):
            kept = False
        else:
            raise VideoRequestError("mp4 must be a boolean", "mp4")
    else:
        raise VideoRequestError("mp4 must be a boolean", "mp4")
    if not kept and _social(body):
        raise VideoRequestError("social encodes an MP4", "mp4")
    return kept


def _social(body: dict[str, Any]) -> bool:
    value = body.get("social")
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in ("1", "true", "yes", "on"):
            return True
        if lowered in ("0", "false", "no", "off", ""):
            return False
    raise VideoRequestError("social must be a boolean", "social")


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


def _still_mark(image: np.ndarray | None) -> str | None:
    if image is None:
        return None
    raw = np.ascontiguousarray(image).tobytes()
    return hashlib.sha256(raw).hexdigest()[:16]


def _last_image(body: dict[str, Any], height: int, width: int, frames: int) -> np.ndarray | None:
    raw = body.get("last_image")
    if raw in (None, ""):
        raw = body.get("end_image")
    if raw in (None, ""):
        return None
    from wan22_i2v.metal41.png import decode_png, resize_rgb

    try:
        image = decode_png(_png_bytes(raw, "last_image"))
    except VideoRequestError:
        raise
    except (ValueError, TypeError) as exc:
        raise VideoRequestError("last_image must be a PNG", "last_image") from exc
    if frames < 5:
        raise VideoRequestError("a last frame needs at least 5 frames", "last_image")
    return resize_rgb(image, height, width)


def _split(body: dict[str, Any]) -> tuple[int | None, int | None]:
    if body.get("context") is not None:
        raise VideoRequestError("pass split instead of context", "context")
    raw = body.get("split")
    if raw is None:
        if body.get("overlap") is not None:
            raise VideoRequestError("overlap needs a split", "overlap")
        return None, None
    if isinstance(raw, str) and raw.lstrip("-").isdigit():
        raw = int(raw)
    if isinstance(raw, bool) or not isinstance(raw, int):
        raise VideoRequestError("split must be an integer", "split")
    if raw < 1:
        raise VideoRequestError(f"split must be at least 1, got {raw}", "split")
    overlap = body.get("overlap", 1)
    if isinstance(overlap, str) and overlap.lstrip("-").isdigit():
        overlap = int(overlap)
    if isinstance(overlap, bool) or not isinstance(overlap, int):
        raise VideoRequestError("overlap must be an integer", "overlap")
    if overlap < 0:
        raise VideoRequestError(f"overlap must be at least 0, got {overlap}", "overlap")
    return raw, overlap


def _image_bytes(body: dict[str, Any]) -> bytes:
    ref = body.get("input_reference", body.get("image"))
    return _png_bytes(ref, "input_reference")


def _png_bytes(ref: Any, param: str) -> bytes:
    if isinstance(ref, dict):
        url = ref.get("image_url")
        if isinstance(url, dict):
            url = url.get("url")
        ref = url
    if not isinstance(ref, str) or not ref:
        raise VideoRequestError(f"{param} is required", param)
    if ref.startswith("file://"):
        from favlo.local_uri import read_file_uri

        try:
            return read_file_uri(ref)
        except (OSError, ValueError) as exc:
            raise VideoRequestError(f"{param} must be a PNG", param) from exc
    named = _named_file(ref, param)
    if named is not None:
        return named
    if ref.startswith("data:"):
        _header, _sep, ref = ref.partition(",")
    try:
        return base64.b64decode(ref)
    except (ValueError, TypeError) as exc:
        raise VideoRequestError(f"{param} must be a PNG", param) from exc


def _named_file(value: str, param: str) -> bytes | None:
    from favlo.media_name import MissingMediaFile, read_plain_filename

    try:
        return read_plain_filename(value)
    except MissingMediaFile as exc:
        raise VideoRequestError(str(exc), param) from exc


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
