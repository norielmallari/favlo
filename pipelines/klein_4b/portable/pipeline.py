"""Text-to-image and image-edit request."""

from __future__ import annotations

import secrets
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from favlo.images import load_image, store_source, sweep_images
from favlo.media_pipeline import MediaPipeline, MediaRequestError, MediaTool

from .constants import (
    ALIGN,
    DEFAULT_SIDE,
    DEFAULT_STEPS,
    MAX_IMAGES,
    MAX_SIDE,
    MAX_STEPS,
    MIN_SIDE,
)
from .png import pack_rgb, write_png


def _device_packages() -> frozenset[str]:
    """Device directories shipped beside this entry. The host picks one."""
    root = Path(__file__).resolve().parent.parent
    return frozenset(
        name for name in ("metal41", "cuda") if (root / name / "__init__.py").is_file()
    )


@dataclass(frozen=True)
class EditRequest:
    prompt: str
    images: tuple[str, ...]
    steps: int
    width: int | None
    height: int | None
    seed: int | None


class Pipeline(MediaPipeline):
    kind = "image"
    id_prefix = "image_"
    object_name = "image"
    table_name = "images"

    def parse(self, body: dict[str, Any]) -> EditRequest:
        prompt = body.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise MediaRequestError("prompt is required", "prompt")
        images = _images(body)
        steps = _int(body.get("steps", DEFAULT_STEPS), "steps", low=1, high=MAX_STEPS)
        width, height = _size(body.get("size"))
        if not images and width is None:
            width = height = DEFAULT_SIDE
        seed = body.get("seed")
        if seed is not None:
            seed = _int(seed, "seed", low=0, high=2**31 - 1)
        return EditRequest(prompt.strip(), images, steps, width, height, seed)

    def identity(self, request: EditRequest) -> tuple[Any, ...]:
        return (
            request.prompt,
            request.images,
            request.steps,
            request.width,
            request.height,
            request.seed,
        )

    def matches(self, request: EditRequest, identity: Any) -> bool:
        return identity == self.identity(request)

    def describe(self, request: EditRequest) -> dict[str, str]:
        if request.width is None or request.height is None:
            return {"size": "auto"}
        return {"size": f"{request.width}x{request.height}"}

    def generate(self, runtime, request, hold, job, stop, call):
        del call
        return runtime.edit(request, hold, stop, job)

    def pack(self, runtime, request, result, job, home, table, call):
        del runtime, request, call
        image_id = job.get("id") if isinstance(job, dict) else None
        if not isinstance(image_id, str) or not image_id.startswith(self.id_prefix):
            image_id = self.id_prefix + secrets.token_hex(self.id_hex)
        source = store_source(image_id, pack_rgb(result), home=home)
        write_png(source)
        sweep_images(table if isinstance(table, dict) else None, home=home)
        stored = load_image(image_id, home=home)
        if stored is None:
            message = "stored image is missing"
            raise RuntimeError(message)
        height, width = result.shape[:2]
        return {
            "created": int(time.time()),
            "data": [{"id": image_id}],
            "size": f"{width}x{height}",
        }

    def main(self, argv: Sequence[str]) -> int:
        from .commands import main as cli_main

        return int(cli_main(list(argv)))

    def open_runtime(self, path: Path, **kwargs: Any) -> Any:
        from favlo.host import select_device
        from favlo.pipelines import bind_open_runtime

        chosen = select_device(_device_packages())
        if chosen == "cuda":
            return bind_open_runtime(_open_cuda, path, **kwargs)
        if chosen != "metal41":
            message = "this runtime cannot run here"
            raise RuntimeError(message)
        from klein_4b.metal41.runtime import open_device

        return bind_open_runtime(open_device, path, **kwargs)

    def mcp_tools(self) -> Sequence[MediaTool]:
        return [
            MediaTool(
                name="generate_image",
                description="Draw one PNG from a text prompt.",
                parameters={
                    "type": "object",
                    "properties": {
                        "prompt": {"type": "string", "description": "Scene to draw."},
                        "size": {"type": "string", "description": "WxH. Default 1024x1024."},
                        "steps": {"type": "integer", "description": "Euler steps. Default 4."},
                        "seed": {"type": "integer", "description": "Omit for a random seed."},
                    },
                    "required": ["prompt"],
                },
                handler="handle_image",
                result="image",
            ),
            MediaTool(
                name="edit_image",
                description="Edit up to 4 stills from an instruction and return one PNG.",
                parameters={
                    "type": "object",
                    "properties": {
                        "prompt": {"type": "string", "description": "Edit instruction."},
                        "image": {
                            "description": "1 to 4 ids, PNGs, file://, or names.",
                            "oneOf": [
                                {"type": "string"},
                                {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "minItems": 1,
                                    "maxItems": MAX_IMAGES,
                                },
                            ],
                        },
                        "size": {"type": "string", "description": "WxH, multiple of 16."},
                        "steps": {"type": "integer", "description": "Euler steps. Default 4."},
                        "seed": {"type": "integer", "description": "Omit for a random seed."},
                    },
                    "required": ["prompt", "image"],
                },
                handler="handle_image",
                result="image",
            )
        ]


class _CudaRuntime:
    """Portable request in, CUDA canvas out. The Metal runtime keeps ``edit``."""

    def __init__(self, path: Path, owner: str | None) -> None:
        from klein_4b.cuda.runtime import KleinRuntime

        self._inner = KleinRuntime(path, owner)
        self._request: Any = None

    def ensure_open(self, request: EditRequest) -> None:
        self._request = _cuda_request(request)
        self._inner.ensure_open(self._request)

    def edit(self, request: EditRequest, hold: Any, stop: Any, job: dict[str, Any] | None) -> Any:
        if self._request is None:
            self.ensure_open(request)
        if self._inner.ops is None or self._inner.store is None:
            message = "runtime is closed"
            raise RuntimeError(message)
        from klein_4b.cuda.generate import generate

        return generate(
            self._inner.ops,
            self._inner.store,
            self._request,
            hold,
            job if isinstance(job, dict) else {},
            stop,
        )

    def close(self) -> None:
        self._inner.close()


def _open_cuda(path: Path, *, owner: str | None = None, **kwargs: Any) -> _CudaRuntime:
    del kwargs
    return _CudaRuntime(path, owner if isinstance(owner, str) else None)


def _cuda_request(request: EditRequest) -> Any:
    from klein_4b.cuda.pixels import read_still
    from klein_4b.cuda.request import Request

    images = tuple(read_still(item) for item in request.images)
    width = request.width
    height = request.height
    if images and (width is None or height is None):
        height, width = int(images[0].shape[0]), int(images[0].shape[1])
    if width is None or height is None:
        width = height = DEFAULT_SIDE
    seed = 0 if request.seed is None else int(request.seed)
    return Request(request.prompt, int(width), int(height), int(request.steps), seed, images)


pipeline = Pipeline()
handle_image = pipeline


def _images(body: dict[str, Any]) -> tuple[str, ...]:
    if "image" not in body and "images" not in body:
        return ()
    raw = body.get("image", body.get("images"))
    if isinstance(raw, str) and raw.strip():
        return (raw.strip(),)
    if isinstance(raw, list) and raw and all(isinstance(item, str) and item.strip() for item in raw):
        images = tuple(item.strip() for item in raw)
        if len(images) > MAX_IMAGES:
            raise MediaRequestError(f"image accepts at most {MAX_IMAGES} images", "image")
        return images
    raise MediaRequestError("image is required", "image")


def _size(value: Any) -> tuple[int | None, int | None]:
    if value is None or value == "auto":
        return None, None
    if not isinstance(value, str) or "x" not in value:
        raise MediaRequestError("size must be WxH", "size")
    width_text, height_text = value.lower().split("x", 1)
    if not width_text.isdigit() or not height_text.isdigit():
        raise MediaRequestError("size must be WxH", "size")
    width = _int(int(width_text), "size", low=MIN_SIDE, high=MAX_SIDE)
    height = _int(int(height_text), "size", low=MIN_SIDE, high=MAX_SIDE)
    if width % ALIGN or height % ALIGN:
        raise MediaRequestError("size must be a multiple of 16", "size")
    return width, height


def _int(value: Any, name: str, *, low: int, high: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < low or value > high:
        raise MediaRequestError(f"{name} must be an integer from {low} to {high}", name)
    return value
