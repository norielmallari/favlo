"""Media pipeline for FLUX.2 [klein] 4B on CUDA."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from favlo.images import store_image, store_source
from favlo.media_pipeline import MediaCall, MediaPipeline, MediaRequestError, MediaTool
from favlo.memory import memory
from klein_4b.cuda.commands import main as commands_main
from klein_4b.cuda.constants import DEFAULT_SIDE, DEFAULT_STEPS, MAX_IMAGES, MAX_STEPS
from klein_4b.cuda.generate import generate
from klein_4b.cuda.pixels import parse_size, read_still
from klein_4b.cuda.png import encode_png
from klein_4b.cuda.request import Request
from klein_4b.cuda.runtime import KleinRuntime


class KleinPipeline(MediaPipeline):
    kind = "image"
    id_prefix = "image_"
    object_name = "image"
    table_name = "images"

    def parse(self, body: dict[str, Any]) -> Request:
        prompt = body.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            message = "prompt is required"
            raise MediaRequestError(message, "prompt")
        raw_images = body.get("image")
        images = _images(raw_images)
        if images:
            fallback = (images[0].shape[1], images[0].shape[0])
        else:
            fallback = (DEFAULT_SIDE, DEFAULT_SIDE)
        width, height = parse_size(body.get("size"), fallback=fallback)
        steps = body.get("steps", DEFAULT_STEPS)
        if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1 or steps > MAX_STEPS:
            message = "steps must be from 1 to 50"
            raise MediaRequestError(message, "steps")
        seed = body.get("seed", 0)
        if isinstance(seed, bool) or not isinstance(seed, int):
            message = "seed must be an integer"
            raise MediaRequestError(message, "seed")
        return Request(prompt, width, height, steps, seed, tuple(images))

    def generate(
        self,
        runtime: KleinRuntime,
        request: Request,
        hold: Any,
        job: dict[str, Any],
        stop: Any,
        call: MediaCall,
    ) -> np.ndarray:
        del call
        if runtime.ops is None or runtime.store is None:
            message = "runtime is closed"
            raise RuntimeError(message)
        return generate(runtime.ops, runtime.store, request, hold, job, stop)

    def pack(
        self,
        runtime: KleinRuntime,
        request: Request,
        result: np.ndarray,
        job: dict[str, Any],
        home: Path | None,
        table: dict[str, Any],
        call: MediaCall,
    ) -> dict[str, Any]:
        del runtime, call
        image_id = str(job["id"])
        rgb = np.ascontiguousarray(result, dtype=np.uint8)
        store_source(image_id, rgb.tobytes(), home=home)
        store_image(image_id, encode_png(rgb), home=home, images=table)
        job["size"] = f"{request.width}x{request.height}"
        job["output_format"] = "png"
        return {"data": [{"id": image_id}]}

    def describe(self, request: Request) -> dict[str, Any]:
        return {
            "size": f"{request.width}x{request.height}",
            "output_format": "png",
            "seed": request.seed,
        }

    def identity(self, request: Request) -> tuple[Any, ...]:
        shapes = tuple((int(image.shape[0]), int(image.shape[1])) for image in request.images)
        return (request.prompt, request.width, request.height, request.steps, shapes)

    def matches(self, request: Request, identity: Any) -> bool:
        return identity == self.identity(request)

    def mcp_tools(self) -> Sequence[MediaTool]:
        return (
            MediaTool(
                name="generate_image",
                description="Draw one PNG from a text prompt.",
                parameters=_schema(False),
                handler="handle_image",
                result="image",
            ),
            MediaTool(
                name="edit_image",
                description="Edit up to 4 stills and return one PNG.",
                parameters=_schema(True),
                handler="handle_image",
                result="image",
            ),
        )

    def main(self, argv: Sequence[str]) -> int:
        return commands_main(argv)

    def open_runtime(self, path: Path, **kwargs: Any) -> KleinRuntime:
        memory.set_max_memory(kwargs.get("max_memory"))
        owner = kwargs.get("owner")
        return KleinRuntime(path, owner if isinstance(owner, str) else None)


def _images(value: Any) -> list[np.ndarray]:
    if value is None:
        return []
    if isinstance(value, str):
        items = [value]
    elif isinstance(value, list) and all(isinstance(item, str) for item in value):
        items = value
    else:
        message = "image must be a PNG or a list of PNGs"
        raise MediaRequestError(message, "image")
    if not items or len(items) > MAX_IMAGES:
        message = "image accepts 1 to 4 stills"
        raise MediaRequestError(message, "image")
    return [read_still(item) for item in items]


def _schema(edit: bool) -> dict[str, Any]:
    properties: dict[str, Any] = {
        "prompt": {"type": "string", "description": "Text to draw."},
        "size": {"type": "string", "description": "WxH, multiple of 16."},
        "steps": {"type": "integer", "description": "Denoise steps."},
        "seed": {"type": "integer", "description": "Noise seed."},
    }
    required = ["prompt"]
    if edit:
        properties["image"] = {
            "description": "One still or up to 4.",
            "oneOf": [
                {"type": "string"},
                {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 4},
            ],
        }
        required.append("image")
    return {"type": "object", "properties": properties, "required": required}


pipeline = KleinPipeline()
