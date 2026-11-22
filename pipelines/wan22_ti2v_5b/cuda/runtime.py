"""CUDA runtime. Windows loads this package. The forward stays on the int4 record."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from favlo.media_memory import open_gate
from favlo.media_pipeline import MediaCall, MediaPipeline, MediaTool
from favlo.pipeline import Runtime
from wan22_ti2v_5b.cuda.layout import BUDGET, fits, opening_bytes
from wan22_ti2v_5b.cuda.request import VideoRequest, list_resolutions, parse_request, video_modes

_TOOLS = (
    (
        "generate_video_t2v",
        "MP4 from text when the user asked for video or motion, not a still.",
        False,
    ),
    (
        "generate_video_i2v",
        "MP4 from a still when the user asked for video or motion, not a still.",
        True,
    ),
)


class CudaRuntime(Runtime):
    def __init__(self, path: Path, owner: str) -> None:
        self.path = path
        self._owner = owner
        self._closed = False

    def ensure_open(self, request: VideoRequest) -> None:
        need = opening_bytes(request.frames, request.height, request.width)
        if need > BUDGET:
            message = (
                f"{request.frames} frames at {request.width}x{request.height} "
                f"exceeds the 6 GiB opening ({need} bytes)"
            )
            raise RuntimeError(message)
        open_gate(need, detail=f"{request.frames} frames at {request.width}x{request.height}")
        if not self.path.is_dir():
            message = f"int4 record is missing: {self.path}"
            raise FileNotFoundError(message)

    def close(self) -> None:
        self._closed = True


class Pipeline(MediaPipeline):
    kind = "video"
    id_prefix = "video_"
    object_name = "video"
    table_name = "videos"
    id_hex = 4

    def parse(self, body: dict) -> VideoRequest:
        return parse_request(body)

    def describe(self, request: VideoRequest) -> dict:
        return {
            "prompt": request.prompt,
            "seconds": request.seconds,
            "size": request.size,
            "frames": request.frames,
            "quality": request.quality,
        }

    def generate(self, runtime, request, hold, job, stop, call: MediaCall):
        del runtime, hold, stop, call
        if not fits(request.frames, request.height, request.width):
            message = f"{request.frames} frames at {request.size} exceeds the 6 GiB opening"
            raise RuntimeError(message)
        job["progress"] = 0
        job["size"] = request.size
        job["seconds"] = request.seconds
        message = "int4 DiT forward is not packed"
        raise RuntimeError(message)

    def pack(self, runtime, request, result, job, home, table, call):
        del runtime, request, job, home, table, call
        if isinstance(result, dict):
            return result
        return None

    def main(self, argv: Sequence[str]) -> int:
        from wan22_ti2v_5b.cuda.commands import main as cli_main

        return int(cli_main(argv))

    def open_runtime(self, path: Path, **kwargs: Any) -> CudaRuntime:
        from favlo.pipelines import bind_open_runtime

        return bind_open_runtime(_open, path, **kwargs)

    def mcp_tools(self) -> Sequence[MediaTool]:
        rows = []
        for name, description, still in _TOOLS:
            properties: dict[str, Any] = {
                "prompt": {"type": "string", "description": "Motion prompt."},
                "size": {"type": "string", "description": "Required WIDTHxHEIGHT."},
                "duration": {"type": "number", "description": "Length in seconds."},
                "seed": {"type": "integer", "description": "Omit for random."},
                "quality": {"type": "string", "description": "auto, low, medium, high, xhigh, max."},
                "steps": {"type": "integer", "description": "Euler 0 or 4."},
            }
            required = ["prompt", "size"]
            if still:
                properties["image"] = {"description": "image_ id, PNG, file://, or name."}
                required.append("image")
            rows.append(
                MediaTool(
                    name=name,
                    description=description,
                    parameters={"type": "object", "properties": properties, "required": required},
                    handler="handle_video",
                    result="video",
                    task=True,
                )
            )
        return rows


def _open(path: Path, **kwargs: Any) -> CudaRuntime:
    owner = kwargs.get("owner")
    if not isinstance(owner, str) or not owner:
        owner = f"wan22_ti2v_5b:{id(path)}"
    return CudaRuntime(Path(path), owner)


pipeline = Pipeline()


__all__ = [
    "pipeline",
    "list_resolutions",
    "video_modes",
]
