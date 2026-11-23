"""Wan2.2 TI2V-5B serve entry. ``open_runtime`` opens Metal or CUDA."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from favlo.bulk_media_pipeline import BulkMediaPipeline
from favlo.media_pipeline import MediaCall, MediaPaused, MediaTool
from wan22_ti2v_5b.portable.request import (
    VideoRequest,
    list_resolutions,
    parse_request,
    video_modes,
)

PIPELINE = "wan22_ti2v_5b"


def _device_packages() -> frozenset[str]:
    """Device directories shipped beside this entry. The host picks one."""
    root = Path(__file__).resolve().parent.parent
    shipped = ("metal41", "cuda")
    return frozenset(name for name in shipped if (root / name / "__init__.py").is_file())


def _video(name: str, description: str, *, still: bool) -> dict[str, Any]:
    properties: dict[str, Any] = {
        "prompt": {"type": "string", "description": "Motion prompt."},
        "size": {"type": "string", "description": "Required WIDTHxHEIGHT."},
        "duration": {"type": "number", "description": "Length in seconds."},
        "frames": {
            "type": "integer",
            "description": "Frame count, 4n+1 and at least 5.",
        },
        "seed": {"type": "integer", "description": "Omit for random."},
        "quality": {"type": "string", "description": "auto, low, medium, high, xhigh, max."},
        "negative_prompt": {"type": "string", "description": "Empty uses the sample negative."},
        "guidance_scale": {"description": "Number or [low, high]; default 1."},
        "steps": {"type": "integer", "description": "Euler 1-40; 0 uses 4."},
    }
    required = ["prompt", "size"]
    if still:
        properties["image"] = {"description": "image_ id, PNG, file://, or name."}
        properties["prompt"] = {"type": "string", "description": "Motion prompt; still is frame 0."}
    return {
        "name": name,
        "description": description,
        "parameters": {"type": "object", "properties": properties, "required": required},
        "task": True,
        "handler": "handle_video",
        "result": "video",
    }


class Pipeline(BulkMediaPipeline):
    """Wan TI2V's listener contract. The clip sampler stays on the Metal device."""

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
        from favlo.host import select_device

        if select_device(_device_packages()) == "cuda":
            from wan22_ti2v_5b.cuda.runtime import pipeline as cuda_pipeline

            return cuda_pipeline.generate(runtime, request, hold, job, stop, call)
        from wan22_ti2v_5b.metal41.clip import ClipHold

        kept = hold if isinstance(hold, ClipHold) else None
        try:
            return runtime.clip.sample(
                request,
                video_id=job["id"],
                home=call.home,
                videos=call.table,
                hold=kept,
                stop=stop,
            )
        except MediaPaused as paused:
            if isinstance(paused.hold, ClipHold):
                job["progress"] = min(99, int(100 * paused.hold.step / 4))
                job["size"] = request.size
                job["seconds"] = request.seconds
            raise

    def pack(self, runtime, request, result, job, home, table, call):
        del runtime, request, job, home, table, call
        if isinstance(result, dict) and result.get("status") == "completed":
            result["progress"] = 100
        if isinstance(result, dict):
            return result
        return None

    def main(self, argv: Sequence[str]) -> int:
        from favlo.host import select_device

        if select_device(_device_packages()) == "cuda":
            from wan22_ti2v_5b.cuda.commands import main as cli_main
        else:
            from wan22_ti2v_5b.metal41.commands import main as cli_main

        return int(cli_main(argv))

    def open_runtime(self, path: Path, **kwargs: Any) -> Any:
        from favlo.host import select_device
        from favlo.pipelines import bind_open_runtime

        chosen = select_device(_device_packages())
        if chosen == "cuda":
            from wan22_ti2v_5b.cuda.runtime import pipeline as cuda_pipeline

            return cuda_pipeline.open_runtime(path, **kwargs)
        if chosen != "metal41":
            message = "this runtime cannot run here"
            raise RuntimeError(message)
        from wan22_ti2v_5b.metal41.runtime import open_device

        return bind_open_runtime(open_device, path, **kwargs)

    def mcp_tools(self) -> Sequence[MediaTool]:
        rows = [
            _video(
                "generate_video_t2v",
                "MP4 from text when the user asked for video or motion, not a still.",
                still=False,
            ),
            _video(
                "generate_video_i2v",
                "MP4 from a still when the user asked for video or motion, not a still.",
                still=True,
            ),
        ]
        return [
            MediaTool(
                name=str(row["name"]),
                description=str(row.get("description") or ""),
                parameters=row.get("parameters") or {"type": "object", "properties": {}},
                handler=str(row["handler"]),
                result=str(row.get("result") or "video"),
                task=bool(row.get("task", True)),
            )
            for row in rows
        ]


pipeline = Pipeline()
handle_video = pipeline
main = pipeline.main
open_runtime = pipeline.open_runtime


def handle_chat(reply, runtime, body, selection, args, state) -> None:
    del runtime, body, selection, args, state
    reply.json(400, {"error": "wan22_ti2v_5b does not complete a chat"})


def mcp_tools():
    return pipeline.mcp_tools()


__all__ = [
    "pipeline",
    "handle_chat",
    "handle_video",
    "list_resolutions",
    "main",
    "mcp_tools",
    "open_runtime",
    "video_modes",
]
