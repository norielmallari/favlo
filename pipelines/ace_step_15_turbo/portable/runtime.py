"""Serve shell. The Metal device opens from ``open_runtime``."""

from __future__ import annotations

import time
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from favlo.media_pipeline import MediaCall, MediaPipeline, MediaRequestError, MediaTool, checkpoint

if TYPE_CHECKING:
    from ace_step_15_turbo.portable.request import MusicRequest


def _device_packages() -> frozenset[str]:
    """Device directories shipped beside this entry. The host picks one."""
    root = Path(__file__).resolve().parent.parent
    return frozenset(
        name for name in ("metal41", "cuda") if (root / name / "__init__.py").is_file()
    )


def handle_chat(
    reply: Any,
    runtime: Any,
    body: dict[str, Any],
    selection: Any,
    args: Any,
    state: dict[str, Any],
) -> None:
    del runtime, body, selection, args, state
    reply.json(
        404,
        {
            "error": {
                "message": "ace_step_15_turbo generates music at POST /v1/audio/music",
                "type": "not_found",
            }
        },
    )


class Pipeline(MediaPipeline):
    """ACE-Step's listener contract. The graph stays in this package."""

    kind = "music"
    id_prefix = "music_"
    object_name = "audio.music"
    table_name = "music"
    id_hex = 16

    def parse(self, body: dict[str, Any]) -> MusicRequest:
        from ace_step_15_turbo.portable.request import parse_music

        model = body.get("model")
        if not isinstance(model, str) or not model:
            raise MediaRequestError("model is required", "model")
        try:
            return parse_music(body)
        except ValueError as exc:
            raise MediaRequestError(str(exc)) from exc

    def describe(self, request: MusicRequest) -> dict[str, Any]:
        from favlo.logs import favlo_home, load_settings, music_ttl_seconds

        seconds = request.seconds
        label = str(int(seconds)) if float(seconds).is_integer() else str(seconds)
        created = int(time.time())
        return {
            "expires_at": created + music_ttl_seconds(load_settings(favlo_home())),
            "stage": "text",
            "prompt": request.prompt,
            "lyrics": request.lyrics,
            "seconds": label,
            "response_format": request.output_format,
        }

    def identity(self, request: MusicRequest) -> tuple[Any, ...]:
        return _identity(request)

    def matches(self, request: MusicRequest, identity: Any) -> bool:
        return identity == self.identity(request)

    def halted(self, runtime: Any) -> bool:
        return bool(getattr(runtime, "_halted", False))

    def generate(
        self,
        runtime: Any,
        request: MusicRequest,
        hold: Any,
        job: dict[str, Any],
        stop: Any,
        call: MediaCall,
    ) -> bytes:
        from ace_step_15_turbo.metal41.hold import ClipHold

        kept = hold if isinstance(hold, ClipHold) else None
        progress = _Progress(job, done=0 if kept is None else kept.done)
        progress.hold = kept
        progress.interrupt = stop
        call.progress = progress
        if kept is not None and kept.wav is not None:
            wav = kept.wav
        else:
            wav = runtime.generate(request, progress, kept)
        checkpoint(stop, ClipHold(stage="encode", wav=wav, done=progress.done))
        return wav

    def pack(
        self,
        runtime: Any,
        request: MusicRequest,
        result: bytes,
        job: dict[str, Any],
        home: Path | None,
        table: dict[str, Any],
        call: MediaCall,
    ) -> None:
        del runtime, call
        from favlo.music import store_wav

        job["stage"] = "encode"
        store_wav(job["id"], result, home=home, music=table, output_format=request.output_format)
        return None

    def main(self, argv: Sequence[str]) -> int:
        from ace_step_15_turbo.metal41.commands import main as cli_main

        return int(cli_main(argv))

    def open_runtime(self, path: Path, **kwargs: Any) -> Any:
        from favlo.host import select_device
        from favlo.pipelines import bind_open_runtime

        chosen = select_device(_device_packages())
        if chosen != "metal41":
            message = "this runtime cannot run here"
            raise RuntimeError(message)
        from ace_step_15_turbo.metal41.runtime import open_device

        return bind_open_runtime(open_device, path, **kwargs)

    def mcp_tools(self) -> Sequence[MediaTool]:
        return [
            MediaTool(
                name="generate_music",
                description="One music clip as mp3, wav, or flac.",
                parameters={
                    "type": "object",
                    "properties": {
                        "prompt": {"type": "string", "description": "Caption or style."},
                        "response_format": {"type": "string", "description": "mp3, wav, or flac."},
                        "lyrics": {"type": "string", "description": "Lyrics, when used."},
                        "seconds": {"type": "number", "description": "Duration in seconds."},
                        "seed": {"type": "integer", "description": "Omit for random."},
                        "vocal_language": {"type": "string", "description": "Vocal language."},
                        "bpm": {"type": "integer", "description": "Tempo."},
                        "keyscale": {"type": "string", "description": "Key."},
                        "timesignature": {"type": "string", "description": "Meter."},
                    },
                    "required": ["prompt"],
                },
                handler="handle_music",
                result="audio",
                task=True,
            )
        ]


pipeline = Pipeline()
handle_music = pipeline


class _Progress:
    def __init__(self, job: dict[str, Any], done: int = 0) -> None:
        self.job = job
        self.done = done
        self.started = time.monotonic()
        self.interrupt = lambda: None

    def __call__(self, stage: str) -> None:
        self.done += 1
        self.job["stage"] = stage
        units = _unit_count()
        self.job["progress"] = min(99, int(100 * self.done / units))
        elapsed = time.monotonic() - self.started
        remain = units - self.done
        self.job["eta"] = 0 if self.done == 0 or remain <= 0 else int(elapsed / self.done * remain)


def _unit_count() -> int:
    from ace_step_15_turbo.metal41.schedule import STEPS

    return 28 + 8 + 4 + 32 * STEPS + 7


def _identity(fields: MusicRequest) -> tuple[Any, ...]:
    return (
        fields.prompt,
        fields.lyrics,
        fields.seconds,
        fields.seed,
        fields.vocal_language,
        fields.bpm,
        fields.keyscale,
        fields.timesignature,
        fields.output_format,
    )
