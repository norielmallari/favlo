"""Listener shell. Speech stays on the device runtime."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, NamedTuple

from favlo.media_pipeline import (
    MediaCall,
    MediaPipeline,
    MediaRequestError,
    MediaTool,
    checkpoint,
)
from voxcpm2.portable.audio import encode_wav


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
    """This pipeline speaks. It does not complete a chat."""
    del runtime, body, selection, args, state
    reply.json(
        500,
        {
            "error": {
                "message": "voxcpm2 does not complete a chat",
                "type": "RuntimeError",
            }
        },
    )


class VoiceRequest(NamedTuple):
    text: str
    persona: str
    output_format: str


class Pipeline(MediaPipeline):
    """VoxCPM2's listener contract. Speech stays in the runtime."""

    kind = "voice"
    id_prefix = "voice_"
    object_name = "audio.voice"
    table_name = "voice"
    seeds = False

    def parameters(self, request: Any) -> dict[str, Any]:
        """Map the spoken line onto ``input`` for stored metadata."""
        fields = super().parameters(request)
        if "input" in fields:
            return fields
        text = getattr(request, "text", None)
        if isinstance(text, str):
            fields = dict(fields)
            fields["input"] = text
            return fields
        if isinstance(request, dict) and isinstance(request.get("text"), str):
            fields = dict(fields)
            fields["input"] = request["text"]
        return fields

    def parse(self, body: dict[str, Any]) -> VoiceRequest:
        text = body.get("input")
        if not isinstance(text, str) or not text.strip():
            raise MediaRequestError("input is required", "input")
        named = body.get("voice")
        if isinstance(named, str) and named.strip():
            raise MediaRequestError(
                "a named voice needs a reference clip; this opening speaks with an empty voice",
                "voice",
            )
        fmt = body.get("response_format") or "mp3"
        if fmt not in {"mp3", "wav", "flac"}:
            raise MediaRequestError("response_format must be mp3, wav, or flac", "response_format")
        persona = body.get("persona")
        if not isinstance(persona, str) or not persona.strip():
            raise MediaRequestError("persona is required", "persona")
        return VoiceRequest(text.strip(), persona.strip(), fmt)

    def describe(self, request: VoiceRequest) -> dict[str, Any]:
        return {"response_format": request.output_format}

    def generate(
        self,
        runtime: Any,
        request: VoiceRequest,
        hold: Any,
        job: dict[str, Any],
        stop: Any,
        call: MediaCall,
    ) -> Any:
        del hold, call

        def gate() -> None:
            checkpoint(stop, None)

        return runtime.speak(request.text, persona=request.persona, stop=gate, job=job)

    def pack(
        self,
        runtime: Any,
        request: VoiceRequest,
        result: Any,
        job: dict[str, Any],
        home: Path | None,
        table: dict[str, Any],
        call: MediaCall,
    ) -> dict[str, Any]:
        del call
        from favlo.voice import store_wav

        payload = encode_wav(result, runtime.config.out_sample_rate)
        store_wav(job["id"], payload, home=home, voice=table, output_format=request.output_format)
        table[job["id"]] = {"response_format": request.output_format}
        return {
            "id": job["id"],
            "object": "audio.voice",
            "status": "completed",
            "response_format": request.output_format,
        }

    def main(self, argv: Sequence[str]) -> int:
        from voxcpm2.metal41.commands import main as cli_main

        return int(cli_main(argv))

    def open_runtime(self, path: Path, **kwargs: Any) -> Any:
        from favlo.host import select_device
        from favlo.pipelines import bind_open_runtime

        chosen = select_device(_device_packages())
        if chosen != "metal41":
            message = "this runtime cannot run here"
            raise RuntimeError(message)
        from voxcpm2.metal41.runtime import open_device

        return bind_open_runtime(open_device, path, **kwargs)

    def mcp_tools(self) -> Sequence[MediaTool]:
        return [
            MediaTool(
                name="generate_voice",
                description="Speak one line as mp3, wav, or flac.",
                parameters={
                    "type": "object",
                    "properties": {
                        "input": {"type": "string", "description": "Line to speak."},
                        "persona": {
                            "type": "string",
                            "description": "Joined in front of the line.",
                        },
                        "voice": {
                            "type": "string",
                            "description": "Empty speaks with no reference.",
                        },
                        "response_format": {
                            "type": "string",
                            "description": "mp3, wav, or flac.",
                        },
                    },
                    "required": ["input", "persona"],
                },
                handler="handle_voice",
                result="audio",
                task=True,
            )
        ]


pipeline = Pipeline()
handle_voice = pipeline
