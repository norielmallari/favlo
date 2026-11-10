"""Open a VoxCPM2 checkout and speak one line."""

from __future__ import annotations

from collections.abc import Sequence

from pathlib import Path
from typing import Any, NamedTuple

import numpy as np

from favlo.media_memory import begin_release, finish_release, open_gate, stamp_owner
from favlo.media_pipeline import MediaTool, MediaCall, MediaPipeline, MediaRequestError, checkpoint
from favlo.memory import memory
from voxcpm2.metal41.audio import encode_wav
from voxcpm2.metal41.config import ModelConfig, load_config
from voxcpm2.metal41.metal import Session
from voxcpm2.metal41.speak import generate
from voxcpm2.metal41.store import Store
from voxcpm2.metal41.text import style_line
from voxcpm2.metal41.tokenize import LineTokenizer
from voxcpm2.metal41.vae import Decoder

_SCRATCH = 256 << 20


class Runtime:
    """The published checkout. Weights move to the device on the first line."""

    def __init__(
        self,
        path: Path,
        config: ModelConfig,
        *,
        owner: str,
        kv_cache: str,
        resident_percent: float | None,
        console: Any,
    ) -> None:
        self.path = path
        self.config = config
        self._owner = owner
        self.kv_cache = kv_cache
        self.resident_percent = resident_percent
        self.console = console
        self._session: Session | None = None
        self._store: Store | None = None
        self._decoder: Decoder | None = None
        self._tokenizer: LineTokenizer | None = None

    def ensure_open(self, request: Any) -> None:
        """Wait for prior releases, then open when the line's working set fits."""
        del request
        need = self.working_set_bytes(None)
        open_gate(need, detail="voice line")
        self._open()

    def working_set_bytes(self, request: Any) -> int:
        """Weight files plus the fixed scratch reservation."""
        del request
        weights = self.path / "model.safetensors"
        vae_path = self.path / "audiovae.pth"
        return weights.stat().st_size + vae_path.stat().st_size + _SCRATCH

    def close(self) -> None:
        if (
            self._session is None
            and self._store is None
            and self._decoder is None
            and self._tokenizer is None
        ):
            return
        before, held = begin_release(self._owner)
        try:
            if self._session is not None:
                self._session.close()
                self._session = None
            if self._store is not None:
                self._store.close()
                self._store = None
            self._decoder = None
            self._tokenizer = None
        finally:
            finish_release(self._owner, before, held)

    def speak(
        self,
        text: str,
        *,
        instructions: str = "",
        max_len: int = 48,
        min_len: int = 2,
        steps: int = 10,
        seed: int | None = None,
        stop: Any = None,
    ) -> np.ndarray:
        """One waveform. ``instructions`` is a parenthetical style prefix."""
        line = style_line(text, instructions).strip()
        if not line:
            message = "input is required"
            raise ValueError(message)
        session, store, decoder, tokenizer = self._open()
        return generate(
            session,
            store,
            decoder,
            tokenizer,
            self.config,
            line,
            max_len=max_len,
            min_len=min_len,
            steps=steps,
            seed=seed,
            stop=stop,
        )

    def _open(self) -> tuple[Session, Store, Decoder, LineTokenizer]:
        ready = (
            self._session is not None
            and self._store is not None
            and self._decoder is not None
            and self._tokenizer is not None
        )
        if ready:
            return self._session, self._store, self._decoder, self._tokenizer
        vae_path = self.path / "audiovae.pth"
        tokenizer_path = self.path / "tokenizer.json"
        if not vae_path.is_file():
            message = f"{self.path.name} has no audiovae.pth"
            raise FileNotFoundError(message)
        if not tokenizer_path.is_file():
            message = f"{self.path.name} has no tokenizer.json"
            raise FileNotFoundError(message)
        need = self.working_set_bytes(None)
        if memory.room() < need:
            message = f"a voice line does not fit in {memory.room()} bytes"
            raise RuntimeError(message)
        self._session = Session(self._owner)
        try:
            self._store = Store(self.path / "model.safetensors")
            self._decoder = Decoder(
                self._session,
                vae_path,
                self.config.decoder_rates,
                self.config.sr_boundaries,
                self.config.out_sample_rate,
            )
            self._tokenizer = LineTokenizer(tokenizer_path)
        except Exception:
            self.close()
            raise
        return self._session, self._store, self._decoder, self._tokenizer


def open_runtime(
    path: str | Path,
    *,
    kv_cache: str = "e4m3",
    console: Any = None,
    resident_percent: float | None = None,
    max_memory: str | None = None,
    owner: str | None = None,
) -> Runtime:
    """Load ``config.json`` beside ``model.safetensors``. A missing file raises."""
    checkout = Path(path)
    weights = checkout / "model.safetensors"
    if not weights.is_file():
        message = f"{checkout.name} has no model.safetensors"
        raise FileNotFoundError(message)
    config_path = checkout / "config.json"
    if not config_path.is_file():
        message = f"{checkout.name} has no config.json"
        raise FileNotFoundError(message)
    memory.set_max_memory(max_memory)
    percent = None if resident_percent is None else float(resident_percent)
    return Runtime(
        checkout,
        load_config(config_path),
        owner=stamp_owner("voxcpm2", owner, id(checkout)),
        kv_cache=kv_cache,
        resident_percent=percent,
        console=console,
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
    instructions: str
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
        instructions = body.get("instructions")
        if instructions is None:
            instructions = ""
        if not isinstance(instructions, str):
            raise MediaRequestError("instructions must be a string", "instructions")
        return VoiceRequest(text, instructions, fmt)

    def describe(self, request: VoiceRequest) -> dict[str, Any]:
        return {"response_format": request.output_format}

    def generate(self, runtime: Runtime, request: VoiceRequest, hold: Any, job: dict[str, Any], stop: Any, call: MediaCall):
        del hold, job, call

        def gate() -> None:
            checkpoint(stop, None)

        return runtime.speak(request.text, instructions=request.instructions, stop=gate)

    def pack(self, runtime: Runtime, request: VoiceRequest, result: Any, job: dict[str, Any], home: Path | None, table: dict[str, Any], call: MediaCall):
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
        from voxcpm2.metal41.runtime import open_runtime as open_rt
        from favlo.pipelines import bind_open_runtime

        return bind_open_runtime(open_rt, path, **kwargs)

    def mcp_tools(self) -> Sequence[MediaTool]:
        return [
            MediaTool(
                name="generate_voice",
                description="Speak one line as mp3, wav, or flac.",
                parameters={
                    "type": "object",
                    "properties": {
                        "input": {"type": "string", "description": "Line to speak."},
                        "voice": {"type": "string", "description": "Empty speaks with no reference."},
                        "instructions": {"type": "string", "description": "Style prefix."},
                        "response_format": {"type": "string", "description": "mp3, wav, or flac."},
                    },
                    "required": ["input"],
                },
                handler="handle_voice",
                result="audio",
                task=True,
            )
        ]

pipeline = Pipeline()
handle_voice = pipeline