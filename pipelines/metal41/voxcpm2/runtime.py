"""Open a VoxCPM2 checkout and speak one line."""

from __future__ import annotations

from pathlib import Path
from typing import Any, NamedTuple

import numpy as np

from favlo.media_pipeline import MediaCall, MediaPaused, MediaPipeline, MediaRequestError
from voxcpm2.audio import encode_wav
from voxcpm2.config import ModelConfig, load_config
from voxcpm2.metal import Session
from voxcpm2.speak import generate
from voxcpm2.store import Store
from voxcpm2.text import style_line
from voxcpm2.tokenize import LineTokenizer
from voxcpm2.vae import Decoder

_SCRATCH = 256 << 20


class Runtime:
    """The published checkout. Weights move to the device on the first line."""

    def __init__(
        self,
        path: Path,
        config: ModelConfig,
        *,
        owner: str,
        max_memory: str | None,
        kv_cache: str,
        resident_percent: float | None,
        console: Any,
    ) -> None:
        self.path = path
        self.config = config
        self._owner = owner
        self.max_memory = max_memory
        self.kv_cache = kv_cache
        self.resident_percent = resident_percent
        self.console = console
        self._session: Session | None = None
        self._store: Store | None = None
        self._decoder: Decoder | None = None
        self._tokenizer: LineTokenizer | None = None

    def close(self) -> None:
        from favlo.memory import memory

        if self._session is not None:
            self._session.close()
            self._session = None
        if self._store is not None:
            self._store.close()
            self._store = None
        self._decoder = None
        self._tokenizer = None
        memory.sweep(self._owner)

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
        weights = self.path / "model.safetensors"
        need = weights.stat().st_size + vae_path.stat().st_size + _SCRATCH
        room = _room(self.max_memory)
        if room < need:
            message = f"a voice line does not fit in {room} bytes"
            raise RuntimeError(message)
        self._session = Session(self._owner)
        try:
            self._store = Store(weights)
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
    percent = None if resident_percent is None else float(resident_percent)
    return Runtime(
        checkout,
        load_config(config_path),
        owner=owner or f"voxcpm2:{id(checkout)}",
        max_memory=max_memory,
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
    seeds = False
    id_prefix = "voice_"
    object_name = "audio.voice"
    table_name = "voice"

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
            reason = stop()
            if reason:
                raise MediaPaused(reason, None)

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


handle_voice = Pipeline()


def _room(max_memory: str | int | None) -> int:
    from favlo.memory import memory

    ceiling = int(memory.budget())
    cap = _parse_memory(max_memory)
    if cap is not None:
        ceiling = min(ceiling, cap)
    return max(0, ceiling - int(memory.live) - int(memory.pending))


def _parse_memory(text: str | int | None) -> int | None:
    """``32GiB``, ``512MiB``, or a byte count. Empty leaves the ledger uncapped."""
    if text is None or text == "":
        return None
    if isinstance(text, int):
        if text < 1:
            message = f"maximum memory {text} is not positive"
            raise ValueError(message)
        return text
    raw = str(text).strip().replace("_", "")
    suffix = ""
    digits = raw
    for ending, name in (
        ("gib", "GiB"),
        ("mib", "MiB"),
        ("gb", "GiB"),
        ("mb", "MiB"),
        ("g", "GiB"),
        ("m", "MiB"),
    ):
        if raw.lower().endswith(ending):
            suffix = name
            digits = raw[: -len(ending)]
            break
    try:
        number = int(digits)
    except ValueError as exc:
        message = f"maximum memory {text} is not a byte count"
        raise ValueError(message) from exc
    if number < 1:
        message = f"maximum memory {text} is not positive"
        raise ValueError(message)
    if suffix == "GiB":
        return number << 30
    if suffix == "MiB":
        return number << 20
    return number
