"""Open the packed record and serve one music job."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from tokenizers import Tokenizer

from ace_step_15_turbo.graph import generate
from ace_step_15_turbo.hold import ClipHold, ClipPaused
from ace_step_15_turbo.metal import Gpu
from ace_step_15_turbo.record import load, weight_bytes
from ace_step_15_turbo.request import MusicRequest, parse_music
from ace_step_15_turbo.schedule import HOP, LYRIC_LIMIT, STEPS, TEXT_LIMIT, latent_frames
from ace_step_15_turbo.tiles import tile_frames
from favlo.media_memory import begin_release, finish_release, open_gate, stamp_owner
from favlo.media_pipeline import MediaCall, MediaPipeline, MediaRequestError
from favlo.memory import memory

_UNITS = 28 + 8 + 4 + 32 * STEPS + 7


def open_runtime(
    path: Path,
    *,
    kv_cache: str,
    console: Any,
    resident_percent: float | None,
    max_memory: str | int | None = None,
    owner: str | None = None,
) -> "Runtime":
    del kv_cache, resident_percent
    memory.set_max_memory(max_memory)
    return Runtime(path, console, owner=owner)


def handle_chat(reply: Any, runtime: Any, body: dict[str, Any], selection: Any, args: Any, state: dict[str, Any]) -> None:
    del runtime, body, selection, args, state
    reply.json(
        404,
        {
            "error": {
                "message": "ace-step-15-turbo generates music at POST /v1/audio/music",
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
        kept = hold if isinstance(hold, ClipHold) else None
        progress = _Progress(job, done=0 if kept is None else kept.done)
        progress.hold = kept
        progress.interrupt = stop
        call.progress = progress
        if kept is not None and kept.wav is not None:
            wav = kept.wav
        else:
            wav = runtime.generate(request, progress, kept)
        reason = stop()
        if reason:
            raise ClipPaused(reason, ClipHold(stage="encode", wav=wav, done=progress.done))
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

handle_music = Pipeline()


class Runtime:
    def __init__(
        self,
        path: Path,
        console: Any = None,
        *,
        owner: str | None = None,
    ) -> None:
        self.console = console
        self.path = Path(path)
        self.gpu: Gpu | None = None
        self.weights: dict[str, Any] = {}
        self._halted = False
        self._closed = True
        self._owner = stamp_owner("ace-step-15-turbo", owner, id(self))
        tokenizer = self.path / "tokenizer.json"
        if not tokenizer.is_file():
            raise FileNotFoundError(tokenizer)
        self.tokenizer = Tokenizer.from_file(str(tokenizer))
        open_gate(weight_bytes(self.path), detail="packed record")
        self._open()

    def ensure_open(self, request: Any) -> None:
        """Wait for prior releases, then load when closed and check this duration."""
        seconds = float(getattr(request, "seconds"))
        need = self.working_set_bytes(request)
        if self._closed:
            open_gate(need, detail=f"{seconds:g} seconds")
            self._open()
        self._reserve_step(seconds)

    def working_set_bytes(self, request: Any) -> int:
        """Weights plus the larger of one step or the decoder peak."""
        seconds = float(getattr(request, "seconds"))
        scratch = max(step_scratch_bytes(seconds), decoder_bytes(seconds))
        return weight_bytes(self.path) + scratch

    def halt(self) -> None:
        """Stop at the next stage boundary. The listener calls this from a signal."""
        self._halted = True

    def generate(self, fields: Any, progress: Any, hold: ClipHold | None = None) -> bytes:
        if hold is None:
            found = getattr(progress, "hold", None)
            if isinstance(found, ClipHold):
                hold = found
        gpu = self.gpu
        if gpu is None or self._closed:
            message = "the music device is closed"
            raise RuntimeError(message)
        self._reserve_step(float(fields.seconds))
        gpu.seconds = 0.0
        gpu.end_clip()
        wav = generate(
            gpu,
            self.weights,
            self.tokenizer,
            fields.prompt,
            fields.lyrics,
            fields.seconds,
            fields.seed,
            fields.vocal_language,
            fields.bpm,
            fields.keyscale,
            fields.timesignature,
            progress,
            hold,
        )
        gpu.end_clip()
        print(f"generation_seconds {gpu.seconds:.3f}", flush=True)
        return wav

    def _open(self) -> None:
        """Open the device and load the record when a pause closed them."""
        if not self._closed:
            return
        record = weight_bytes(self.path)
        if record > self._room():
            message = "the packed record does not fit"
            raise RuntimeError(message)
        self.gpu = Gpu(self._owner)
        try:
            self.weights = load(self.gpu, self.path)
            self.gpu.reclaim()
            self._closed = False
        except Exception:
            self._closed = False
            self.close()
            raise

    def close(self) -> None:
        """Drop the device, sweep the owner, and wait until those pages return."""
        if self._closed and self.gpu is None and not self.weights:
            return
        before, held = begin_release(self._owner)
        self._closed = True
        gpu = self.gpu
        self.gpu = None
        self.weights = {}
        try:
            if gpu is not None:
                gpu.close()
        finally:
            finish_release(self._owner, before, held)

    def _room(self) -> int:
        """Bytes a new span may take. The serve ledger applies ``--max-memory``."""
        return memory.room()

    def _reserve_step(self, seconds: float) -> None:
        """Require room for one step or the decoder peak. The duration stays the request."""
        if max(step_scratch_bytes(seconds), decoder_bytes(seconds)) > self._room():
            message = f"scratch for {seconds:g} seconds does not fit"
            raise RuntimeError(message)


class _Progress:
    def __init__(self, job: dict[str, Any], done: int = 0) -> None:
        self.job = job
        self.done = done
        self.started = time.monotonic()
        self.interrupt = lambda: None

    def __call__(self, stage: str) -> None:
        self.done += 1
        self.job["stage"] = stage
        self.job["progress"] = min(99, int(100 * self.done / _UNITS))
        elapsed = time.monotonic() - self.started
        remain = _UNITS - self.done
        self.job["eta"] = 0 if self.done == 0 or remain <= 0 else int(elapsed / self.done * remain)


def decoder_bytes(seconds: float) -> int:
    """Bytes of the skip plus the branch being written, at the widest tile."""
    frames = latent_frames(seconds)
    if frames % 2:
        frames += 1
    one = 128 * tile_frames(frames) * HOP * 4
    return one * 3


def step_scratch_bytes(seconds: float) -> int:
    """Bytes of one diffusion step at this duration, plus the condition cache."""
    frames = latent_frames(seconds)
    if frames % 2:
        frames += 1
    tokens = frames // 2
    width = 2560
    ffn = 9728
    heads = 32
    kv_heads = 8
    dim = 128
    cond = TEXT_LIMIT + 1 + LYRIC_LIMIT
    total = 0

    def add(nbytes: int) -> None:
        nonlocal total
        total += max(128, (int(nbytes) + 127) // 128 * 128)

    def linear(rows: int, inner: int, cols: int) -> None:
        padded = ((rows + 63) // 64) * 64
        add(padded * inner * 4)
        add(padded * inner * 2)
        add(padded * cols * 4)

    add(frames * 64 * 4)
    add(frames * 64 * 4)
    add(frames * 64 * 4)
    add(frames * 64 * 4)
    add(3 * 64 * frames * 4)
    add(width * tokens * 4)
    add(width * tokens * 4)
    add(tokens * dim * 4)
    add(tokens * dim * 4)
    add(6 * width * 4)
    for _chunk in range(6):
        add(width * 4)
    for _norm in range(5):
        add(tokens * width * 4)
    add(tokens * width * 4)
    add(tokens * width * 4)
    linear(tokens, width, heads * dim)
    linear(tokens, width, kv_heads * dim)
    linear(tokens, width, kv_heads * dim)
    add(tokens * heads * dim * 4)
    add(tokens * kv_heads * dim * 4)
    add(tokens * heads * dim * 4)
    add(tokens * kv_heads * dim * 4)
    for _quant in range(2):
        count = tokens * kv_heads * dim
        add(count)
        add(count // 32)
    add(tokens * heads * dim * 4)
    linear(tokens, heads * dim, width)
    linear(tokens, width, heads * dim)
    add(tokens * heads * dim * 4)
    add(tokens * heads * dim * 4)
    linear(tokens, heads * dim, width)
    linear(tokens, width, ffn)
    linear(tokens, width, ffn)
    add(tokens * ffn * 4)
    linear(tokens, ffn, width)
    add(TEXT_LIMIT * 2048 * 4)
    add(LYRIC_LIMIT * 2048 * 4)
    add(2048 * 4)
    add(cond * 2048 * 4)
    for _layer in range(32):
        count = cond * kv_heads * dim
        add(count)
        add(max(count // 32, 1))
        add(count)
        add(max(count // 32, 1))
    return total


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
