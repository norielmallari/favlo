"""Open the packed record and serve one music job."""

from __future__ import annotations

import secrets
import time
from pathlib import Path
from typing import Any

from tokenizers import Tokenizer

from ace_step_15_turbo.graph import generate
from ace_step_15_turbo.hold import ClipHold, ClipPaused, interrupt
from ace_step_15_turbo.metal import Gpu
from ace_step_15_turbo.record import load
from ace_step_15_turbo.request import MusicRequest, parse_music
from ace_step_15_turbo.schedule import STEPS

_UNITS = 28 + 8 + 4 + 32 * STEPS + 7


def open_runtime(
    path: Path,
    *,
    kv_cache: str,
    console: Any,
    resident_percent: float | None,
    max_memory: str | int | None = None,
) -> "Runtime":
    del kv_cache, resident_percent, max_memory
    return Runtime(path, console)


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


def handle_music(
    reply: Any,
    runtime: "Runtime",
    body: dict[str, Any],
    selection: Any,
    args: Any,
    state: dict[str, Any],
) -> None:
    del args
    model = body.get("model")
    if not isinstance(model, str) or not model:
        _reject(reply, "model is required")
        return
    try:
        fields = parse_music(body)
    except ValueError as exc:
        _reject(reply, str(exc))
        return
    if selection is not None:
        requested = getattr(selection, "requested_id", "")
        if isinstance(requested, str) and requested:
            model = requested
    jobs = state.setdefault("music", {})
    if not isinstance(jobs, dict):
        jobs = {}
        state["music"] = jobs
    held = _held(state, fields)
    if held is None:
        job = _job(model, fields)
        jobs[job["id"]] = job
        hold = None
    else:
        job = held["job"]
        hold = held["hold"]
        jobs[job["id"]] = job
    if getattr(reply, "payload", None) is not job:
        reply.json(200, job)
    progress = _Progress(job, done=0 if hold is None else hold.done)
    progress.hold = hold
    progress.interrupt = lambda: _stop_reason(state, runtime)
    home = state.get("music_home")
    root = home if isinstance(home, Path) else None
    try:
        if hold is not None and hold.wav is not None:
            wav = hold.wav
        else:
            wav = runtime.generate(fields, progress, hold)
        reason = interrupt(progress)
        if reason:
            raise ClipPaused(reason, ClipHold(stage="encode", wav=wav, done=progress.done))
        job["stage"] = "encode"
        from favlo.music import store_wav

        store_wav(job["id"], wav, home=root, music=jobs, output_format=fields.output_format)
    except ClipPaused as paused:
        _release(runtime)
        if paused.reason == "cancel":
            state.pop("music_hold", None)
            return
        state["music_hold"] = {"job": job, "hold": paused.hold, "request": _identity(fields)}
        return
    except Exception as exc:
        state.pop("music_hold", None)
        job["status"] = "failed"
        job["error"] = {"message": str(exc), "type": type(exc).__name__}
        return
    state.pop("music_hold", None)
    job["completed_at"] = int(time.time())
    job["progress"] = 100
    job["eta"] = 0
    job["status"] = "completed"
    job["stage"] = "encode"


class Runtime:
    def __init__(self, path: Path, console: Any) -> None:
        self.console = console
        self.path = Path(path)
        self.gpu: Gpu | None = None
        self.weights: dict[str, Any] = {}
        self.tokenizer = Tokenizer.from_file(str(self.path / "tokenizer.json"))
        self._halted = False
        self._open()

    def halt(self) -> None:
        """Stop at the next stage boundary. The listener calls this from a signal."""
        self._halted = True

    def generate(self, fields: Any, progress: Any, hold: ClipHold | None = None) -> bytes:
        if hold is None:
            found = getattr(progress, "hold", None)
            if isinstance(found, ClipHold):
                hold = found
        self._open()
        gpu = self.gpu
        if gpu is None:
            message = "the music device is closed"
            raise RuntimeError(message)
        gpu.seconds = 0.0
        gpu.end_clip()
        try:
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
        except ClipPaused:
            self.close()
            raise
        gpu.end_clip()
        print(f"generation_seconds {gpu.seconds:.3f}", flush=True)
        return wav

    def _open(self) -> None:
        """Open the device and load the record when a pause closed them."""
        if self.weights:
            return
        self.gpu = Gpu()
        self.weights = load(self.gpu, self.path)
        self.gpu.reclaim()

    def close(self) -> None:
        """Drop the device. A later ``generate`` opens it again."""
        gpu = self.gpu
        self.gpu = None
        self.weights = {}
        if gpu is not None:
            gpu.close()


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


def _job(model: str, fields: Any) -> dict[str, Any]:
    from favlo.logs import favlo_home, load_settings, music_ttl_seconds

    created = int(time.time())
    seconds = fields.seconds
    label = str(int(seconds)) if float(seconds).is_integer() else str(seconds)
    return {
        "id": "music_" + secrets.token_hex(16),
        "object": "audio.music",
        "created_at": created,
        "completed_at": None,
        "expires_at": created + music_ttl_seconds(load_settings(favlo_home())),
        "status": "in_progress",
        "model": model,
        "progress": 0,
        "eta": None,
        "stage": "text",
        "prompt": fields.prompt,
        "lyrics": fields.lyrics,
        "seconds": label,
        "response_format": fields.output_format,
        "error": None,
    }


def _reject(reply: Any, message: str) -> None:
    reply.json(400, {"error": {"message": message, "type": "invalid_request_error"}})


def _stop_reason(state: dict[str, Any], runtime: Any) -> str | None:
    if getattr(runtime, "_halted", False):
        return "cancel"
    return _reason(state)


def _reason(state: dict[str, Any]) -> str | None:
    cancel = state.get("cancel")
    if getattr(cancel, "is_set", lambda: False)():
        return "cancel"
    preempt = state.get("preempt")
    if getattr(preempt, "is_set", lambda: False)():
        return "preempt"
    return None


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


def _held(state: dict[str, Any], fields: MusicRequest) -> dict[str, Any] | None:
    """The paused clip when this request is that same job. A mismatch is dropped."""
    held = state.get("music_hold")
    if not isinstance(held, dict):
        return None
    job = held.get("job")
    hold = held.get("hold")
    same = (
        isinstance(job, dict)
        and isinstance(hold, ClipHold)
        and job.get("status") == "in_progress"
        and held.get("request") == _identity(fields)
    )
    state.pop("music_hold", None)
    if not same or not isinstance(job, dict) or not isinstance(hold, ClipHold):
        return None
    return {"job": job, "hold": hold}


def _release(runtime: Any) -> None:
    """Close the device at a pause so the next call opens it again."""
    close = getattr(runtime, "close", None)
    if callable(close):
        close()
        return
    gpu = getattr(runtime, "gpu", None)
    end = getattr(gpu, "end_clip", None)
    if callable(end):
        end()
