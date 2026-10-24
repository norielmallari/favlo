"""Resident record. Speech is ``handle_voice``."""

from __future__ import annotations

import secrets
from pathlib import Path
from typing import Any

import numpy as np

from zonos2.audio import encode_wav, text_ids


class Runtime:
    def __init__(self, path: Path, *, kv_cache: str, resident_percent: float) -> None:
        self.path = path
        self.kv_cache = kv_cache
        self.resident_percent = resident_percent
        self._model = None

    def speak(self, text: str, speaker: np.ndarray | None = None, max_frames: int | None = None) -> np.ndarray:
        if self._model is None:
            from zonos2.forward import Model

            self._model = Model.open(self.path)
        return self._model.speak(text, speaker, max_frames=max_frames)

    def close(self) -> None:
        model = self._model
        self._model = None
        session = getattr(model, "session", None)
        if session is not None:
            session.close()
        from favlo.memory import memory

        memory.sweep("zonos")


def open_runtime(
    path: str | Path,
    *,
    kv_cache: str,
    console: Any,
    resident_percent: float,
    max_memory: str | None = None,
) -> Runtime:
    del console, max_memory
    if kv_cache == "bf16":
        message = "zonos2 loads the packed record"
        raise ValueError(message)
    record = Path(path)
    if (record / "model.pth").is_file():
        message = "zonos2 loads the packed record"
        raise FileNotFoundError(message)
    manifest = record / "manifest.json"
    weights = record / "weights"
    if not manifest.is_file() or not weights.is_dir():
        message = "zonos2 record is not packed"
        raise FileNotFoundError(message)
    return Runtime(record, kv_cache=kv_cache, resident_percent=resident_percent)


def handle_chat(reply: Any, runtime: Runtime, body: dict, selection: Any, args: Any, state: dict) -> None:
    del reply, runtime, body, selection, args, state
    message = "zonos2 does not serve chat completions"
    raise RuntimeError(message)


def handle_voice(reply: Any, runtime: Runtime, body: dict, selection: Any, args: Any, state: dict) -> None:
    """Write one spoken WAV, then the reply names that id."""
    del selection, args
    text = body.get("input")
    if not isinstance(text, str):
        message = "input is required"
        raise ValueError(message)
    text_ids(text)
    fmt = body.get("response_format") or "mp3"
    if fmt not in {"mp3", "wav", "flac"}:
        message = "response_format must be mp3, wav, or flac"
        raise ValueError(message)
    samples = runtime.speak(text)
    from favlo.voice import store_wav

    voice_id = "voice_" + secrets.token_hex(12)
    home = state.get("voice_home")
    root = home if isinstance(home, Path) else None
    jobs = state.setdefault("voice", {})
    store_wav(voice_id, encode_wav(samples), home=root, voice=jobs, output_format=fmt)
    jobs[voice_id] = {"response_format": fmt}
    reply.json(
        200,
        {
            "id": voice_id,
            "object": "audio.voice",
            "status": "completed",
            "response_format": fmt,
        },
    )
