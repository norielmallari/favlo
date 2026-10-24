"""Stored voice clips. The process entry owns this module.

Each clip keeps ``<home>/voice/<id>.wav``. The served file is that WAV, or
an MP3 or FLAC encoded from it. ``~/.favlo/settings.json`` key
``voice_ttl_seconds`` is how long they stay. The default is one day. A sweep
at startup and after each stored clip deletes an older file and drops the
matching job.
"""

from __future__ import annotations

import subprocess
import threading
import time
from pathlib import Path
from typing import Any

from favlo.logs import favlo_home, load_settings, voice_ttl_seconds

_DIR = "voice"
_FORMATS = {"wav": "audio/wav", "mp3": "audio/mpeg", "flac": "audio/flac"}
_encode_lock = threading.Lock()


def voice_dir(home: Path) -> Path:
    return home / _DIR


def store_wav(
    voice_id: str,
    payload: bytes,
    *,
    home: Path | None = None,
    voice: dict[str, Any] | None = None,
    output_format: str = "mp3",
) -> Path:
    """Write the WAV, encode the served file, then drop expired clips."""
    root = favlo_home() if home is None else home
    path = _path(root, voice_id, "wav")
    _replace(path, payload)
    if output_format != "wav":
        encode(path, output_format)
    sweep_voice(voice, home=root)
    return path


def encode(source: Path, output_format: str) -> Path:
    """MP3 at 256 kbps, or FLAC, from one saved WAV."""
    if output_format not in {"mp3", "flac"}:
        message = "output_format must be mp3, wav, or flac"
        raise ValueError(message)
    target = source.with_suffix(f".{output_format}")
    temporary = target.with_name(target.stem + f".tmp.{output_format}")
    codec = ["-c:a", "libmp3lame", "-b:a", "256k"] if output_format == "mp3" else ["-c:a", "flac"]
    result = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(source), *codec, str(temporary)],
        capture_output=True,
        check=False,
    )
    if result.returncode != 0 or not temporary.is_file():
        temporary.unlink(missing_ok=True)
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        message = detail or "ffmpeg failed"
        raise RuntimeError(message)
    temporary.replace(target)
    return target


def load_voice(
    voice_id: str,
    output_format: str | None,
    *,
    home: Path | None = None,
    default: str = "mp3",
) -> tuple[bytes, str] | None:
    """The served bytes and media type, or None when the id or file is missing."""
    chosen = default if output_format is None else output_format
    if chosen not in _FORMATS:
        message = "response_format must be mp3, wav, or flac"
        raise ValueError(message)
    root = favlo_home() if home is None else home
    try:
        path = _path(root, voice_id, chosen)
        if chosen != "wav" and not path.is_file():
            source = _path(root, voice_id, "wav")
            if not source.is_file():
                return None
            with _encode_lock:
                if not path.is_file():
                    encode(source, chosen)
    except ValueError:
        return None
    if not path.is_file():
        return None
    return path.read_bytes(), _FORMATS[chosen]


def sweep_voice(
    voice: dict[str, Any] | None = None,
    *,
    home: Path | None = None,
    now: float | None = None,
) -> None:
    """Delete a clip once its WAV has reached the expiry."""
    root = favlo_home() if home is None else home
    folder = voice_dir(root)
    if not folder.is_dir():
        return
    ttl = voice_ttl_seconds(load_settings(root))
    clock = time.time() if now is None else now
    for path in folder.glob("*.wav"):
        try:
            modified = path.stat().st_mtime
        except OSError:
            continue
        if clock - modified < ttl:
            continue
        stem = path.stem
        for suffix in ("wav", "mp3", "flac"):
            sibling = folder / f"{stem}.{suffix}"
            try:
                sibling.unlink()
            except OSError:
                continue
        if isinstance(voice, dict):
            voice.pop(stem, None)


def _path(home: Path, voice_id: str, suffix: str) -> Path:
    if (
        not isinstance(voice_id, str)
        or not voice_id
        or voice_id in {".", ".."}
        or "/" in voice_id
        or "\\" in voice_id
        or "." in voice_id
        or "\x00" in voice_id
    ):
        message = "refusing voice id"
        raise ValueError(message)
    return voice_dir(home) / f"{voice_id}.{suffix}"


def _replace(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)
