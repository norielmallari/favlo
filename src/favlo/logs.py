"""Rolling server log. The process entry owns this module.

The file is ``~/.favlo/logs/favlo.log``. ``~/.favlo/settings.json``
key ``log_max_bytes`` is the most bytes that file may hold. The default is
1 MiB. A line that would pass the cap leaves the newest bytes that still fit.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

DEFAULT_LOG_MAX_BYTES = 1024 * 1024
DEFAULT_VIDEO_TTL_SECONDS = 24 * 60 * 60
DEFAULT_IMAGE_TTL_SECONDS = 24 * 60 * 60
DEFAULT_MUSIC_TTL_SECONDS = 24 * 60 * 60
DEFAULT_VOICE_TTL_SECONDS = 24 * 60 * 60
DEFAULT_HEARTBEAT_SECONDS = 5 * 60
_SETTINGS_NAME = "settings.json"
_LOG_NAME = "favlo.log"


def favlo_home() -> Path:
    return Path.home() / ".favlo"


def settings_path(home: Path) -> Path:
    return home / _SETTINGS_NAME


def log_path(home: Path) -> Path:
    return home / "logs" / _LOG_NAME


def load_settings(home: Path) -> dict[str, Any]:
    """Read settings. A missing or unreadable file uses the defaults."""
    path = settings_path(home)
    if not path.is_file():
        return _default_settings()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeError):
        return _default_settings()
    if not isinstance(raw, dict):
        return _default_settings()
    return raw


def _configured_int(settings: dict[str, Any], key: str) -> int | None:
    """An integer stored under ``key``, or None when the key is absent."""
    if key not in settings or settings[key] is None:
        return None
    try:
        return int(settings[key])
    except (TypeError, ValueError):
        return None


def output_token_limit(
    body: dict[str, Any], settings: dict[str, Any] | None = None
) -> int | None:
    """Completion cap from the request, then from settings. An omission is uncapped."""
    if body.get("max_tokens") is not None:
        return int(body["max_tokens"])
    if body.get("max_completion_tokens") is not None:
        return int(body["max_completion_tokens"])
    source = load_settings(favlo_home()) if settings is None else settings
    return _configured_int(source, "max_tokens")


def configured_thinking_budget(settings: dict[str, Any] | None = None) -> int | None:
    """Think-token cap from settings. An omission does not close the think block on a count."""
    source = load_settings(favlo_home()) if settings is None else settings
    return _configured_int(source, "thinking_budget")


def video_ttl_seconds(settings: dict[str, Any]) -> int:
    """How long a stored MP4 stays, in seconds. The default is one day."""
    return _ttl_seconds(settings, "video_ttl_seconds", DEFAULT_VIDEO_TTL_SECONDS)


def image_ttl_seconds(settings: dict[str, Any]) -> int:
    """How long a stored PNG stays, in seconds. The default is one day."""
    return _ttl_seconds(settings, "image_ttl_seconds", DEFAULT_IMAGE_TTL_SECONDS)


def music_ttl_seconds(settings: dict[str, Any]) -> int:
    """How long a stored WAV stays, in seconds. The default is one day."""
    return _ttl_seconds(settings, "music_ttl_seconds", DEFAULT_MUSIC_TTL_SECONDS)


def voice_ttl_seconds(settings: dict[str, Any]) -> int:
    """How long a stored spoken WAV stays, in seconds. The default is one day."""
    return _ttl_seconds(settings, "voice_ttl_seconds", DEFAULT_VOICE_TTL_SECONDS)


def _ttl_seconds(settings: dict[str, Any], key: str, default: int) -> int:
    value = settings.get(key, default)
    try:
        ttl = int(value)
    except (TypeError, ValueError):
        return default
    if ttl < 1:
        return default
    return ttl


def _default_settings() -> dict[str, Any]:
    return {
        "log_max_bytes": DEFAULT_LOG_MAX_BYTES,
        "video_ttl_seconds": DEFAULT_VIDEO_TTL_SECONDS,
        "image_ttl_seconds": DEFAULT_IMAGE_TTL_SECONDS,
        "music_ttl_seconds": DEFAULT_MUSIC_TTL_SECONDS,
        "voice_ttl_seconds": DEFAULT_VOICE_TTL_SECONDS,
        "heartbeat_seconds": DEFAULT_HEARTBEAT_SECONDS,
    }


def heartbeat_seconds(settings: dict[str, Any]) -> int:
    """Quiet seconds before the terminal's own check. The default is 5 minutes."""
    value = settings.get("heartbeat_seconds", DEFAULT_HEARTBEAT_SECONDS)
    try:
        seconds = int(value)
    except (TypeError, ValueError):
        return DEFAULT_HEARTBEAT_SECONDS
    if seconds < 60:
        return DEFAULT_HEARTBEAT_SECONDS
    return seconds


def set_heartbeat_seconds(home: Path, seconds: int) -> int:
    """Store ``heartbeat_seconds``. A value below 60 is refused."""
    if seconds < 60:
        message = "heartbeat_seconds is at least 60"
        raise ValueError(message)
    path = settings_path(home)
    current: dict[str, Any]
    if path.is_file():
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, UnicodeError):
            current = _default_settings()
        else:
            current = raw if isinstance(raw, dict) else _default_settings()
    else:
        current = _default_settings()
        home.mkdir(parents=True, exist_ok=True)
    current["heartbeat_seconds"] = seconds
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(current, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)
    return seconds


def log_max_bytes(settings: dict[str, Any]) -> int:
    """Positive ``log_max_bytes``, or the 1 MiB default."""
    value = settings.get("log_max_bytes", DEFAULT_LOG_MAX_BYTES)
    try:
        size = int(value)
    except (TypeError, ValueError):
        return DEFAULT_LOG_MAX_BYTES
    if size < 1:
        return DEFAULT_LOG_MAX_BYTES
    return size


def ensure_settings(home: Path) -> dict[str, Any]:
    """Create ``settings.json`` when it is absent. An existing file is left as it is."""
    path = settings_path(home)
    if path.is_file():
        return load_settings(home)
    home.mkdir(parents=True, exist_ok=True)
    payload = _default_settings()
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)
    return payload


class RollingLog:
    """One log file whose size stays at or under ``max_bytes``."""

    def __init__(self, path: Path, max_bytes: int) -> None:
        if max_bytes < 1:
            message = "log_max_bytes must be positive"
            raise ValueError(message)
        self.path = path
        self.max_bytes = max_bytes
        self._lock = threading.Lock()

    def write(self, text: str) -> None:
        if not text:
            return
        payload = text.encode("utf-8")
        if not payload.endswith(b"\n"):
            payload += b"\n"
        with self._lock:
            try:
                self._append(payload)
            except OSError:
                return

    def _append(self, payload: bytes) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        existing = self.path.read_bytes() if self.path.is_file() else b""
        if existing and len(existing) + len(payload) <= self.max_bytes:
            with self.path.open("ab") as handle:
                handle.write(payload)
            return
        combined = existing + payload
        if len(combined) > self.max_bytes:
            combined = _tail(combined, self.max_bytes)
        temporary = self.path.with_name(self.path.name + ".tmp")
        temporary.write_bytes(combined)
        temporary.replace(self.path)


def _tail(blob: bytes, max_bytes: int) -> bytes:
    """The newest ``max_bytes`` of ``blob``, starting at a line when one fits."""
    kept = blob[-max_bytes:]
    newline = kept.find(b"\n")
    if newline != -1 and newline < len(kept) - 1:
        return kept[newline + 1 :]
    return kept


def attach_process_log(console: Any, home: Path | None = None) -> None:
    """Point ``console`` at the rolling log for ``home`` (default ``~/.favlo``)."""
    root = favlo_home() if home is None else home
    try:
        settings = ensure_settings(root)
        console.attach_log(log_path(root), log_max_bytes(settings))
    except OSError:
        return
