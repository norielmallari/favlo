"""Rolling server log. The process entry owns this module.

The file is ``~/.madmetal/logs/madmetal.log``. ``~/.madmetal/settings.json``
key ``log_max_bytes`` is the most bytes that file may hold. The default is
1 MiB. A line that would pass the cap leaves the newest bytes that still fit.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

DEFAULT_LOG_MAX_BYTES = 1024 * 1024
_SETTINGS_NAME = "settings.json"
_LOG_NAME = "madmetal.log"


def madmetal_home() -> Path:
    return Path.home() / ".madmetal"


def settings_path(home: Path) -> Path:
    return home / _SETTINGS_NAME


def log_path(home: Path) -> Path:
    return home / "logs" / _LOG_NAME


def load_settings(home: Path) -> dict[str, Any]:
    """Read settings. A missing or unreadable file uses the defaults."""
    path = settings_path(home)
    if not path.is_file():
        return {"log_max_bytes": DEFAULT_LOG_MAX_BYTES}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeError):
        return {"log_max_bytes": DEFAULT_LOG_MAX_BYTES}
    if not isinstance(raw, dict):
        return {"log_max_bytes": DEFAULT_LOG_MAX_BYTES}
    return raw


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
    payload = {"log_max_bytes": DEFAULT_LOG_MAX_BYTES}
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
    """Point ``console`` at the rolling log for ``home`` (default ``~/.madmetal``)."""
    root = madmetal_home() if home is None else home
    try:
        settings = ensure_settings(root)
        console.attach_log(log_path(root), log_max_bytes(settings))
    except OSError:
        return
