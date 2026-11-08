"""Absolute ``file://`` URIs for bytes on this machine.

A value that starts with ``file://`` is a local file. The path is absolute,
the URI has no host, and the bytes are read once. A bare path is not a URI.
"""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit


def is_file_uri(value: str) -> bool:
    return value.startswith("file://")


def file_uri(path: Path) -> str:
    """``file:///`` for an absolute local path."""
    return path.resolve().as_uri()


def read_file_uri(value: str) -> bytes:
    """Bytes at an absolute ``file:///`` path."""
    parsed = urlsplit(value)
    path = Path(unquote(parsed.path))
    if parsed.scheme != "file" or parsed.netloc or not path.is_absolute():
        message = "file URI must be an absolute path"
        raise ValueError(message)
    if not path.is_file():
        message = "local file is missing"
        raise ValueError(message)
    return path.read_bytes()


def materialize(value: Any) -> Any:
    """Replace a ``file://`` input with the base64 of that file."""
    if isinstance(value, str) and is_file_uri(value):
        return base64.b64encode(read_file_uri(value)).decode("ascii")
    if not isinstance(value, dict):
        return value
    for key in ("uri", "url", "b64_json", "data"):
        found = value.get(key)
        if isinstance(found, str) and is_file_uri(found):
            encoded = base64.b64encode(read_file_uri(found)).decode("ascii")
            if key in {"uri", "url"}:
                return {"b64_json": encoded}
            copied = dict(value)
            copied[key] = encoded
            return copied
    return value
