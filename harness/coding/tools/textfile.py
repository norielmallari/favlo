"""UTF-8 file bytes. Binary content is rejected before it is decoded."""

from __future__ import annotations

import hashlib
from pathlib import Path

from harness.coding.tools.errors import ToolError
from harness.coding.tools.limits import READ_FILE_MAX_BYTES


def content_hash(data: bytes) -> str:
    """Lowercase hex SHA-256 of the complete file bytes."""
    return hashlib.sha256(data).hexdigest()


def read_bytes(path: Path) -> bytes:
    """The complete file, or an error that names why it cannot be text."""
    if not path.exists():
        raise ToolError("not_found", "file does not exist")
    if path.is_dir():
        raise ToolError("invalid_arguments", "path is a directory")
    try:
        data = path.read_bytes()
    except PermissionError as exc:
        raise ToolError("permission_denied", "file cannot be read") from exc
    except OSError as exc:
        raise ToolError("execution_failed", exc.strerror or "file cannot be read") from exc
    if len(data) > READ_FILE_MAX_BYTES:
        raise ToolError("execution_failed", "file exceeds the read cap")
    if b"\0" in data:
        raise ToolError("binary_content", "file is binary")
    try:
        data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ToolError("binary_content", "file is not utf-8 text") from exc
    return data
