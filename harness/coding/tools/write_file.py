"""Create a file. An existing path is left unchanged."""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any

from harness.coding.tools.args import optional_bool, reject_unknown, require_text
from harness.coding.tools.context import ToolContext
from harness.coding.tools.errors import ToolError, dump
from harness.coding.tools.idempotency import check_key
from harness.coding.tools.limits import WRITE_MAX_BYTES
from harness.coding.tools.paths import assert_writable, display_path, resolve_inside
from harness.coding.tools.textfile import content_hash

_FIELDS = {"content", "create_directories", "file_path", "idempotency_key"}


def perform(ctx: ToolContext, arguments: dict[str, Any]) -> str:
    reject_unknown(arguments, _FIELDS)
    relative_raw = require_text(arguments, "file_path", allow_empty=False)
    content = require_text(arguments, "content", allow_empty=True)
    key = check_key(require_text(arguments, "idempotency_key", allow_empty=False))
    create_directories = optional_bool(arguments, "create_directories", default=False)
    data = content.encode("utf-8")
    if len(data) > WRITE_MAX_BYTES:
        raise ToolError("invalid_arguments", "content exceeds the write cap")
    path = resolve_inside(ctx.workspace, relative_raw)
    relative = display_path(ctx.workspace, path)
    assert_writable(relative)
    if relative == ".":
        raise ToolError("invalid_arguments", "path is a directory")
    payload = {
        "content": content,
        "create_directories": create_directories,
        "file_path": relative,
    }
    reserved = ctx.idempotency.reserve(key, payload)
    if reserved.kind == "replay" and reserved.outcome is not None:
        return reserved.outcome
    if reserved.kind == "pending":
        recovered = _recover(ctx, path, relative, data, key)
        if recovered is not None:
            return recovered
    with ctx.mutations.file(path):
        try:
            _create(ctx, path, data, create_directories=create_directories)
        except Exception:
            if not _published(path, data):
                ctx.idempotency.abandon(key)
            raise
        body = dump({"content_hash": content_hash(data), "file_path": relative})
        ctx.idempotency.finish(key, body)
        return body


def _recover(ctx: ToolContext, path: Path, relative: str, data: bytes, key: str) -> str | None:
    if not path.is_file():
        return None
    try:
        current = path.read_bytes()
    except OSError:
        return None
    if current != data:
        raise ToolError("conflict", "attempt did not finish")
    body = dump({"content_hash": content_hash(data), "file_path": relative})
    ctx.idempotency.finish(key, body)
    return body


def _create(ctx: ToolContext, path: Path, data: bytes, *, create_directories: bool) -> None:
    parent = path.parent
    if parent.exists() and not parent.is_dir():
        raise ToolError("invalid_arguments", "parent path is not a directory")
    if not parent.exists():
        if not create_directories:
            raise ToolError("not_found", "parent directory does not exist")
        try:
            parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            detail = exc.strerror or "parent directory was not created"
            raise ToolError("execution_failed", detail) from exc
    if path.exists():
        raise ToolError("conflict", "file already exists")
    temporary = _temporary(ctx, path)
    try:
        fd = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        try:
            written = os.write(fd, data)
            if written != len(data):
                raise ToolError("execution_failed", "file was not fully written")
            os.fsync(fd)
        finally:
            os.close(fd)
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise ToolError("conflict", "file already exists") from exc
    except ToolError:
        raise
    except OSError as exc:
        raise ToolError("execution_failed", exc.strerror or "file was not created") from exc
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
    if not path.is_file():
        raise ToolError("execution_failed", "file was not created")


def _temporary(ctx: ToolContext, path: Path) -> Path:
    directory = ctx.state_dir / "tmp"
    directory.mkdir(parents=True, exist_ok=True)
    token = f"{os.getpid()}.{threading.get_ident()}.{path.name}"
    return directory / token


def _published(path: Path, data: bytes) -> bool:
    try:
        return path.is_file() and path.read_bytes() == data
    except OSError:
        return False
