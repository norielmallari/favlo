"""Replace an exact string. The tool does not guess a nearby match."""

from __future__ import annotations

import difflib
import os
from pathlib import Path
from typing import Any

from harness.coding.tools.args import optional_bool, optional_str, reject_unknown, require_text
from harness.coding.tools.context import ToolContext
from harness.coding.tools.errors import ToolError, dump
from harness.coding.tools.idempotency import check_key
from harness.coding.tools.limits import DIFF_MAX_LINES, WRITE_MAX_BYTES
from harness.coding.tools.paths import assert_writable, display_path, resolve_inside
from harness.coding.tools.textfile import content_hash, read_bytes

_FIELDS = {
    "expected_hash",
    "file_path",
    "idempotency_key",
    "new_string",
    "old_string",
    "replace_all",
}


def perform(ctx: ToolContext, arguments: dict[str, Any]) -> str:
    reject_unknown(arguments, _FIELDS)
    raw_path = require_text(arguments, "file_path", allow_empty=False)
    old = require_text(arguments, "old_string", allow_empty=False)
    new = require_text(arguments, "new_string", allow_empty=True)
    key = check_key(require_text(arguments, "idempotency_key", allow_empty=False))
    replace_all = optional_bool(arguments, "replace_all", default=False)
    expected = optional_str(arguments, "expected_hash")
    if expected is not None and not _sha256(expected):
        raise ToolError("invalid_arguments", "expected_hash must be a sha256 hex digest")
    path = resolve_inside(ctx.workspace, raw_path)
    relative = display_path(ctx.workspace, path)
    assert_writable(relative)
    payload = {
        "expected_hash": expected,
        "file_path": relative,
        "new_string": new,
        "old_string": old,
        "replace_all": replace_all,
    }
    reserved = ctx.idempotency.reserve(key, payload)
    if reserved.kind == "replay" and reserved.outcome is not None:
        return reserved.outcome
    if reserved.kind == "pending":
        raise ToolError("conflict", "attempt did not finish")
    with ctx.mutations.file(path):
        published = False
        try:
            before = read_bytes(path)
            before_hash = content_hash(before)
            if expected is not None and expected != before_hash:
                raise ToolError("conflict", "file hash does not match expected_hash")
            text = before.decode("utf-8")
            count = text.count(old)
            if count == 0:
                raise ToolError("match_not_found", "old_string is not in the file")
            if count > 1 and not replace_all:
                raise ToolError("ambiguous_match", "old_string occurs more than once")
            updated = text.replace(old, new) if replace_all else text.replace(old, new, 1)
            data = updated.encode("utf-8")
            if len(data) > WRITE_MAX_BYTES:
                raise ToolError("invalid_arguments", "content exceeds the write cap")
            _replace(ctx, path, data)
            published = True
        except Exception:
            if not published:
                ctx.idempotency.abandon(key)
            raise
        after_hash = content_hash(data)
        diff, truncated, artifact = _diff(ctx, relative, text, updated)
        body: dict[str, Any] = {
            "after_hash": after_hash,
            "before_hash": before_hash,
            "diff": diff,
            "file_path": relative,
            "replacements": count,
        }
        if truncated:
            body["diff_truncated"] = True
            if artifact is not None:
                body["diff_artifact"] = artifact
        encoded = dump(body)
        ctx.idempotency.finish(key, encoded)
        return encoded


def _sha256(value: str) -> bool:
    return len(value) == 64 and all(char in "0123456789abcdef" for char in value)


def _replace(ctx: ToolContext, path: Path, data: bytes) -> None:
    mode = path.stat().st_mode & 0o777
    directory = ctx.state_dir / "tmp"
    directory.mkdir(parents=True, exist_ok=True)
    temporary = directory / f"{os.getpid()}.{path.name}.edit"
    try:
        fd = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY, mode or 0o644)
        try:
            written = os.write(fd, data)
            if written != len(data):
                raise ToolError("execution_failed", "file was not fully written")
            os.fsync(fd)
        finally:
            os.close(fd)
        os.replace(temporary, path)
    except ToolError:
        _drop(temporary)
        raise
    except OSError as exc:
        _drop(temporary)
        raise ToolError("execution_failed", exc.strerror or "file was not replaced") from exc


def _drop(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass


def _diff(ctx: ToolContext, relative: str, before: str, after: str) -> tuple[str, bool, str | None]:
    lines = list(
        difflib.unified_diff(
            before.splitlines(keepends=True),
            after.splitlines(keepends=True),
            fromfile=relative,
            tofile=relative,
        )
    )
    if len(lines) <= DIFF_MAX_LINES:
        return "".join(lines), False, None
    directory = ctx.state_dir / "artifacts"
    directory.mkdir(parents=True, exist_ok=True)
    number = 1
    while (directory / f"edit-{number}.diff").exists():
        number += 1
    artifact = directory / f"edit-{number}.diff"
    artifact.write_text("".join(lines), encoding="utf-8")
    return "".join(lines[:DIFF_MAX_LINES]), True, f".favlo/artifacts/edit-{number}.diff"
