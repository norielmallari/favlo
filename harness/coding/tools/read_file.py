"""Read a window of a text file and hash the whole file."""

from __future__ import annotations

from typing import Any

from harness.coding.tools.args import optional_int, reject_unknown, require_str
from harness.coding.tools.context import ToolContext
from harness.coding.tools.errors import ToolError, dump
from harness.coding.tools.limits import READ_MAX_BYTES, READ_MAX_LINES
from harness.coding.tools.paths import assert_readable, display_path, resolve_inside
from harness.coding.tools.textfile import content_hash, read_bytes

_FIELDS = {"file_path", "limit", "offset"}


def perform(ctx: ToolContext, arguments: dict[str, Any]) -> str:
    reject_unknown(arguments, _FIELDS)
    path = resolve_inside(ctx.workspace, require_str(arguments, "file_path"))
    relative = display_path(ctx.workspace, path)
    assert_readable(relative)
    offset = optional_int(arguments, "offset")
    if offset is None:
        offset = 0
    if offset < 0:
        raise ToolError("invalid_arguments", "offset must be zero or greater")
    limit = optional_int(arguments, "limit")
    if limit is not None and limit < 1:
        raise ToolError("invalid_arguments", "limit must be at least 1")
    page_limit = READ_MAX_LINES if limit is None else min(limit, READ_MAX_LINES)
    data = read_bytes(path)
    lines = data.decode("utf-8").splitlines()
    if offset > len(lines):
        raise ToolError("invalid_range", "offset is past the end of the file")
    selected: list[dict[str, Any]] = []
    size = 0
    index = offset
    while index < len(lines) and len(selected) < page_limit:
        piece = lines[index]
        extra = len(piece.encode("utf-8")) + 1
        if selected and size + extra > READ_MAX_BYTES:
            break
        selected.append({"line": index + 1, "text": piece})
        size += extra
        index += 1
    body: dict[str, Any] = {
        "content_hash": content_hash(data),
        "file_path": relative,
        "lines": selected,
        "offset": offset,
    }
    if index < len(lines):
        body["next_offset"] = index
    return dump(body)
