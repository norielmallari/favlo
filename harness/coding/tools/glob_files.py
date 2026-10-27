"""List files with ripgrep. Results stay under the requested root."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from harness.coding.tools.args import (
    bounded_limit,
    cursor_offset,
    optional_int,
    optional_str,
    reject_unknown,
    require_str,
)
from harness.coding.tools.context import ToolContext
from harness.coding.tools.errors import ToolError, dump
from harness.coding.tools.globs import glob_match
from harness.coding.tools.limits import GLOB_DEFAULT_LIMIT, GLOB_MAX_LIMIT
from harness.coding.tools.paths import display_path, resolve_inside
from harness.coding.tools.rg import run_rg

_FIELDS = {"cursor", "limit", "path", "pattern"}
# Exclusions come last so a caller glob cannot put .git or .favlo back.
_EXCLUDE = ["--glob", "!.git/**", "--glob", "!.favlo/**"]


def perform(ctx: ToolContext, arguments: dict[str, Any]) -> str:
    reject_unknown(arguments, _FIELDS)
    pattern = require_str(arguments, "pattern")
    if ".." in pattern or pattern.startswith("/"):
        raise ToolError("invalid_arguments", "pattern must be a workspace glob")
    root = ctx.workspace.resolve()
    raw_root = optional_str(arguments, "path")
    search = root if raw_root is None else resolve_inside(ctx.workspace, raw_root)
    if not search.exists():
        raise ToolError("not_found", "path does not exist")
    relative_root = "." if search == root else display_path(ctx.workspace, search)
    limit = bounded_limit(
        optional_int(arguments, "limit"),
        default=GLOB_DEFAULT_LIMIT,
        maximum=GLOB_MAX_LIMIT,
    )
    start = cursor_offset(arguments)
    paths = _list(root, relative_root, pattern)
    page = paths[start : start + limit]
    truncated = start + limit < len(paths)
    body: dict[str, Any] = {"paths": page, "truncated": truncated}
    if truncated:
        body["next_cursor"] = str(start + len(page))
    return dump(body)


def _list(root: Path, relative_root: str, pattern: str) -> list[str]:
    args = [
        "--files",
        "--hidden",
        "--no-require-git",
        "--color",
        "never",
        *_EXCLUDE,
    ]
    if relative_root != ".":
        args.append(relative_root)
    completed = run_rg(args, cwd=root)
    if completed.returncode not in (0, 1):
        detail = completed.stderr.strip().splitlines()
        message = detail[0][:400] if detail else "rg failed"
        raise ToolError("execution_failed", message)
    found: list[str] = []
    for line in completed.stdout.splitlines():
        relative = _kept(root, line)
        if relative is not None and glob_match(relative, pattern):
            found.append(relative)
    return sorted(set(found))


def _kept(root: Path, line: str) -> str | None:
    text = line.strip()
    if text == "" or text.startswith(".favlo/") or text.startswith(".git/"):
        return None
    candidate = Path(text)
    if candidate.is_absolute():
        resolved = candidate.resolve(strict=False)
    else:
        resolved = (root / candidate).resolve(strict=False)
    if resolved != root and root not in resolved.parents:
        return None
    if not resolved.is_file():
        return None
    relative = resolved.relative_to(root).as_posix()
    if relative.startswith(".favlo/") or relative.startswith(".git/"):
        return None
    return relative
