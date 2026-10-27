"""Text search over ripgrep. No matches is a successful result."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from harness.coding.tools.args import (
    bounded_limit,
    cursor_offset,
    optional_bool,
    optional_int,
    optional_str,
    reject_unknown,
    require_str,
)
from harness.coding.tools.context import ToolContext
from harness.coding.tools.errors import ToolError, dump
from harness.coding.tools.globs import glob_match
from harness.coding.tools.limits import (
    GREP_DEFAULT_LIMIT,
    GREP_MAX_CONTEXT,
    GREP_MAX_LIMIT,
    GREP_RESPONSE_BUDGET,
)
from harness.coding.tools.paths import display_path, resolve_inside
from harness.coding.tools.rg import regex_error, run_rg

_FIELDS = {
    "case_sensitive",
    "context_lines",
    "cursor",
    "glob",
    "limit",
    "literal",
    "path",
    "pattern",
}
_EXCLUDE = ["--glob", "!.git/**", "--glob", "!.favlo/**"]


def perform(ctx: ToolContext, arguments: dict[str, Any]) -> str:
    reject_unknown(arguments, _FIELDS)
    pattern = require_str(arguments, "pattern")
    literal = optional_bool(arguments, "literal", default=False)
    case_sensitive = optional_bool(arguments, "case_sensitive", default=True)
    context_lines = optional_int(arguments, "context_lines")
    if context_lines is None:
        context_lines = 0
    if context_lines < 0:
        raise ToolError("invalid_arguments", "context_lines must be zero or greater")
    if context_lines > GREP_MAX_CONTEXT:
        raise ToolError("invalid_arguments", "context_lines exceeds the harness cap")
    file_glob = optional_str(arguments, "glob")
    root = ctx.workspace.resolve()
    raw_path = optional_str(arguments, "path")
    search = root if raw_path is None else resolve_inside(ctx.workspace, raw_path)
    if not search.exists():
        raise ToolError("not_found", "path does not exist")
    relative_search = "." if search == root else display_path(ctx.workspace, search)
    limit = bounded_limit(
        optional_int(arguments, "limit"),
        default=GREP_DEFAULT_LIMIT,
        maximum=GREP_MAX_LIMIT,
    )
    start = cursor_offset(arguments)
    matches = _search(
        root,
        relative_search,
        pattern,
        literal=literal,
        case_sensitive=case_sensitive,
        context_lines=context_lines,
        file_glob=file_glob,
    )
    artifact = _artifact(ctx, matches)
    page = matches[start : start + limit]
    while len(page) > 1 and len(dump(page).encode("utf-8")) > GREP_RESPONSE_BUDGET:
        page = page[:-1]
    truncated = start + len(page) < len(matches)
    body: dict[str, Any] = {
        "match_count": len(matches),
        "matches": page,
        "truncated": truncated,
    }
    if truncated:
        body["next_cursor"] = str(start + len(page))
    if artifact is not None:
        body["artifact"] = artifact
    return dump(body)


def _search(
    root: Path,
    relative_search: str,
    pattern: str,
    *,
    literal: bool,
    case_sensitive: bool,
    context_lines: int,
    file_glob: str | None,
) -> list[dict[str, Any]]:
    args = ["--json", "--hidden", "--no-require-git", "--color", "never", *_EXCLUDE]
    if literal:
        args.append("--fixed-strings")
    args.append("--case-sensitive" if case_sensitive else "--ignore-case")
    if context_lines:
        args.extend(["-C", str(context_lines)])
    args.extend(["-e", pattern, "--", relative_search])
    completed = run_rg(args, cwd=root)
    if completed.returncode == 2:
        raise ToolError("invalid_arguments", regex_error(completed.stderr))
    if completed.returncode not in (0, 1):
        detail = completed.stderr.strip().splitlines()
        message = detail[0][:400] if detail else "rg failed"
        raise ToolError("execution_failed", message)
    matches = _matches(root, completed.stdout, context_lines)
    if file_glob:
        matches = [item for item in matches if glob_match(str(item["path"]), file_glob)]
    return matches


def _matches(root: Path, stdout: str, context_lines: int) -> list[dict[str, Any]]:
    events: list[tuple[str, dict[str, Any]]] = []
    for line in stdout.splitlines():
        if line.strip() == "":
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ToolError("execution_failed", "rg returned malformed output") from exc
        if not isinstance(event, dict):
            continue
        kind = event.get("type")
        data = event.get("data")
        if kind not in {"match", "context"} or not isinstance(data, dict):
            continue
        item = _event(root, data)
        if item is None:
            continue
        events.append((str(kind), item))
    matches: list[dict[str, Any]] = []
    for index, (kind, item) in enumerate(events):
        if kind != "match":
            continue
        body = {"line_number": item["line_number"], "path": item["path"], "text": item["text"]}
        if context_lines:
            body["context_before"] = _context(events, index, -1, context_lines)
            body["context_after"] = _context(events, index, 1, context_lines)
        matches.append(body)
    return matches


def _context(
    events: list[tuple[str, dict[str, Any]]],
    index: int,
    step: int,
    count: int,
) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    cursor = index + step
    while 0 <= cursor < len(events) and len(found) < count:
        kind, item = events[cursor]
        if kind == "match":
            break
        if kind == "context":
            found.append({"line_number": item["line_number"], "text": item["text"]})
        cursor += step
    if step < 0:
        found.reverse()
    return found


def _event(root: Path, data: dict[str, Any]) -> dict[str, Any] | None:
    path_body = data.get("path")
    lines = data.get("lines")
    number = data.get("line_number")
    if (
        not isinstance(path_body, dict)
        or not isinstance(lines, dict)
        or not isinstance(number, int)
    ):
        return None
    raw_path = path_body.get("text")
    raw_line = lines.get("text")
    if not isinstance(raw_path, str) or not isinstance(raw_line, str):
        return None
    relative = _relative(root, raw_path)
    if relative is None:
        return None
    return {"line_number": number, "path": relative, "text": _line(raw_line)}


def _relative(root: Path, raw: str) -> str | None:
    candidate = Path(raw)
    if candidate.is_absolute():
        resolved = candidate.resolve(strict=False)
    else:
        resolved = (root / candidate).resolve()
    if resolved != root and root not in resolved.parents:
        return None
    relative = resolved.relative_to(root).as_posix()
    if relative.startswith(".favlo/") or relative.startswith(".git/"):
        return None
    return relative


def _line(raw: str) -> str:
    if raw.endswith("\n"):
        raw = raw[:-1]
    if raw.endswith("\r"):
        raw = raw[:-1]
    return raw


def _artifact(ctx: ToolContext, matches: list[dict[str, Any]]) -> str | None:
    if len(dump({"matches": matches}).encode("utf-8")) <= GREP_RESPONSE_BUDGET:
        return None
    directory = ctx.state_dir / "artifacts"
    directory.mkdir(parents=True, exist_ok=True)
    number = 1
    while (directory / f"grep-{number}.json").exists():
        number += 1
    path = directory / f"grep-{number}.json"
    path.write_text(dump({"matches": matches}), encoding="utf-8")
    return f".favlo/artifacts/grep-{number}.json"
