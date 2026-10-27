"""Resolve a path and its symlinks before any tool uses it."""

from __future__ import annotations

from pathlib import Path

from harness.coding.tools.errors import ToolError


def resolve_inside(workspace: Path, raw: object) -> Path:
    """Absolute or workspace-relative path that stays inside ``workspace``."""
    if not isinstance(raw, str) or raw == "" or "\x00" in raw:
        raise ToolError("invalid_arguments", "path is required")
    root = workspace.resolve()
    candidate = Path(raw)
    if not candidate.is_absolute():
        candidate = root / candidate
    try:
        resolved = candidate.resolve(strict=False)
    except OSError as exc:
        raise ToolError("permission_denied", "path cannot be resolved") from exc
    if resolved != root and root not in resolved.parents:
        raise ToolError("permission_denied", "path is outside the workspace")
    return resolved


def display_path(workspace: Path, resolved: Path) -> str:
    """Workspace-relative POSIX path."""
    root = workspace.resolve()
    relative = resolved.resolve().relative_to(root).as_posix()
    return relative if relative != "." else "."


def assert_readable(relative: str) -> None:
    """Harness state stays hidden. Search artifacts are readable."""
    if relative == ".favlo" or relative.startswith(".favlo/"):
        if relative.startswith(".favlo/artifacts/"):
            return
        raise ToolError("permission_denied", "path is reserved")


def assert_writable(relative: str) -> None:
    """File tools cannot create or edit harness state."""
    if relative == ".favlo" or relative.startswith(".favlo/"):
        raise ToolError("permission_denied", "path is reserved")
