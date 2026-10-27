"""Route one tool call to the module that owns it."""

from __future__ import annotations

from typing import Any

from harness.coding.tools.context import ToolContext
from harness.coding.tools.edit import perform as edit
from harness.coding.tools.errors import ToolError
from harness.coding.tools.glob_files import perform as glob_files
from harness.coding.tools.grep_search import perform as grep_search
from harness.coding.tools.read_file import perform as read_file
from harness.coding.tools.shell import ShellJobs
from harness.coding.tools.write_file import perform as write_file

_RUNNERS = {
    "edit": edit,
    "glob": glob_files,
    "grep_search": grep_search,
    "read_file": read_file,
    "write_file": write_file,
}


def perform(ctx: ToolContext, name: str, arguments: dict[str, Any]) -> str:
    if not isinstance(arguments, dict):
        raise ToolError("invalid_arguments", "arguments must be an object")
    if name == "run_shell_command":
        shells = ctx.shells
        if not isinstance(shells, ShellJobs):
            raise ToolError("execution_failed", "shell jobs are not available")
        return shells.perform(arguments, ctx)
    runner = _RUNNERS.get(name)
    if runner is None:
        raise ToolError("invalid_arguments", f"unknown tool {name}")
    return runner(ctx, arguments)
