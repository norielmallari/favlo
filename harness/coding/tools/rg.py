"""Run ripgrep. Callers pass the search root and do not widen it."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from harness.coding.tools.errors import ToolError


def run_rg(args: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    binary = shutil.which("rg")
    if binary is None:
        raise ToolError("execution_failed", "rg is not available")
    try:
        return subprocess.run(
            [binary, *args],
            cwd=cwd,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        detail = exc.strerror or "rg could not be started"
        raise ToolError("execution_failed", detail) from exc


def regex_error(stderr: str) -> str:
    for line in stderr.splitlines():
        if line.startswith("error:"):
            detail = line.removeprefix("error:").strip()
            if detail:
                return detail[:400]
    return "invalid regular expression"
