"""Run the hidden suite against a workspace the model cannot edit."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

_CHECKS = Path(__file__).resolve().parent / "checks" / "test_ledger.py"


def grade(workspace: Path) -> subprocess.CompletedProcess[str]:
    """Pytest on the hidden suite. Exit 0 is a pass."""
    env = os.environ.copy()
    env["CODING_WORKSPACE"] = str(workspace.resolve())
    return subprocess.run(
        [sys.executable, "-m", "pytest", str(_CHECKS), "-q", "--noconftest"],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
