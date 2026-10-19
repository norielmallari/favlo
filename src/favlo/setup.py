"""First serve with an empty registry. The pipeline owns the download.

A terminal with no registered rows checks Metal 4.1, then runs
``favlo --qwen36-35b-a3b setup``. A pipe leaves the registry empty.
"""

from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from favlo.registry import load_registry

FIRST_PIPELINE = "qwen36-35b-a3b"
_SHADER = """
#include <metal_stdlib>
using namespace metal;
kernel void favlo_ready(uint id [[thread_position_in_grid]]) {}
"""


def registry_empty(root: Path | None = None) -> bool:
    return not load_registry(root)["models"]


def metal_ready() -> bool:
    """True when a small shader compiles at Metal language version 4.1."""
    try:
        import Metal
    except ImportError:
        return False
    device = Metal.MTLCreateSystemDefaultDevice()
    if device is None:
        return False
    options = Metal.MTLCompileOptions.alloc().init()
    options.setLanguageVersion_((4 << 16) + 1)
    library, error = device.newLibraryWithSource_options_error_(_SHADER, options, None)
    del error
    return library is not None


def prepare(
    root: Path,
    *,
    stdin_tty: bool,
    console: str,
    metal: Callable[[], bool] = metal_ready,
    run: Callable[..., Any] = subprocess.run,
) -> int:
    """Download and pack when this terminal has no models. ``0`` continues to listen."""
    if not registry_empty(root):
        return 0
    if not stdin_tty:
        return 0
    if not metal():
        sys.stderr.write("this runtime cannot run here\n")
        return 1
    env = os.environ.copy()
    env["FAVLO_ROOT"] = str(root)
    env["FAVLO_SKIP_PLATE"] = "1"
    command = [
        sys.executable,
        "-m",
        "favlo",
        f"--{FIRST_PIPELINE}",
        "setup",
        "--console",
        console,
    ]
    completed = run(command, cwd=root, env=env)
    return int(completed.returncode)
