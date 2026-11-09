"""Put each pipeline directory on the import path.

A package lives beside its entry file. Pytest imports that package by name.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_RUNTIMES = ("metal41", "cuda")


def pytest_configure() -> None:
    for runtime in _RUNTIMES:
        tree = _ROOT / "pipelines" / runtime
        if not tree.is_dir():
            continue
        for child in sorted(path for path in tree.iterdir() if path.is_dir()):
            text = str(child)
            if text not in sys.path:
                sys.path.insert(0, text)
