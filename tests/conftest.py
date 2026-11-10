"""Put ``pipelines/`` on the import path once."""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]


def pytest_configure() -> None:
    text = str(_ROOT / "pipelines")
    if text not in sys.path:
        sys.path.insert(0, text)
