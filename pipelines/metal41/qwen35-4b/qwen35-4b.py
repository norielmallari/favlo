"""Qwen3.5-4B pipeline.

``favlo --qwen35-4b`` loads this file by path. The model lives in the
``qwen35_4b`` package beside it.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

PIPELINE = "qwen35-4b"

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import qwen35_4b as _pipeline  # noqa: E402


def __getattr__(name: str) -> Any:
    return getattr(_pipeline, name)
