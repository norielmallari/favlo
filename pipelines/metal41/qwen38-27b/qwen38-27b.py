"""Qwen3.8-27B pipeline.

``favlo --qwen38-27b`` loads this file by path. The model lives in the
``qwen38_27b`` package beside it. This file does not import Flash Next.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

PIPELINE = "qwen38-27b"

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import qwen38_27b as _pipeline  # noqa: E402


def __getattr__(name: str) -> Any:
    return getattr(_pipeline, name)
