"""Gemma 4 26B-A4B pipeline.

``favlo --gemma4-26b-a4b`` loads this file by path. The model lives in the
``gemma4_26b_a4b`` package beside it. This file does not import another pipeline.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

PIPELINE = "gemma4-26b-a4b"

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import gemma4_26b_a4b as _pipeline  # noqa: E402


def __getattr__(name: str) -> Any:
    return getattr(_pipeline, name)
