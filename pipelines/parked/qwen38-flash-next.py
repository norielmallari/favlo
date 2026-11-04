"""Parked Qwen3.8-Flash-Next pipeline.

The command scan does not load this file. The package beside it is
``qwen38_flash_next``. The note is ``docs/parked/qwen38-flash-next``.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

PIPELINE = "qwen38-flash-next"

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import qwen38_flash_next as _pipeline  # noqa: E402


def __getattr__(name: str) -> Any:
    return getattr(_pipeline, name)
