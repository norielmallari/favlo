"""Qwen3.8-Flash-Next pipeline.

``madmetal --qwen38-flash-next`` loads this file by path. The model lives in
the ``qwen38_flash_next`` package beside it.
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
