"""Qwen3.6-35B-A3B pipeline.

``madmetal --qwen36-35b-a3b`` loads this file by path. The model lives in the
``qwen36_35b_a3b`` package beside it. This file does not import another pipeline.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

PIPELINE = "qwen36-35b-a3b"

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import qwen36_35b_a3b as _pipeline  # noqa: E402


def __getattr__(name: str) -> Any:
    return getattr(_pipeline, name)
