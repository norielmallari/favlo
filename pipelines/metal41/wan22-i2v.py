"""Wan 2.2 I2V pipeline.

``favlo --wan22-i2v`` loads this file by path. The model lives in the
``wan22_i2v`` package beside it. This file does not import another pipeline.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

PIPELINE = "wan22-i2v"

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import wan22_i2v as _pipeline  # noqa: E402


def __getattr__(name: str) -> Any:
    return getattr(_pipeline, name)
