"""Wan2.2 TI2V-5B pipeline.

``favlo --wan22-ti2v-5b`` loads this file by path. The model lives in the
``wan22_ti2v_5b`` package beside it. This file does not import another pipeline.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

PIPELINE = "wan22-ti2v-5b"

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import wan22_ti2v_5b as _pipeline  # noqa: E402


def __getattr__(name: str) -> Any:
    return getattr(_pipeline, name)
