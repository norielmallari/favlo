"""Failed ZONOS2 pipeline.

The command scan does not load this file. The package beside it is
``zonos2``. The note is ``docs/zonos2``.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

PIPELINE = "zonos2"

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import zonos2 as _pipeline  # noqa: E402


def __getattr__(name: str) -> Any:
    return getattr(_pipeline, name)
