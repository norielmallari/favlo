"""Failed Ternary Bonsai 2 pipeline.

The command scan does not load this file. The package beside it is
``ternary_bonsai_2``. The note is ``docs/ternary-bonsai-2``.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

PIPELINE = "ternary-bonsai-2"

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import ternary_bonsai_2 as _pipeline  # noqa: E402


def __getattr__(name: str) -> Any:
    return getattr(_pipeline, name)
