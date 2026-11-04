"""Parked Spark-X2.6-1.7B pipeline.

The command scan does not load this file. The package beside it is
``spark_x26_1_7b``. The note is ``docs/parked/spark-x26-1-7b``.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

PIPELINE = "spark-x26-1-7b"

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import spark_x26_1_7b as _pipeline  # noqa: E402


def __getattr__(name: str) -> Any:
    return getattr(_pipeline, name)
