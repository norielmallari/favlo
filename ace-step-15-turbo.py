"""ACE-Step 1.5 XL Turbo pipeline.

``favlo --ace-step-15-turbo`` loads this file by path. The model lives in the
``ace_step_15_turbo`` package beside it. This file does not import another pipeline.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

PIPELINE = "ace-step-15-turbo"

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import ace_step_15_turbo as _pipeline  # noqa: E402


def __getattr__(name: str) -> Any:
    return getattr(_pipeline, name)
