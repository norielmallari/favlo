"""One image request after parse."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Request:
    prompt: str
    width: int
    height: int
    steps: int
    seed: int
    images: tuple[np.ndarray, ...]
