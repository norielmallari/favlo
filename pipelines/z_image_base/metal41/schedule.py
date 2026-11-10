"""Flow-matching knots for the base checkpoint.

The scheduler file sets shift 6. A request may choose from 1 to 100
velocities. When it omits a sampler field, the registry row supplies
``steps``, ``guidance``, ``shift``, ``cfg_truncation``, and
``cfg_normalization``. A row without a key uses 28, 4, 6, 1, and 0.
"""

from __future__ import annotations

import numpy as np
from z_image.metal41.schedule import T_SCALE
from z_image.metal41.schedule import sigmas as _sigmas
from z_image.metal41.schedule import timestep as timestep

SHIFT = 6.0
STEPS = 28
GUIDANCE = 4.0
CFG_TRUNCATION = 1.0
CFG_NORMALIZATION = 0.0
MAX_STEPS = 100

__all__ = [
    "CFG_NORMALIZATION",
    "CFG_TRUNCATION",
    "GUIDANCE",
    "MAX_STEPS",
    "SHIFT",
    "STEPS",
    "T_SCALE",
    "sampler_defaults",
    "sigmas",
    "timestep",
]

_FALLBACK = {
    "steps": STEPS,
    "guidance": GUIDANCE,
    "shift": SHIFT,
    "cfg_truncation": CFG_TRUNCATION,
    "cfg_normalization": CFG_NORMALIZATION,
}


def sampler_defaults(model: str | None = None) -> dict[str, int | float]:
    """Sampler fields for a request that omits them. The registry row wins."""
    found: dict[str, int | float] = dict(_FALLBACK)
    row = _row(model)
    if row is None:
        return found
    for key in _FALLBACK:
        if key not in row:
            continue
        found[key] = _checked(key, row[key])
    return found


def _row(model: str | None) -> dict[str, object] | None:
    if not isinstance(model, str) or not model.strip():
        return None
    from favlo.registry import load_registry

    models = load_registry().get("models")
    if not isinstance(models, dict):
        return None
    name = model.strip()
    entry = models.get(name)
    if not isinstance(entry, dict):
        entry = next(
            (
                row
                for row in models.values()
                if isinstance(row, dict) and row.get("alias") == name
            ),
            None,
        )
    if not isinstance(entry, dict):
        return None
    return entry


def _checked(key: str, value: object) -> int | float:
    if key == "steps":
        if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= MAX_STEPS:
            message = f"registry steps must be from 1 to {MAX_STEPS}, got {value}"
            raise ValueError(message)
        return value
    if key == "cfg_normalization" and isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        message = f"registry {key} must be a number, got {value}"
        raise ValueError(message)
    number = float(value)
    if key == "shift":
        if not 0.0 < number <= 100.0:
            message = f"registry shift must be greater than 0 and at most 100, got {value}"
            raise ValueError(message)
        return number
    if not 0.0 <= number <= 100.0:
        message = f"registry {key} must be from 0 to 100, got {value}"
        raise ValueError(message)
    return number


def sigmas(steps: int = STEPS, shift: float = SHIFT) -> np.ndarray:
    if steps > MAX_STEPS:
        message = f"steps must be from 1 to {MAX_STEPS}, got {steps}"
        raise ValueError(message)
    return _sigmas(steps, shift)
