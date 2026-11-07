"""The 128 calibration passages.

Each one is shorter than the 128-token function, so every function pads the
same list. The strings are fixed. Quantize does not draw a new set.
"""

from __future__ import annotations

_SUBJECTS = (
    "a stone bridge",
    "the north harbor",
    "a quiet library",
    "an empty platform",
    "a glass workshop",
    "the river lock",
    "a cedar cabin",
    "the market square",
)


def passages() -> list[str]:
    """128 sentences, index order, no draw."""
    rows: list[str] = []
    for index in range(128):
        subject = _SUBJECTS[index % len(_SUBJECTS)]
        rows.append(f"Passage {index} describes {subject} under a clear sky.")
    return rows
