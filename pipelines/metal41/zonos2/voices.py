"""Shipped speaker vectors. Each file is 2048 floats from the encoder."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from zonos2.layout import SPEAKER_DIM

_ROOT = Path(__file__).resolve().parent / "voices"
_CATALOG = (
    {
        "id": "heather-lawrence",
        "label": "Heather Lawrence",
        "source": "LibriVox, A Daughter of the Land, section 4",
    },
    {
        "id": "soprano-harmony",
        "label": "SopranoHarmony",
        "source": "LibriVox, The Mystic Will, chapter 8",
    },
)


def list_voices() -> list[dict[str, str]]:
    """Ids a client passes as ``voice``."""
    return [dict(item) for item in _CATALOG]


def load(name: str) -> np.ndarray:
    """The 2048-float vector for one collection id."""
    known = {item["id"] for item in _CATALOG}
    if name not in known:
        message = "voice is not in the collection"
        raise ValueError(message)
    vector = np.load(_ROOT / f"{name}.npy")
    if getattr(vector, "shape", None) != (SPEAKER_DIM,) or not np.isfinite(vector).all():
        message = "the speaker vector is 2048 floats"
        raise ValueError(message)
    return np.ascontiguousarray(vector, dtype=np.float32)
