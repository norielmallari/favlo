"""PCM for one spoken line."""

from __future__ import annotations

import io
import wave

import numpy as np


def encode_wav(samples: np.ndarray, sample_rate: int) -> bytes:
    """16-bit mono WAV. Values outside [-1, 1] are clamped at the integer encode."""
    pcm = np.clip(np.asarray(samples, dtype=np.float32), -1.0, 1.0)
    ints = np.round(pcm * 32767.0).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(int(sample_rate))
        handle.writeframes(ints.tobytes())
    return buf.getvalue()
