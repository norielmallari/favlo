"""Line tokens. Published inference tokenizes without the BOS id, then appends audio-start."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

AUDIO_START = 101


class LineTokenizer:
    """Encode one styled line and append the audio-start token."""

    def __init__(self, path: Path) -> None:
        self._tokenizer = Tokenizer.from_file(str(path))

    def encode(self, text: str) -> np.ndarray:
        ids = list(self._tokenizer.encode(text, add_special_tokens=False).ids)
        ids.append(AUDIO_START)
        return np.asarray(ids, dtype=np.int32)
