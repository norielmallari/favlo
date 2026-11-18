"""Qwen3 prompt wrap. Thinking is off and the generation prompt is open."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .constants import PAD_ID, PROMPT_LENGTH


def wrap(prompt: str) -> str:
    """One user turn, then the assistant header with an empty think block."""
    return (
        "<|im_start|>user\n"
        f"{prompt}<|im_end|>\n"
        "<|im_start|>assistant\n"
        "<think>\n\n</think>\n\n"
    )


def encode(prompt: str, tokenizer_file: Path) -> tuple[np.ndarray, int]:
    """Pad or truncate to 512. The second value is the unpadded length."""
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(tokenizer_file))
    ids = list(tokenizer.encode(wrap(prompt)).ids)
    valid = min(len(ids), PROMPT_LENGTH)
    ids = ids[:PROMPT_LENGTH]
    if len(ids) < PROMPT_LENGTH:
        ids.extend([PAD_ID] * (PROMPT_LENGTH - len(ids)))
    return np.asarray(ids, dtype=np.int32), valid
