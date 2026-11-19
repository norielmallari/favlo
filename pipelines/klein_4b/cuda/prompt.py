"""Qwen3 user turn with thinking closed, padded to 512."""

from __future__ import annotations

from pathlib import Path

from klein_4b.cuda.constants import PAD_ID, PROMPT_LENGTH


def wrap(prompt: str) -> str:
    return (
        "<|im_start|>user\n"
        f"{prompt}<|im_end|>\n"
        "<|im_start|>assistant\n"
        "<think>\n\n"
        "</think>\n\n"
    )


def encode(prompt: str, tokenizer_path: Path) -> tuple[list[int], int]:
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    ids = tokenizer.encode(wrap(prompt), add_special_tokens=False).ids
    valid = min(len(ids), PROMPT_LENGTH)
    if len(ids) > PROMPT_LENGTH:
        ids = ids[:PROMPT_LENGTH]
    elif len(ids) < PROMPT_LENGTH:
        ids = ids + [PAD_ID] * (PROMPT_LENGTH - len(ids))
    return ids, valid
