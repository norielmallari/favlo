"""A seeded passkey file. The same seed, length, and depth always write the same text."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import random

FILE_NAME = "haystack.txt"
QUESTION = "What is the special magic number in haystack.txt?"

_SENTENCES = (
    "The grass is green.",
    "The sky is blue.",
    "The sun is yellow.",
    "Here we go.",
    "There and back again.",
)
_LINE_CHARS = 160


@dataclass(frozen=True)
class Haystack:
    """One generated file and the value a reply must report."""

    text: str
    number: str
    line: int
    tokens: int


def needle_line(number: str) -> str:
    """The one sentence that carries the passkey."""
    return f"The special magic number is {number}."


def generate(
    *,
    tokens: int,
    depth: float,
    seed: int,
    encode: Callable[[str], int],
) -> Haystack:
    """Filler lines and one needle line. The file encodes to exactly ``tokens``.

    ``depth`` is 0 at the first line and 1 at the last. ``encode`` counts tokens
    in the file text. The needle is a seven-digit number the filler never contains.
    """
    if tokens < 1:
        message = "tokens must be at least 1"
        raise ValueError(message)
    if depth < 0.0 or depth > 1.0:
        message = "depth must be from 0 to 1"
        raise ValueError(message)
    rng = random.Random(seed)
    number = f"{rng.randrange(1_000_000, 10_000_000)}"
    pool = _filler_pool(rng, tokens, encode)
    count = _line_count(pool, tokens, encode)
    lines = pool[:count]
    index = round(depth * (len(lines) - 1))
    lines[index] = needle_line(number)
    added = 0
    while encode(_join(lines)) < tokens and added < tokens:
        if index == len(lines) - 1:
            lines.insert(index, _filler_line(rng))
            index += 1
        else:
            lines.append(_filler_line(rng))
        added += 1
    lines = _fit_tokens(lines, index, tokens, encode)
    text = _join(lines)
    return Haystack(text=text, number=number, line=index + 1, tokens=encode(text))


def _filler_pool(
    rng: random.Random,
    tokens: int,
    encode: Callable[[str], int],
) -> list[str]:
    pool: list[str] = []
    batch = max(8, tokens // 8)
    while True:
        pool.extend(_filler_line(rng) for _ in range(batch))
        if encode(_join(pool)) >= tokens:
            return pool
        if len(pool) > max(tokens * 4, 8):
            message = "the filler pool is shorter than the token target"
            raise RuntimeError(message)


def _line_count(pool: list[str], tokens: int, encode: Callable[[str], int]) -> int:
    if encode(_join(pool[:1])) >= tokens:
        return 1
    if encode(_join(pool)) < tokens:
        message = "the filler pool is shorter than the token target"
        raise RuntimeError(message)
    lo = 1
    hi = len(pool)
    while lo < hi:
        mid = (lo + hi) // 2
        if encode(_join(pool[:mid])) < tokens:
            lo = mid + 1
        else:
            hi = mid
    return lo


def _fit_tokens(
    lines: list[str],
    needle: int,
    tokens: int,
    encode: Callable[[str], int],
) -> list[str]:
    """Shorten one filler line until the file encodes to exactly ``tokens``."""
    count = encode(_join(lines))
    if count == tokens:
        return lines
    if count < tokens:
        message = "the file is shorter than the token target"
        raise RuntimeError(message)
    filler = _filler_index(lines, needle)
    original = lines[filler]
    lo = 0
    hi = len(original)
    under: tuple[int, int] | None = None
    while lo <= hi:
        mid = (lo + hi) // 2
        lines[filler] = original[:mid]
        got = encode(_join(lines))
        if got == tokens:
            return lines
        if got > tokens:
            hi = mid - 1
        else:
            under = (mid, got)
            lo = mid + 1
    if under is None:
        lines[filler] = original
        message = "the file cannot be trimmed to the token target"
        raise RuntimeError(message)
    start, _got = under
    for end in range(start + 1, len(original) + 1):
        lines[filler] = original[:end]
        got = encode(_join(lines))
        if got == tokens:
            return lines
        if got > tokens:
            break
    lines[filler] = original
    message = "the file cannot be trimmed to the token target"
    raise RuntimeError(message)


def _filler_index(lines: list[str], needle: int) -> int:
    """A filler line whose tail can shrink without moving the needle."""
    if needle != len(lines) - 1:
        return len(lines) - 1
    if needle > 0:
        return needle - 1
    message = "the file cannot be trimmed to the token target"
    raise RuntimeError(message)


def _filler_line(rng: random.Random) -> str:
    parts: list[str] = []
    text = ""
    while len(text) < _LINE_CHARS:
        piece = _SENTENCES[rng.randrange(len(_SENTENCES))]
        text = piece if not text else f"{text} {piece}"
        parts.append(piece)
    return " ".join(parts)


def _join(lines: list[str]) -> str:
    return "\n".join(lines) + "\n"
