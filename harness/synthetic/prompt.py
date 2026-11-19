"""One user message whose chat encoding is an exact token count.

The label is the first line, so two lengths do not share a token prefix.
``encode`` counts the prompt the server will prefill, including the chat
template. The same seed and label always write the same text.
"""

from __future__ import annotations

import importlib
import inspect
import random
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

_SENTENCES = (
    "The grass is green.",
    "The sky is blue.",
    "The sun is yellow.",
    "Here we go.",
    "There and back again.",
)
_LINE_CHARS = 160
_PADS = ("0", "1", "x", ".")
_PAIR_PADS = ("00", "10", "x0", ".0")
_PAD_STEPS = 8
_WALK_BACK = 16


def user_text(
    *,
    tokens: int,
    label: str,
    encode: Callable[[str], int],
    seed: int = 0,
) -> str:
    """Filler under ``label`` that ``encode`` counts as exactly ``tokens``."""
    if tokens < 1:
        message = "tokens must be at least 1"
        raise ValueError(message)
    if not label.strip():
        message = "label must not be empty"
        raise ValueError(message)
    label_text = label.strip()
    count = encode(_join([label_text]))
    if count > tokens:
        message = "the label is longer than the token target"
        raise ValueError(message)
    if count == tokens:
        return _join([label_text])
    short = 0
    added = 1
    found = False
    while added <= tokens:
        count = encode(_join(_lines_with(label_text, seed, added)))
        if count >= tokens:
            found = True
            break
        short = added
        added *= 2
    if not found:
        message = "the filler is shorter than the token target"
        raise RuntimeError(message)
    lo = short
    hi = added
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        got = encode(_join(_lines_with(label_text, seed, mid)))
        if got < tokens:
            lo = mid
        else:
            hi = mid
    lines = _fit_tokens(_lines_with(label_text, seed, hi), tokens, encode)
    return _join(lines)


def chat_encode(selection: Any, *, thinking: bool) -> Callable[[str], int]:
    """Count tokens the way this pipeline's ``encode_chat`` does.

    The callable takes the user message. Thinking follows the request, tools
    stay empty, and a kept-thinking flag is off. That is the one-turn prompt
    the listener renders.
    """
    from favlo.cli import pipeline_tree, repo_root

    tree = pipeline_tree(
        selection.runtime,
        pipeline=str(selection.pipeline),
        pipeline_root=getattr(selection, "pipeline_root", None),
        root=repo_root(),
    )
    from favlo.cli import ensure_pipelines_path

    ensure_pipelines_path()
    module = importlib.import_module(f"{selection.pipeline}.{selection.runtime}.chat")
    encode_chat = module.encode_chat
    root = Path(selection.path)

    def encode(text: str) -> int:
        messages = [{"role": "user", "content": text}]
        return len(_call_encode(encode_chat, root, messages, thinking))

    return encode


def _call_encode(fn: Any, root: Path, messages: list[dict[str, str]], thinking: bool) -> list[int]:
    arguments: dict[str, Any] = {}
    for name, param in inspect.signature(fn).parameters.items():
        if name == "root":
            arguments[name] = root
        elif name == "messages":
            arguments[name] = messages
        elif name == "tools":
            arguments[name] = None
        elif name in {"enable_thinking", "thinking"}:
            arguments[name] = thinking
        elif name == "reasoning_effort":
            arguments[name] = "high" if thinking else "none"
        elif name == "preserve_thinking":
            arguments[name] = False
        elif param.default is inspect.Parameter.empty:
            message = f"encode_chat argument {name} is not filled"
            raise TypeError(message)
    ids = fn(**arguments)
    if not isinstance(ids, list):
        message = "encode_chat did not return token ids"
        raise TypeError(message)
    return ids


def _fit_tokens(
    lines: list[str],
    tokens: int,
    encode: Callable[[str], int],
) -> list[str]:
    """Shorten the last filler line until the prompt encodes to ``tokens``."""
    count = encode(_join(lines))
    if count == tokens:
        return lines
    if count < tokens:
        message = "the prompt is shorter than the token target"
        raise RuntimeError(message)
    if len(lines) < 2:
        message = "the prompt cannot be trimmed to the token target"
        raise RuntimeError(message)
    original = lines[-1]
    lo = 0
    hi = len(original)
    under: int | None = None
    while lo <= hi:
        mid = (lo + hi) // 2
        lines[-1] = original[:mid]
        got = encode(_join(lines))
        if got == tokens:
            return lines
        if got > tokens:
            hi = mid - 1
        else:
            under = mid
            lo = mid + 1
    if under is None:
        lines[-1] = original
        message = "the prompt cannot be trimmed to the token target"
        raise RuntimeError(message)
    for end in range(under + 1, len(original) + 1):
        lines[-1] = original[:end]
        got = encode(_join(lines))
        if got == tokens:
            return lines
        if got > tokens:
            break
    if _close_gap(lines, original, under, tokens, encode):
        return lines
    lines[-1] = original
    message = "the prompt cannot be trimmed to the token target"
    raise RuntimeError(message)


def _close_gap(
    lines: list[str],
    original: str,
    under: int,
    tokens: int,
    encode: Callable[[str], int],
) -> bool:
    """Pad after a trim that stepped over ``tokens``.

    A character can add more than one token, so the prefixes of the filler
    line miss the target. A short walk back plus a one-piece pad closes that
    gap. When the first character of the last line still steps over, the
    previous line is retokenized.
    """
    saved = list(lines)
    stop = max(-1, under - _WALK_BACK)
    for end in range(under, stop, -1):
        lines[-1] = original[:end]
        count = encode(_join(lines))
        if count == tokens:
            return True
        if count > tokens:
            continue
        if _pad_exact(lines, tokens, encode, count):
            return True
    lines[:] = saved
    if _retokenize_previous(lines, tokens, encode):
        return True
    lines[:] = saved
    return False


def _retokenize_previous(
    lines: list[str],
    tokens: int,
    encode: Callable[[str], int],
) -> bool:
    """Replace the end of the previous line when a new line cannot add one token."""
    if len(lines) < 2:
        return False
    saved = list(lines)
    lines.pop()
    if not lines[-1]:
        lines[:] = saved
        return False
    original = lines[-1]
    stem = original[:-1]
    last = original[-1]
    for pair in _PAIR_PADS:
        lines[-1] = stem + pair
        if encode(_join(lines)) == tokens:
            return True
    for digit in "0123456789":
        lines[-1] = stem + digit + last
        if encode(_join(lines)) == tokens:
            return True
    lines[:] = saved
    return False


def _pad_exact(
    lines: list[str],
    tokens: int,
    encode: Callable[[str], int],
    count: int,
) -> bool:
    """Append a pad piece until ``encode`` lands on ``tokens``."""
    stem = lines[-1]
    for _ in range(_PAD_STEPS):
        if count >= tokens:
            break
        moved = False
        for piece in _PADS:
            lines[-1] = stem + piece
            got = encode(_join(lines))
            if got == tokens:
                return True
            if count < got < tokens:
                stem = lines[-1]
                count = got
                moved = True
                break
        if not moved:
            lines[-1] = stem
            return False
    lines[-1] = stem
    return count == tokens


def _lines_with(label: str, seed: int, added: int) -> list[str]:
    """The label plus ``added`` filler lines from ``seed``.

    Rebuilding from the seed keeps a length search on the same text the
    one-line loop would have written.
    """
    rng = random.Random(seed)
    lines = [label]
    for _ in range(added):
        lines.append(_filler_line(rng))
    return lines


def _filler_line(rng: random.Random) -> str:
    text = ""
    parts: list[str] = []
    while len(text) < _LINE_CHARS:
        piece = _SENTENCES[rng.randrange(len(_SENTENCES))]
        parts.append(piece)
        text = " ".join(parts)
    return text


def _join(lines: list[str]) -> str:
    return "\n".join(lines) + "\n"
