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
    lines = [label.strip()]
    count = encode(_join(lines))
    if count > tokens:
        message = "the label is longer than the token target"
        raise ValueError(message)
    rng = random.Random(seed)
    added = 0
    while count < tokens and added <= tokens:
        lines.append(_filler_line(rng))
        count = encode(_join(lines))
        added += 1
    if count < tokens:
        message = "the filler is shorter than the token target"
        raise RuntimeError(message)
    lines = _fit_tokens(lines, tokens, encode)
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
    lines[-1] = original
    message = "the prompt cannot be trimmed to the token target"
    raise RuntimeError(message)


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
