"""Persona prefix. One parenthetical phrase, then the line."""

from __future__ import annotations

_PARENS = str.maketrans("", "", "()（）")
_SENTENCE = str.maketrans("", "", ".!?。！？…")


def style_line(text: str, persona: str) -> str:
    """Join the persona as a phrase the model does not speak.

    A space before an ASCII letter or digit keeps that word's own token.
    ``)The`` is the fragment ``The``; ``) The`` is the word ``▁The``.
    """
    words = " ".join(persona.strip().translate(_PARENS).translate(_SENTENCE).split())
    if not words:
        message = "persona is required"
        raise ValueError(message)
    spoken = text.lstrip()
    gap = " " if _needs_word_gap(spoken) else ""
    return f"({words}){gap}{spoken}"


def _needs_word_gap(text: str) -> bool:
    first = text[:1]
    return first.isascii() and first.isalnum()
