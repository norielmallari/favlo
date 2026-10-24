"""Style prefix. A parenthetical is ordinary text in front of the line."""

from __future__ import annotations


def style_line(text: str, instructions: str) -> str:
    """Prepend ``instructions`` as ``(text)`` when they are not already wrapped."""
    words = instructions.strip()
    if not words:
        return text
    if words.startswith("(") and words.endswith(")"):
        prefix = words
    else:
        prefix = f"({words})"
    return f"{prefix}{text}"
