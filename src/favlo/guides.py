"""Setup guides the terminal chat reads back to the operator."""

from __future__ import annotations

from favlo.cli import repo_root

GUIDES = ("clients", "models", "history")


def read_guide(name: str) -> str:
    """Return one guide. ``name`` is ``clients``, ``models``, or ``history``."""
    if name not in GUIDES:
        known = ", ".join(GUIDES)
        message = f"unknown guide {name}; known: {known}"
        raise ValueError(message)
    path = repo_root() / "docs" / "guides" / f"{name}.md"
    if not path.is_file():
        message = f"guide file is missing: {path}"
        raise FileNotFoundError(message)
    return path.read_text(encoding="utf-8")
