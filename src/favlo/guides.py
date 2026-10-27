"""Setup guides the configuration tools read back."""

from __future__ import annotations

from pathlib import Path

from favlo.cli import pipeline_names, repo_root

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


def setup_pipelines(root: Path | None = None) -> tuple[str, ...]:
    """Pipeline ids that have ``docs/<pipeline>/SETUP.md``."""
    base = repo_root() if root is None else root
    known = set(pipeline_names(base))
    docs = base / "docs"
    if not docs.is_dir():
        return ()
    found = [
        path.name
        for path in sorted(docs.iterdir())
        if path.name in known and (path / "SETUP.md").is_file()
    ]
    return tuple(found)


def read_setup(pipeline: str, root: Path | None = None) -> str:
    """Return one pipeline's setup guide. ``pipeline`` is the flag stem."""
    name = pipeline.strip()
    known = setup_pipelines(root)
    if name not in known:
        listed = ", ".join(known) if known else "none"
        message = f"unknown setup guide {name}; known: {listed}"
        raise ValueError(message)
    base = repo_root() if root is None else root
    return (base / "docs" / name / "SETUP.md").read_text(encoding="utf-8")
