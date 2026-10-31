"""Workspace skills shipped with the app, one directory per workspace type."""

from __future__ import annotations

from pathlib import Path

from web_api.global_skill_catalog import read_skill_file


def skills_for(kind: str) -> list[tuple[str, str, str]]:
    """Name, description, and definition for that workspace type."""
    root = Path(__file__).resolve().parent / "skills-workspace" / kind
    if not root.is_dir():
        return []
    skills: list[tuple[str, str, str]] = []
    for path in sorted(root.glob("*/SKILL.md")):
        name, description, definition = read_skill_file(path)
        if name and description:
            skills.append((name, description, definition))
    return skills
