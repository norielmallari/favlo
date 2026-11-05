"""Workspace skills appended on a turn. A workspace type has no standing prompt."""

from __future__ import annotations

from typing import Any


def skills_text(skills: list[dict[str, Any]]) -> str | None:
    """Markdown block for stored skills, or none when the list is empty."""
    blocks = ["# Skills"]
    for skill in skills:
        name = str(skill.get("name") or "").strip()
        if not name:
            continue
        description = str(skill.get("description") or "").strip()
        definition = str(skill.get("definition") or "").strip()
        parts = [f"## {name}"]
        if description:
            parts.append(description)
        if definition:
            parts.append(definition)
        blocks.append("\n\n".join(parts))
    if len(blocks) == 1:
        return None
    return "\n\n".join(blocks)
