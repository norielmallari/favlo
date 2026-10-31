"""Fixed system text for a workspace session, by type."""

from __future__ import annotations

from typing import Any

INSTRUCTIONS = {
    "gallery": (
        "You are operating in a gallery workspace. "
        "This conversation is that workspace, not the home channel."
    ),
    "coding": (
        "You are operating in a coding workspace. "
        "This conversation is that workspace, not the home channel."
    ),
}


def text_for(kind: str) -> str | None:
    """The standing prompt for that workspace type, or none when the type is unknown."""
    text = INSTRUCTIONS.get(kind)
    return text if text else None


def skills_text(skills: list[dict[str, Any]]) -> str | None:
    """Markdown block for stored skills, or none when the list is empty."""
    blocks = ["# Skills"]
    for skill in skills:
        name = str(skill.get("name") or "").strip()
        if not name:
            continue
        description = str(skill.get("description") or "").strip()
        block = f"## {name}"
        if description:
            block = f"{block}\n\n{description}"
        blocks.append(block)
    if len(blocks) == 1:
        return None
    return "\n\n".join(blocks)
