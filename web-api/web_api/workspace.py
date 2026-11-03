"""Fixed system text for a workspace session, by type."""

from __future__ import annotations

from typing import Any

INSTRUCTIONS = {
    "gallery": (
        "You are operating in a gallery workspace. "
        "This conversation is that gallery's workspace session, not the home channel. "
        "The user sees this gallery's shell (preview and filmstrip) beside the chat. "
        "Stay in this gallery: use its local folder and gallery tools, "
        "place the filmstrip with workspace_place, and do not treat the turn as a home-channel chat."
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
