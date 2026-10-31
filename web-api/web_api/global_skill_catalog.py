"""Global skills shipped with the app. Each directory is one Agent Skill."""

from __future__ import annotations

from pathlib import Path


def catalog_skills() -> list[tuple[str, str, str]]:
    """Name, description, and definition from each skills-global/*/SKILL.md."""
    root = Path(__file__).resolve().parents[2] / "skills-global"
    if not root.is_dir():
        return []
    skills: list[tuple[str, str, str]] = []
    for path in sorted(root.glob("*/SKILL.md")):
        name, description, definition = read_skill_file(path)
        if name and description:
            skills.append((name, description, definition))
    return skills


def read_skill_file(path: Path) -> tuple[str, str, str]:
    text = path.read_text(encoding="utf-8").strip()
    description = ""
    name = path.parent.name
    body = text
    if text.startswith("---"):
        rest = text[3:]
        marker = rest.find("\n---")
        if marker >= 0:
            front = rest[:marker]
            body = rest[marker + 4 :].strip()
            for line in front.splitlines():
                if ":" not in line:
                    continue
                key, value = line.split(":", 1)
                value = value.strip().strip("'").strip('"')
                if key.strip() == "name" and value:
                    name = value
                elif key.strip() == "description":
                    description = value
    if not description:
        description = body
        body = ""
    return name, description, body
