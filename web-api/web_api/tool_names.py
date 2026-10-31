"""Prefix a tool with its source so names stay unique on one turn."""

from __future__ import annotations


def qualified(source: str, name: str) -> str:
    stem = slug(source)
    if name.startswith(f"{stem}_"):
        return name
    return f"{stem}_{name}"


def unique(source: str, name: str, taken: set[str]) -> str:
    candidate = qualified(source, name)
    if candidate not in taken:
        taken.add(candidate)
        return candidate
    extra = 2
    while True:
        candidate = qualified(f"{source}_{extra}", name)
        if candidate not in taken:
            taken.add(candidate)
            return candidate
        extra += 1


def slug(source: str) -> str:
    cleaned = "".join(char.lower() if char.isalnum() else "_" for char in source.strip())
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned.strip("_") or "tool"
