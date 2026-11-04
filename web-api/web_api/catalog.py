"""Closed catalog of workspace shell elements."""

from __future__ import annotations

from typing import Any

ELEMENTS: dict[str, dict[str, Any]] = {
    "preview": {
        "name": "preview",
        "version": 1,
        "slots": ["stage"],
        "max": 1,
        "agent_props": ["fit"],
        "local_props": [],
        "gestures": ["download", "add_to_chat", "delete"],
        "props": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "fit": {"type": "string", "enum": ["contain", "cover"]},
            },
        },
    },
    "filmstrip": {
        "name": "filmstrip",
        "version": 1,
        "slots": ["rail"],
        "max": 1,
        "agent_props": ["title", "items"],
        "local_props": [],
        "gestures": ["select", "ArrowLeft", "ArrowRight"],
        "props": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "title": {"type": "string", "maxLength": 80},
                "items": {
                    "type": "array",
                    "maxItems": 60,
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["path"],
                        "properties": {
                            "path": {"type": "string"},
                            "caption": {"type": "string", "maxLength": 80},
                        },
                    },
                },
            },
        },
    },
}

RESERVED = ("transcript", "file_list", "file_view", "diff", "terminal_log")


def names() -> list[str]:
    return sorted(ELEMENTS)


def get(name: str) -> dict[str, Any] | None:
    return ELEMENTS.get(name)
