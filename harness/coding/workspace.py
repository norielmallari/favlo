"""Files the model may read and write. Paths stay inside one directory."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

PROMPT = """\
Write ledger.py in this workspace.

The module defines collect(before, after).
before and after are integers.
collect returns after minus before.

Call write_file once. path is ledger.py. content is the whole module.
After that call, stop. Do not call another tool.
"""

TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Create or replace a file in the workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file in the workspace.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    },
]


def locate(workspace: Path, raw: str) -> Path:
    """The file ``raw`` names, inside ``workspace``."""
    path = Path(raw)
    if path.is_absolute() or ".." in path.parts:
        message = "path stays inside the workspace"
        raise ValueError(message)
    root = workspace.resolve()
    resolved = (root / path).resolve()
    if resolved != root and root not in resolved.parents:
        message = "path stays inside the workspace"
        raise ValueError(message)
    return resolved


def dispatch(workspace: Path, name: str, arguments: dict[str, Any]) -> str:
    """Run one tool and return the text the model sees."""
    try:
        if name == "write_file":
            path = locate(workspace, str(arguments.get("path") or ""))
            content = arguments.get("content")
            if not isinstance(content, str) or content == "":
                message = "content is required"
                raise ValueError(message)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
            return f"wrote {path.name}"
        if name == "read_file":
            path = locate(workspace, str(arguments.get("path") or ""))
            if not path.is_file():
                message = "file is missing"
                raise ValueError(message)
            return path.read_text(encoding="utf-8")
        message = f"unknown tool {name}"
        raise ValueError(message)
    except (OSError, ValueError) as exc:
        return str(exc)


def arguments_of(call: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Function name and argument object from one OpenAI tool call."""
    function = call.get("function") if isinstance(call.get("function"), dict) else {}
    name = function.get("name") if isinstance(function, dict) else ""
    raw = function.get("arguments") if isinstance(function, dict) else {}
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = {}
    else:
        parsed = raw
    if not isinstance(parsed, dict):
        parsed = {}
    return str(name or ""), parsed
