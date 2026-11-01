"""Files the model may read and write. Paths stay inside one directory."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from harness.coding.session import missing_tools, session_for
from harness.coding.tools.schemas import TOOLS

BRIEF = "collect(before, after) returns after minus before.\n"
LEDGER_DRAFT = "def collect(before, after):\n    return before - after\n"
LEDGER_RETURN_OLD = "    return before - after"
LEDGER_RETURN_NEW = "    return after - before"
SHELL_CHECK = 'python3 -c "import ledger; print(ledger.collect(15, 21))"'

PROMPT = (
    "Build ledger.py. Do one step, then wait for its result. Follow this order.\n"
    "\n"
    "Find the text files under notes.\n"
    "\n"
    "Search notes/brief.txt for the word collect.\n"
    "\n"
    "Read notes/brief.txt.\n"
    "\n"
    "Create ledger.py. The content is exactly this text, including the final newline:\n"
    f"{LEDGER_DRAFT}"
    "\n"
    "In ledger.py, replace this line, including the four leading spaces:\n"
    f"{LEDGER_RETURN_OLD}\n"
    "with this line, including the four leading spaces:\n"
    f"{LEDGER_RETURN_NEW}\n"
    "\n"
    "From the workspace root, run exactly this command:\n"
    f"{SHELL_CHECK}\n"
    "Wait until that command has succeeded and its output contains 6. Then stop.\n"
)


def prepare_workspace(workspace: Path) -> None:
    """Seed the note the coding task reads. An existing note is left as it is."""
    notes = workspace / "notes"
    notes.mkdir(parents=True, exist_ok=True)
    brief = notes / "brief.txt"
    if not brief.exists():
        brief.write_text(BRIEF, encoding="utf-8")


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


def dispatch(
    workspace: Path,
    name: str,
    arguments: dict[str, Any],
    *,
    tool_call_id: str = "call",
) -> str:
    """Run one tool and return the text the model sees."""
    return session_for(workspace).dispatch(name, arguments, tool_call_id=tool_call_id)


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


__all__ = [
    "BRIEF",
    "LEDGER_DRAFT",
    "LEDGER_RETURN_NEW",
    "LEDGER_RETURN_OLD",
    "PROMPT",
    "SHELL_CHECK",
    "TOOLS",
    "arguments_of",
    "dispatch",
    "locate",
    "missing_tools",
    "prepare_workspace",
]
