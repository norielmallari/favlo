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
    "Build ledger.py in this workspace. Use every tool, in this order. Then stop.\n"
    "\n"
    "The workspace already contains notes/brief.txt.\n"
    "\n"
    "1. Call glob with pattern **/*.txt and path notes.\n"
    "2. Call grep_search with pattern collect, path notes/brief.txt, and literal true.\n"
    "3. Call read_file with file_path notes/brief.txt.\n"
    "4. Call write_file with file_path ledger.py, idempotency_key create-ledger,\n"
    "and content that is exactly this text, including the final newline,\n"
    "with no markdown fence:\n"
    f"{LEDGER_DRAFT}"
    "5. Call edit with file_path ledger.py and idempotency_key fix-ledger.\n"
    "old_string is exactly this line, including the leading spaces:\n"
    f"{LEDGER_RETURN_OLD}\n"
    "new_string is exactly this line, including the leading spaces:\n"
    f"{LEDGER_RETURN_NEW}\n"
    "6. Call run_shell_command with directory . and idempotency_key check-ledger.\n"
    "command is exactly:\n"
    f"{SHELL_CHECK}\n"
    "7. Call run_shell_command with action status and that job_id\n"
    "until status is succeeded and the output contains 6.\n"
    "\n"
    "After the job succeeds, stop. Do not call another tool.\n"
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
