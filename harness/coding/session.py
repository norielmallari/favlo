"""One coding session: identity, idempotency, edits, and shell jobs."""

from __future__ import annotations

import uuid
from pathlib import Path

from harness.coding.tools.catalog import perform
from harness.coding.tools.context import ToolContext
from harness.coding.tools.errors import ToolError
from harness.coding.tools.idempotency import IdempotencyStore
from harness.coding.tools.journal import Journal
from harness.coding.tools.mutations import MutationCoordinator
from harness.coding.tools.shell import ShellJobs

REQUIRED_TOOLS = (
    "read_file",
    "glob",
    "grep_search",
    "write_file",
    "edit",
    "run_shell_command",
)

_SESSIONS: dict[str, ToolSession] = {}


class ToolSession:
    def __init__(self, workspace: Path) -> None:
        self.workspace = workspace.resolve()
        self.session_id = uuid.uuid4().hex
        self.run_id = uuid.uuid4().hex
        self.state_dir = self.workspace / ".favlo"
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.journal = Journal(self.state_dir / "journal.jsonl")
        self.idempotency = IdempotencyStore(self.state_dir / "idempotency")
        self.mutations = MutationCoordinator()
        self.shells = ShellJobs(self.workspace, self.state_dir, self.mutations)
        self.calls: list[str] = []
        self._closed = False
        self.ctx = ToolContext(
            workspace=self.workspace,
            state_dir=self.state_dir,
            session_id=self.session_id,
            run_id=self.run_id,
            tool_call_id="",
            idempotency=self.idempotency,
            mutations=self.mutations,
            shells=self.shells,
        )

    def dispatch(self, name: str, arguments: dict[str, object], *, tool_call_id: str) -> str:
        self.ctx.tool_call_id = tool_call_id
        try:
            result = perform(self.ctx, name, arguments)
        except ToolError as exc:
            self.journal.record(
                session_id=self.session_id,
                run_id=self.run_id,
                tool_call_id=tool_call_id,
                tool=name,
                error=exc.error,
                message=exc.message,
            )
            return exc.payload()
        except OSError as exc:
            self.journal.record(
                session_id=self.session_id,
                run_id=self.run_id,
                tool_call_id=tool_call_id,
                tool=name,
                error="execution_failed",
            )
            message = exc.strerror or "the tool failed"
            return ToolError("execution_failed", message).payload()
        self.calls.append(name)
        self.journal.record(
            session_id=self.session_id,
            run_id=self.run_id,
            tool_call_id=tool_call_id,
            tool=name,
            error=None,
        )
        return result

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.shells.shutdown()


def session_for(workspace: Path) -> ToolSession:
    key = str(workspace.resolve())
    session = _SESSIONS.get(key)
    if session is None:
        session = ToolSession(workspace)
        _SESSIONS[key] = session
    return session


def missing_tools(calls: list[str]) -> list[str]:
    """Tool names that have not yet returned successfully, in contract order."""
    seen = set(calls)
    return [name for name in REQUIRED_TOOLS if name not in seen]
