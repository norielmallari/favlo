"""Values a tool call needs. Tool modules do not import the session."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from harness.coding.tools.idempotency import IdempotencyStore
from harness.coding.tools.mutations import MutationCoordinator


class ShellControl(Protocol):
    def perform(self, arguments: dict[str, Any], ctx: ToolContext) -> str: ...


@dataclass
class ToolContext:
    workspace: Path
    state_dir: Path
    session_id: str
    run_id: str
    tool_call_id: str
    idempotency: IdempotencyStore
    mutations: MutationCoordinator
    shells: ShellControl
