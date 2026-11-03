"""The coding read, bound for this agent, with each result kept for the grade."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from agent_framework import FunctionTool

from harness.coding.tools.context import ToolContext
from harness.coding.tools.errors import ToolError
from harness.coding.tools.idempotency import IdempotencyStore
from harness.coding.tools.mutations import MutationCoordinator
from harness.coding.tools.read_file import perform
from harness.coding.tools.schemas import TOOLS


class _NoShell:
    def perform(self, arguments: dict[str, Any], ctx: ToolContext) -> str:
        del arguments, ctx
        message = "this agent has no shell"
        raise ToolError("execution_failed", message)


class _TextIntegers(FunctionTool):
    """Accept an offset or a limit that arrives as text."""

    def _prepare_arguments(self, arguments: Any) -> dict[str, Any]:
        mapped = self._arguments_as_mapping(arguments)
        if isinstance(mapped, dict):
            arguments = _coerce_integers(mapped, self.parameters())
        return super()._prepare_arguments(arguments)


def read_tool(workspace: Path, reads: list[dict[str, Any]]) -> FunctionTool:
    """One ``read_file``. Successful bodies are appended to ``reads``."""
    spec = next(item["function"] for item in TOOLS if item["function"]["name"] == "read_file")
    state = workspace / ".favlo"
    state.mkdir(parents=True, exist_ok=True)
    ctx = ToolContext(
        workspace=workspace,
        state_dir=state,
        session_id="needle",
        run_id="needle",
        tool_call_id="needle",
        idempotency=IdempotencyStore(state / "idempotency"),
        mutations=MutationCoordinator(),
        shells=_NoShell(),
    )

    def read_file(**arguments: Any) -> str:
        cleaned = {
            key: value
            for key, value in arguments.items()
            if value is not None and key != "limit"
        }
        try:
            body = perform(ctx, cleaned)
        except ToolError as exc:
            return exc.payload()
        parsed = json.loads(body)
        if isinstance(parsed, dict):
            reads.append(parsed)
        return body

    read_file.__name__ = "read_file"
    read_file.__doc__ = str(spec["description"])
    return _TextIntegers(
        name="read_file",
        description=str(spec["description"]),
        func=read_file,
        input_model=spec["parameters"],
        approval_mode="never_require",
    )


def _coerce_integers(arguments: dict[str, Any], schema: dict[str, Any]) -> dict[str, Any]:
    properties = schema.get("properties")
    if not isinstance(properties, dict):
        return arguments
    coerced = dict(arguments)
    for key, value in arguments.items():
        spec = properties.get(key)
        if not isinstance(value, str) or not isinstance(spec, dict):
            continue
        if spec.get("type") != "integer":
            continue
        text = value.strip()
        digits = text[1:] if text.startswith("-") else text
        if digits.isdigit() and digits != "":
            coerced[key] = int(text)
        elif _finite_number(text):
            coerced[key] = int(float(text))
    return coerced


def _finite_number(text: str) -> bool:
    try:
        number = float(text)
    except ValueError:
        return False
    return math.isfinite(number) and number == int(number)
