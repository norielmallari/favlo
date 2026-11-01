"""Coding tools as Agent Framework functions.

The schemas stay the ones in ``tools/schemas.py``. The framework calls each
function and sends the JSON result back to the model. A boolean, integer, or
number that arrives as text is coerced before the schema type check. A write,
an edit, or a shell start or input that omits ``idempotency_key`` still runs.
The key is the SHA-256 of the tool name and the other arguments, so the same
call returns the recorded outcome. A call that sends a key keeps that key.
"""

from __future__ import annotations

import hashlib
import json
import math
import uuid
from typing import Any

from agent_framework import FunctionTool

from harness.coding.session import ToolSession
from harness.coding.tools.schemas import TOOLS


def coding_tools(session: ToolSession) -> list[FunctionTool]:
    """The six coding tools, bound to ``session``."""
    bound: list[FunctionTool] = []
    for spec in TOOLS:
        function = spec["function"]
        name = str(function["name"])
        description = str(function["description"])
        schema = function["parameters"]
        bound.append(_bind(session, name, description, schema))
    return bound


def _bind(
    session: ToolSession,
    name: str,
    description: str,
    schema: dict[str, Any],
) -> FunctionTool:
    def run(**arguments: Any) -> str:
        cleaned = {key: value for key, value in arguments.items() if value is not None}
        return session.dispatch(name, cleaned, tool_call_id=uuid.uuid4().hex)

    run.__name__ = name
    run.__doc__ = description
    return _TextScalars(
        name=name,
        description=description,
        func=run,
        input_model=schema,
        approval_mode="never_require",
    )


class _TextScalars(FunctionTool):
    """Accept the text forms the tool contract already allows."""

    def _prepare_arguments(self, arguments: Any) -> dict[str, Any]:
        mapped = self._arguments_as_mapping(arguments)
        if isinstance(mapped, dict):
            arguments = _supply_key(self.name, _coerce_text(mapped, self.parameters()))
        return super()._prepare_arguments(arguments)


def _coerce_text(arguments: dict[str, Any], schema: dict[str, Any]) -> dict[str, Any]:
    properties = schema.get("properties")
    if not isinstance(properties, dict):
        return arguments
    coerced = dict(arguments)
    for key, value in arguments.items():
        spec = properties.get(key)
        if not isinstance(value, str) or not isinstance(spec, dict):
            continue
        kind = spec.get("type")
        text = value.strip()
        if kind == "boolean":
            lowered = text.lower()
            if lowered == "true":
                coerced[key] = True
            elif lowered == "false":
                coerced[key] = False
        elif kind == "integer" and _integer_text(text):
            coerced[key] = int(text)
        elif kind == "number":
            try:
                number = float(text)
            except ValueError:
                continue
            if math.isfinite(number):
                coerced[key] = number
    return coerced


def _integer_text(value: str) -> bool:
    text = value[1:] if value.startswith("-") else value
    return text.isdigit() and text != ""


def _supply_key(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    if not _needs_key(name, arguments):
        return arguments
    current = arguments.get("idempotency_key")
    if isinstance(current, str) and current != "":
        return arguments
    filled = dict(arguments)
    filled["idempotency_key"] = _derived_key(name, arguments)
    return filled


def _needs_key(name: str, arguments: dict[str, Any]) -> bool:
    if name in {"write_file", "edit"}:
        return True
    if name != "run_shell_command":
        return False
    action = arguments.get("action")
    return action in {None, "start", "input"}


def _derived_key(name: str, arguments: dict[str, Any]) -> str:
    body = {key: value for key, value in arguments.items() if key != "idempotency_key"}
    raw = json.dumps(
        {"arguments": body, "tool": name},
        ensure_ascii=False,
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(raw.encode()).hexdigest()
