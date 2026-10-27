"""Actionable tool failures. The payload is the text the model sees."""

from __future__ import annotations

import json
from typing import Any


class ToolError(Exception):
    """A tool call that did not run. ``error`` is the stable code."""

    def __init__(self, error: str, message: str) -> None:
        super().__init__(message)
        self.error = error
        self.message = message

    def payload(self) -> str:
        return dump({"error": self.error, "message": self.message})


def dump(body: dict[str, Any]) -> str:
    """Stable JSON for a tool result."""
    return json.dumps(body, ensure_ascii=False, sort_keys=True)
