"""Per-action argument checks. Unknown keys are rejected."""

from __future__ import annotations

import math
from typing import Any

from harness.coding.tools.errors import ToolError


def reject_unknown(arguments: dict[str, Any], allowed: set[str]) -> None:
    extra = sorted(set(arguments) - allowed)
    if extra:
        raise ToolError("invalid_arguments", f"unexpected argument {extra[0]}")


def require_str(arguments: dict[str, Any], key: str) -> str:
    value = arguments.get(key)
    if not isinstance(value, str) or value == "":
        raise ToolError("invalid_arguments", f"{key} is required")
    return value


def optional_str(arguments: dict[str, Any], key: str) -> str | None:
    if key not in arguments or arguments[key] is None:
        return None
    value = arguments[key]
    if not isinstance(value, str):
        raise ToolError("invalid_arguments", f"{key} must be a string")
    return value


def require_text(arguments: dict[str, Any], key: str, *, allow_empty: bool) -> str:
    if key not in arguments or arguments[key] is None:
        raise ToolError("invalid_arguments", f"{key} is required")
    value = arguments[key]
    if not isinstance(value, str):
        raise ToolError("invalid_arguments", f"{key} must be a string")
    if value == "" and not allow_empty:
        raise ToolError("invalid_arguments", f"{key} must not be empty")
    return value


def optional_bool(arguments: dict[str, Any], key: str, *, default: bool) -> bool:
    if key not in arguments or arguments[key] is None:
        return default
    value = arguments[key]
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        text = value.strip().lower()
        if text == "true":
            return True
        if text == "false":
            return False
    raise ToolError("invalid_arguments", f"{key} must be a boolean")


def optional_int(arguments: dict[str, Any], key: str) -> int | None:
    if key not in arguments or arguments[key] is None:
        return None
    value = arguments[key]
    if isinstance(value, bool):
        raise ToolError("invalid_arguments", f"{key} must be an integer")
    if isinstance(value, int):
        return value
    if isinstance(value, str) and _integer_text(value):
        return int(value.strip())
    raise ToolError("invalid_arguments", f"{key} must be an integer")


def optional_number(arguments: dict[str, Any], key: str) -> float | None:
    if key not in arguments or arguments[key] is None:
        return None
    value = arguments[key]
    if isinstance(value, bool):
        raise ToolError("invalid_arguments", f"{key} must be a number")
    if isinstance(value, (int, float)):
        number = float(value)
    elif isinstance(value, str):
        try:
            number = float(value.strip())
        except ValueError:
            raise ToolError("invalid_arguments", f"{key} must be a number") from None
    else:
        raise ToolError("invalid_arguments", f"{key} must be a number")
    if not math.isfinite(number):
        raise ToolError("invalid_arguments", f"{key} must be a number")
    return number


def _integer_text(value: str) -> bool:
    text = value.strip()
    if text.startswith("-"):
        text = text[1:]
    return text.isdigit()


def bounded_limit(value: int | None, *, default: int, maximum: int) -> int:
    if value is None:
        return default
    if value < 1:
        raise ToolError("invalid_arguments", "limit must be at least 1")
    return min(value, maximum)


def cursor_offset(arguments: dict[str, Any]) -> int:
    if "cursor" not in arguments or arguments["cursor"] is None:
        return 0
    value = arguments["cursor"]
    if isinstance(value, bool):
        raise ToolError("invalid_arguments", "cursor must be a decimal string")
    if isinstance(value, int):
        text = str(value)
    elif isinstance(value, str):
        text = value
    else:
        raise ToolError("invalid_arguments", "cursor must be a decimal string")
    if text == "" or not text.isdigit():
        raise ToolError("invalid_arguments", "cursor must be a decimal string")
    return int(text)
