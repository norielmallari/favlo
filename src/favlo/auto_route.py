"""Public model ``auto``. The listener picks the concrete row.

``~/.favlo/settings.json`` maps ``default`` and ``coding`` to registry
aliases. ``auto.tools`` maps a task to the tool names that select it. The
latest listed call selects that task. Any other call, and a turn with no
tool call, uses ``default``. One open follows, on that alias.
"""

from __future__ import annotations

from typing import Any

from favlo.registry import resolve

PUBLIC_ID = "auto"
TASKS = ("default", "coding")


class AutoRouteError(LookupError):
    """``auto`` has no usable settings."""


def is_auto(body: dict[str, Any]) -> bool:
    """True when this request asks for the public auto id."""
    model = body.get("model")
    return isinstance(model, str) and model.strip() == PUBLIC_ID


def auto_map(settings: dict[str, Any] | None = None) -> dict[str, str] | None:
    """Task name to registry alias. ``default`` is required. A missing map is none."""
    raw = _auto(settings)
    if raw is None:
        return None
    found: dict[str, str] = {}
    for name in TASKS:
        value = raw.get(name)
        if isinstance(value, str) and value.strip():
            found[name] = value.strip()
    if "default" not in found:
        return None
    return found


def tool_tasks(settings: dict[str, Any] | None = None) -> dict[str, str]:
    """Tool name to the task that lists it. A missing table is empty."""
    raw = _auto(settings)
    listed = raw.get("tools") if isinstance(raw, dict) else None
    if not isinstance(listed, dict):
        return {}
    found: dict[str, str] = {}
    for task in TASKS:
        names = listed.get(task)
        if not isinstance(names, list):
            continue
        for name in names:
            if isinstance(name, str) and name.strip():
                found[name.strip()] = task
    return found


def request_label(concrete: str) -> str:
    """Log owner for a turn the client addressed as ``auto``."""
    if concrete:
        return f"{PUBLIC_ID}:{concrete}"
    return PUBLIC_ID


def prepare(body: dict[str, Any], state: dict[str, Any] | None = None) -> dict[str, Any]:
    """The pipeline body. The model is the task the current tool calls select."""
    mapping = auto_map()
    if mapping is None:
        raise AutoRouteError("auto is not configured")
    task = select_task(body, state, tool_tasks())
    if task not in mapping:
        task = "default"
    alias = mapping[task]
    routed = dict(body)
    routed["model"] = alias
    _log(f"auto {task} {alias}")
    return routed


def select_task(
    body: dict[str, Any],
    state: dict[str, Any] | None = None,
    tools: dict[str, str] | None = None,
) -> str:
    """The latest listed call after the last user message. No call is ``default``."""
    if tools is None:
        tools = tool_tasks()
    pending: list[str] = []
    for item in _history(body, state):
        if _is_user(item):
            pending = []
            continue
        name = _call_name(item)
        if name:
            pending.append(name)
    if not pending:
        return "default"
    return tools.get(pending[-1], "default")


def auto_row(created: int | None = None, settings: dict[str, Any] | None = None) -> dict[str, Any]:
    """The synthetic ``GET /v1/models`` row. It is not a registry folder."""
    if created is None:
        from favlo.registry import _created

        created = _created(None)
    row: dict[str, Any] = {
        "id": PUBLIC_ID,
        "object": "model",
        "created": created,
        "owned_by": "favlo",
        "pipeline": PUBLIC_ID,
        "loaded": False,
        "context_length": 0,
        "mtp": False,
        "mtp_depth": 0,
    }
    mapping = auto_map(settings)
    alias = mapping.get("default") if mapping else None
    if not isinstance(alias, str):
        return row
    try:
        selection = resolve(alias)
    except LookupError:
        return row
    row["context_length"] = selection.context_length
    return row


def _auto(settings: dict[str, Any] | None) -> dict[str, Any] | None:
    if settings is None:
        from favlo.logs import favlo_home, load_settings

        settings = load_settings(favlo_home())
    raw = settings.get("auto")
    return raw if isinstance(raw, dict) else None


def _history(body: dict[str, Any], state: dict[str, Any] | None) -> list[Any]:
    items: list[Any] = []
    previous = body.get("previous_response_id")
    if isinstance(previous, str) and previous and isinstance(state, dict):
        inputs = state.get("response_inputs")
        if isinstance(inputs, dict) and isinstance(inputs.get(previous), list):
            items.extend(inputs[previous])
        store = state.get("responses")
        found = store.get(previous) if isinstance(store, dict) else None
        if isinstance(found, dict) and isinstance(found.get("output"), list):
            items.extend(found["output"])
    fresh = body.get("input")
    if isinstance(fresh, list):
        items.extend(fresh)
    return items


def _is_user(item: Any) -> bool:
    if not isinstance(item, dict):
        return False
    if item.get("type") == "function_call_output":
        return False
    if item.get("role") == "user":
        return True
    return item.get("type") == "message" and item.get("role") in (None, "user")


def _call_name(item: Any) -> str:
    if not isinstance(item, dict) or item.get("type") != "function_call":
        return ""
    name = item.get("name")
    if isinstance(name, str) and name:
        return name
    function = item.get("function")
    if isinstance(function, dict):
        nested = function.get("name")
        if isinstance(nested, str) and nested:
            return nested
    return ""


def _log(message: str) -> None:
    from favlo.listener import _log as write

    write(message)
