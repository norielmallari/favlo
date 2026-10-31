"""Heartbeat tools. One switch lets an agent store its own schedules."""

from __future__ import annotations

import uuid
from typing import Any

from web_api.tool_names import qualified

GROUP_ID = "heartbeats"
GROUP_NAME = "Heartbeat tools"
SOURCE = "heartbeat"

INSTRUCTIONS = (
    "When you should check something on a schedule, use the heartbeat tools. "
    f"Call {qualified(SOURCE, 'list_heartbeats')} to see the schedules already stored for you. "
    f"Call {qualified(SOURCE, 'add_heartbeat')} with a five-field cron expression, a timezone, and the "
    f"instructions for that run. Call {qualified(SOURCE, 'update_heartbeat')} to change one of yours. "
    f"Call {qualified(SOURCE, 'delete_heartbeat')} to remove one of yours. The schedule runs when that minute arrives."
)


def tools(store: Any, agent_id: str) -> list[object]:
    return [
        _list_tool(store, agent_id),
        _add_tool(store, agent_id),
        _update_tool(store, agent_id),
        _delete_tool(store, agent_id),
    ]


def list_heartbeats(store: Any, agent_id: str) -> str:
    rows = [row for row in store.heartbeats() if row["agent_id"] == agent_id]
    if not rows:
        return "No heartbeats are stored."
    lines = []
    for row in rows:
        state = "enabled" if row["enabled"] else "disabled"
        lines.append(
            f"{row['id']} {row['cron_expression']} {row['timezone']} {state}: {row['instructions']}"
        )
    return "\n".join(lines)


def add_heartbeat(
    store: Any,
    agent_id: str,
    cron_expression: str,
    instructions: str,
    timezone: str,
) -> str:
    text = instructions.strip()
    if not text:
        return "Instructions are required."
    zone = timezone.strip() or "UTC"
    from web_api.store import NotFound, Rejected

    try:
        _status, created, _events = store.create_heartbeat(
            str(uuid.uuid4()),
            {
                "agent_id": agent_id,
                "cron_expression": cron_expression,
                "timezone": zone,
                "instructions": text,
                "enabled": True,
            },
        )
    except (Rejected, NotFound) as exc:
        return str(exc)
    return (
        f"Heartbeat {created['id']} runs {created['cron_expression']} "
        f"in {created['timezone']}."
    )


def update_heartbeat(
    store: Any,
    agent_id: str,
    heartbeat_id: str,
    cron_expression: str | None = None,
    timezone: str | None = None,
    instructions: str | None = None,
    enabled: bool | None = None,
) -> str:
    row = _owned(store, agent_id, heartbeat_id)
    if row is None:
        return "That heartbeat does not exist."
    if cron_expression is None and timezone is None and instructions is None and enabled is None:
        return "Name a cron expression, a timezone, instructions, or whether it is enabled."
    text = row["instructions"]
    if instructions is not None:
        text = instructions.strip()
        if not text:
            return "Instructions are required."
    zone = row["timezone"] if timezone is None else (timezone.strip() or "UTC")
    cron = row["cron_expression"] if cron_expression is None else cron_expression
    flag = row["enabled"] if enabled is None else enabled
    from web_api.store import Conflict, NotFound, Rejected

    try:
        updated, _events = store.update_heartbeat(
            heartbeat_id,
            {
                "expected_revision": row["revision"],
                "cron_expression": cron,
                "timezone": zone,
                "instructions": text,
                "enabled": flag,
            },
        )
    except (Rejected, NotFound, Conflict) as exc:
        return str(exc)
    state = "enabled" if updated["enabled"] else "disabled"
    return (
        f"Heartbeat {updated['id']} runs {updated['cron_expression']} "
        f"in {updated['timezone']}, and it is {state}."
    )


def delete_heartbeat(store: Any, agent_id: str, heartbeat_id: str) -> str:
    from web_api.store import NotFound

    try:
        store.delete_heartbeat(heartbeat_id, agent_id)
    except NotFound as exc:
        return str(exc)
    return f"Heartbeat {heartbeat_id} was deleted."


def _owned(store: Any, agent_id: str, heartbeat_id: str) -> dict[str, Any] | None:
    for row in store.heartbeats():
        if row["id"] == heartbeat_id and row["agent_id"] == agent_id:
            return row
    return None


def _list_tool(store: Any, agent_id: str) -> object:
    from agent_framework import tool

    @tool(approval_mode="never_require", name=qualified(SOURCE, "list_heartbeats"))
    def list_heartbeats_tool() -> str:
        """List the heartbeats stored for you."""
        return list_heartbeats(store, agent_id)

    return list_heartbeats_tool


def _add_tool(store: Any, agent_id: str) -> object:
    from typing import Annotated

    from agent_framework import tool
    from pydantic import Field

    @tool(approval_mode="never_require", name=qualified(SOURCE, "add_heartbeat"))
    def add_heartbeat_tool(
        cron_expression: Annotated[
            str,
            Field(description="Five cron fields: minute hour day month weekday."),
        ],
        instructions: Annotated[str, Field(description="What to do when this heartbeat runs.")],
        timezone: Annotated[
            str,
            Field(description="IANA timezone. UTC when omitted."),
        ] = "UTC",
    ) -> str:
        """Store one heartbeat for yourself.

        The cron expression has five fields. The smallest interval is one minute.
        """
        return add_heartbeat(store, agent_id, cron_expression, instructions, timezone)

    return add_heartbeat_tool


def _update_tool(store: Any, agent_id: str) -> object:
    from typing import Annotated

    from agent_framework import tool
    from pydantic import Field

    @tool(approval_mode="never_require", name=qualified(SOURCE, "update_heartbeat"))
    def update_heartbeat_tool(
        heartbeat_id: Annotated[str, Field(description="The heartbeat id from list_heartbeats.")],
        cron_expression: Annotated[
            str | None,
            Field(description="Five cron fields. Omit to keep the current expression."),
        ] = None,
        timezone: Annotated[
            str | None,
            Field(description="IANA timezone. Omit to keep the current timezone."),
        ] = None,
        instructions: Annotated[
            str | None,
            Field(description="What to do when this heartbeat runs. Omit to keep them."),
        ] = None,
        enabled: Annotated[
            bool | None,
            Field(description="True enables the schedule. False disables it. Omit to leave it."),
        ] = None,
    ) -> str:
        """Change one of your heartbeats.

        Pass the id and only the fields that should change.
        """
        return update_heartbeat(
            store,
            agent_id,
            heartbeat_id,
            cron_expression,
            timezone,
            instructions,
            enabled,
        )

    return update_heartbeat_tool


def _delete_tool(store: Any, agent_id: str) -> object:
    from typing import Annotated

    from agent_framework import tool
    from pydantic import Field

    @tool(approval_mode="never_require", name=qualified(SOURCE, "delete_heartbeat"))
    def delete_heartbeat_tool(
        heartbeat_id: Annotated[str, Field(description="The heartbeat id from list_heartbeats.")],
    ) -> str:
        """Remove one of your heartbeats."""
        return delete_heartbeat(store, agent_id, heartbeat_id)

    return delete_heartbeat_tool
