"""Attach an agent's enabled MCP tools to one turn."""

from __future__ import annotations

import asyncio
from typing import Any

from web_api.mcp_client import McpError, render_result, rpc, task_handle
from web_api.mcp_groups import LIST_BEFORE, ORDER, group_for
from web_api.mcp_tasks import spawn, watch

INSTRUCTIONS = (
    "Enabled MCP tools are available on every turn. "
    "Before generate_image, call list_image_models and pass a model id that list returned. "
    "Before generate_video_i2v, call list_video_models_i2v and pass a model id that list returned. "
    "Before generate_video_t2v, call list_video_models_t2v and pass a model id that list returned. "
    "Before generate_music, call list_music_models and pass a model id that list returned. "
    "Before generate_voice, call list_voice_models and pass a model id that list returned. "
    "Do not invent a model id, and do not reuse a model id from an earlier message. "
    "When you call a generate tool, tell the user you are generating that image, video, music, or voice. "
    "Do not write a favlo:// link or an image id, and do not say the file is already ready. "
    "A media tool may return a task id instead of the file. "
    "The finished result arrives in this session. "
    "Do not call that tool again for the same request while the task is working. "
    "When a task result says the tool failed or names an unknown model, "
    "call the matching list tool and call the generate tool again with an id from that list."
)


def bundle(
    store: Any,
    agent_id: str,
    session_id: str,
    bus: Any,
    pending: set[asyncio.Task[Any]],
) -> tuple[str, list[object]]:
    rows = store.enabled_mcp_tools(agent_id)
    if not rows:
        return "", []
    return INSTRUCTIONS, [_tool(store, row, session_id, agent_id, bus, pending) for row in rows]


def prepare_catalog(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    prepared = []
    for tool in tools:
        group_id, group_name = group_for(tool["name"])
        prepared.append({**tool, "group_id": group_id, "group_name": group_name})
    return prepared


def groups_from(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[str, dict[str, Any]] = {}
    for tool in tools:
        group_id = str(tool["group_id"])
        bucket = buckets.setdefault(
            group_id,
            {"id": group_id, "name": tool["group_name"], "enabled": True, "tools": []},
        )
        bucket["tools"].append(tool["name"])
    rank = {group_id: index for index, group_id in enumerate(ORDER)}
    ordered = sorted(buckets.values(), key=lambda item: rank.get(str(item["id"]), len(rank)))
    for bucket in ordered:
        bucket["tools"] = sorted(bucket["tools"])
    return ordered


async def call_tool(
    store: Any,
    row: dict[str, Any],
    arguments: dict[str, Any],
    session_id: str,
    agent_id: str,
    bus: Any,
    pending: set[asyncio.Task[Any]],
) -> str:
    server = {
        "id": row["server_id"],
        "transport": row["transport"],
        "command": row["command"],
        "url": row["url"],
    }
    try:
        result = await rpc(
            server,
            "tools/call",
            {"name": row["name"], "arguments": arguments},
            name=row["name"],
            timeout=60.0,
        )
    except McpError as exc:
        return str(exc)
    handle = task_handle(result)
    if handle is None:
        return render_result(result)
    remote_id, interval = handle
    task = store.begin_mcp_task(
        {
            "remote_id": remote_id,
            "server_id": row["server_id"],
            "agent_id": agent_id,
            "session_id": session_id,
            "tool_name": row["name"],
            "poll_interval_ms": interval,
        }
    )
    spawn(pending, watch(store, bus, task))
    return f"Task {remote_id} is working. The result will arrive in this session."


def _tool(
    store: Any,
    row: dict[str, Any],
    session_id: str,
    agent_id: str,
    bus: Any,
    pending: set[asyncio.Task[Any]],
) -> object:
    from agent_framework import tool

    schema = row["schema"] if isinstance(row.get("schema"), dict) else {"type": "object"}

    async def invoke(**arguments: Any) -> str:
        return await call_tool(store, row, arguments, session_id, agent_id, bus, pending)

    return tool(
        invoke,
        name=row["name"],
        description=_description(row),
        schema=schema,
        approval_mode="never_require",
    )


def _description(row: dict[str, Any]) -> str:
    text = str(row.get("description") or row["name"])
    listed = LIST_BEFORE.get(str(row.get("name") or ""))
    if listed is None:
        return text
    note = f"Call {listed} first and pass a model id that {listed} returned."
    if note in text:
        return text
    return f"{text} {note}"
