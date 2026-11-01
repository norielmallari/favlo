"""Accept an MCP task without waiting, then deliver the result on a later turn."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

from web_api.bus import Bus
from web_api.mcp_client import McpError, render_result, rpc
from web_api.mcp_groups import GENERATE
from web_api.store import Store

_FAILURES = 30
_CORRECTION_WAIT_S = 180.0
_CORRECTION_POLL_S = 0.2
_Follow = Callable[[str], Awaitable[None]]
_follow: _Follow | None = None


def bind_followup(follow: _Follow | None) -> None:
    global _follow
    _follow = follow


async def watch(store: Store, bus: Bus, task: dict[str, Any]) -> None:
    failures = 0
    interval = max(200, int(task["poll_interval_ms"]))
    while True:
        await asyncio.sleep(interval / 1000)
        server = store.mcp_server(task["server_id"])
        if server is None:
            await _finish(store, bus, task, "failed", "The MCP server is gone.")
            return
        try:
            result = await rpc(
                server,
                "tasks/get",
                {"taskId": task["remote_id"]},
                timeout=15.0,
            )
        except McpError:
            failures += 1
            if failures >= _FAILURES:
                await _finish(store, bus, task, "failed", "The task stopped responding.")
                return
            continue
        failures = 0
        posted = result.get("pollIntervalMs")
        if isinstance(posted, int) and posted > 0:
            interval = posted
        status = result.get("status")
        if status == "completed":
            body = render_result(result)
            await _finish(store, bus, task, "completed", body, error=_payload_error(result))
            return
        if status == "failed":
            await _finish(store, bus, task, "failed", render_result(result))
            return
        if status == "cancelled":
            await _finish(store, bus, task, "cancelled", "")
            return
        if status == "input_required":
            await _finish(store, bus, task, "failed", "The server asked for more input.")
            return


def resume(store: Store, bus: Bus, pending: set[asyncio.Task[Any]]) -> None:
    for task in store.working_mcp_tasks():
        spawn(pending, watch(store, bus, task))


def started(tool: str, task_id: str) -> str:
    """What the model sees when a call is accepted as a task."""
    return (
        f"{tool} is working as task {task_id}. "
        "Tell the user it has started. "
        "The result arrives in a later turn. Do not wait for the file."
    )


def accept(
    store: Store,
    bus: Bus,
    pending: set[asyncio.Task[Any]],
    holder: Any,
    *,
    server_id: str,
    agent_id: str,
    session_id: str,
    tool_name: str,
    create_result: Any,
) -> None:
    """Record a task and poll it on the same client. The tool call does not wait."""
    from fastmcp_tasks.client import ToolTask

    interval = getattr(create_result, "poll_interval_ms", None)
    milliseconds = int(interval) if isinstance(interval, int) and interval > 0 else 2000
    task = store.begin_mcp_task(
        {
            "remote_id": create_result.task_id,
            "server_id": server_id,
            "agent_id": agent_id,
            "session_id": session_id,
            "tool_name": tool_name,
            "poll_interval_ms": milliseconds,
        }
    )
    handle = ToolTask(holder.owned_client, tool_name, create_result, raise_on_error=False)
    holder.pending += 1
    holder.idle.clear()
    spawn(pending, _guard(store, bus, holder, task, handle))


async def watch_handle(store: Store, bus: Bus, task: dict[str, Any], handle: Any) -> None:
    """Poll ``tasks/get`` on the client that accepted the task."""
    failures = 0
    interval = max(200, int(task["poll_interval_ms"]))
    while True:
        await asyncio.sleep(interval / 1000)
        try:
            current = await handle.status()
        except Exception:
            failures += 1
            if failures >= _FAILURES:
                await _finish(store, bus, task, "failed", "The task stopped responding.")
                return
            continue
        failures = 0
        posted = current.poll_interval_ms
        if isinstance(posted, int) and posted > 0:
            interval = posted
        status = current.status
        if status == "completed":
            payload = {"result": current.result} if current.result is not None else {}
            await _finish(
                store,
                bus,
                task,
                "completed",
                render_result(payload),
                error=_payload_error(payload),
            )
            return
        if status == "failed":
            payload = {"error": current.error} if current.error else {}
            body = render_result(payload) if payload else "The tool failed."
            await _finish(store, bus, task, "failed", body)
            return
        if status == "cancelled":
            await _finish(store, bus, task, "cancelled", "")
            return
        if status == "input_required":
            await _finish(store, bus, task, "failed", "The server asked for more input.")
            return


async def _guard(
    store: Store,
    bus: Bus,
    holder: Any,
    task: dict[str, Any],
    handle: Any,
) -> None:
    try:
        await watch_handle(store, bus, task, handle)
    finally:
        holder.pending = max(0, holder.pending - 1)
        if holder.pending == 0:
            holder.idle.set()


def spawn(pending: set[asyncio.Task[Any]], work: Any) -> None:
    job = asyncio.create_task(work)
    pending.add(job)
    job.add_done_callback(pending.discard)


def settle(tool: str, status: str, body: str, *, error: bool = False) -> tuple[str, str]:
    if status == "cancelled":
        return "cancelled", f"{tool} was cancelled."
    if status == "completed" and not error and not _error_text(body):
        return "completed", f"{tool} finished.\n{body}"
    return "failed", f"{tool} failed.\n{body}"


def follow_cue(tool: str, status: str, text: str) -> str:
    """The prompt for the turn that answers a finished task. Not a chat message."""
    if status == "failed":
        return correction_cue(tool, text)
    return (
        f"{text}\n"
        "Reply to the user about this finished task. "
        "Include each favlo:// link so the file appears. "
        "Do not paste the JSON, the tool name, or a file id."
    )


def correction_cue(tool: str, text: str) -> str:
    pair = GENERATE.get(tool)
    if pair:
        listed, setter = pair
        return (
            f"{text}\n"
            f"Call {listed}, then {setter}. Repeat the returned session id in your reply, "
            f"then call {tool} again with that session."
        )
    return f"{text}\nCorrect the failed call."


def _error_text(body: str) -> bool:
    lowered = body.lower()
    return (
        "error calling tool" in lowered
        or "unknown model" in lowered
        or lowered.strip() == "the tool failed."
    )


def _payload_error(result: dict[str, Any]) -> bool:
    if result.get("isError") is True:
        return True
    nested = result.get("result")
    return isinstance(nested, dict) and nested.get("isError") is True


async def _finish(
    store: Store,
    bus: Bus,
    task: dict[str, Any],
    status: str,
    body: str,
    *,
    error: bool = False,
) -> None:
    tool = str(task["tool_name"])
    status, text = settle(tool, status, body, error=error)
    if not store.deliver_mcp_result(task["id"], status):
        return
    await _open_correction(
        store,
        bus,
        str(task["session_id"]),
        str(task["agent_id"]),
        follow_cue(tool, status, text),
    )


async def _open_correction(
    store: Store, bus: Bus, session_id: str, agent_id: str, cue: str
) -> None:
    deadline = asyncio.get_running_loop().time() + _CORRECTION_WAIT_S
    while True:
        run_id, events, waiting = store.begin_task_correction(session_id, agent_id, cue)
        for event in events:
            await bus.publish(event)
        if run_id:
            if _follow is not None:
                await _follow(run_id)
            return
        if not waiting or asyncio.get_running_loop().time() >= deadline:
            return
        await asyncio.sleep(_CORRECTION_POLL_S)
