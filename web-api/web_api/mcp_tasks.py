"""Poll an MCP task and deliver the result to the calling session."""

from __future__ import annotations

import asyncio
from typing import Any

from web_api.bus import Bus
from web_api.mcp_client import McpError, render_result, rpc
from web_api.store import Store

_FAILURES = 30


async def watch(store: Store, bus: Bus, task: dict[str, Any]) -> None:
    failures = 0
    interval = max(200, int(task["poll_interval_ms"]))
    while True:
        await asyncio.sleep(interval / 1000)
        server = store.mcp_server(task["server_id"])
        if server is None:
            await _finish(
                store,
                bus,
                task,
                "failed",
                f"{task['tool_name']} failed.\nThe MCP server is gone.",
            )
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
                await _finish(
                    store,
                    bus,
                    task,
                    "failed",
                    f"{task['tool_name']} failed.\nThe task stopped responding.",
                )
                return
            continue
        failures = 0
        posted = result.get("pollIntervalMs")
        if isinstance(posted, int) and posted > 0:
            interval = posted
        status = result.get("status")
        if status == "completed":
            body = render_result(result)
            await _finish(store, bus, task, "completed", f"{task['tool_name']} finished.\n{body}")
            return
        if status == "failed":
            detail = render_result(result)
            await _finish(store, bus, task, "failed", f"{task['tool_name']} failed.\n{detail}")
            return
        if status == "cancelled":
            await _finish(store, bus, task, "cancelled", f"{task['tool_name']} was cancelled.")
            return
        if status == "input_required":
            await _finish(
                store,
                bus,
                task,
                "failed",
                f"{task['tool_name']} failed.\nThe server asked for more input.",
            )
            return


def resume(store: Store, bus: Bus, pending: set[asyncio.Task[Any]]) -> None:
    for task in store.working_mcp_tasks():
        spawn(pending, watch(store, bus, task))


def spawn(pending: set[asyncio.Task[Any]], work: Any) -> None:
    job = asyncio.create_task(work)
    pending.add(job)
    job.add_done_callback(pending.discard)


async def _finish(
    store: Store,
    bus: Bus,
    task: dict[str, Any],
    status: str,
    text: str,
) -> None:
    events = store.deliver_mcp_result(task["id"], status, text)
    for event in events:
        await bus.publish(event)
