"""Channel sessions and the user message that starts a run."""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, Request

from web_api.http import json_body, key
from web_api.ids import guid
from web_api.runs import perform
from web_api.schemas import ArchiveRequest, MessageCreate

router = APIRouter()


@router.get("/sessions/{session_id}")
async def snapshot(session_id: str, request: Request):
    body = await asyncio.to_thread(request.app.state.store.session_snapshot, session_id)
    return json_body(200, body)


@router.post("/sessions/{session_id}/clear")
async def clear(session_id: str, request: Request):
    result, events = await asyncio.to_thread(request.app.state.store.clear_chat, session_id)
    bus = request.app.state.bus
    for event in events:
        await bus.publish(event)
    for run_id in result.get("cancelled_run_ids") or []:
        pending = request.app.state.cancels.pop(run_id, None)
        if pending is not None:
            pending.set()
    return json_body(200, result)


@router.post("/sessions/{session_id}/cancel")
async def cancel(session_id: str, request: Request):
    """Stop every running MAF turn on this chat. Safe when none are running."""
    run_ids = await asyncio.to_thread(request.app.state.store.running_run_ids, session_id)
    bus = request.app.state.bus
    for run_id in run_ids:
        pending = request.app.state.cancels.pop(run_id, None)
        if pending is not None:
            pending.set()
        for event in await asyncio.to_thread(
            request.app.state.store.finish_run, run_id, "cancelled", None
        ):
            await bus.publish(event)
    return json_body(200, {"status": "cancelled", "cancelled_run_ids": run_ids})


@router.post("/sessions/{session_id}/archive")
async def archive(session_id: str, body: ArchiveRequest, request: Request):
    run_id = body.run_id.strip() or guid()
    result = await asyncio.to_thread(request.app.state.store.arm_archive, session_id, run_id)
    return json_body(202, result)


@router.post("/sessions/{session_id}/messages")
async def send(session_id: str, body: MessageCreate, request: Request):
    status, result, events = await asyncio.to_thread(
        request.app.state.store.begin_message,
        key(request),
        session_id,
        body.text,
    )
    bus = request.app.state.bus
    for event in events:
        await bus.publish(event)
    if result.get("run_id"):
        snap = await asyncio.to_thread(request.app.state.store.session_snapshot, session_id)
        workspace_id = snap["session"].get("workspace_id")
        cancel = asyncio.Event()
        request.app.state.cancels[result["run_id"]] = cancel

        async def _run_then_index() -> None:
            try:
                await perform(request.app, cancel, result["run_id"])
            finally:
                if workspace_id:
                    from web_api.gallery_embed import request_wake

                    request_wake(request.app, workspace_id, vision=False)

        task = asyncio.create_task(_run_then_index())
        request.app.state.tasks.add(task)
        task.add_done_callback(request.app.state.tasks.discard)
    for run_id in result.get("cancelled_run_ids") or []:
        pending = request.app.state.cancels.pop(run_id, None)
        if pending is not None:
            pending.set()
    return json_body(status, result)
