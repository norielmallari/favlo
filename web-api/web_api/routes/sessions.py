"""Channel sessions and the user message that starts a run."""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, Request

from web_api.http import json_body, key
from web_api.runs import perform
from web_api.schemas import MessageCreate

router = APIRouter()


@router.get("/sessions/{session_id}")
async def snapshot(session_id: str, request: Request):
    body = await asyncio.to_thread(request.app.state.store.session_snapshot, session_id)
    return json_body(200, body)


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
    if events:
        cancel = asyncio.Event()
        request.app.state.cancels[result["run_id"]] = cancel
        task = asyncio.create_task(
            perform(
                request.app.state.store,
                bus,
                cancel,
                result["run_id"],
                request.app.state.reply,
                request.app.state.tasks,
            )
        )
        request.app.state.tasks.add(task)
        task.add_done_callback(request.app.state.tasks.discard)
    return json_body(status, result)
