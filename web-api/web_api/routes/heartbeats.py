"""Heartbeat schedules stored for an agent."""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, Request

from web_api.http import json_body, key
from web_api.runs import perform
from web_api.schemas import HeartbeatCreate, HeartbeatUpdate

router = APIRouter()


@router.get("/heartbeats")
async def list_heartbeats(request: Request):
    rows = await asyncio.to_thread(request.app.state.store.heartbeats)
    return json_body(200, {"heartbeats": rows})


@router.post("/heartbeats")
async def create_heartbeat(body: HeartbeatCreate, request: Request):
    status, result, events = await asyncio.to_thread(
        request.app.state.store.create_heartbeat,
        key(request),
        body.model_dump(),
    )
    for event in events:
        await request.app.state.bus.publish(event)
    return json_body(status, result)


@router.patch("/heartbeats/{heartbeat_id}")
async def update_heartbeat(heartbeat_id: str, body: HeartbeatUpdate, request: Request):
    updated, events = await asyncio.to_thread(
        request.app.state.store.update_heartbeat,
        heartbeat_id,
        body.model_dump(),
    )
    for event in events:
        await request.app.state.bus.publish(event)
    return json_body(200, updated)


@router.post("/heartbeats/{heartbeat_id}/run")
async def run_heartbeat(heartbeat_id: str, request: Request):
    status, result, events = await asyncio.to_thread(
        request.app.state.store.begin_heartbeat_run,
        key(request),
        heartbeat_id,
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
