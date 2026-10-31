"""WebSocket replay and live run deltas."""

from __future__ import annotations

import asyncio
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

router = APIRouter()

BROWSER_ORIGIN = "http://127.0.0.1:8012"


@router.websocket("/events")
async def events(socket: WebSocket) -> None:
    if socket.headers.get("origin") != BROWSER_ORIGIN:
        await socket.close(code=1008)
        return
    await socket.accept()
    bus = socket.app.state.bus
    store = socket.app.state.store
    incoming: asyncio.Queue[dict[str, Any]] = bus.subscribe()
    outgoing: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
    streams: set[str] = set()
    seen: set[str] = set()

    async def pump() -> None:
        while True:
            event = await incoming.get()
            if event.get("stream_id") in streams and event.get("event_id") not in seen:
                await outgoing.put(event)

    async def send() -> None:
        while True:
            event = await outgoing.get()
            await socket.send_json(event)

    pumping = asyncio.create_task(pump())
    sending = asyncio.create_task(send())
    try:
        while True:
            message = await socket.receive_json()
            action = message.get("action")
            stream_id = message.get("stream_id")
            if not isinstance(stream_id, str) or not stream_id:
                continue
            if action == "unsubscribe":
                streams.discard(stream_id)
                continue
            if action != "subscribe":
                continue
            cursor = message.get("cursor")
            after = cursor if isinstance(cursor, int) and cursor >= 0 else 0
            streams.add(stream_id)
            for event in store.events_after(stream_id, after):
                seen.add(event["event_id"])
                await outgoing.put(event)
    except WebSocketDisconnect:
        return
    finally:
        pumping.cancel()
        sending.cancel()
        bus.unsubscribe(incoming)
