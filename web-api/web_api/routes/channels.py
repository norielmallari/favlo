"""Agent Channels."""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, Request

from web_api.http import json_body

router = APIRouter()


@router.get("/channels")
async def list_channels(request: Request):
    rows = await asyncio.to_thread(request.app.state.store.channels)
    return json_body(200, {"channels": rows})
