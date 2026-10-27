"""Application settings. Theme stays in the browser."""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, Request

from favlo.logs import favlo_home, heartbeat_seconds, load_settings, set_heartbeat_seconds
from web_api.http import json_body
from web_api.schemas import HeartbeatInterval, SettingsUpdate

router = APIRouter()


@router.get("/settings")
async def read_settings(request: Request):
    body = await asyncio.to_thread(request.app.state.store.settings)
    seconds = await asyncio.to_thread(heartbeat_seconds, load_settings(favlo_home()))
    body["heartbeat_minutes"] = seconds // 60
    return json_body(200, body)


@router.patch("/settings")
async def update_settings(body: SettingsUpdate, request: Request):
    updated = await asyncio.to_thread(
        request.app.state.store.update_settings,
        body.expected_revision,
        body.default_model_id,
        body.exchange_budget,
    )
    return json_body(200, updated)


@router.put("/server-heartbeat")
async def write_heartbeat(body: HeartbeatInterval):
    seconds = await asyncio.to_thread(set_heartbeat_seconds, favlo_home(), body.minutes * 60)
    return json_body(200, {"minutes": seconds // 60})
