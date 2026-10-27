"""Application settings. Theme stays in the browser."""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, Request

from web_api.http import json_body
from web_api.schemas import SettingsUpdate

router = APIRouter()


@router.get("/settings")
async def read_settings(request: Request):
    body = await asyncio.to_thread(request.app.state.store.settings)
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
