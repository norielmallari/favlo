"""Providers and the models each one hosts."""

from __future__ import annotations

import asyncio

import httpx
from fastapi import APIRouter, Request

from web_api.discovery import DiscoveryFailed, fetch_model_ids
from web_api.http import json_body, key
from web_api.schemas import ModelChoice, ModelCreate, ModelUpdate
from web_api.store import NotFound

router = APIRouter()


@router.get("/providers")
async def list_providers(request: Request):
    rows = await asyncio.to_thread(request.app.state.store.providers)
    seeded: set[str] = request.app.state.seeded_providers
    for row in rows:
        if row["models"] or row["id"] in seeded:
            continue
        seeded.add(row["id"])
        try:
            ids = await fetch_model_ids(str(row["base_url"]))
        except DiscoveryFailed:
            continue
        row["models"] = await asyncio.to_thread(
            request.app.state.store.sync_models,
            row["id"],
            ids,
        )
    return json_body(200, {"providers": rows})


@router.post("/providers/{provider_id}/test")
async def test_provider(provider_id: str, request: Request):
    try:
        base_url = await asyncio.to_thread(request.app.state.store.provider_url, provider_id)
    except NotFound:
        return json_body(404, {"error": "not_found", "message": "That provider does not exist."})
    url = base_url.rstrip("/") + "/models"
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(url)
        ok = response.status_code == 200
    except httpx.HTTPError:
        ok = False
    if ok:
        return json_body(200, {"ok": True, "message": "Connection succeeded"})
    return json_body(200, {"ok": False, "message": "Connection failed"})


@router.post("/providers/{provider_id}/refresh")
async def refresh_models(provider_id: str, request: Request):
    try:
        base_url = await asyncio.to_thread(request.app.state.store.provider_url, provider_id)
    except NotFound:
        return json_body(404, {"error": "not_found", "message": "That provider does not exist."})
    try:
        ids = await fetch_model_ids(base_url)
    except DiscoveryFailed:
        current = await asyncio.to_thread(request.app.state.store.providers)
        hosted = _enabled(current, provider_id)
        return json_body(200, {"models": hosted, "message": "Connection failed"})
    models = await asyncio.to_thread(request.app.state.store.sync_models, provider_id, ids)
    return json_body(200, {"models": models, "message": "Models refreshed"})


@router.get("/providers/{provider_id}/models")
async def list_models(provider_id: str, request: Request):
    rows = await asyncio.to_thread(request.app.state.store.providers)
    if not any(row["id"] == provider_id for row in rows):
        return json_body(404, {"error": "not_found", "message": "That provider does not exist."})
    return json_body(200, {"models": [model["model_id"] for model in _enabled(rows, provider_id)]})


@router.post("/providers/{provider_id}/models")
async def choose_model(provider_id: str, body: ModelChoice, request: Request):
    try:
        await asyncio.to_thread(request.app.state.store.provider_url, provider_id)
    except NotFound:
        return json_body(404, {"error": "not_found", "message": "That provider does not exist."})
    updated = await asyncio.to_thread(
        request.app.state.store.use_model,
        provider_id,
        body.model_id,
        body.expected_revision,
    )
    return json_body(200, updated)


def _enabled(rows: list[dict], provider_id: str) -> list[dict]:
    for row in rows:
        if row["id"] == provider_id:
            return [model for model in row["models"] if model["enabled"]]
    return []


@router.post("/provider-models")
async def add_model(body: ModelCreate, request: Request):
    status, result, events = await asyncio.to_thread(
        request.app.state.store.add_model,
        key(request),
        body.model_dump(),
    )
    for event in events:
        await request.app.state.bus.publish(event)
    return json_body(status, result)


@router.patch("/provider-models/{model_id}")
async def update_model(model_id: str, body: ModelUpdate, request: Request):
    updated = await asyncio.to_thread(
        request.app.state.store.update_model,
        model_id,
        body.expected_revision,
        body.enabled,
    )
    return json_body(200, updated)
