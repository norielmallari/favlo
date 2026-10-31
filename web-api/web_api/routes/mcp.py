"""Per-agent MCP servers. A test reads the live tool catalog."""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, Request

from web_api.http import json_body, key
from web_api.mcp_cache import catalog
from web_api.mcp_client import McpError
from web_api.mcp_tools import groups_from, prepare_catalog, with_catalog
from web_api.schemas import McpCreate, McpTest, McpUpdate
from web_api.store import NotFound, Rejected, _mcp_fields

router = APIRouter()


@router.get("/mcp-servers")
async def list_servers(request: Request):
    rows = await asyncio.to_thread(request.app.state.store.mcp_servers)
    servers = await asyncio.gather(*(with_catalog(row) for row in rows))
    return json_body(200, {"servers": servers})


@router.post("/mcp-servers")
async def create_server(body: McpCreate, request: Request):
    status, result, events = await asyncio.to_thread(
        request.app.state.store.create_mcp_server,
        key(request),
        body.model_dump(),
    )
    for event in events:
        await request.app.state.bus.publish(event)
    return json_body(status, result)


@router.patch("/mcp-servers/{server_id}")
async def update_server(server_id: str, body: McpUpdate, request: Request):
    updated = await asyncio.to_thread(
        request.app.state.store.update_mcp_server,
        server_id,
        body.model_dump(),
    )
    return json_body(200, updated)


@router.post("/mcp-servers/test")
async def test_connection(body: McpTest):
    payload = body.model_dump()
    failed = {"ok": False, "message": "Connection failed", "tools": [], "groups": []}
    try:
        transport, command, url = _mcp_fields(payload)
    except Rejected:
        return json_body(200, failed)
    server = {"id": "preview", "transport": transport, "command": command, "url": url}
    try:
        tools = await catalog.get(server, refresh=True)
    except McpError:
        return json_body(200, failed)
    prepared = prepare_catalog(tools)
    return json_body(
        200,
        {
            "ok": True,
            "message": "Connection succeeded",
            "tools": prepared,
            "groups": groups_from(prepared),
            "url": url,
        },
    )


@router.post("/mcp-servers/{server_id}/test")
async def test_server(server_id: str, request: Request):
    server = await asyncio.to_thread(request.app.state.store.mcp_server, server_id)
    if server is None:
        raise NotFound("That MCP server does not exist.")
    try:
        fresh = await with_catalog(server, refresh=True)
    except McpError:
        failed = await asyncio.to_thread(request.app.state.store.mark_mcp_failed, server_id)
        return json_body(200, {"ok": False, "message": "Connection failed", "server": failed})
    marked = await asyncio.to_thread(request.app.state.store.mark_mcp_connected, server_id)
    marked["groups"] = fresh["groups"]
    return json_body(200, {"ok": True, "message": "Connection succeeded", "server": marked})
