"""Agent records and the Agent Channel each one owns."""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, Request

from favlo.config_tools import GROUP_ID, GROUP_NAME
from web_api.conversation import GROUP_ID as CONVERSATION_ID
from web_api.conversation import GROUP_NAME as CONVERSATION_NAME
from web_api.heartbeat_tools import GROUP_ID as HEARTBEAT_ID
from web_api.heartbeat_tools import GROUP_NAME as HEARTBEAT_NAME
from web_api.http import json_body, key
from web_api.schemas import AgentCreate, AgentUpdate
from web_api.session_tools import GROUP_ID as SESSION_ID
from web_api.session_tools import GROUP_NAME as SESSION_NAME

router = APIRouter()


@router.get("/tool-groups")
async def list_tool_groups():
    return json_body(
        200,
        {
            "groups": [
                {"id": GROUP_ID, "name": GROUP_NAME},
                {"id": CONVERSATION_ID, "name": CONVERSATION_NAME},
                {"id": HEARTBEAT_ID, "name": HEARTBEAT_NAME},
                {"id": SESSION_ID, "name": SESSION_NAME},
            ]
        },
    )


@router.get("/agents")
async def list_agents(request: Request):
    rows = await asyncio.to_thread(request.app.state.store.agents)
    return json_body(200, {"agents": rows})


@router.post("/agents")
async def create_agent(body: AgentCreate, request: Request):
    status, result, events = await asyncio.to_thread(
        request.app.state.store.create_agent,
        key(request),
        body.model_dump(),
    )
    for event in events:
        await request.app.state.bus.publish(event)
    return json_body(status, result)


@router.patch("/agents/{agent_id}")
async def update_agent(agent_id: str, body: AgentUpdate, request: Request):
    updated, events = await asyncio.to_thread(
        request.app.state.store.update_agent,
        agent_id,
        body.expected_revision,
        body.name,
        body.persona,
        body.default_model_id,
        body.tool_groups,
    )
    for event in events:
        await request.app.state.bus.publish(event)
    return json_body(200, updated)


@router.delete("/agents/{agent_id}")
async def delete_agent(agent_id: str, request: Request):
    result, events, run_ids = await asyncio.to_thread(
        request.app.state.store.delete_agent,
        agent_id,
    )
    for run_id in run_ids:
        pending = request.app.state.cancels.get(run_id)
        if pending is not None:
            pending.set()
    for event in events:
        await request.app.state.bus.publish(event)
    return json_body(200, result)
