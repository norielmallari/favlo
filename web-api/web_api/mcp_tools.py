"""Attach an agent's enabled MCP tools to one turn."""

from __future__ import annotations

import asyncio
import contextlib
import shlex
from typing import Any

from agent_framework import MCPStdioTool, MCPStreamableHTTPTool
from agent_framework.exceptions import ToolExecutionException

from web_api.mcp_cache import catalog
from web_api.mcp_client import McpError
from web_api.mcp_groups import GENERATE, ORDER, group_for
from web_api.mcp_tasks import accept, spawn, started
from web_api.tool_names import slug

_MEMORY = {"remember", "recall", "forget"}

INSTRUCTIONS = (
    "Enabled MCP tools are available on every turn. "
    "remember, recall, and forget use only this agent's memories. "
    "Each MCP tool name starts with the server name, then the tool. "
    "generate_image, generate_video_i2v, generate_video_t2v, generate_music, and generate_voice "
    "use the registry default for that media type. "
    "To change a model, call the matching list tool, then set_image_model, set_video_i2v_model, "
    "set_video_t2v_model, set_music_model, or set_voice_model, each with that server prefix. "
    "Repeat the returned session id in your reply. "
    "Pass that same id as session on later set and generate calls. "
    "Omit session to use the registry defaults. "
    "An omitted seed is chosen at random. "
    "The generate result includes the parameters used, including the seed and the prompts. "
    "read_media(name) returns that record. "
    "When you call a generate tool, tell the user you are generating "
    "that image, video, music, or voice. "
    "The tool returns when the task is accepted. "
    "Do not wait for the file in this turn. "
    "A later turn brings the result. Include each favlo:// link from that result. "
    "Do not paste the JSON, the tool name, or a file id. "
    "When a task result says the tool failed or names an unknown model, "
    "call the matching list and set tools, repeat the session id, "
    "and call the generate tool again with that session."
)


async def bundle(
    store: Any,
    agent_id: str,
    session_id: str,
    bus: Any,
    pending: set[asyncio.Task[Any]],
) -> tuple[str, list[object]]:
    """Tools for one turn. Definitions come from the catalog cache."""
    servers = store.enabled_mcp_servers(agent_id)
    if not servers:
        return "", []
    taken: set[str] = set()
    tools = []
    names = []
    for server in servers:
        try:
            listed = await catalog.get(server)
        except McpError:
            continue
        switches = server.get("switches")
        enabled = switches if isinstance(switches, dict) else {}
        allowed = [
            str(tool["name"])
            for tool in prepare_catalog(listed)
            if enabled.get(tool["group_id"], True)
        ]
        if not allowed:
            continue
        prefix = _prefix(str(server["server_name"] or server["server_id"]), taken)
        opened = _server(
            server,
            agent_id,
            prefix,
            allowed,
            _Turn(store, bus, pending, str(server["server_id"]), agent_id, session_id),
        )
        if opened is None:
            continue
        tools.append(opened)
        names.extend(f"{prefix}_{name}" for name in allowed)
    if not names:
        return "", []
    joined = ", ".join(names)
    return f"{INSTRUCTIONS} This turn's MCP tools are: {joined}.", tools


def overlay(server: dict[str, Any], tools: list[dict[str, Any]]) -> dict[str, Any]:
    """Group ``tools`` and keep each saved switch. A missing switch stays on."""
    prepared = prepare_catalog(tools)
    groups = groups_from(prepared)
    prior = {
        item["id"]: bool(item["enabled"])
        for item in server.get("groups") or []
        if isinstance(item, dict) and isinstance(item.get("id"), str)
    }
    for group in groups:
        if group["id"] in prior:
            group["enabled"] = prior[group["id"]]
    return {**server, "groups": groups}


async def with_catalog(server: dict[str, Any], *, refresh: bool = False) -> dict[str, Any]:
    """Attach the cached catalog. A failed read leaves the saved groups."""
    try:
        tools = await catalog.get(server, refresh=refresh)
    except McpError:
        if refresh:
            raise
        return server
    return overlay(server, tools)


def prepare_catalog(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    prepared = []
    for tool in tools:
        group_id, group_name = group_for(tool["name"])
        prepared.append({**tool, "group_id": group_id, "group_name": group_name})
    return prepared


def groups_from(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[str, dict[str, Any]] = {}
    for tool in tools:
        group_id = str(tool["group_id"])
        bucket = buckets.setdefault(
            group_id,
            {"id": group_id, "name": tool["group_name"], "enabled": True, "tools": []},
        )
        bucket["tools"].append(tool["name"])
    rank = {group_id: index for index, group_id in enumerate(ORDER)}
    ordered = sorted(buckets.values(), key=lambda item: rank.get(str(item["id"]), len(rank)))
    for bucket in ordered:
        bucket["tools"] = sorted(bucket["tools"])
    return ordered


def _prefix(source: str, taken: set[str]) -> str:
    stem = slug(source)
    if stem not in taken:
        taken.add(stem)
        return stem
    extra = 2
    while f"{stem}_{extra}" in taken:
        extra += 1
    chosen = f"{stem}_{extra}"
    taken.add(chosen)
    return chosen


def _memory_arguments(tool_name: str, arguments: dict[str, Any], agent_id: str) -> dict[str, Any]:
    if tool_name not in _MEMORY:
        return arguments
    kept = {key: value for key, value in arguments.items() if key != "client"}
    kept["client"] = agent_id
    return kept


def _mcp_client(server: dict[str, Any]) -> Any | None:
    """A caller-owned client. The tasks extension is not on the framework's own client."""
    from fastmcp_tasks.client import TasksClientExtension
    from mcp import Client

    extension = TasksClientExtension()
    if server.get("transport") == "stdio":
        from mcp.client.stdio import StdioServerParameters

        argv = shlex.split(str(server.get("command") or ""))
        if not argv:
            return None
        target = StdioServerParameters(command=argv[0], args=argv[1:])
    else:
        target = str(server.get("url") or "").strip()
        if not target:
            return None
    return Client(target, extensions=[extension], read_timeout_seconds=3600)


async def open_clients(tools: list[object], stack: contextlib.AsyncExitStack) -> None:
    """Enter each caller-owned client before the framework connects its tool."""
    for tool in tools:
        client = getattr(tool, "owned_client", None)
        if client is not None:
            await stack.enter_async_context(_Lease(client, tool))


class _Turn:
    def __init__(
        self,
        store: Any,
        bus: Any,
        jobs: set[asyncio.Task[Any]],
        server_id: str,
        agent_id: str,
        session_id: str,
    ) -> None:
        self.store = store
        self.bus = bus
        self.jobs = jobs
        self.server_id = server_id
        self.agent_id = agent_id
        self.session_id = session_id


class _Lease:
    """Close the client when the turn ends, unless a task is still running on it."""

    def __init__(self, client: Any, server: Any) -> None:
        self._client = client
        self._server = server

    async def __aenter__(self) -> Any:
        return await self._client.__aenter__()

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        pending = getattr(self._server, "pending", 0)
        jobs = getattr(self._server, "jobs", None)
        if pending and jobs is not None:
            spawn(jobs, self._close())
            return False
        await self._client.__aexit__(exc_type, exc, tb)
        return False

    async def _close(self) -> None:
        await self._server.idle.wait()
        with contextlib.suppress(Exception):
            await self._client.__aexit__(None, None, None)


def _server(
    server: dict[str, Any],
    agent_id: str,
    prefix: str,
    allowed: list[str],
    turn: _Turn | None = None,
) -> MCPStdioTool | MCPStreamableHTTPTool | None:
    client = _mcp_client(server)
    if client is None:
        return None
    shared = {
        "name": str(server["server_name"] or server["server_id"]),
        "tool_name_prefix": prefix,
        "approval_mode": "never_require",
        "allowed_tools": allowed,
        "load_prompts": False,
    }
    kind = _StdioServer if server.get("transport") == "stdio" else _HttpServer
    return kind(agent_id, client, turn, **shared)


def _arm(server: Any, agent_id: str, client: Any, turn: _Turn | None) -> None:
    server._agent_id = agent_id
    server.owned_client = client
    server.turn = turn
    server.jobs = None if turn is None else turn.jobs
    server.pending = 0
    server.idle = asyncio.Event()
    server.idle.set()


def _live_session(client: Any) -> Any | None:
    try:
        return client.session
    except RuntimeError:
        return None


def _result_text(parsed: Any) -> str:
    if isinstance(parsed, str):
        return parsed
    if isinstance(parsed, list):
        lines = [item.text for item in parsed if getattr(item, "text", None)]
        if lines:
            return "\n".join(lines)
    return "The tool failed."


async def _invoke(server: Any, tool_name: str, arguments: dict[str, Any]) -> Any:
    """One ``tools/call``. A task handle returns immediately."""
    from fastmcp_tasks.client_models import ClientCreateTaskResult

    filtered, meta = server._prepare_call_kwargs(tool_name, arguments)
    raw = await server.owned_client.session.call_tool(
        tool_name,
        filtered,
        meta=meta,
        allow_claimed=True,
        read_timeout_seconds=server.owned_client.read_timeout_seconds,
    )
    if isinstance(raw, ClientCreateTaskResult):
        turn = server.turn
        if turn is not None:
            accept(
                turn.store,
                turn.bus,
                turn.jobs,
                server,
                server_id=turn.server_id,
                agent_id=turn.agent_id,
                session_id=turn.session_id,
                tool_name=tool_name,
                create_result=raw,
            )
        return started(tool_name, raw.task_id)
    if getattr(raw, "is_error", False):
        raise ToolExecutionException(_result_text(server._parse_tool_result_from_mcp(raw)))
    return server._parse_tool_result_from_mcp(raw)


class _HttpServer(MCPStreamableHTTPTool):
    """One HTTP MCP server the agent connects for the run."""

    def __init__(
        self,
        agent_id: str,
        client: Any,
        turn: _Turn | None = None,
        **kwargs: Any,
    ) -> None:
        name = str(kwargs.pop("name"))
        super().__init__(name, mcp_client=client, **kwargs)
        _arm(self, agent_id, client, turn)

    async def call_tool(self, tool_name: str, **kwargs: Any) -> Any:
        arguments = _memory_arguments(tool_name, kwargs, self._agent_id)
        if _live_session(self.owned_client) is None:
            return await super().call_tool(tool_name, **arguments)
        return await _invoke(self, tool_name, arguments)


class _StdioServer(MCPStdioTool):
    """One stdio MCP server the agent connects for the run."""

    def __init__(
        self,
        agent_id: str,
        client: Any,
        turn: _Turn | None = None,
        **kwargs: Any,
    ) -> None:
        name = str(kwargs.pop("name"))
        super().__init__(name, mcp_client=client, **kwargs)
        _arm(self, agent_id, client, turn)

    async def call_tool(self, tool_name: str, **kwargs: Any) -> Any:
        arguments = _memory_arguments(tool_name, kwargs, self._agent_id)
        if _live_session(self.owned_client) is None:
            return await super().call_tool(tool_name, **arguments)
        return await _invoke(self, tool_name, arguments)


def _description(row: dict[str, Any], exposed: str | None = None) -> str:
    text = str(row.get("description") or row["name"])
    pair = GENERATE.get(str(row.get("name") or ""))
    if pair is None:
        return text
    original = str(row.get("name") or "")
    prefix = ""
    if exposed and original and exposed.endswith(original):
        prefix = exposed[: -len(original)]
    listed, setter = pair
    note = (
        f"The registry default is already selected. "
        f"To use another model, call {prefix}{listed}, then {prefix}{setter}, "
        f"repeat the returned session id in your reply, and pass it as session."
    )
    if note in text:
        return text
    return f"{text} {note}"
