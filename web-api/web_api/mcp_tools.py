"""Attach an agent's enabled MCP tools to one turn."""

from __future__ import annotations

import asyncio
import contextlib
import shlex
import warnings
from typing import Any

from agent_framework import MCPStdioTool, MCPStreamableHTTPTool
from agent_framework._feature_stage import ExperimentalWarning

from web_api.mcp_cache import catalog
from web_api.mcp_client import McpError
from web_api.mcp_groups import GENERATE, ORDER, group_for
from web_api.tool_names import slug

_MEMORY = {"remember", "recall", "forget"}

# Eager under MAF progressive disclosure. Cold media groups stay in allowed_tools.
ALWAYS_LOAD = frozenset(
    {
        "generate_image",
        "list_image_models",
        "set_image_model",
        "remember",
        "recall",
        "forget",
        "media_status",
        "media_cancel",
        "read_media",
    }
)

INSTRUCTIONS = (
    "Enabled MCP tools use progressive disclosure. "
    "Image generate, memory, and media_status/cancel/read_media are available immediately. "
    "Each MCP tool name starts with the server name, then the tool. "
    "Before calling a video, music, or voice tool, call that server's load_tool with the "
    "remote tool name (or a list of names). Prefer loading a whole media group together: "
    "list, set, and generate for that kind (i2v and t2v are separate). "
    "Do not call list_mcp_tools just to browse; it returns full schemas. "
    "remember, recall, and forget use only this agent's memories. "
    "Match the generate tool to what the user asked for. "
    "generate_image is for a still (image, picture, photo, illustration, drawing). "
    "Do not call generate_video_i2v or generate_video_t2v unless the user asked for "
    "video, a clip, motion, animation, or to animate a still. "
    "Do not offer or insist on video when they asked for an image. "
    "generate_music and generate_voice only when they asked for music or spoken audio. "
    "Those generate tools use the registry default for that media type. "
    "To change a model, call the matching list tool, then set_image_model, set_video_i2v_model, "
    "set_video_t2v_model, set_music_model, or set_voice_model, each with that server prefix. "
    "Repeat the returned session id in your reply. "
    "Pass that same id as session on later set and generate calls. "
    "Omit session to use the registry defaults. "
    "An omitted seed is chosen at random. "
    "read_media(name) returns that record. "
    "When you call a generate tool, tell the user you are generating "
    "that image, video, music, or voice. "
    "The tool returns a task id while the file is still being made. "
    "That result says the file is not ready and names the id. "
    "There is no favlo:// link in it. "
    "Reply to the user in that same turn. "
    "Do not call media_status, read_media, or another tool after that generate. "
    "The chat shows each favlo:// link when the file is ready. "
    "Do not paste the JSON, the tool name, or a file id. "
    "When the result says the tool failed or names an unknown model, "
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
    eager_names: list[str] = []
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
        always = [name for name in allowed if name in ALWAYS_LOAD]
        opened = _server(
            server,
            agent_id,
            prefix,
            allowed,
            always,
            _Turn(store, bus, pending, str(server["server_id"]), agent_id, session_id),
        )
        if opened is None:
            continue
        tools.append(opened)
        eager_names.extend(f"{prefix}_{name}" for name in always)
        eager_names.append(f"{prefix}_load_tool")
    if not tools:
        return "", []
    joined = ", ".join(eager_names)
    return (
        f"{INSTRUCTIONS} This turn's eager MCP tools include: {joined}. "
        "Load cold tools with that server's load_tool before calling them.",
        tools,
    )


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
    """A caller-owned client. ``call_tool`` resolves a task. A generate tool returns the task id."""
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


async def restore_loaded(tools: list[object], store: Any, session_id: str) -> None:
    """Re-apply session-persisted progressive loads after MCP tools have connected."""
    saved = store.loaded_mcp_tools(session_id)
    if not saved:
        return
    for tool in tools:
        turn = getattr(tool, "turn", None)
        if turn is None:
            continue
        remotes = saved.get(str(turn.server_id)) or []
        if not remotes:
            continue
        connect = getattr(tool, "connect", None)
        if connect is not None:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ExperimentalWarning)
                await connect()
        _apply_loaded_remotes(tool, remotes)


def persist_loaded(tools: list[object], store: Any, session_id: str) -> None:
    """Write progressively loaded remote names for the next turn."""
    payload: dict[str, list[str]] = dict(store.loaded_mcp_tools(session_id))
    for tool in tools:
        turn = getattr(tool, "turn", None)
        if turn is None:
            continue
        remotes = _loaded_remotes(tool)
        if remotes:
            payload[str(turn.server_id)] = sorted(remotes)
        elif str(turn.server_id) in payload:
            del payload[str(turn.server_id)]
    store.save_loaded_mcp_tools(session_id, payload)


def _apply_loaded_remotes(tool: Any, remotes: list[str]) -> None:
    wanted = set(remotes)
    names = getattr(tool, "_progressive_loaded_tool_names", None)
    if not isinstance(names, set):
        return
    filtered = getattr(tool, "_filtered_functions", None)
    if filtered is None:
        return
    for func in filtered():
        props = getattr(func, "additional_properties", None) or {}
        remote = props.get("_mcp_remote_name")
        if isinstance(remote, str) and remote in wanted:
            names.add(func.name)


def _loaded_remotes(tool: Any) -> list[str]:
    names = getattr(tool, "_progressive_loaded_tool_names", None)
    if not isinstance(names, set) or not names:
        return []
    filtered = getattr(tool, "_filtered_functions", None)
    if filtered is None:
        return []
    remotes: list[str] = []
    for func in filtered():
        if func.name not in names:
            continue
        props = getattr(func, "additional_properties", None) or {}
        remote = props.get("_mcp_remote_name")
        if isinstance(remote, str):
            remotes.append(remote)
    return remotes


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
    """Enter and close a caller-owned MCP client on one task.

    MAF does not enter or close a client passed as ``mcp_client``. The AG-UI
    stream finishes on another task, and that task must not exit the client's
    anyio cancel scope.
    """

    def __init__(self, client: Any, server: Any) -> None:
        self._client = client
        self._server = server
        self._ready = asyncio.Event()
        self._stop = asyncio.Event()
        self._task: asyncio.Task[None] | None = None
        self._error: BaseException | None = None

    async def __aenter__(self) -> Any:
        task = asyncio.create_task(self._own())
        self._task = task
        jobs = getattr(self._server, "jobs", None)
        if isinstance(jobs, set):
            jobs.add(task)
            task.add_done_callback(jobs.discard)
        try:
            await self._ready.wait()
        except BaseException:
            self._stop.set()
            task.cancel()
            raise
        if self._error is not None:
            with contextlib.suppress(Exception):
                await task
            raise self._error
        return self._client

    async def _own(self) -> None:
        entered = False
        try:
            await self._client.__aenter__()
            entered = True
            self._ready.set()
            await self._stop.wait()
        except BaseException as exc:
            if not entered:
                self._error = exc
                self._ready.set()
            raise
        finally:
            if entered:
                with contextlib.suppress(Exception):
                    await self._client.__aexit__(None, None, None)

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        self._stop.set()
        task = self._task
        if task is None:
            return False
        current = asyncio.current_task()
        cancelled = exc_type is asyncio.CancelledError or (
            current is not None and current.cancelling()
        )
        if cancelled:
            return False
        with contextlib.suppress(Exception):
            await task
        return False


def _server(
    server: dict[str, Any],
    agent_id: str,
    prefix: str,
    allowed: list[str],
    always_load: list[str],
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
        "always_load": always_load,
        "use_progressive_disclosure": True,
        "load_prompts": False,
    }
    kind = _StdioServer if server.get("transport") == "stdio" else _HttpServer
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ExperimentalWarning)
        return kind(agent_id, client, turn, **shared)


def _arm(server: Any, agent_id: str, client: Any, turn: _Turn | None) -> None:
    server._agent_id = agent_id
    server.owned_client = client
    server.turn = turn
    server.jobs = None if turn is None else turn.jobs


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
        started = await _generate_task(self, tool_name, arguments)
        if started is not None:
            return started
        return await super().call_tool(tool_name, **arguments)


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
        started = await _generate_task(self, tool_name, arguments)
        if started is not None:
            return started
        return await super().call_tool(tool_name, **arguments)


async def _generate_task(server: Any, tool_name: str, arguments: dict[str, Any]) -> str | None:
    """Return a generate task id. Other tools stay on ``Client.call_tool``.

    ``session.call_tool(..., allow_claimed=True)`` is the early return.
    ``Client.call_tool`` would poll ``tasks/get`` until the file exists.
    """
    if tool_name not in GENERATE:
        return None
    client = getattr(server, "owned_client", None)
    if client is None:
        return None
    from fastmcp_tasks.client_models import ClientCreateTaskResult

    filtered, meta = server._prepare_call_kwargs(tool_name, arguments)
    raw = await client.session.call_tool(
        tool_name,
        arguments=filtered or None,
        meta=meta if isinstance(meta, dict) else None,
        allow_claimed=True,
    )
    if not isinstance(raw, ClientCreateTaskResult):
        parser = server.parse_tool_results or server._parse_tool_result_from_mcp
        return parser(raw)
    task_id = str(raw.task_id)
    target = getattr(client, "server", None)
    if isinstance(target, str):
        from web_api.media_delivery import kind_for_tool, note_task

        note_task(task_id, target, kind=kind_for_tool(tool_name))
    return (
        f"Task {task_id} is working. The file is not ready. "
        "There is no favlo:// link yet. "
        "Reply to the user now. Do not call another tool on this turn. "
        "The chat shows the file when it is ready."
    )


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
        f"Default model selected. For another: {prefix}{listed}, "
        f"then {prefix}{setter}; pass session."
    )
    if note in text:
        return text
    return f"{text} {note}"
