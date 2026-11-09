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
from web_api.mcp_groups import group_for
from web_api.tool_names import slug

_MEMORY = {"remember", "recall", "forget"}

# Eager under MAF progressive disclosure. Non-generate pipeline tools stay cold.
_EAGER = frozenset(
    {
        "remember",
        "recall",
        "forget",
        "media_status",
        "media_cancel",
    }
)


def task_tool(name: str) -> bool:
    """A published generate tool returns a task id instead of polling."""
    return name.split("__")[-1].startswith("generate_")


def eager(name: str) -> bool:
    """Every generate_*, memory, and media-task tool loads with the turn."""
    if name in _EAGER:
        return True
    return name.split("__")[-1].startswith("generate_")


async def bundle(
    store: Any,
    agent_id: str,
    session_id: str,
    bus: Any,
    pending: set[asyncio.Task[Any]],
) -> tuple[str, list[object]]:
    """Tools for one turn. Every enabled server is attached the same way.

    The text is empty. Descriptions come from that server. Every ``generate_*``
    tool, memory, and media task is eager. Any other tool stays on ``load_tool``.
    """
    servers = store.enabled_mcp_servers(agent_id)
    if not servers:
        return "", []
    taken: set[str] = set()
    tools = []
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
        always = [name for name in allowed if eager(name)]
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
    if not tools:
        return "", []
    return "", tools


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
        group_id, group_name = group_for(tool)
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
    def _rank(group_id: str) -> tuple[int, str]:
        if group_id == "media":
            return (1, "")
        if group_id == "memory":
            return (2, "")
        if group_id == "tools":
            return (3, "")
        return (0, group_id)

    ordered = sorted(buckets.values(), key=lambda item: _rank(str(item["id"])))
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
    """A caller-owned client. ``call_tool`` resolves a task. A named file returns that name."""
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

    async def load_tools(self) -> None:
        await super().load_tools()
        _declare_filename(self)

    async def call_tool(self, tool_name: str, **kwargs: Any) -> Any:
        arguments = _memory_arguments(tool_name, kwargs, self._agent_id)
        _drop_runtime_session(arguments)
        refusal = _rewrite_inputs(self, arguments)
        if refusal is not None:
            return refusal
        started = await _generate_task(self, tool_name, arguments)
        if started is not None:
            return started
        return await _finish_call(self, tool_name, arguments)


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

    async def load_tools(self) -> None:
        await super().load_tools()
        _declare_filename(self)

    async def call_tool(self, tool_name: str, **kwargs: Any) -> Any:
        arguments = _memory_arguments(tool_name, kwargs, self._agent_id)
        _drop_runtime_session(arguments)
        refusal = _rewrite_inputs(self, arguments)
        if refusal is not None:
            return refusal
        started = await _generate_task(self, tool_name, arguments)
        if started is not None:
            return started
        return await _finish_call(self, tool_name, arguments)


async def _finish_call(server: Any, tool_name: str, arguments: dict[str, Any]) -> Any:
    """Call the listener. A status miss is a sentence, not a raised tool error."""
    try:
        return await super(type(server), server).call_tool(tool_name, **arguments)
    except Exception:
        if tool_name in {"media_status", "media_cancel"}:
            return "That task is not here."
        raise


def _declare_filename(server: Any) -> None:
    """A workspace root adds a required file name the listener never sees."""
    turn = getattr(server, "turn", None)
    if turn is None:
        return
    from web_api.workspace_files import local_root

    if local_root(turn.store, turn.session_id) is None:
        return
    for func in getattr(server, "_functions", []):
        extra = func.additional_properties or {}
        remote = str(extra.get("_mcp_remote_name") or "")
        if not task_tool(remote):
            continue
        schema = dict(func._input_schema_cached or {})
        properties = dict(schema.get("properties") or {})
        properties["filename"] = {
            "type": "string",
            "description": "File name at the folder root.",
        }
        required = [item for item in schema.get("required") or [] if item != "filename"]
        required.append("filename")
        schema["properties"] = properties
        schema["required"] = required
        func._input_schema_cached = schema
        func._cached_parameters = schema


def _drop_runtime_session(arguments: dict[str, Any]) -> None:
    """Drop ``session``. MCP media tools do not take a model session."""
    arguments.pop("session", None)


def _rewrite_inputs(server: Any, arguments: dict[str, Any]) -> str | None:
    """Turn a root image name into bytes before the listener sees it."""
    turn = getattr(server, "turn", None)
    if turn is None:
        return None
    from web_api.store import Rejected
    from web_api.workspace_files import local_root, rewrite_still

    root = local_root(turn.store, turn.session_id)
    if root is None:
        return None
    try:
        rewritten = rewrite_still(root, arguments)
    except Rejected as exc:
        return str(exc)
    if rewritten is not arguments:
        arguments.clear()
        arguments.update(rewritten)
    return None


async def _generate_task(server: Any, tool_name: str, arguments: dict[str, Any]) -> str | None:
    """Return a generate task id. Other tools stay on ``Client.call_tool``.

    ``session.call_tool(..., allow_claimed=True)`` is the early return.
    ``Client.call_tool`` would poll ``tasks/get`` until the file exists.
    """
    if not task_tool(tool_name):
        return None
    client = getattr(server, "owned_client", None)
    if client is None:
        return None
    from fastmcp_tasks.client_models import ClientCreateTaskResult

    from web_api.store import Rejected
    from web_api.workspace_files import local_root, release_output, reserve_output

    turn = getattr(server, "turn", None)
    root = None if turn is None else local_root(turn.store, turn.session_id)
    held = ""
    if root is None:
        arguments.pop("filename", None)
    else:
        try:
            held = reserve_output(root, tool_name, arguments)
        except Rejected as exc:
            return str(exc)
    try:
        filtered, meta = server._prepare_call_kwargs(tool_name, arguments)
        raw = await client.session.call_tool(
            tool_name,
            arguments=filtered or None,
            meta=meta if isinstance(meta, dict) else None,
            allow_claimed=True,
        )
    except Exception:
        if root is not None and held:
            release_output(root, held)
        raise
    if not isinstance(raw, ClientCreateTaskResult):
        if root is not None and held:
            release_output(root, held)
        parser = server.parse_tool_results or server._parse_tool_result_from_mcp
        return parser(raw)
    task_id = str(raw.task_id)
    target = getattr(client, "server", None)
    if isinstance(target, str):
        from web_api.media_delivery import kind_for_tool, note_task

        note_task(task_id, target, kind=kind_for_tool(tool_name), filename=held)
    named = held.strip()
    label = named or f"Task {task_id}"
    return (
        f"{label} is working. The file is not ready. "
        "There is no favlo:// link yet. Do not call media_status."
    )


def _description(row: dict[str, Any], exposed: str | None = None) -> str:
    del exposed
    return str(row.get("description") or row["name"])
