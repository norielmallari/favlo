"""MCP servers and the Agent Framework session."""

from __future__ import annotations

import asyncio
import contextlib
import sqlite3
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "web-api"))

from web_api.ids import FAVLO_AGENT_ID, FAVLO_MCP_ID, guid  # noqa: E402
from web_api.mcp_client import list_tools  # noqa: E402
from web_api.mcp_tools import _description, groups_from, prepare_catalog  # noqa: E402
from web_api.store import Store, _mcp_fields  # noqa: E402


def _home(store: Store) -> str:
    return next(row["session_id"] for row in store.channels() if row["agent_id"] == FAVLO_AGENT_ID)


def test_a_favlo_listener_address_uses_the_mcp_path() -> None:
    assert _mcp_fields({"transport": "http", "url": "http://127.0.0.1:8129"})[2] == (
        "http://127.0.0.1:8129/mcp"
    )
    assert _mcp_fields({"transport": "http", "url": "http://127.0.0.1:8129/v1"})[2] == (
        "http://127.0.0.1:8129/mcp"
    )
    assert _mcp_fields({"transport": "http", "url": "http://localhost:8129/mcp/"})[2] == (
        "http://localhost:8129/mcp"
    )
    kept = _mcp_fields({"transport": "http", "url": "http://127.0.0.1:9000/v1"})[2]
    assert kept == "http://127.0.0.1:9000/v1"


def test_a_stdio_server_lists_its_tools(tmp_path: Path) -> None:
    script = tmp_path / "server.py"
    script.write_text(
        "\n".join(
            [
                "import json, sys",
                "for line in sys.stdin:",
                "    msg = json.loads(line)",
                "    result = {'resultType': 'complete', 'tools': [",
                "        {'name': 'echo', 'description': 'Echo.',",
                "         'inputSchema': {'type': 'object', 'properties': {}}}],",
                "    }",
                "    payload = json.dumps({'jsonrpc': '2.0', 'id': msg['id'], 'result': result})",
                "    sys.stdout.write(payload + '\\n')",
                "    sys.stdout.flush()",
            ]
        ),
        encoding="utf-8",
    )
    tools = asyncio.run(
        list_tools(
            {
                "id": "stdio-test",
                "transport": "stdio",
                "command": f"{sys.executable} {script}",
                "url": "",
            }
        )
    )
    assert [tool["name"] for tool in tools] == ["echo"]


def test_the_framework_session_advertises_tasks() -> None:
    from fastmcp.utilities.tasks import TASKS_EXTENSION_ID

    from web_api.mcp_tools import _HttpServer, _mcp_client

    client = _mcp_client({"transport": "http", "url": "http://127.0.0.1:9/mcp"})
    assert client is not None
    assert client.read_timeout_seconds == 3600
    identifiers = [getattr(item, "identifier", None) for item in client.extensions]
    assert TASKS_EXTENSION_ID in identifiers
    server = _HttpServer(
        FAVLO_AGENT_ID,
        client,
        name="Favlo",
        tool_name_prefix="favlo",
        approval_mode="never_require",
        allowed_tools=["echo"],
        load_prompts=False,
    )
    assert server.owned_client is client


def test_save_keeps_a_disabled_group_and_lists_its_tools(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    catalog = prepare_catalog(
        [
            {
                "name": "z_image_turbo__generate_image",
                "description": "Generate.",
                "schema": {"type": "object"},
                "pipeline": "z-image-turbo",
            },
            {"name": "echo", "description": "Echo.", "schema": {"type": "object"}},
        ]
    )
    groups = groups_from(catalog)
    assert groups[0]["id"] == "z-image-turbo"
    assert groups[0]["tools"] == ["z_image_turbo__generate_image"]
    store = Store(tmp_path / "app.db")
    try:
        _status, created, _events = store.create_mcp_server(
            "mcp-save",
            {
                "agent_id": FAVLO_AGENT_ID,
                "name": "Favlo media",
                "transport": "http",
                "command": "",
                "url": "http://127.0.0.1:9/mcp",
                "enabled": True,
                "groups": [{"id": "z-image-turbo", "enabled": False}],
                "tools": catalog,
            },
        )
        image = next(group for group in created["groups"] if group["id"] == "z-image-turbo")
        assert image["enabled"] is False
        assert image["tools"] == []
        from web_api.mcp_tools import bundle

        async def listed(_server: object, *, refresh: bool = False) -> list[dict[str, object]]:
            return [
                {
                    "name": "z_image_turbo__generate_image",
                    "description": "Generate.",
                    "schema": {"type": "object"},
                    "pipeline": "z-image-turbo",
                },
                {"name": "echo", "description": "Echo.", "schema": {"type": "object"}},
            ]

        monkeypatch.setattr("web_api.mcp_cache.catalog.get", listed)
        _text, enabled = asyncio.run(bundle(store, FAVLO_AGENT_ID, _home(store), None, set()))
        created_tool = next(
            item for item in enabled if list(item.allowed_tools) == ["echo"]
        )
        assert created_tool.tool_name_prefix.startswith("favlo_media")
        assert list(created_tool.allowed_tools) == ["echo"]
    finally:
        store.close()


def test_media_instructions_keep_the_server_description() -> None:
    described = _description(
        {"name": "z_image_turbo__generate_image", "description": "Generate one or more PNGs."}
    )
    assert described == "Generate one or more PNGs."


def test_mcp_always_load_covers_generate_memory_and_media_tasks() -> None:
    from web_api.mcp_tools import eager

    assert eager("z_image_turbo__generate_image")
    assert eager("wan22_i2v__generate_video")
    assert eager("wan22_ti2v_5b__generate_video_i2v")
    assert eager("wan22_ti2v_5b__generate_video_t2v")
    assert eager("ace_step_15_turbo__generate_music")
    assert eager("voxcpm2__generate_voice")
    assert eager("remember")
    assert eager("media_status")
    assert not eager("example_pipeline__custom_tool")


def test_dumps_for_model_omits_spaces() -> None:
    from favlo.json_model import dumps_for_model

    text = dumps_for_model({"a": 1, "b": "x"})
    assert text == '{"a":1,"b":"x"}'


def test_dumps_for_model_drops_empty_fields() -> None:
    from favlo.json_model import dumps_for_model

    text = dumps_for_model({"path": "a.png", "title": "", "tags": [], "score": 0.1})
    assert text == '{"path":"a.png","score":0.1}'


_EAGER = frozenset(
    {
        "remember",
        "recall",
        "forget",
        "media_status",
        "media_cancel",
        "z_image_turbo__generate_image",
        "wan22_i2v__generate_video",
        "ace_step_15_turbo__generate_music",
        "voxcpm2__generate_voice",
    }
)
_COLD = frozenset({"example_pipeline__custom_tool"})


def test_progressive_mcp_server_flags_and_eager_mask() -> None:
    from web_api.mcp_tools import _server

    class _Client:
        pass

    opened = _server(
        {
            "server_id": "s1",
            "server_name": "favlo",
            "transport": "http",
            "url": "http://127.0.0.1:8129/mcp",
        },
        "agent",
        "favlo",
        sorted(_EAGER | _COLD),
        sorted(_EAGER),
        None,
    )
    assert opened is not None
    # Replace the real client so construction stayed side-effect free for flags.
    opened.owned_client = _Client()
    assert opened.use_progressive_disclosure is True
    assert set(opened.always_load or []) == set(_EAGER)
    assert "example_pipeline__custom_tool" in set(opened.allowed_tools or [])
    assert "example_pipeline__custom_tool" not in set(opened.always_load or [])


def test_progressive_initial_functions_include_generate_tools() -> None:
    """Eager progressive surface has loaders, generate_*, and cold non-generate remotes stay out."""
    from agent_framework import FunctionTool

    from web_api.mcp_tools import _server

    opened = _server(
        {
            "server_id": "s1",
            "server_name": "favlo",
            "transport": "http",
            "url": "http://127.0.0.1:8129/mcp",
        },
        "agent",
        "favlo",
        sorted(_EAGER | _COLD),
        sorted(_EAGER),
        None,
    )
    assert opened is not None

    def _fake(name: str, remote: str) -> FunctionTool:
        async def _run() -> str:
            return name

        return FunctionTool(
            name=name,
            description=remote,
            approval_mode="never_require",
            additional_properties={
                "_mcp_normalized_name": remote,
                "_mcp_remote_name": remote,
            },
            func=_run,
        )

    fakes = [_fake(f"favlo_{remote}", remote) for remote in sorted(_EAGER | _COLD)]
    opened._filtered_functions = lambda: fakes  # type: ignore[method-assign]
    opened._progressive_loader_functions = None
    names = {func.name for func in opened._progressive_functions()}
    assert "favlo_load_tool" in names
    assert "favlo_list_mcp_tools" in names
    assert "favlo_z_image_turbo__generate_image" in names
    assert "favlo_wan22_i2v__generate_video" in names
    assert "favlo_ace_step_15_turbo__generate_music" in names
    assert "favlo_voxcpm2__generate_voice" in names
    assert "favlo_example_pipeline__custom_tool" not in names


def test_mcp_tools_from_two_servers_keep_unique_names(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from web_api.mcp_tools import bundle
    from web_api.store import Store

    async def listed(_server: object, *, refresh: bool = False) -> list[dict[str, object]]:
        return [
            {
                "name": "z_image_turbo__generate_image",
                "description": "Generate.",
                "schema": {"type": "object", "properties": {"prompt": {"type": "string"}}},
                "pipeline": "z-image-turbo",
            }
        ]

    monkeypatch.setattr("web_api.mcp_cache.catalog.get", listed)
    store = Store(tmp_path / "app.db")
    try:
        store.create_mcp_server(
            "mcp-a",
            {
                "agent_id": FAVLO_AGENT_ID,
                "name": "Favlo media",
                "transport": "http",
                "command": "",
                "url": "http://127.0.0.1:8129/mcp",
                "enabled": True,
            },
        )
        store.create_mcp_server(
            "mcp-b",
            {
                "agent_id": FAVLO_AGENT_ID,
                "name": "Favlo media",
                "transport": "http",
                "command": "",
                "url": "http://127.0.0.1:9000/mcp",
                "enabled": True,
            },
        )
        text, tools = asyncio.run(bundle(store, FAVLO_AGENT_ID, _home(store), None, set()))
        prefixes = [item.tool_name_prefix for item in tools]
        assert prefixes[0] == "favlo_media"
        assert "favlo_media_2" in prefixes
        assert len(prefixes) == len(set(prefixes))
        assert all(list(item.allowed_tools) == ["z_image_turbo__generate_image"] for item in tools)
        assert text == ""
        described = _description(
            {"name": "z_image_turbo__generate_image", "description": "Generate."},
            "favlo_media_z_image_turbo__generate_image",
        )
        assert described == "Generate."
        stored = sqlite3.connect(tmp_path / "app.db")
        try:
            tables = stored.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
            found = {row[0] for row in tables}
            assert "mcp_tool" not in found
            assert "mcp_tool_group" not in found
        finally:
            stored.close()
    finally:
        store.close()


def test_any_configured_server_attaches_with_no_instruction_text(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from web_api.mcp_tools import bundle
    from web_api.store import Store

    async def listed(
        server: dict[str, object], *, refresh: bool = False
    ) -> list[dict[str, object]]:
        if server.get("url") == "http://127.0.0.1:9001/mcp":
            return [
                {
                    "name": "lookup_note",
                    "description": "Look up a note.",
                    "schema": {"type": "object"},
                }
            ]
        return [
            {
                "name": "z_image_turbo__generate_image",
                "description": "Generate.",
                "schema": {"type": "object"},
                "pipeline": "z-image-turbo",
            }
        ]

    monkeypatch.setattr("web_api.mcp_cache.catalog.get", listed)
    store = Store(tmp_path / "app.db")
    try:
        store.create_mcp_server(
            "mcp-notes",
            {
                "agent_id": FAVLO_AGENT_ID,
                "name": "Notes",
                "transport": "http",
                "command": "",
                "url": "http://127.0.0.1:9001/mcp",
                "enabled": True,
            },
        )
        text, tools = asyncio.run(bundle(store, FAVLO_AGENT_ID, _home(store), None, set()))
    finally:
        store.close()
    by_prefix = {item.tool_name_prefix: item for item in tools}
    assert text == ""
    assert set(by_prefix) == {"favlo_media", "notes"}
    assert list(by_prefix["notes"].allowed_tools) == ["lookup_note"]
    assert list(by_prefix["notes"].always_load or []) == []
    assert "z_image_turbo__generate_image" in list(by_prefix["favlo_media"].always_load or [])


def test_an_unstamped_database_is_rebuilt(tmp_path: Path) -> None:
    path = tmp_path / "app.db"
    server_id = guid()
    raw = sqlite3.connect(path)
    raw.execute(
        """
        CREATE TABLE mcp_server (
            id TEXT PRIMARY KEY,
            agent_id TEXT NOT NULL,
            name TEXT NOT NULL,
            transport TEXT NOT NULL,
            command TEXT NOT NULL,
            url TEXT NOT NULL,
            enabled INTEGER NOT NULL,
            status TEXT NOT NULL,
            status_message TEXT NOT NULL,
            revision INTEGER NOT NULL
        )
        """
    )
    raw.execute(
        """
        CREATE TABLE mcp_tool (
            server_id TEXT NOT NULL,
            name TEXT NOT NULL,
            group_id TEXT NOT NULL,
            description TEXT NOT NULL,
            schema TEXT NOT NULL
        )
        """
    )
    raw.execute(
        """
        CREATE TABLE mcp_tool_group (
            server_id TEXT NOT NULL,
            group_id TEXT NOT NULL,
            name TEXT NOT NULL,
            enabled INTEGER NOT NULL
        )
        """
    )
    raw.execute(
        """
        INSERT INTO mcp_server (
            id, agent_id, name, transport, command, url, enabled,
            status, status_message, revision
        ) VALUES (?, ?, 'Favlo media', 'http', '', ?, 1, 'connected', '', 1)
        """,
        (server_id, FAVLO_AGENT_ID, "http://127.0.0.1:8129/mcp"),
    )
    raw.execute(
        """
        INSERT INTO mcp_tool (server_id, name, group_id, description, schema)
        VALUES (?, 'generate_image', 'image', 'Generate.', ?)
        """,
        (server_id, '{"type":"object","required":["model"]}'),
    )
    raw.execute(
        """
        INSERT INTO mcp_tool_group (server_id, group_id, name, enabled)
        VALUES (?, 'image', 'Image', 0)
        """,
        (server_id,),
    )
    raw.commit()
    raw.close()
    store = Store(path)
    try:
        assert store.mcp_server(server_id) is None
        seeded = store.mcp_server(FAVLO_MCP_ID)
        assert seeded is not None
        assert seeded["name"] == "Favlo media"
        assert {group["id"]: group["enabled"] for group in seeded["groups"]} == {
            "media": True,
            "memory": True,
        }
    finally:
        store.close()
    check = sqlite3.connect(path)
    try:
        listed = check.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
        tables = {row[0] for row in listed}
        assert "mcp_tool" not in tables
        assert "mcp_tool_group" not in tables
    finally:
        check.close()


def test_the_catalog_cache_reuses_a_fresh_list_and_refetches_when_it_expires() -> None:
    from web_api.mcp_cache import CatalogCache

    now = [0.0]
    calls = {"n": 0}

    async def fetch(_server: dict[str, object]) -> list[dict[str, object]]:
        calls["n"] += 1
        return [{"name": "echo", "description": "", "schema": {"type": "object"}}]

    cache = CatalogCache(ttl=10, fetch=fetch, clock=lambda: now[0])
    server = {"transport": "http", "url": "http://127.0.0.1:8129/mcp", "command": ""}

    async def run() -> None:
        first = await cache.get(server)
        second = await cache.get(server)
        assert calls["n"] == 1
        assert first == second
        assert first is not second
        first[0]["name"] = "changed"
        third = await cache.get(server)
        assert third[0]["name"] == "echo"
        now[0] = 11
        await cache.get(server)
        assert calls["n"] == 2
        await cache.get(server, refresh=True)
        assert calls["n"] == 3

    asyncio.run(run())


def test_one_catalog_fetch_serves_concurrent_readers() -> None:
    from web_api.mcp_cache import CatalogCache

    started = asyncio.Event()
    release = asyncio.Event()
    calls = {"n": 0}

    async def fetch(_server: dict[str, object]) -> list[dict[str, object]]:
        calls["n"] += 1
        started.set()
        await release.wait()
        return [{"name": "echo", "description": "", "schema": {"type": "object"}}]

    cache = CatalogCache(fetch=fetch)
    server = {"transport": "http", "url": "http://127.0.0.1:8129/mcp", "command": ""}

    async def run() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
        first = asyncio.create_task(cache.get(server))
        second = asyncio.create_task(cache.get(server))
        await started.wait()
        release.set()
        return await first, await second

    left, right = asyncio.run(run())
    assert calls["n"] == 1
    assert left == right == [{"name": "echo", "description": "", "schema": {"type": "object"}}]


def test_a_framework_server_keeps_the_agents_own_client() -> None:
    from agent_framework import MCPStreamableHTTPTool

    from web_api.mcp_tools import _HttpServer, _mcp_client

    seen: dict[str, object] = {}

    async def fake(self: object, tool_name: str, **kwargs: object) -> str:
        seen["tool"] = tool_name
        seen["arguments"] = kwargs
        return "ok"

    monkeypatch_base = MCPStreamableHTTPTool.call_tool
    MCPStreamableHTTPTool.call_tool = fake  # type: ignore[method-assign]
    try:
        client = _mcp_client({"transport": "http", "url": "http://127.0.0.1:9/mcp"})
        assert client is not None
        server = _HttpServer(
            FAVLO_AGENT_ID,
            client,
            name="Favlo",
            tool_name_prefix="favlo",
            approval_mode="never_require",
            allowed_tools=["recall"],
            load_prompts=False,
        )

        async def run() -> None:
            await server.call_tool("recall", query="tea", client="agent_other")

        asyncio.run(run())
    finally:
        MCPStreamableHTTPTool.call_tool = monkeypatch_base  # type: ignore[method-assign]
    assert seen["tool"] == "recall"
    assert seen["arguments"] == {"query": "tea", "client": FAVLO_AGENT_ID}


def _peer(tmp_path: Path, tasks_get: str, tool: str = "echo") -> str:
    script = tmp_path / "peer.py"
    script.write_text(
        "import json, sys\n"
        "STAMP = '2026-10-31T00:00:00Z'\n"
        "def reply(msg, result=None, error=None):\n"
        "    if 'id' not in msg:\n"
        "        return\n"
        "    body = {'jsonrpc': '2.0', 'id': msg['id']}\n"
        "    key = 'error' if error else 'result'\n"
        "    body[key] = error or result\n"
        "    sys.stdout.write(json.dumps(body) + '\\n')\n"
        "    sys.stdout.flush()\n"
        "for line in sys.stdin:\n"
        "    line = line.strip()\n"
        "    if not line:\n"
        "        continue\n"
        "    msg = json.loads(line)\n"
        "    method = msg.get('method')\n"
        "    if method == 'server/discover':\n"
        "        reply(msg, {\n"
        "            'supportedVersions': ['2026-07-28'],\n"
        "            'capabilities': {'tools': {}},\n"
        "            'resultType': 'complete', 'ttlMs': 0,\n"
        "            'cacheScope': 'private'})\n"
        "    elif method == 'initialize':\n"
        "        reply(msg, error={\n"
        "            'code': -32022,\n"
        "            'message': 'Unsupported protocol version',\n"
        "            'data': {'supported': ['2026-07-28'],\n"
        "                     'requested': '2025-11-25'}})\n"
        "    elif method == 'tools/list':\n"
        "        reply(msg, {'tools': [{\n"
        f"            'name': {tool!r}, 'description': 'Echo.',\n"
        "            'inputSchema': {'type': 'object',\n"
        "                'properties': {'text': {'type': 'string'}}}}],\n"
        "            'resultType': 'complete', 'ttlMs': 0,\n"
        "            'cacheScope': 'private'})\n"
        "    elif method == 'tools/call':\n"
        "        reply(msg, {'resultType': 'task', 'taskId': 'task-1',\n"
        "            'status': 'working', 'createdAt': STAMP,\n"
        "            'lastUpdatedAt': STAMP, 'ttlMs': 60000,\n"
        "            'pollIntervalMs': 20})\n"
        "    elif method == 'tasks/get':\n"
        f"        {tasks_get}\n"
        "    else:\n"
        "        reply(msg, error={'code': -32601,\n"
        "            'message': 'Method not found'})\n",
        encoding="utf-8",
    )
    import shlex

    return f"{shlex.quote(sys.executable)} {shlex.quote(str(script))}"


def _shown(result: object) -> str:
    if isinstance(result, str):
        return result
    return "\n".join(getattr(item, "text", "") for item in result)  # type: ignore[arg-type]


def test_a_task_result_returns_in_the_tool_call(tmp_path: Path) -> None:
    from web_api.mcp_tools import _mcp_client, _StdioServer

    command = _peer(
        tmp_path,
        "reply(msg, {'resultType': 'complete', 'taskId': 'task-1',\n"
        "            'status': 'completed', 'createdAt': STAMP,\n"
        "            'lastUpdatedAt': STAMP, 'ttlMs': 60000,\n"
        "            'pollIntervalMs': 20,\n"
        "            'result': {'content': [\n"
        "                {'type': 'text', 'text': 'ready'}],\n"
        "                'isError': False}})",
    )

    async def run() -> str:
        client = _mcp_client({"transport": "stdio", "command": command})
        assert client is not None
        server = _StdioServer(
            FAVLO_AGENT_ID,
            client,
            name="Peer",
            tool_name_prefix="peer",
            approval_mode="never_require",
            allowed_tools=["echo"],
            load_prompts=False,
        )
        async with client:
            await server.connect()
            assert client.protocol_version == "2026-07-28"
            result = await asyncio.wait_for(server.call_tool("echo", text="hi"), 2)
            await server.close()
        return _shown(result)

    text = asyncio.run(run())
    assert text == "ready"


def test_a_generate_tool_returns_the_task_id(tmp_path: Path) -> None:
    from web_api.mcp_tools import _mcp_client, _StdioServer

    command = _peer(
        tmp_path,
        "import time; time.sleep(30)",
        tool="generate_image",
    )

    async def run() -> str:
        client = _mcp_client({"transport": "stdio", "command": command})
        assert client is not None
        server = _StdioServer(
            FAVLO_AGENT_ID,
            client,
            name="Peer",
            tool_name_prefix="peer",
            approval_mode="never_require",
            allowed_tools=["generate_image"],
            load_prompts=False,
        )
        async with client:
            await server.connect()
            result = await asyncio.wait_for(
                server.call_tool("generate_image", prompt="a horse"),
                2,
            )
            await server.close()
        return _shown(result)

    text = asyncio.run(run())
    assert "task-1" in text
    assert "not ready" in text
    assert "no favlo:// link yet" in text
    assert "Do not call media_status." in text
    assert "Do not call another tool" not in text
    assert "filmstrip" not in text


def test_a_workspace_generate_keeps_the_file_name(tmp_path: Path) -> None:
    from types import SimpleNamespace

    from web_api.mcp_tools import _mcp_client, _StdioServer, _Turn

    command = _peer(tmp_path, "import time; time.sleep(30)", tool="generate_image")
    root = tmp_path / "folder"
    root.mkdir()
    store = SimpleNamespace(
        workspace_for_session=lambda _sid: {"id": "ws"},
        has_local_folder=lambda _wid: True,
        local_location=lambda _wid: str(root),
    )

    async def run() -> str:
        client = _mcp_client({"transport": "stdio", "command": command})
        assert client is not None
        server = _StdioServer(
            FAVLO_AGENT_ID,
            client,
            _Turn(store, None, set(), "srv", FAVLO_AGENT_ID, "sess"),
            name="Peer",
            tool_name_prefix="peer",
            approval_mode="never_require",
            allowed_tools=["generate_image"],
            load_prompts=False,
        )
        async with client:
            await server.connect()
            func = next(
                item
                for item in server._functions
                if (item.additional_properties or {}).get("_mcp_remote_name") == "generate_image"
            )
            result = await asyncio.wait_for(
                func.invoke(arguments={"prompt": "a horse", "filename": "star.png"}),
                2,
            )
            await server.close()
        return _shown(result)

    text = asyncio.run(run())
    assert "star.png" in text
    assert "task-1" not in text
    assert "not ready" in text
    assert "Pass a file name" not in text


class _ScopedClient:
    """An anyio cancel scope, the same constraint as the streamable HTTP client."""

    def __init__(self) -> None:
        self.enter_task: asyncio.Task[object] | None = None
        self.exit_task: asyncio.Task[object] | None = None
        self._scope: object | None = None

    async def __aenter__(self) -> _ScopedClient:
        import anyio

        self.enter_task = asyncio.current_task()
        self._scope = anyio.CancelScope()
        self._scope.__enter__()
        return self

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.exit_task = asyncio.current_task()
        assert self._scope is not None
        self._scope.__exit__(None, None, None)


def test_the_mcp_client_closes_on_the_task_that_opened_it() -> None:
    asyncio.run(_lease_survives_a_cancelled_turn())


async def _lease_survives_a_cancelled_turn() -> None:
    from types import SimpleNamespace

    from web_api.mcp_tools import _Lease

    client = _ScopedClient()
    idle = asyncio.Event()
    idle.set()
    server = SimpleNamespace(pending=0, idle=idle, jobs=set())
    lease = _Lease(client, server)
    opened = asyncio.Event()

    async def turn() -> None:
        await lease.__aenter__()
        opened.set()
        try:
            await asyncio.Event().wait()
        finally:
            await lease.__aexit__(None, None, None)

    job = asyncio.create_task(turn())
    await opened.wait()
    job.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await job
    assert lease._task is not None
    await lease._task
    await asyncio.sleep(0)
    assert client.enter_task is client.exit_task
    assert client.enter_task is not job
    assert not server.jobs



