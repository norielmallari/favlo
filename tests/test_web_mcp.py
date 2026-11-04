"""MCP servers and the Agent Framework session."""

from __future__ import annotations

import asyncio
import sqlite3
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "web-api"))

from web_api.bus import Bus  # noqa: E402
from web_api.ids import FAVLO_AGENT_ID, FAVLO_MCP_ID, guid  # noqa: E402
from web_api.mcp_client import list_tools  # noqa: E402
from web_api.mcp_tasks import _finish, settle  # noqa: E402
from web_api.mcp_tools import INSTRUCTIONS, _description, groups_from, prepare_catalog  # noqa: E402
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
            {"name": "generate_image", "description": "Generate.", "schema": {"type": "object"}},
            {"name": "list_image_models", "description": "List.", "schema": {"type": "object"}},
            {"name": "set_image_model", "description": "Set.", "schema": {"type": "object"}},
            {"name": "echo", "description": "Echo.", "schema": {"type": "object"}},
        ]
    )
    groups = groups_from(catalog)
    assert groups[0]["tools"] == ["generate_image", "list_image_models", "set_image_model"]
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
                "groups": [{"id": "image", "enabled": False}],
                "tools": catalog,
            },
        )
        image = next(group for group in created["groups"] if group["id"] == "image")
        assert image["enabled"] is False
        assert image["tools"] == []
        from web_api.mcp_tools import bundle

        async def listed(_server: object, *, refresh: bool = False) -> list[dict[str, object]]:
            return [
                {
                    "name": "generate_image",
                    "description": "Generate.",
                    "schema": {"type": "object"},
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


def test_media_instructions_require_a_listed_model_id() -> None:
    assert "set_image_model" in INSTRUCTIONS
    assert "set_video_i2v_model" in INSTRUCTIONS
    assert "set_video_t2v_model" in INSTRUCTIONS
    assert "set_music_model" in INSTRUCTIONS
    assert "set_voice_model" in INSTRUCTIONS
    assert "progressive disclosure" in INSTRUCTIONS
    assert "load_tool" in INSTRUCTIONS
    assert "Match the generate tool to what the user asked for" in INSTRUCTIONS
    assert "Do not offer or insist on video when they asked for an image" in INSTRUCTIONS
    assert "Repeat the returned session id" in INSTRUCTIONS
    assert "returns when the task is accepted" in INSTRUCTIONS
    assert "Do not paste the JSON" in INSTRUCTIONS
    assert "tell the user you are generating" in INSTRUCTIONS
    described = _description(
        {"name": "generate_image", "description": "Generate one or more PNGs."}
    )
    assert "set_image_model" in described
    assert "session" in described


def test_mcp_always_load_covers_image_memory_and_media_tasks() -> None:
    from web_api.mcp_tools import ALWAYS_LOAD

    assert "generate_image" in ALWAYS_LOAD
    assert "remember" in ALWAYS_LOAD
    assert "media_status" in ALWAYS_LOAD
    assert "generate_video_i2v" not in ALWAYS_LOAD
    assert "generate_music" not in ALWAYS_LOAD
    assert "generate_voice" not in ALWAYS_LOAD


def test_dumps_for_model_omits_spaces() -> None:
    from favlo.json_model import dumps_for_model

    text = dumps_for_model({"a": 1, "b": "x"})
    assert text == '{"a":1,"b":"x"}'


def test_dumps_for_model_drops_empty_fields() -> None:
    from favlo.json_model import dumps_for_model

    text = dumps_for_model({"path": "a.png", "title": "", "tags": [], "score": 0.1})
    assert text == '{"path":"a.png","score":0.1}'


def test_progressive_mcp_server_flags_and_eager_mask() -> None:
    from web_api.mcp_tools import ALWAYS_LOAD, _server

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
        sorted(ALWAYS_LOAD | {"generate_video_i2v", "generate_music"}),
        sorted(ALWAYS_LOAD),
        None,
    )
    assert opened is not None
    # Replace the real client so construction stayed side-effect free for flags.
    opened.owned_client = _Client()
    assert opened.use_progressive_disclosure is True
    assert set(opened.always_load or []) == set(ALWAYS_LOAD)
    assert "generate_video_i2v" in set(opened.allowed_tools or [])
    assert "generate_video_i2v" not in set(opened.always_load or [])


def test_progressive_initial_functions_exclude_cold_media() -> None:
    """Eager progressive surface has loaders + always_load, not video/music generate."""
    from agent_framework import FunctionTool

    from web_api.mcp_tools import ALWAYS_LOAD, _server

    opened = _server(
        {
            "server_id": "s1",
            "server_name": "favlo",
            "transport": "http",
            "url": "http://127.0.0.1:8129/mcp",
        },
        "agent",
        "favlo",
        sorted(ALWAYS_LOAD | {"generate_video_i2v", "generate_music", "generate_voice"}),
        sorted(ALWAYS_LOAD),
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

    fakes = [
        _fake(f"favlo_{remote}", remote)
        for remote in sorted(ALWAYS_LOAD | {"generate_video_i2v", "generate_music", "generate_voice"})
    ]
    opened._filtered_functions = lambda: fakes  # type: ignore[method-assign]
    opened._progressive_loader_functions = None
    names = {func.name for func in opened._progressive_functions()}
    assert "favlo_load_tool" in names
    assert "favlo_list_mcp_tools" in names
    assert "favlo_generate_image" in names
    assert "favlo_generate_video_i2v" not in names
    assert "favlo_generate_music" not in names
    assert "favlo_generate_voice" not in names


def test_a_finished_image_stays_finished() -> None:
    status, text = settle("generate_image", "completed", "favlo://images/image_abc.png")
    assert status == "completed"
    assert text.startswith("generate_image finished.")
    cancelled, note = settle("generate_image", "cancelled", "")
    assert cancelled == "cancelled"
    assert note == "generate_image was cancelled."


def test_an_unknown_model_result_starts_a_correction_turn(tmp_path: Path) -> None:
    store = Store(tmp_path / "app.db")
    store.ensure_initial_model("example-model")
    try:
        task = store.begin_mcp_task(
            {
                "remote_id": "task-1",
                "server_id": "server-1",
                "agent_id": FAVLO_AGENT_ID,
                "session_id": _home(store),
                "tool_name": "generate_image",
                "poll_interval_ms": 1000,
            }
        )
        asyncio.run(
            _finish(
                store,
                Bus(),
                task,
                "completed",
                "Error calling tool 'generate_image': unknown model flux-schnell",
            )
        )
        snapshot = store.session_snapshot(_home(store))
        bodies = [message["body"] for message in snapshot["messages"]]
        assert not any("unknown model" in body for body in bodies)
        running = [row for row in snapshot["runs"] if row["status"] == "running"]
        assert len(running) == 1
        context = store.run_context(running[0]["id"])
        assert "list_image_models" in context["user_text"]
        assert "generate_image again" in context["user_text"]
        assert "unknown model" in context["user_text"]
        assert not any("unknown model" in row["text"] for row in context["prior"])
        again, _events, waiting = store.begin_task_correction(
            _home(store), FAVLO_AGENT_ID, "again"
        )
        assert again is None
        assert waiting is True
    finally:
        store.close()


def test_mcp_tools_from_two_servers_keep_unique_names(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from web_api.mcp_tools import bundle
    from web_api.store import Store

    async def listed(_server: object, *, refresh: bool = False) -> list[dict[str, object]]:
        return [
            {
                "name": "generate_image",
                "description": "Generate.",
                "schema": {"type": "object", "properties": {"prompt": {"type": "string"}}},
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
        assert all(list(item.allowed_tools) == ["generate_image"] for item in tools)
        assert "favlo_media_generate_image" in text
        described = _description(
            {"name": "generate_image", "description": "Generate."},
            "favlo_media_generate_image",
        )
        assert "favlo_media_list_image_models" in described
        assert "favlo_media_set_image_model" in described
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
        assert seeded["groups"] == []
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


def _peer(tmp_path: Path, tasks_get: str) -> str:
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
        "            'name': 'echo', 'description': 'Echo.',\n"
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


def test_a_modern_server_returns_when_the_task_is_accepted(tmp_path: Path) -> None:
    from web_api.mcp_tools import _mcp_client, _StdioServer

    command = _peer(
        tmp_path,
        "import time; time.sleep(30)\n        reply(msg, {'resultType': 'complete'})",
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
    assert "working" in text
    assert "task-1" in text
    assert "ready" not in text


def test_a_task_result_arrives_on_a_later_turn(tmp_path: Path) -> None:
    from web_api.mcp_tools import _mcp_client, _StdioServer, _Turn

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
    store = Store(tmp_path / "app.db")
    store.ensure_initial_model("example-model")
    jobs: set[asyncio.Task[object]] = set()
    try:

        async def run() -> str:
            client = _mcp_client({"transport": "stdio", "command": command})
            assert client is not None
            server = _StdioServer(
                FAVLO_AGENT_ID,
                client,
                _Turn(store, Bus(), jobs, "peer", FAVLO_AGENT_ID, _home(store)),
                name="Peer",
                tool_name_prefix="peer",
                approval_mode="never_require",
                allowed_tools=["echo"],
                load_prompts=False,
            )
            async with client:
                await server.connect()
                result = await asyncio.wait_for(server.call_tool("echo", text="hi"), 2)
                shown = _shown(result)
                for _ in range(40):
                    snapshot = store.session_snapshot(_home(store))
                    running = [row for row in snapshot["runs"] if row["status"] == "running"]
                    if running:
                        return shown + "\n" + store.run_context(running[0]["id"])["user_text"]
                    await asyncio.sleep(0.05)
            raise AssertionError("task did not finish")

        text = asyncio.run(run())
    finally:
        store.close()
    call, cue = text.split("\n", 1)
    assert "working" in call
    assert "ready" not in call
    assert "ready" in cue

