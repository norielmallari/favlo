"""MCP servers, the tasks poll, and the session result."""

from __future__ import annotations

import asyncio
import json
import sqlite3
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "web-api"))

from web_api.bus import Bus  # noqa: E402
from web_api.db import FAVLO_AGENT_ID, FAVLO_SESSION_ID  # noqa: E402
from web_api.mcp_client import list_tools  # noqa: E402
from web_api.mcp_tasks import _finish, settle  # noqa: E402
from web_api.mcp_tools import INSTRUCTIONS, _description, call_tool, groups_from, prepare_catalog  # noqa: E402
from web_api.store import Store, _mcp_fields  # noqa: E402


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


def test_a_finished_task_starts_a_turn_and_stays_out_of_the_chat(tmp_path: Path) -> None:
    asyncio.run(_finished_task(tmp_path))


async def _finished_task(tmp_path: Path) -> None:
    seen = {"gets": 0}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802
            length = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(length).decode())
            method = body["method"]
            if method == "tools/call":
                result = {
                    "resultType": "task",
                    "taskId": "task-1",
                    "status": "working",
                    "pollIntervalMs": 20,
                }
            else:
                seen["gets"] += 1
                result = {
                    "resultType": "complete",
                    "taskId": "task-1",
                    "status": "completed",
                    "pollIntervalMs": 20,
                    "result": {
                        "content": [
                            {"type": "text", "text": "image_abc"},
                            {"type": "resource_link", "uri": "favlo://images/image_abc.png"},
                        ]
                    },
                }
            raw = json.dumps({"jsonrpc": "2.0", "id": body["id"], "result": result}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def log_message(self, _format: str, *_args: object) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]
    store = Store(tmp_path / "app.db")
    try:
        _status, created, _events = store.create_mcp_server(
            "mcp-key",
            {
                "agent_id": "agent_favlo",
                "name": "Favlo media",
                "transport": "http",
                "command": "",
                "url": f"http://127.0.0.1:{port}/mcp",
                "enabled": True,
            },
        )
        row = {
            "server_id": created["id"],
            "server_name": created["name"],
            "transport": created["transport"],
            "command": created["command"],
            "url": created["url"],
            "name": "generate_image",
            "description": "Generate one or more PNGs.",
            "schema": {"type": "object", "properties": {}},
        }
        pending: set[asyncio.Task[object]] = set()
        text = await call_tool(
            store,
            row,
            {"prompt": "a horse"},
            "session_favlo",
            "agent_favlo",
            Bus(),
            pending,
        )
        assert "task-1" in text
        assert pending
        await asyncio.wait_for(next(iter(pending)), timeout=2)
        snapshot = store.session_snapshot("session_favlo")
        bodies = [message["body"] for message in snapshot["messages"]]
        assert not any("favlo://images/image_abc.png" in body for body in bodies)
        assert not any(body.strip().startswith("{") for body in bodies)
        running = [row for row in snapshot["runs"] if row["status"] == "running"]
        assert len(running) == 1
        assert "favlo://images/image_abc.png" in running[0]["cue"]
        assert "Do not paste the JSON" in running[0]["cue"]
        assert seen["gets"] >= 1
    finally:
        server.shutdown()
        store.close()


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
                "agent_id": "agent_favlo",
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
        _text, enabled = asyncio.run(bundle(store, "agent_favlo", "session_favlo", None, set()))
        assert [item.name for item in enabled] == ["favlo_media_echo"]
    finally:
        store.close()


def test_media_instructions_require_a_listed_model_id() -> None:
    assert "set_image_model" in INSTRUCTIONS
    assert "set_video_i2v_model" in INSTRUCTIONS
    assert "set_video_t2v_model" in INSTRUCTIONS
    assert "set_music_model" in INSTRUCTIONS
    assert "set_voice_model" in INSTRUCTIONS
    assert "Repeat the returned session id" in INSTRUCTIONS
    assert "do not write a favlo:// link" in INSTRUCTIONS
    assert "Do not paste the JSON" in INSTRUCTIONS
    assert "tell the user you are generating" in INSTRUCTIONS
    described = _description(
        {"name": "generate_image", "description": "Generate one or more PNGs."}
    )
    assert "set_image_model" in described
    assert "session" in described


def test_a_finished_image_stays_finished() -> None:
    status, text = settle("generate_image", "completed", "favlo://images/image_abc.png")
    assert status == "completed"
    assert text.startswith("generate_image finished.")
    cancelled, note = settle("generate_image", "cancelled", "")
    assert cancelled == "cancelled"
    assert note == "generate_image was cancelled."


def test_an_unknown_model_result_starts_a_correction_turn(tmp_path: Path) -> None:
    store = Store(tmp_path / "app.db")
    try:
        task = store.begin_mcp_task(
            {
                "remote_id": "task-1",
                "server_id": "server-1",
                "agent_id": FAVLO_AGENT_ID,
                "session_id": FAVLO_SESSION_ID,
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
        snapshot = store.session_snapshot(FAVLO_SESSION_ID)
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
            FAVLO_SESSION_ID, FAVLO_AGENT_ID, "again"
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
                "agent_id": "agent_favlo",
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
                "agent_id": "agent_favlo",
                "name": "Favlo media",
                "transport": "http",
                "command": "",
                "url": "http://127.0.0.1:9000/mcp",
                "enabled": True,
            },
        )
        text, tools = asyncio.run(bundle(store, "agent_favlo", "session_favlo", None, set()))
        names = [item.name for item in tools]
        assert names == ["favlo_media_generate_image", "favlo_media_2_generate_image"]
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


def test_an_old_catalog_table_becomes_group_switches(tmp_path: Path) -> None:
    path = tmp_path / "app.db"
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
        ) VALUES (
            'mcp_old', 'agent_favlo', 'Favlo media', 'http', '',
            'http://127.0.0.1:8129/mcp', 1, 'connected', '', 1
        )
        """
    )
    raw.execute(
        """
        INSERT INTO mcp_tool (server_id, name, group_id, description, schema)
        VALUES (
            'mcp_old', 'generate_image', 'image', 'Generate.',
            '{"type":"object","required":["model"]}'
        )
        """
    )
    raw.execute(
        """
        INSERT INTO mcp_tool_group (server_id, group_id, name, enabled)
        VALUES ('mcp_old', 'image', 'Image', 0)
        """
    )
    raw.commit()
    raw.close()
    store = Store(path)
    try:
        server = store.mcp_server("mcp_old")
        assert server is not None
        assert server["groups"] == [
            {"id": "image", "name": "Image", "enabled": False, "tools": []}
        ]
        kept = store.enabled_mcp_servers("agent_favlo")
        assert kept[0]["switches"] == {"image": False}
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


def test_a_memory_call_keeps_the_agents_own_client(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, object] = {}

    async def fake_rpc(_server: dict[str, object], _method: str, params: dict[str, object], **_kwargs: object) -> dict[str, object]:
        seen["arguments"] = params["arguments"]
        return {"content": [{"type": "text", "text": "[]"}]}

    monkeypatch.setattr("web_api.mcp_tools.rpc", fake_rpc)

    async def run() -> None:
        await call_tool(
            None,
            {
                "server_id": "server",
                "server_name": "Favlo",
                "transport": "http",
                "command": "",
                "url": "http://127.0.0.1:9/mcp",
                "name": "recall",
                "description": "Recall.",
                "schema": {},
            },
            {"query": "tea", "client": "agent_other"},
            "session_favlo",
            "agent_favlo",
            None,
            set(),
        )

    asyncio.run(run())
    assert seen["arguments"] == {"query": "tea", "client": "agent_favlo"}
