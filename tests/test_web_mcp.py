"""MCP servers, the tasks poll, and the session result."""

from __future__ import annotations

import asyncio
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

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


def test_a_finished_task_is_written_to_the_calling_session(tmp_path: Path) -> None:
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
        catalog = [
            {
                "name": "generate_image",
                "description": "Generate one or more PNGs.",
                "schema": {"type": "object", "properties": {}},
            }
        ]
        stored = store.store_mcp_catalog(created["id"], prepare_catalog(catalog))
        assert stored["groups"][0]["id"] == "image"
        assert stored["groups"][0]["tools"] == ["generate_image"]
        row = store.enabled_mcp_tools("agent_favlo")[0]
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
        assert any("favlo://images/image_abc.png" in body for body in bodies)
        assert seen["gets"] >= 1
    finally:
        server.shutdown()
        store.close()


def test_save_keeps_a_disabled_group_and_lists_its_tools(tmp_path: Path) -> None:
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
        assert image["tools"] == ["generate_image", "list_image_models", "set_image_model"]
        assert [row["name"] for row in store.enabled_mcp_tools("agent_favlo")] == ["echo"]
    finally:
        store.close()


def test_media_instructions_require_a_listed_model_id() -> None:
    assert "set_image_model" in INSTRUCTIONS
    assert "set_video_i2v_model" in INSTRUCTIONS
    assert "set_video_t2v_model" in INSTRUCTIONS
    assert "set_music_model" in INSTRUCTIONS
    assert "set_voice_model" in INSTRUCTIONS
    assert "Repeat the returned session id" in INSTRUCTIONS
    assert "Do not write a favlo:// link" in INSTRUCTIONS
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
        assert any(
            body.startswith("generate_image failed.") and "unknown model" in body for body in bodies
        )
        running = [row for row in snapshot["runs"] if row["status"] == "running"]
        assert len(running) == 1
        context = store.run_context(running[0]["id"])
        assert "list_image_models" in context["user_text"]
        assert "generate_image again" in context["user_text"]
        assert context["prior"][-1]["role"] == "assistant"
        assert "unknown model" in context["prior"][-1]["text"]
        again, _events, waiting = store.begin_task_correction(
            FAVLO_SESSION_ID, FAVLO_AGENT_ID, "again"
        )
        assert again is None
        assert waiting is True
    finally:
        store.close()
