"""The web API keeps each agent on its own channel."""

from __future__ import annotations

import asyncio
import json
import sqlite3
import sys
import threading
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "web-api"))

from web_api.ids import (  # noqa: E402
    FAVLO_AGENT_ID,
    FAVLO_MCP_ID,
    LOCAL_PROVIDER_ID,
    is_guid,
)
from web_api.api import create_app  # noqa: E402
from web_api.discovery import DiscoveryFailed, ids_from_payload  # noqa: E402



def _loc(tmp_path: Path, name: str = "gallery-root") -> str:
    root = tmp_path / name
    root.mkdir(exist_ok=True)
    return str(root)


def _favlo_session(store_or_client) -> str:
    """The Home chat GUID for the seeded Favlo agent."""
    if hasattr(store_or_client, "channels"):
        channels = store_or_client.channels()
    else:
        channels = store_or_client.get("/channels").json()["channels"]
    match = next(row for row in channels if row["agent_id"] == FAVLO_AGENT_ID)
    return str(match["session_id"])



async def _reply(**_kwargs: object):
    yield "Hello"
    yield " there"


@pytest.fixture
def client(tmp_path: Path):
    app = create_app(tmp_path, reply=_reply)
    with TestClient(app) as test_client:
        yield test_client


def test_the_default_agent_is_the_horse(client: TestClient) -> None:
    from favlo.persona import PERSONA

    agents = client.get("/agents").json()["agents"]
    favlo = next(row for row in agents if row["id"] == FAVLO_AGENT_ID)
    assert favlo["name"] == "Favlo"
    assert favlo["persona"] == PERSONA
    assert favlo["default_model_id"] is None
    assert favlo["enable_thinking"] is None
    assert favlo["preserve_thinking"] is None
    assert favlo["reasoning_effort"] is None
    assert favlo["tool_groups"] == ["favlo-config"]
    channels = client.get("/channels").json()["channels"]
    match = next(row for row in channels if row["agent_id"] == FAVLO_AGENT_ID)
    assert match["name"] == "Favlo"
    assert is_guid(match["session_id"])


def test_the_initial_model_comes_from_the_registry(tmp_path: Path) -> None:
    from web_api.store import Store

    path = tmp_path / "app.db"
    store = Store(path)
    store.ensure_initial_model("example-model")
    settings = store.settings()
    listed = store.providers()[0]
    store.ensure_initial_model("other-model")
    again = store.settings()
    store.close()
    assert settings["default_model_id"]
    assert [row["model_id"] for row in listed["models"]] == ["example-model"]
    assert again["default_model_id"] == settings["default_model_id"]


def test_the_default_agent_is_seeded_once(tmp_path: Path) -> None:
    from web_api.store import Store

    path = tmp_path / "app.db"
    first = Store(path)
    first.close()
    second = Store(path)
    names = [row["name"] for row in second.agents()]
    second.close()
    assert names.count("Favlo") == 1


def test_favlo_media_mcp_is_seeded_once(tmp_path: Path) -> None:
    from web_api.db import LOCAL_MCP_URL
    from web_api.store import Store

    path = tmp_path / "app.db"
    first = Store(path)
    servers = first.mcp_servers()
    first.close()
    assert len(servers) == 1
    assert servers[0]["id"] == FAVLO_MCP_ID
    assert servers[0]["agent_id"] == FAVLO_AGENT_ID
    assert servers[0]["name"] == "Favlo media"
    assert servers[0]["transport"] == "http"
    assert servers[0]["url"] == LOCAL_MCP_URL
    assert servers[0]["enabled"] is True
    second = Store(path)
    again = second.mcp_servers()
    second.close()
    assert [row["id"] for row in again] == [FAVLO_MCP_ID]


def test_session_commands_stay_on_the_session_that_sent_them(tmp_path: Path) -> None:
    from web_api.session_commands import ARCHIVE_CUE
    from web_api.store import Store

    store = Store(tmp_path / "app.db")
    try:
        home_id = _favlo_session(store)
        channel = next(row for row in store.channels() if row["session_id"] == home_id)
        _status, workspace, _events = store.create_workspace(
            "ws",
            {"channel_id": channel["id"], "name": "Stills", "type": "gallery", "location": _loc(tmp_path)},
        )
        workspace_id = workspace["session_id"]
        _status, channel_turn, _events = store.begin_message("channel-note", home_id, "Channel note")
        store.complete_run(channel_turn["run_id"], "Hi")
        _status, workspace_turn, _events = store.begin_message("workspace-note", workspace_id, "Workspace note")
        store.complete_run(workspace_turn["run_id"], "Hi")

        status, cleared, events = store.begin_message("clear-workspace", workspace_id, " /session clear ")
        assert status == 200
        assert cleared["cleared"] is True
        assert "run_id" not in cleared
        workspace_id = cleared["session_id"]
        assert events[0]["type"] == "session.cleared"
        assert events[0]["stream_id"] == f"session:{workspace_id}"
        assert store.session_snapshot(workspace_id)["messages"] == []
        assert _bodies(store, home_id) == ["Channel note", "Hi"]

        _status, again, _events = store.begin_message("workspace-again", workspace_id, "Still here")
        store.complete_run(again["run_id"], "Hi")
        status, cleared, events = store.begin_message("clear-channel", home_id, "/session clear")
        assert status == 200
        home_id = cleared["session_id"]
        assert events[0]["stream_id"] == f"session:{home_id}"
        assert store.session_snapshot(home_id)["messages"] == []
        assert _bodies(store, workspace_id) == ["Still here", "Hi"]

        _status, remembered, _events = store.begin_message("remember", workspace_id, "Remember the red door")
        store.complete_run(remembered["run_id"], "Noted")
        status, archived, events = store.begin_message("archive", workspace_id, "/session archive")
        assert status == 202
        assert archived["run_id"]
        assert "message_id" not in archived
        assert events[0]["type"] == "run.started"
        context = store.run_context(archived["run_id"])
        assert context["user_text"] == ARCHIVE_CUE
        assert [item["text"] for item in context["prior"]] == ["Still here", "Hi", "Remember the red door", "Noted"]
        done = store.complete_run(archived["run_id"], "Stored.")
        assert [event["type"] for event in done] == ["run.completed", "session.cleared"]
        workspace_id = done[-1]["payload"]["session_id"]
        assert done[-1]["stream_id"] == f"session:{workspace_id}"
        assert store.session_snapshot(workspace_id)["messages"] == []
        assert store.session_snapshot(home_id)["messages"] == []

        _status, kept, _events = store.begin_message("keep", workspace_id, "Keep me")
        store.complete_run(kept["run_id"], "Ok")
        _status, failed_archive, _events = store.begin_message("archive-fail", workspace_id, "/session archive")
        failed = store.finish_run(failed_archive["run_id"], "failed", "The listener stopped.")
        assert "session.cleared" not in [event["type"] for event in failed]
        assert "Keep me" in _bodies(store, workspace_id)

        _status, empty_archive, _events = store.begin_message("archive-empty", workspace_id, "/session archive")
        emptied = store.finish_archive(empty_archive["run_id"])
        assert emptied is not None
        assert emptied[-1]["type"] == "session.cleared"
        workspace_id = emptied[-1]["payload"]["session_id"]
        assert store.session_snapshot(workspace_id)["messages"] == []
        assert store.finish_archive(failed_archive["run_id"]) is None

        status, ordinary, _events = store.begin_message(
            "ordinary",
            home_id,
            "please /session clear",
        )
        assert status == 202
        assert ordinary["text"] == "please /session clear"
    finally:
        store.close()


def _bodies(store: object, session_id: str) -> list[str]:
    from web_api.store import Store

    assert isinstance(store, Store)
    snapshot = store.session_snapshot(session_id)
    return [row["body"] for row in snapshot["messages"]]


def test_the_configuration_group_adds_its_instructions() -> None:
    from web_api.chat import turn_setup

    plain, idle = turn_setup("You are River.", "River", [], "http://127.0.0.1:8129/v1")
    assert "read_setup" not in plain
    assert idle == []
    text, enabled = turn_setup(
        "You are Favlo.",
        "Favlo",
        ["favlo-config"],
        "http://127.0.0.1:8129/v1",
    )
    assert "read_setup" in text
    assert "You live in this terminal" not in text
    assert [item.name for item in enabled] == ["config_load_tools"]
    assert "config_load_tools" in text


def test_conversation_tools_read_every_transcript(tmp_path: Path) -> None:
    from web_api.chat import turn_setup
    from web_api.conversation import (
        READ_MAX,
        SEARCH_MAX,
        _cap,
        read_recent_messages,
        search_messages,
    )
    from web_api.store import Store

    assert _cap(1000, 20, READ_MAX) == 50
    assert _cap(1000, 10, SEARCH_MAX) == 20
    path = tmp_path / "app.db"
    store = Store(path)
    created = store.create_agent(
        "river-key",
        {"name": "River", "persona": "A short persona.", "default_model_id": None},
    )
    river = created[1]["channel"]["session_id"]
    home = _favlo_session(store)
    store.close()
    _note(path, home, "m-favlo", "user", "The garden is open.", "2026-10-27T01:00:00Z")
    _note(path, river, "m-river-1", "user", "Bring the oats.", "2026-10-27T02:00:00Z")
    _note(path, river, "m-river-2", "assistant", "The oats are in.", "2026-10-27T02:01:00Z")
    _session(path, "session_hall", "channel", "channel_hall", "group", "Hall")
    _note(path, "session_hall", "m-hall", "user", "The hall is quiet.", "2026-10-27T03:00:00Z")
    _session(path, "session_desk", "workspace_chat", None, None, None)
    _note(path, "session_desk", "m-desk", "user", "Bring the carrots.", "2026-10-27T04:00:00Z")
    _session(path, "session_hall_2", "channel", "channel_hall_2", "group", "Hall")
    store = Store(path)
    found = search_messages(store, "garden", "agent", 10)
    assert home in found
    assert "The garden is open." in found
    assert search_messages(store, "garden", "workspace_chat", 10) == "No messages matched."
    carrots = search_messages(store, "carrots", "workspace_chat", 10)
    assert "session_desk" in carrots
    assert home not in carrots
    hall = search_messages(store, "hall", "channel", 10)
    assert "session_hall" in hall
    assert search_messages(store, "hall", "agent", 10) == "No messages matched."
    assert search_messages(store, "%", "all", 10) == "No messages matched."
    latest = read_recent_messages(store, "River", 1)
    assert "The oats are in." in latest
    assert "Bring the oats." not in latest
    both = read_recent_messages(store, river, 20)
    assert both.index("Bring the oats.") < both.index("The oats are in.")
    ambiguous = read_recent_messages(store, "Hall", 5)
    assert "More than one conversation matches." in ambiguous
    text, enabled = turn_setup(
        "You are River.",
        "River",
        ["conversation"],
        "http://127.0.0.1:8129/v1",
        store,
    )
    assert "search_messages" in text
    assert "read_recent_messages" in text
    assert [item.name for item in enabled] == ["conversation_load_tools"]
    assert "conversation_load_tools" in text
    bare, idle = turn_setup(
        "You are River.",
        "River",
        ["conversation"],
        "http://127.0.0.1:8129/v1",
    )
    assert idle == []
    assert "search_messages" not in bare
    store.close()


def _note(path: Path, session_id: str, message_id: str, role: str, body: str, created: str) -> None:
    conn = sqlite3.connect(path)
    conn.execute(
        """
        INSERT INTO message (id, session_id, role, body, actor_kind, actor_id, created_at)
        VALUES (?, ?, ?, ?, 'user', 'local', ?)
        """,
        (message_id, session_id, role, body, created),
    )
    conn.commit()
    conn.close()


def _session(
    path: Path,
    session_id: str,
    kind: str,
    channel_id: str | None,
    channel_kind: str | None,
    name: str | None,
) -> None:
    conn = sqlite3.connect(path)
    if channel_id is not None:
        conn.execute(
            """
            INSERT INTO channel (id, kind, name, revision)
            VALUES (?, ?, ?, 1)
            """,
            (channel_id, channel_kind, name),
        )
    conn.execute(
        """
        INSERT INTO session (id, kind, channel_id, participant_id, created_at)
        VALUES (?, ?, ?, NULL, '2026-10-27T00:00:00Z')
        """,
        (session_id, kind, channel_id),
    )
    conn.commit()
    conn.close()


def test_a_new_agent_stores_the_chosen_tools(client: TestClient) -> None:
    created = client.post(
        "/agents",
        headers={"Idempotency-Key": "tools-on"},
        json={
            "name": "Reed",
            "persona": "A short persona.",
            "default_model_id": None,
            "tool_groups": ["heartbeats", "conversation"],
        },
    )
    assert created.status_code == 201
    assert created.json()["agent"]["tool_groups"] == ["conversation", "heartbeats"]
    rejected = client.post(
        "/agents",
        headers={"Idempotency-Key": "tools-bad"},
        json={
            "name": "Reed",
            "persona": "A short persona.",
            "default_model_id": None,
            "tool_groups": ["missing"],
        },
    )
    assert rejected.status_code == 400
    assert rejected.json()["message"] == "That tool group is not available."


def test_a_new_agent_leaves_configuration_tools_off(client: TestClient) -> None:
    created = _agent(client, "River", "tools-off")
    assert created["agent"]["tool_groups"] == []
    listed = client.get("/tool-groups").json()["groups"]
    assert listed == [
        {"id": "favlo-config", "name": "Configuration tools"},
        {"id": "conversation", "name": "Conversation tools"},
        {"id": "heartbeats", "name": "Heartbeat tools"},
    ]
    agents = client.get("/agents").json()["agents"]
    favlo = next(row for row in agents if row["id"] == FAVLO_AGENT_ID)
    saved = client.patch(
        f"/agents/{favlo['id']}",
        json={
            "expected_revision": favlo["revision"],
            "name": favlo["name"],
            "persona": favlo["persona"],
            "default_model_id": None,
            "tool_groups": [],
        },
    )
    assert saved.status_code == 200
    assert saved.json()["tool_groups"] == []
    unknown = client.patch(
        f"/agents/{created['agent']['id']}",
        json={
            "expected_revision": 1,
            "name": "River",
            "persona": "A short persona.",
            "default_model_id": None,
            "tool_groups": ["missing"],
        },
    )
    assert unknown.status_code == 400


def test_disabling_configuration_tools_stays_off(tmp_path: Path) -> None:
    from web_api.store import Store

    path = tmp_path / "app.db"
    first = Store(path)
    favlo = next(row for row in first.agents() if row["id"] == FAVLO_AGENT_ID)
    first.update_agent(favlo["id"], favlo["revision"], "Favlo", favlo["persona"], None, [])
    first.close()
    second = Store(path)
    again = next(row for row in second.agents() if row["id"] == FAVLO_AGENT_ID)
    second.close()
    assert again["tool_groups"] == []


def test_favlo_runs_with_the_configuration_group(tmp_path: Path) -> None:
    seen: list[list[str]] = []

    async def reply(**kwargs: object):
        groups = kwargs.get("tool_groups")
        seen.append(list(groups) if isinstance(groups, list) else [])
        yield "Hello"

    app = create_app(tmp_path, reply=reply)
    with TestClient(app) as test_client:
        model = test_client.post(
            "/provider-models",
            headers={"Idempotency-Key": "model-tools"},
            json={
                "provider_id": LOCAL_PROVIDER_ID,
                "model_id": "example-model",
                "display_name": "Example",
            },
        )
        assert model.status_code == 201
        settings = test_client.get("/settings").json()
        saved = test_client.patch(
            "/settings",
            json={
                "expected_revision": settings["revision"],
                "default_model_id": model.json()["id"],
                "exchange_budget": 10,
            },
        )
        assert saved.status_code == 200
        sent = test_client.post(
            "/sessions/" + _favlo_session(test_client) + "/messages",
            headers={"Idempotency-Key": "favlo-tools"},
            json={"text": "Hello"},
        )
        assert sent.status_code == 202
        _wait(test_client, _favlo_session(test_client))
    assert seen == [["favlo-config"]]


def test_two_agents_get_separate_channels(client: TestClient) -> None:
    first = _agent(client, "River", "key-river")
    second = _agent(client, "Cedar", "key-cedar")
    assert first["channel"]["session_id"] != second["channel"]["session_id"]
    listed = client.get("/channels").json()["channels"]
    names = {row["name"]: row["session_id"] for row in listed}
    assert names["River"] == first["channel"]["session_id"]
    assert names["Cedar"] == second["channel"]["session_id"]


def test_the_same_idempotency_key_returns_the_stored_agent(client: TestClient) -> None:
    first = _agent(client, "River", "same-key")
    second = _agent(client, "River", "same-key")
    assert first["agent"]["id"] == second["agent"]["id"]
    changed = client.post(
        "/agents",
        headers={"Idempotency-Key": "same-key"},
        json={"name": "Other", "persona": "", "default_model_id": None},
    )
    assert changed.status_code == 409


def test_an_agent_stores_its_own_model(client: TestClient) -> None:
    created_model = client.post(
        "/provider-models",
        headers={"Idempotency-Key": "agent-model"},
        json={
            "provider_id": LOCAL_PROVIDER_ID,
            "model_id": "example-model",
            "display_name": "Example",
        },
    )
    assert created_model.status_code == 201
    created = _agent(client, "River", "agent-model-key")
    saved = client.patch(
        f"/agents/{created['agent']['id']}",
        json={
            "expected_revision": 1,
            "name": "River",
            "persona": "A short persona.",
            "default_model_id": created_model.json()["id"],
            "tool_groups": [],
        },
    )
    assert saved.status_code == 200
    assert saved.json()["default_model_id"] == created_model.json()["id"]
    listed = client.get("/agents").json()["agents"]
    river = next(row for row in listed if row["id"] == created["agent"]["id"])
    assert river["default_model_id"] == created_model.json()["id"]


def test_a_stale_agent_revision_is_a_conflict(client: TestClient) -> None:
    created = _agent(client, "River", "rev-key")
    agent_id = created["agent"]["id"]
    stale = client.patch(
        f"/agents/{agent_id}",
        json={
            "expected_revision": 4,
            "name": "River",
            "persona": "Updated",
            "default_model_id": None,
        },
    )
    assert stale.status_code == 409


def test_the_browser_origin_is_the_only_cors_origin(client: TestClient) -> None:
    allowed = client.get("/health", headers={"Origin": "http://127.0.0.1:8012"})
    assert allowed.headers["access-control-allow-origin"] == "http://127.0.0.1:8012"
    blocked = client.get("/health", headers={"Origin": "http://127.0.0.1:9999"})
    assert "access-control-allow-origin" not in blocked.headers


def test_a_missing_model_fails_in_view(client: TestClient, tmp_path: Path) -> None:
    created = _agent(client, "River", "no-model")
    session_id = created["channel"]["session_id"]
    sent = client.post(
        f"/sessions/{session_id}/messages",
        headers={"Idempotency-Key": "msg-1"},
        json={"text": "Hello"},
    )
    assert sent.status_code == 202
    snapshot = _wait(client, session_id)
    assert snapshot["runs"][-1]["status"] == "failed"
    assert "model" in snapshot["runs"][-1]["error"].lower()
    stored = sqlite3.connect(tmp_path / "app.db")
    kinds = [row[0] for row in stored.execute("SELECT type FROM event")]
    assert "run.delta" not in kinds


def test_a_reply_is_stored_on_that_agents_session(client: TestClient, tmp_path: Path) -> None:
    model = client.post(
        "/provider-models",
        headers={"Idempotency-Key": "model-1"},
        json={
            "provider_id": LOCAL_PROVIDER_ID,
            "model_id": "example-model",
            "display_name": "Example",
        },
    )
    assert model.status_code == 201
    settings = client.get("/settings").json()
    saved = client.patch(
        "/settings",
        json={
            "expected_revision": settings["revision"],
            "default_model_id": model.json()["id"],
            "exchange_budget": 10,
        },
    )
    assert saved.status_code == 200
    created = _agent(client, "River", "with-model")
    other = _agent(client, "Cedar", "other-agent")
    session_id = created["channel"]["session_id"]
    sent = client.post(
        f"/sessions/{session_id}/messages",
        headers={"Idempotency-Key": "msg-2"},
        json={"text": "Hello"},
    )
    assert sent.status_code == 202
    snapshot = _wait(client, session_id)
    bodies = [row["body"] for row in snapshot["messages"]]
    assert bodies == ["Hello", "Hello there"]
    assert [row["reasoning"] for row in snapshot["messages"]] == [None, None]
    other_snapshot = client.get(f"/sessions/{other['channel']['session_id']}").json()
    assert other_snapshot["messages"] == []
    stored = sqlite3.connect(tmp_path / "app.db")
    kinds = [row[0] for row in stored.execute("SELECT type FROM event")]
    assert "run.delta" not in kinds


def test_reasoning_is_stored_beside_the_reply(tmp_path: Path) -> None:
    from web_api.chat import ReplyPiece

    async def reply(**_kwargs: object):
        yield ReplyPiece("reasoning", "Looking")
        yield ReplyPiece("content", "Hello")

    app = create_app(tmp_path, reply=reply)
    with TestClient(app) as test_client:
        model = test_client.post(
            "/provider-models",
            headers={"Idempotency-Key": "model-reason"},
            json={
                "provider_id": LOCAL_PROVIDER_ID,
                "model_id": "example-model",
                "display_name": "Example",
            },
        )
        assert model.status_code == 201
        settings = test_client.get("/settings").json()
        saved = test_client.patch(
            "/settings",
            json={
                "expected_revision": settings["revision"],
                "default_model_id": model.json()["id"],
                "exchange_budget": 10,
            },
        )
        assert saved.status_code == 200
        sent = test_client.post(
            "/sessions/" + _favlo_session(test_client) + "/messages",
            headers={"Idempotency-Key": "reason-1"},
            json={"text": "Hello"},
        )
        assert sent.status_code == 202
        snapshot = _wait(test_client, _favlo_session(test_client))
    assistant = snapshot["messages"][-1]
    assert assistant["body"] == "Hello"
    assert assistant["reasoning"] == "Looking"
    assert assistant["trace"] == [
        {"kind": "thinking", "text": "Looking"},
        {"kind": "reply", "text": "Hello"},
    ]
    stored = sqlite3.connect(tmp_path / "app.db")
    payload = stored.execute(
        "SELECT body FROM event WHERE type = 'message.created' ORDER BY sequence DESC LIMIT 1"
    ).fetchone()[0]
    assert json.loads(payload)["payload"]["reasoning"] == "Looking"


def test_a_streamed_tool_call_joins_on_its_index() -> None:
    from agent_framework import Content

    from web_api.chat import record_tool, tool_pieces

    class Update:
        contents = [
            Content.from_function_call(
                "call_1",
                "conversation_list_conversations",
                arguments='{"kind":',
                additional_properties={"tool_call_index": 0},
            )
        ]

    calls: list[dict[str, object]] = []
    record_tool(calls, tool_pieces(Update())[0])
    follow = Content.from_function_call(
        "",
        "",
        arguments=' "all"}',
        additional_properties={"tool_call_index": 0},
    )
    record_tool(calls, tool_pieces(type("Later", (), {"contents": [follow]})())[0])
    done = Content.from_function_result("call_1", result="No conversations are stored.")
    record_tool(calls, tool_pieces(type("Result", (), {"contents": [done]})())[0])
    assert calls[0]["name"] == "conversation_list_conversations"
    assert calls[0]["arguments"] == '{"kind": "all"}'
    assert calls[0]["result"] == "No conversations are stored."


def test_a_tool_call_is_stored_on_the_message(tmp_path: Path) -> None:
    from web_api.chat import ReplyPiece

    async def reply(**_kwargs: object):
        yield ReplyPiece("tool", '{"kind":', "call_1", "conversation_list_conversations", 0)
        yield ReplyPiece("tool", ' "all"}', "", "", 0)
        yield ReplyPiece("tool_result", "No conversations are stored.", "call_1")
        yield ReplyPiece("content", "Done")

    app = create_app(tmp_path, reply=reply)
    with TestClient(app) as test_client:
        model = test_client.post(
            "/provider-models",
            headers={"Idempotency-Key": "model-tool"},
            json={
                "provider_id": LOCAL_PROVIDER_ID,
                "model_id": "example-model",
                "display_name": "Example",
            },
        )
        assert model.status_code == 201
        settings = test_client.get("/settings").json()
        saved = test_client.patch(
            "/settings",
            json={
                "expected_revision": settings["revision"],
                "default_model_id": model.json()["id"],
                "exchange_budget": 10,
            },
        )
        assert saved.status_code == 200
        sent = test_client.post(
            "/sessions/" + _favlo_session(test_client) + "/messages",
            headers={"Idempotency-Key": "tool-1"},
            json={"text": "Hello"},
        )
        assert sent.status_code == 202
        snapshot = _wait(test_client, _favlo_session(test_client))
    assistant = snapshot["messages"][-1]
    assert assistant["body"] == "Done"
    assert assistant["trace"][0]["kind"] == "tool"
    assert assistant["trace"][1] == {"kind": "reply", "text": "Done"}
    assert assistant["tool_calls"] == [
        {
            "id": "call_1",
            "name": "conversation_list_conversations",
            "arguments": '{"kind": "all"}',
            "result": "No conversations are stored.",
        }
    ]


def test_reasoning_without_text_finishes_the_turn(tmp_path: Path) -> None:
    from web_api.store import Store

    store = Store(tmp_path / "app.db")
    home = _favlo_session(store)
    _status, started, _events = store.begin_message("empty", home, "hello")
    events = store.complete_run(started["run_id"], "", "User handed over carrots.")
    prior = store.run_context(started["run_id"])
    store.close()
    created = next(event for event in events if event["type"] == "message.created")
    completed = next(event for event in events if event["type"] == "run.completed")
    assert created["payload"]["reasoning"] == "User handed over carrots."
    assert created["payload"]["text"] == ""
    assert completed["payload"]["status"] == "completed"
    assert prior["run"]["status"] == "completed"


def test_unset_sampling_stays_off_the_request() -> None:
    from web_api.chat import sampling_options

    assert sampling_options(None, None, None) is None
    assert sampling_options(None, None, "") is None


def test_a_set_sampling_field_is_the_only_extra() -> None:
    from web_api.chat import sampling_options

    assert sampling_options(True, None, None) == {"reasoning": {"effort": "medium"}}
    assert sampling_options(False, None, None) == {"reasoning": {"effort": "none"}}
    assert sampling_options(False, True, "high") == {
        "reasoning": {"effort": "high"},
        "extra_body": {"preserve_thinking": True},
    }
    assert sampling_options(None, False, None) == {
        "extra_body": {"preserve_thinking": False},
    }


def test_agent_sampling_fields_round_trip(client: TestClient) -> None:
    created = client.post(
        "/agents",
        headers={"Idempotency-Key": "sample-fields"},
        json={
            "name": "Reed",
            "persona": "",
            "enable_thinking": True,
            "preserve_thinking": False,
            "reasoning_effort": "high",
        },
    )
    assert created.status_code == 201
    agent = created.json()["agent"]
    assert agent["enable_thinking"] is True
    assert agent["preserve_thinking"] is False
    assert agent["reasoning_effort"] == "high"
    saved = client.patch(
        f"/agents/{agent['id']}",
        json={
            "expected_revision": agent["revision"],
            "name": "Reed",
            "persona": "",
            "enable_thinking": None,
            "preserve_thinking": None,
            "reasoning_effort": None,
        },
    )
    assert saved.status_code == 200
    body = saved.json()
    assert body["enable_thinking"] is None
    assert body["preserve_thinking"] is None
    assert body["reasoning_effort"] is None


def test_the_next_turn_includes_stored_reasoning(tmp_path: Path) -> None:
    from web_api.store import Store

    store = Store(tmp_path / "app.db")
    home = _favlo_session(store)
    _status, first, _events = store.begin_message("one", home, "hello")
    store.complete_run(first["run_id"], "Hi", "Looking")
    _status, second, _events = store.begin_message("two", home, "again")
    prior = store.run_context(second["run_id"])["prior"]
    store.close()
    assert prior == [
        {"role": "user", "text": "hello", "reasoning": ""},
        {"role": "assistant", "text": "Hi", "reasoning": "Looking"},
    ]


def test_reasoning_content_stays_out_of_the_reply() -> None:
    from agent_framework import Content

    from favlo.chat_client import read_reasoning
    from web_api.chat import assistant_piece, reasoning_piece

    class Delta:
        reasoning_content = "Looking"

    parsed = read_reasoning(Delta(), [Content.from_text(text="Hi")])

    class Update:
        role = "assistant"
        contents = parsed

        @property
        def text(self) -> str:
            return "Hi"

    assert reasoning_piece(Update()) == "Looking"
    assert assistant_piece(Update()) == "Hi"
    assert read_reasoning(object(), parsed) == parsed


def test_session_cancel_stops_every_running_turn(tmp_path: Path) -> None:
    started = threading.Event()
    released = threading.Event()

    async def hanging(**kwargs: object):
        started.set()
        cancel = kwargs["cancel"]
        assert isinstance(cancel, asyncio.Event)
        await cancel.wait()
        released.set()
        return
        yield "nope"

    app = create_app(tmp_path, reply=hanging)
    with TestClient(app) as client:
        model = client.post(
            "/provider-models",
            headers={"Idempotency-Key": "model-session-cancel"},
            json={
                "provider_id": LOCAL_PROVIDER_ID,
                "model_id": "example-model",
                "display_name": "Example",
            },
        )
        assert model.status_code == 201
        settings = client.get("/settings").json()
        saved = client.patch(
            "/settings",
            json={
                "expected_revision": settings["revision"],
                "default_model_id": model.json()["id"],
                "exchange_budget": 10,
            },
        )
        assert saved.status_code == 200
        session_id = _favlo_session(client)
        sent = client.post(
            f"/sessions/{session_id}/messages",
            headers={"Idempotency-Key": "session-cancel-1"},
            json={"text": "Hello"},
        )
        assert sent.status_code == 202
        assert started.wait(2)
        stopped = client.post(f"/sessions/{session_id}/cancel")
        assert stopped.status_code == 200
        assert stopped.json()["cancelled_run_ids"] == [sent.json()["run_id"]]
        assert released.wait(2)
        snapshot = _wait(client, session_id)
    assert snapshot["runs"][-1]["status"] == "cancelled"


def test_stop_ends_a_run_that_is_still_waiting(tmp_path: Path) -> None:
    started = threading.Event()
    released = threading.Event()

    async def hanging(**kwargs: object):
        started.set()
        cancel = kwargs["cancel"]
        assert isinstance(cancel, asyncio.Event)
        await cancel.wait()
        released.set()
        return
        yield "nope"

    app = create_app(tmp_path, reply=hanging)
    with TestClient(app) as client:
        model = client.post(
            "/provider-models",
            headers={"Idempotency-Key": "model-stop"},
            json={
                "provider_id": LOCAL_PROVIDER_ID,
                "model_id": "example-model",
                "display_name": "Example",
            },
        )
        assert model.status_code == 201
        settings = client.get("/settings").json()
        saved = client.patch(
            "/settings",
            json={
                "expected_revision": settings["revision"],
                "default_model_id": model.json()["id"],
                "exchange_budget": 10,
            },
        )
        assert saved.status_code == 200
        sent = client.post(
            "/sessions/" + _favlo_session(client) + "/messages",
            headers={"Idempotency-Key": "stop-1"},
            json={"text": "Hello"},
        )
        assert sent.status_code == 202
        assert started.wait(2)
        stopped = client.post(f"/runs/{sent.json()['run_id']}/cancel")
        assert stopped.status_code == 202
        assert released.wait(2)
        snapshot = _wait(client, _favlo_session(client))
    assert snapshot["runs"][-1]["status"] == "cancelled"


def test_stop_closes_the_agent_framework_stream(monkeypatch: pytest.MonkeyPatch) -> None:
    from web_api.chat import stream_reply

    closed = {"stream": False, "http": False}

    class Stream:
        def __init__(self) -> None:
            self._wait = asyncio.Event()

        def __aiter__(self):
            return self

        async def __anext__(self):
            await self._wait.wait()
            raise StopAsyncIteration

        async def close(self) -> None:
            closed["stream"] = True
            self._wait.set()

    class Session:
        def __init__(self) -> None:
            self.state: dict[str, object] = {}

    class Agent:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def create_session(self) -> Session:
            return Session()

        def run(self, *_args: object, **_kwargs: object) -> Stream:
            return Stream()

    class HTTP:
        def __init__(self, **_kwargs: object) -> None:
            pass

        async def close(self) -> None:
            closed["http"] = True

    class Client:
        def __init__(self) -> None:
            self.client = HTTP()
            self.function_invocation_configuration: dict[str, object] = {}

        def close(self) -> None:
            return None

    monkeypatch.setattr("agent_framework.Agent", Agent)
    monkeypatch.setattr("web_api.chat.open_chat_client", lambda *_args, **_kwargs: Client())

    async def once() -> None:
        cancel = asyncio.Event()
        task = asyncio.create_task(_collect_reply(cancel))
        await asyncio.sleep(0.05)
        cancel.set()
        pieces = await asyncio.wait_for(task, 1)
        assert pieces == []
        assert closed["stream"]
        assert closed["http"]

    async def _collect_reply(cancel: asyncio.Event) -> list[object]:
        pieces = []
        async for piece in stream_reply(
            base_url="http://127.0.0.1:8129/v1",
            model="example-model",
            persona="A short persona.",
            agent_name="Favlo",
            user_text="Hello",
            cancel=cancel,
        ):
            pieces.append(piece)
        return pieces

    asyncio.run(once())


def test_a_cancel_event_still_streams_the_reply(monkeypatch: pytest.MonkeyPatch) -> None:
    from types import SimpleNamespace

    from web_api.chat import stream_reply

    class Update:
        role = "assistant"
        contents: list[object] = []
        raw_representation = None

        def __init__(self, text: str) -> None:
            self.text = text

    async def pieces():
        yield Update("Hel")
        yield Update("lo")

    class Stream:
        def __init__(self) -> None:
            self._it = pieces()

        def __aiter__(self):
            return self._it

        async def close(self) -> None:
            return None

        async def get_final_response(self) -> SimpleNamespace:
            return SimpleNamespace(text="Hello", messages=[])

    class Session:
        def __init__(self) -> None:
            self.state: dict[str, object] = {}

    class Agent:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def create_session(self) -> Session:
            return Session()

        def run(self, *_args: object, **_kwargs: object) -> Stream:
            return Stream()

    class HTTP:
        def __init__(self, **_kwargs: object) -> None:
            pass

        async def close(self) -> None:
            return None

    class Client:
        def __init__(self) -> None:
            self.client = HTTP()
            self.function_invocation_configuration: dict[str, object] = {}

        def close(self) -> None:
            return None

    monkeypatch.setattr("agent_framework.Agent", Agent)
    monkeypatch.setattr("web_api.chat.open_chat_client", lambda *_args, **_kwargs: Client())

    async def once() -> None:
        collected = []
        async for piece in stream_reply(
            base_url="http://127.0.0.1:8129/v1",
            model="example-model",
            persona="A short persona.",
            agent_name="Favlo",
            user_text="Hello",
            cancel=asyncio.Event(),
        ):
            collected.append(piece.text)
        assert collected == ["Hel", "lo"]

    asyncio.run(once())


def test_a_turn_sends_the_stored_agent_framework_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from types import SimpleNamespace

    from agent_framework import AgentSession, InMemoryHistoryProvider, Message

    from web_api.chat import stream_reply
    from web_api.gallery_tools import turn_note
    from web_api.store import Store

    store = Store(tmp_path / "app.db")
    home = _favlo_session(store)
    session = AgentSession(session_id=home)
    provider = InMemoryHistoryProvider()

    async def seed() -> None:
        state = session.state.setdefault(provider.source_id, {})
        await provider.save_messages(
            session.session_id,
            [Message("user", ["Hello"]), Message("assistant", ["Hi"])],
            state=state,
        )

    asyncio.run(seed())
    store.save_maf_session(home, session.to_dict())
    seen: dict[str, object] = {}

    class Stream:
        def __aiter__(self):
            return self

        async def __anext__(self):
            raise StopAsyncIteration

        async def close(self) -> None:
            return None

        async def get_final_response(self) -> SimpleNamespace:
            return SimpleNamespace(text="Again", messages=[])

    class Agent:
        def __init__(self, **kwargs: object) -> None:
            seen["instructions"] = kwargs.get("instructions")
            seen["options"] = kwargs.get("default_options")
            seen["persist"] = kwargs.get("require_per_service_call_history_persistence")
            seen["providers"] = kwargs.get("context_providers")

        def create_session(self, **_kwargs: object) -> object:
            raise AssertionError("a stored session is restored")

        def run(self, prompt: object, session: object = None, stream: bool = True) -> Stream:
            seen["prompt"] = prompt
            seen["session"] = session
            return Stream()

    class HTTP:
        async def close(self) -> None:
            return None

    class Client:
        def __init__(self) -> None:
            self.client = HTTP()
            self.function_invocation_configuration: dict[str, object] = {}

        def close(self) -> None:
            return None

    monkeypatch.setattr("agent_framework.Agent", Agent)
    monkeypatch.setattr("web_api.chat.open_chat_client", lambda *_args, **_kwargs: Client())
    note = turn_note("", "photos/horse.png", strip_count=2)

    async def once() -> None:
        collected = []
        async for piece in stream_reply(
            base_url="http://127.0.0.1:8129/v1",
            model="example-model",
            persona="A short persona.",
            agent_name="Favlo",
            user_text="Next",
            store=store,
            session_id=home,
            cancel=asyncio.Event(),
            gallery_note=note,
        ):
            collected.append(piece.text)
        assert collected == ["Again"]

    try:
        asyncio.run(once())
        prompt = str(seen["prompt"])
        restored = seen["session"]
        assert isinstance(restored, AgentSession)
        assert "Next" in prompt
        assert "photos/horse.png" in prompt
        assert "photos/horse.png" not in str(seen["instructions"])
        assert seen["options"] == {"store": False}
        assert seen["persist"] is True
        providers = seen["providers"]
        assert isinstance(providers, list)
        assert type(providers[0]).__name__ == "InMemoryHistoryProvider"
        assert providers[0].skip_excluded is True
        assert type(providers[1]).__name__ == "PercentWindow"
        messages = restored.state[provider.source_id]["messages"]
        assert [item.text for item in messages] == ["Hello", "Hi"]
        saved = store.maf_session(home)
        assert saved is not None
        assert saved["session_id"] == home
        assert "maf_state" not in store.session_snapshot(home)["session"]
    finally:
        store.close()


def test_a_cancelled_turn_leaves_the_stored_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from agent_framework import AgentSession

    from web_api.chat import stream_reply
    from web_api.store import Store

    store = Store(tmp_path / "app.db")
    home = _favlo_session(store)
    store.save_maf_session(
        home,
        AgentSession(session_id=home).to_dict(),
    )

    class Stream:
        def __init__(self) -> None:
            self._wait = asyncio.Event()

        def __aiter__(self):
            return self

        async def __anext__(self):
            await self._wait.wait()
            raise StopAsyncIteration

        async def close(self) -> None:
            self._wait.set()

    class Agent:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def create_session(self, **_kwargs: object) -> object:
            raise AssertionError("a stored session is restored")

        def run(self, _prompt: object, session: object = None, stream: bool = True) -> Stream:
            assert isinstance(session, AgentSession)
            session.state["marked"] = True
            return Stream()

    class HTTP:
        async def close(self) -> None:
            return None

    class Client:
        def __init__(self) -> None:
            self.client = HTTP()
            self.function_invocation_configuration: dict[str, object] = {}

        def close(self) -> None:
            return None

    monkeypatch.setattr("agent_framework.Agent", Agent)
    monkeypatch.setattr("web_api.chat.open_chat_client", lambda *_args, **_kwargs: Client())

    async def once() -> None:
        cancel = asyncio.Event()
        task = asyncio.create_task(_collect(cancel))
        await asyncio.sleep(0.05)
        cancel.set()
        await asyncio.wait_for(task, 1)

    async def _collect(cancel: asyncio.Event) -> None:
        async for _piece in stream_reply(
            base_url="http://127.0.0.1:8129/v1",
            model="example-model",
            persona="A short persona.",
            agent_name="Favlo",
            user_text="Hello",
            store=store,
            session_id=home,
            cancel=cancel,
        ):
            pass

    try:
        asyncio.run(once())
        saved = store.maf_session(home)
        assert saved is not None
        assert "marked" not in saved["state"]
    finally:
        store.close()


def test_clearing_a_session_drops_its_agent_framework_session(tmp_path: Path) -> None:
    from web_api.store import Store

    store = Store(tmp_path / "app.db")
    try:
        home = _favlo_session(store)
        store.save_maf_session(
            home,
            {"type": "session", "session_id": home, "state": {}},
        )
        assert store.maf_session(home)["session_id"] == home
        cleared, _events = store.clear_chat(home)
        assert store.maf_session(home) is None
        assert store.maf_session(cleared["session_id"]) is None
    finally:
        store.close()


def test_an_older_message_table_gains_reasoning(tmp_path: Path) -> None:
    from web_api.db import connect, migrate

    path = tmp_path / "app.db"
    conn = connect(path)
    conn.execute(
        """
        CREATE TABLE message (
            id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            body TEXT NOT NULL,
            actor_kind TEXT NOT NULL,
            actor_id TEXT,
            created_at TEXT NOT NULL
        )
        """
    )
    conn.commit()
    conn.close()
    again = connect(path)
    migrate(again)
    names = {row[1] for row in again.execute("PRAGMA table_info(message)")}
    again.close()
    assert "reasoning" in names
    assert "trace" in names


def test_model_ids_come_from_the_models_payload() -> None:
    assert ids_from_payload(
        {"data": [{"id": "example-model"}, {"id": "example-model"}, {"id": ""}, "nope"]}
    ) == ["example-model"]
    assert ids_from_payload({"data": None}) == []


def test_the_favlo_provider_is_present_and_uses_its_models(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def advertised(_base_url: str) -> list[str]:
        return ["example-model"]

    monkeypatch.setattr("web_api.routes.providers.fetch_model_ids", advertised)
    listed = client.get("/providers")
    assert listed.status_code == 200
    favlo = listed.json()["providers"][0]
    assert favlo["name"] == "Favlo"
    assert favlo["base_url"] == "http://127.0.0.1:8129/v1"
    assert [row["model_id"] for row in favlo["models"]] == ["example-model"]
    again = client.get("/providers")
    assert [row["model_id"] for row in again.json()["providers"][0]["models"]] == ["example-model"]
    settings = client.get("/settings").json()
    chosen = client.post(
        "/providers/" + LOCAL_PROVIDER_ID + "/models",
        json={"model_id": "example-model", "expected_revision": settings["revision"]},
    )
    assert chosen.status_code == 200
    assert chosen.json()["model_id"] == "example-model"
    rejected = client.post(
        "/providers/" + LOCAL_PROVIDER_ID + "/models",
        json={"model_id": "other-model", "expected_revision": chosen.json()["revision"]},
    )
    assert rejected.status_code == 400


def test_refresh_replaces_the_hosted_models(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = {"count": 0}

    async def advertised(_base_url: str) -> list[str]:
        calls["count"] += 1
        if calls["count"] == 1:
            return ["example-model"]
        return ["example-model-2"]

    monkeypatch.setattr("web_api.routes.providers.fetch_model_ids", advertised)
    client.get("/providers")
    refreshed = client.post("/providers/" + LOCAL_PROVIDER_ID + "/refresh")
    assert refreshed.status_code == 200
    assert refreshed.json()["message"] == "Models refreshed"
    assert [row["model_id"] for row in refreshed.json()["models"]] == ["example-model-2"]
    stored = client.get("/providers/" + LOCAL_PROVIDER_ID + "/models")
    assert stored.json()["models"] == ["example-model-2"]


def test_a_failed_discovery_leaves_the_list_empty(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = {"count": 0}

    async def down(_base_url: str) -> list[str]:
        calls["count"] += 1
        raise DiscoveryFailed("Connection failed")

    monkeypatch.setattr("web_api.routes.providers.fetch_model_ids", down)
    listed = client.get("/providers")
    favlo = listed.json()["providers"][0]
    assert favlo["name"] == "Favlo"
    assert favlo["models"] == []
    client.get("/providers")
    assert calls["count"] == 1
    refreshed = client.post("/providers/" + LOCAL_PROVIDER_ID + "/refresh")
    assert refreshed.json()["message"] == "Connection failed"
    assert refreshed.json()["models"] == []
    assert calls["count"] == 2


def test_a_heartbeat_is_stored_on_its_agent(client: TestClient) -> None:
    created = client.post(
        "/heartbeats",
        headers={"Idempotency-Key": "beat"},
        json={
            "agent_id": FAVLO_AGENT_ID,
            "cron_expression": "15 * * * *",
            "timezone": "UTC",
            "instructions": "Check the garden.",
            "enabled": True,
        },
    )
    assert created.status_code == 201
    body = created.json()
    assert body["agent_id"] == FAVLO_AGENT_ID
    assert body["cron_expression"] == "15 * * * *"
    assert body["timezone"] == "UTC"
    assert body["enabled"] is True
    listed = client.get("/heartbeats").json()["heartbeats"]
    assert listed[0]["id"] == body["id"]
    saved = client.patch(
        f"/heartbeats/{body['id']}",
        json={
            "expected_revision": 1,
            "cron_expression": "*/15 * * * *",
            "timezone": "UTC",
            "instructions": "Check the garden.",
            "enabled": False,
        },
    )
    assert saved.status_code == 200
    assert saved.json()["cron_expression"] == "*/15 * * * *"
    assert saved.json()["enabled"] is False
    assert saved.json()["revision"] == 2


def test_a_heartbeat_rejects_a_schedule_finer_than_one_minute(client: TestClient) -> None:
    rejected = client.post(
        "/heartbeats",
        headers={"Idempotency-Key": "fine"},
        json={
            "agent_id": FAVLO_AGENT_ID,
            "cron_expression": "* * * * * *",
            "timezone": "UTC",
            "instructions": "",
            "enabled": True,
        },
    )
    assert rejected.status_code == 400
    assert rejected.json()["message"] == "A cron expression has five fields."
    minute = client.post(
        "/heartbeats",
        headers={"Idempotency-Key": "minute"},
        json={
            "agent_id": FAVLO_AGENT_ID,
            "cron_expression": "* * * * *",
            "timezone": "UTC",
            "instructions": "",
            "enabled": True,
        },
    )
    assert minute.status_code == 201
    unknown = client.post(
        "/heartbeats",
        headers={"Idempotency-Key": "zone"},
        json={
            "agent_id": FAVLO_AGENT_ID,
            "cron_expression": "0 * * * *",
            "timezone": "Not/AZone",
            "instructions": "",
            "enabled": True,
        },
    )
    assert unknown.status_code == 400
    assert unknown.json()["message"] == "That timezone is not recognized."
    missing = client.post(
        "/heartbeats",
        headers={"Idempotency-Key": "gone"},
        json={
            "agent_id": "agent_missing",
            "cron_expression": "0 * * * *",
            "timezone": "UTC",
            "instructions": "",
            "enabled": True,
        },
    )
    assert missing.status_code == 404


def test_run_now_keeps_one_session_and_skips_a_busy_one(tmp_path: Path) -> None:
    from web_api.heartbeat_tools import add_heartbeat, list_heartbeats
    from web_api.store import Conflict

    app = create_app(tmp_path, reply=_reply)
    with TestClient(app):
        store = app.state.store
        added = add_heartbeat(store, FAVLO_AGENT_ID, "15 * * * *", "Check the garden.", "UTC")
        assert "runs 15 * * * *" in added
        assert "Check the garden." in list_heartbeats(store, FAVLO_AGENT_ID)
        assert list_heartbeats(store, "agent_other") == "No heartbeats are stored."
        assert "five fields" in add_heartbeat(store, FAVLO_AGENT_ID, "nope", "Later.", "UTC")
        row = store.heartbeats()[0]
        status, result, events = store.begin_heartbeat_run("run-1", row["id"])
        assert status == 202
        assert events
        assert result["text"] == "Check the garden."
        again = store.begin_heartbeat_run("run-1", row["id"])
        assert again[0] == 202
        assert again[1]["run_id"] == result["run_id"]
        with pytest.raises(Conflict):
            store.begin_heartbeat_run("run-2", row["id"])
        second = store.begin_heartbeat_run("run-1", row["id"])
        assert second[1]["session_id"] == result["session_id"]


def test_an_agent_changes_and_deletes_only_its_own_heartbeat(tmp_path: Path) -> None:
    from web_api.heartbeat_tools import (
        add_heartbeat,
        delete_heartbeat,
        list_heartbeats,
        update_heartbeat,
    )

    app = create_app(tmp_path, reply=_reply)
    with TestClient(app) as client:
        store = app.state.store
        other = _agent(client, "Cedar", "cedar-beat")
        add_heartbeat(store, FAVLO_AGENT_ID, "15 * * * *", "Check the garden.", "UTC")
        row = next(item for item in store.heartbeats() if item["agent_id"] == FAVLO_AGENT_ID)
        changed = update_heartbeat(
            store,
            FAVLO_AGENT_ID,
            row["id"],
            cron_expression="0 9 * * *",
            instructions="Water the garden.",
            enabled=False,
        )
        assert "0 9 * * *" in changed
        assert "disabled" in changed
        assert "Water the garden." in list_heartbeats(store, FAVLO_AGENT_ID)
        assert "five fields" in update_heartbeat(
            store, FAVLO_AGENT_ID, row["id"], cron_expression="nope"
        )
        add_heartbeat(store, other["agent"]["id"], "0 * * * *", "Stay.", "UTC")
        foreign = next(
            item for item in store.heartbeats() if item["agent_id"] == other["agent"]["id"]
        )
        missing = "That heartbeat does not exist."
        assert update_heartbeat(store, FAVLO_AGENT_ID, foreign["id"], instructions="No.") == missing
        assert delete_heartbeat(store, FAVLO_AGENT_ID, foreign["id"]) == missing
        removed = delete_heartbeat(store, FAVLO_AGENT_ID, row["id"])
        assert removed == f"Heartbeat {row['id']} was deleted."
        assert list_heartbeats(store, FAVLO_AGENT_ID) == "No heartbeats are stored."
        assert "Stay." in list_heartbeats(store, other["agent"]["id"])


def test_deleting_an_agent_removes_its_downstream_rows(client: TestClient, tmp_path: Path) -> None:
    model = client.post(
        "/provider-models",
        headers={"Idempotency-Key": "delete-model"},
        json={
            "provider_id": LOCAL_PROVIDER_ID,
            "model_id": "example-model",
            "display_name": "Example",
        },
    )
    assert model.status_code == 201
    settings = client.get("/settings").json()
    saved = client.patch(
        "/settings",
        json={
            "expected_revision": settings["revision"],
            "default_model_id": model.json()["id"],
            "exchange_budget": 10,
        },
    )
    assert saved.status_code == 200
    created = _agent(client, "Reed", "delete-reed")
    kept = _agent(client, "Cedar", "delete-cedar")
    agent_id = created["agent"]["id"]
    session_id = created["channel"]["session_id"]
    channel_id = created["channel"]["id"]
    patched = client.patch(
        f"/agents/{agent_id}",
        json={
            "expected_revision": created["agent"]["revision"],
            "name": "Reed",
            "persona": "A short persona.",
            "default_model_id": None,
            "tool_groups": ["conversation"],
        },
    )
    assert patched.status_code == 200
    beat = client.post(
        "/heartbeats",
        headers={"Idempotency-Key": "delete-beat"},
        json={
            "agent_id": agent_id,
            "cron_expression": "15 * * * *",
            "timezone": "UTC",
            "instructions": "Check the garden.",
            "enabled": True,
        },
    )
    assert beat.status_code == 201
    skill = client.post(
        "/agent-skills",
        headers={"Idempotency-Key": "delete-skill"},
        json={
            "agent_id": agent_id,
            "name": "Photo captions",
            "description": "Write a one-line caption.",
        },
    )
    assert skill.status_code == 201
    started = client.post(
        f"/heartbeats/{beat.json()['id']}/run",
        headers={"Idempotency-Key": "delete-run"},
    )
    assert started.status_code == 202
    heartbeat_session = started.json()["session_id"]
    _wait(client, heartbeat_session)
    server = client.post(
        "/mcp-servers",
        headers={"Idempotency-Key": "delete-mcp"},
        json={
            "agent_id": agent_id,
            "name": "Notes",
            "transport": "http",
            "url": "http://127.0.0.1:9/mcp",
            "tools": [
                {
                    "name": "generate_image",
                    "description": "Draw",
                    "schema": {"type": "object"},
                }
            ],
        },
    )
    assert server.status_code == 201
    client.app.state.store.begin_mcp_task(
        {
            "remote_id": "task-1",
            "server_id": server.json()["id"],
            "agent_id": agent_id,
            "session_id": session_id,
            "tool_name": "generate_image",
            "poll_interval_ms": 1000,
        }
    )
    sent = client.post(
        f"/sessions/{session_id}/messages",
        headers={"Idempotency-Key": "delete-message"},
        json={"text": "Hello"},
    )
    assert sent.status_code == 202
    _wait(client, session_id)

    removed = client.delete(f"/agents/{agent_id}")
    assert removed.status_code == 200
    assert removed.json()["name"] == "Reed"
    again = client.delete(f"/agents/{agent_id}")
    assert again.status_code == 404
    refused = client.delete("/agents/" + FAVLO_AGENT_ID)
    assert refused.status_code == 400
    assert refused.json()["message"] == "Favlo stays."
    missing = client.delete("/agents/agent_missing")
    assert missing.status_code == 404

    names = [row["name"] for row in client.get("/agents").json()["agents"]]
    assert "Reed" not in names
    assert "Cedar" in names
    assert "Favlo" in names
    channels = client.get("/channels").json()["channels"]
    assert channel_id not in [row["id"] for row in channels]
    assert kept["channel"]["id"] in [row["id"] for row in channels]

    stored = sqlite3.connect(tmp_path / "app.db")
    for table, column, value in (
        ("agent", "id", agent_id),
        ("agent_tool_group", "agent_id", agent_id),
        ("channel", "id", channel_id),
        ("channel_member", "agent_id", agent_id),
        ("session", "id", session_id),
        ("session", "id", heartbeat_session),
        ("message", "session_id", session_id),
        ("message", "session_id", heartbeat_session),
        ("run", "agent_id", agent_id),
        ("heartbeat", "agent_id", agent_id),
        ("agent_skill", "agent_id", agent_id),
        ("mcp_server", "agent_id", agent_id),
        ("mcp_task", "agent_id", agent_id),
        ("event", "stream_id", f"session:{session_id}"),
        ("event", "stream_id", f"session:{heartbeat_session}"),
    ):
        count = stored.execute(
            f"SELECT COUNT(*) FROM {table} WHERE {column} = ?",
            (value,),
        ).fetchone()[0]
        assert count == 0, table
    stored.close()


def _agent(client: TestClient, name: str, key: str) -> dict:
    response = client.post(
        "/agents",
        headers={"Idempotency-Key": key},
        json={"name": name, "persona": "A short persona.", "default_model_id": None},
    )
    assert response.status_code == 201
    return response.json()


def _wait(client: TestClient, session_id: str) -> dict:
    for _ in range(50):
        snapshot = client.get(f"/sessions/{session_id}").json()
        if snapshot["runs"] and snapshot["runs"][-1]["status"] != "running":
            return snapshot
        time.sleep(0.02)
    message = "the run did not finish"
    raise AssertionError(message)
