"""AG-UI SSE turns on the Favlo web API."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "web-api"))

from web_api.api import create_app  # noqa: E402
from web_api.chat import ReplyPiece  # noqa: E402
from web_api.ids import guid, FAVLO_AGENT_ID, LOCAL_PROVIDER_ID, is_guid  # noqa: E402


def _favlo_session(client: TestClient) -> str:
    channels = client.get("/channels").json()["channels"]
    match = next(row for row in channels if row["agent_id"] == FAVLO_AGENT_ID)
    return str(match["session_id"])


async def _reply(**_kwargs: object):
    yield ReplyPiece("content", "Hello")
    yield ReplyPiece("content", " there")


@pytest.fixture
def client(tmp_path: Path):
    app = create_app(tmp_path, reply=_reply)
    with TestClient(app) as test_client:
        _enable_model(test_client)
        yield test_client


def _enable_model(client: TestClient) -> None:
    model = client.post(
        "/provider-models",
        headers={"Idempotency-Key": "agui-model"},
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
            "exchange_budget": settings.get("exchange_budget", 10),
        },
    )
    assert saved.status_code == 200


def _sse_types(body: str) -> list[str]:
    found: list[str] = []
    for line in body.splitlines():
        if not line.startswith("data:"):
            continue
        payload = json.loads(line[5:].strip())
        event_type = payload.get("type")
        if isinstance(event_type, str):
            found.append(event_type)
    return found


def test_agui_text_turn_order(client: TestClient) -> None:
    thread = _favlo_session(client)
    run_id = guid()
    response = client.post(
        "/ag-ui",
        json={
            "threadId": thread,
            "runId": run_id,
            "messages": [{"role": "user", "content": "Hi"}],
        },
    )
    assert response.status_code == 200
    types = _sse_types(response.text)
    assert types[0] == "RUN_STARTED"
    assert "TEXT_MESSAGE_START" in types
    assert "TEXT_MESSAGE_CONTENT" in types
    assert "TEXT_MESSAGE_END" in types
    assert types[-1] == "RUN_FINISHED"
    snapshot = client.get(f"/sessions/{thread}").json()
    assert any(row["role"] == "user" and row["body"] == "Hi" for row in snapshot["messages"])
    assert any(row["role"] == "assistant" and "Hello" in row["body"] for row in snapshot["messages"])
    assert "state" in snapshot


async def _reply_with_reasoning(**_kwargs: object):
    yield ReplyPiece("reasoning", "Looking")
    yield ReplyPiece("content", "Hello")


def test_agui_reasoning_lands_on_the_assistant_row(tmp_path: Path) -> None:
    app = create_app(tmp_path, reply=_reply_with_reasoning)
    with TestClient(app) as client:
        _enable_model(client)
        thread = _favlo_session(client)
        response = client.post(
            "/ag-ui",
            json={
                "threadId": thread,
                "runId": guid(),
                "messages": [{"role": "user", "content": "Hi"}],
            },
        )
        assert response.status_code == 200
        types = _sse_types(response.text)
        assert "REASONING_MESSAGE_CONTENT" in types
        assert "TEXT_MESSAGE_CONTENT" in types
        snapshot = client.get(f"/sessions/{thread}").json()
        assistant = next(row for row in snapshot["messages"] if row["role"] == "assistant")
        assert assistant["reasoning"] == "Looking"
        assert "Hello" in assistant["body"]


def test_agui_rejects_second_running_turn(client: TestClient) -> None:
    thread = _favlo_session(client)
    store = client.app.state.store
    run_id = guid()
    store.begin_agui_run(thread, run_id, "first")
    response = client.post(
        "/ag-ui",
        json={
            "threadId": thread,
            "runId": guid(),
            "messages": [{"role": "user", "content": "second"}],
        },
    )
    assert response.status_code == 200
    assert "RUN_ERROR" in _sse_types(response.text)


def test_clear_route_mints_new_guid(client: TestClient) -> None:
    thread = _favlo_session(client)
    cleared = client.post(f"/sessions/{thread}/clear")
    assert cleared.status_code == 200
    body = cleared.json()
    assert body["status"] == "cleared"
    assert is_guid(body["session_id"])
    assert body["session_id"] != thread
    missing = client.get(f"/sessions/{thread}")
    assert missing.status_code == 404
    ok = client.get(f"/sessions/{body['session_id']}")
    assert ok.status_code == 200
    assert ok.json()["messages"] == []
    assert ok.json()["state"] == {}


def test_fenced_host_cues_stay_off_the_transcript() -> None:
    from web_api.host_cues import host_cue_message
    from web_api.snapshots import project_messages

    rows = project_messages(
        [
            {"id": "user", "role": "user", "content": "Show the meadow"},
            host_cue_message(
                "The selected media on this turn is horse.png. This image means that path.",
                message_id="cue",
            ),
            {"id": "reply", "role": "assistant", "content": "Here it is."},
        ]
    )
    assert [row["role"] for row in rows] == ["user", "assistant"]
    assert rows[0]["body"] == "Show the meadow"
    assert "horse.png" not in rows[0]["body"]
    assert rows[1]["body"] == "Here it is."


def test_empty_silent_user_rows_stay_off_the_transcript() -> None:
    from web_api.host_cues import host_cue_message
    from web_api.snapshots import project_messages

    rows = project_messages(
        [
            host_cue_message(
                "The user just created this gallery, named Demo.",
                message_id="cue",
            ),
            {"id": "pad", "role": "user", "content": ""},
            {"id": "reply", "role": "assistant", "content": "Welcome."},
        ]
    )
    assert [row["role"] for row in rows] == ["assistant"]
    assert rows[0]["body"] == "Welcome."


def test_tool_rounds_stay_on_the_transcript() -> None:
    from web_api.snapshots import project_messages

    earlier = {
        "id": "call-a",
        "type": "function",
        "function": {"name": "conversation_load_tools", "arguments": "{}"},
    }
    later = {
        "id": "call-b",
        "type": "function",
        "function": {"name": "conversation_list_conversations", "arguments": "{\"kind\": \"all\"}"},
    }
    rows = project_messages(
        [
            {"id": "user", "role": "user", "content": "List"},
            {"id": "tool-a", "role": "assistant", "tool_calls": [earlier]},
            {"id": "result-a", "role": "tool", "toolCallId": "call-a", "content": "loaded"},
            {"id": "tool-b", "role": "assistant", "tool_calls": [later]},
            {"id": "result-b", "role": "tool", "toolCallId": "call-b", "content": "[]"},
            {"id": "reply", "role": "assistant", "content": "Done"},
        ]
    )
    assistant = rows[-1]
    assert assistant["body"] == "Done"
    assert [call["name"] for call in assistant["tool_calls"]] == [
        "conversation_load_tools",
        "conversation_list_conversations",
    ]
    assert assistant["tool_calls"][1]["result"] == "[]"

    trailing = project_messages(
        [
            {"id": "user", "role": "user", "content": "List"},
            {"id": "reply", "role": "assistant", "content": "Now let me list the conversations: done"},
            {"id": "tool-b", "role": "assistant", "tool_calls": [later]},
            {"id": "result-b", "role": "tool", "toolCallId": "call-b", "content": "[]"},
        ]
    )
    stored = trailing[-1]
    assert stored["tool_calls"][0]["name"] == "conversation_list_conversations"
    assert stored["trace"][0]["kind"] == "tool"


def test_archive_arm_returns_run_id(client: TestClient) -> None:
    thread = _favlo_session(client)
    run_id = guid()
    armed = client.post(f"/sessions/{thread}/archive", json={"run_id": run_id})
    assert armed.status_code == 202
    assert armed.json()["run_id"] == run_id
    snapshot = client.get(f"/sessions/{thread}").json()
    assert any(row["id"] == run_id and row["status"] == "running" for row in snapshot["runs"])
