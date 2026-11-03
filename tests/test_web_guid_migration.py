"""Every application record id is a GUID."""

from __future__ import annotations

import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "web-api"))

from web_api.ids import (  # noqa: E402
    FAVLO_AGENT_ID,
    FAVLO_CHANNEL_ID,
    FAVLO_MCP_ID,
    LOCAL_PROVIDER_ID,
    is_guid,
)
from web_api.store import Store  # noqa: E402


def test_seeded_rows_use_stable_guids(tmp_path: Path) -> None:
    store = Store(tmp_path / "app.db")
    try:
        assert is_guid(FAVLO_AGENT_ID)
        assert is_guid(FAVLO_CHANNEL_ID)
        assert is_guid(FAVLO_MCP_ID)
        assert is_guid(LOCAL_PROVIDER_ID)
        agent = next(row for row in store.agents() if row["id"] == FAVLO_AGENT_ID)
        assert agent["name"] == "Favlo"
        channel = next(row for row in store.channels() if row["id"] == FAVLO_CHANNEL_ID)
        assert channel["agent_id"] == FAVLO_AGENT_ID
        assert is_guid(channel["session_id"])
        provider = next(row for row in store.providers() if row["id"] == LOCAL_PROVIDER_ID)
        assert provider["name"] == "Favlo"
        servers = [row for row in store.mcp_servers() if row["id"] == FAVLO_MCP_ID]
        assert len(servers) == 1
        assert servers[0]["agent_id"] == FAVLO_AGENT_ID
    finally:
        store.close()


def test_clear_chat_mints_a_new_guid(tmp_path: Path) -> None:
    store = Store(tmp_path / "app.db")
    try:
        channel = next(row for row in store.channels() if row["agent_id"] == FAVLO_AGENT_ID)
        old_id = channel["session_id"]
        assert is_guid(old_id)
        store.begin_message("note", old_id, "keep me")
        result, events = store.clear_chat(old_id)
        new_id = result["session_id"]
        assert is_guid(new_id)
        assert new_id != old_id
        assert result["previous_session_id"] == old_id
        assert events[0]["type"] == "session.cleared"
        assert events[0]["payload"]["session_id"] == new_id
        assert store.session_snapshot(new_id)["messages"] == []
        try:
            store.session_snapshot(old_id)
            raised = False
        except Exception:
            raised = True
        assert raised
        again = next(row for row in store.channels() if row["agent_id"] == FAVLO_AGENT_ID)
        assert again["session_id"] == new_id
    finally:
        store.close()


def test_new_records_use_guids(tmp_path: Path) -> None:
    store = Store(tmp_path / "app.db")
    try:
        _status, created, _events = store.create_agent(
            "create-river",
            {"name": "River", "persona": "A guide."},
        )
        assert is_guid(created["agent"]["id"])
        assert is_guid(created["session"]["id"])
        assert is_guid(created["session"]["channel_id"])
        session_id = created["session"]["id"]
        _status, turn, _events = store.begin_message("hello", session_id, "hello")
        assert is_guid(turn["run_id"])
        uuid.UUID(turn["run_id"])
    finally:
        store.close()
