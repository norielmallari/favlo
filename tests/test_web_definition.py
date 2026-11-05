"""A chat freezes its agent definition on the first turn."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "web-api"))

from web_api.agent_factory import from_definition, rebuild_tools  # noqa: E402
from web_api.ids import FAVLO_AGENT_ID  # noqa: E402
from web_api.definition import build, hash_of  # noqa: E402
from web_api.store import Store  # noqa: E402


def test_freeze_definition_writes_once(tmp_path: Path) -> None:
    store = Store(tmp_path / "app.db")
    home = next(row["session_id"] for row in store.channels() if row["agent_id"] == FAVLO_AGENT_ID)
    first = {
        "version": 1,
        "instructions": "You are Favlo.",
        "hash": "abc",
        "options": {"store": False},
    }
    assert store.freeze_definition(home, first) is True
    assert store.definition(home)["instructions"] == "You are Favlo."
    second = {
        "version": 1,
        "instructions": "Changed.",
        "hash": "def",
        "options": {"store": False},
    }
    assert store.freeze_definition(home, second) is False
    assert store.definition(home)["instructions"] == "You are Favlo."
    cleared, _events = store.clear_chat(home)
    assert store.definition(home) is None
    assert store.definition(cleared["session_id"]) is None
    store.close()


def test_definition_build_sets_store_true(tmp_path: Path) -> None:
    store = Store(tmp_path / "app.db")
    home = next(row["session_id"] for row in store.channels() if row["agent_id"] == FAVLO_AGENT_ID)
    agent = next(row for row in store.agents() if row["id"] == FAVLO_AGENT_ID)
    payload = build(
        store=store,
        agent=agent,
        session_id=home,
        base_url="http://127.0.0.1:8129/v1",
        model="example-model",
    )
    assert payload["options"]["store"] is True
    assert payload["hash"] == hash_of(payload)
    assert isinstance(payload["instructions"], str) and payload["instructions"]
    store.close()


def test_agent_factory_uses_frozen_instructions(tmp_path: Path, monkeypatch) -> None:
    store = Store(tmp_path / "app.db")
    home = next(row["session_id"] for row in store.channels() if row["agent_id"] == FAVLO_AGENT_ID)
    client = MagicMock()
    client.function_invocation_configuration = {}
    monkeypatch.setattr("web_api.agent_factory.open_chat_client", lambda *_a, **_k: client)
    monkeypatch.setattr(
        "web_api.agent_factory.percent_window",
        lambda _percent: MagicMock(name="window"),
    )
    captured: dict[str, object] = {}

    class FakeAgent:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.client = kwargs["client"]

    monkeypatch.setattr("agent_framework.Agent", FakeAgent)
    monkeypatch.setattr("agent_framework.InMemoryHistoryProvider", lambda **_k: MagicMock())
    definition = {
        "instructions": "Frozen persona text.",
        "tool_groups": [],
        "options": {"reasoning": {"effort": "none"}, "store": True},
        "model": {"model_id": "example-model", "base_url": "http://127.0.0.1:8129/v1"},
        "agent": {"id": FAVLO_AGENT_ID, "name": "Favlo"},
        "max_iterations": 6,
        "mcp": [],
        "workspace": None,
    }
    agent = from_definition(definition=definition, store=store, session_id=home)
    assert captured["instructions"] == "Frozen persona text."
    assert captured["default_options"]["store"] is True
    assert agent.client is client
    store.close()


def test_rebuild_tools_skips_live_skills(tmp_path: Path) -> None:
    store = Store(tmp_path / "app.db")
    home = next(row["session_id"] for row in store.channels() if row["agent_id"] == FAVLO_AGENT_ID)
    tools = rebuild_tools(
        {"tool_groups": [], "workspace": None, "mcp": []},
        store=store,
        session_id=home,
        bus=None,
        base_url="http://127.0.0.1:8129/v1",
        agent_id=FAVLO_AGENT_ID,
        mcp_tools=[],
    )
    names = [getattr(item, "name", None) or getattr(item, "__name__", "") for item in tools]
    assert names == ["display_media"]
    store.close()
