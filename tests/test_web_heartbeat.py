"""The heartbeat clock starts a run when the cron minute matches."""

from __future__ import annotations

import asyncio
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "web-api"))

from web_api.api import create_app  # noqa: E402
from web_api.ids import FAVLO_AGENT_ID, LOCAL_PROVIDER_ID  # noqa: E402
from web_api.bus import Bus  # noqa: E402
from web_api.cron import ScheduleError, due  # noqa: E402
from web_api.heartbeat_clock import seconds_until_next_minute, tick  # noqa: E402
from web_api.heartbeat_tools import add_heartbeat  # noqa: E402
from web_api.store import Store  # noqa: E402

UTC = ZoneInfo("UTC")
DUE = datetime(2024, 6, 15, 3, 30, tzinfo=UTC)


async def _reply(**_kwargs: object):
    yield "Hello"


def test_due_matches_a_minute_and_rejects_a_neighbor() -> None:
    assert due("30 3 15 6 *", DUE)
    assert due("0,30 3 15 6 *", DUE)
    assert due("*/15 3 15 6 *", DUE)
    assert due("0-45/15 3 15 6 *", DUE)
    assert not due("31 3 15 6 *", DUE)
    assert not due("30 4 15 6 *", DUE)
    assert not due("30 3 16 6 *", DUE)
    assert not due("30 3 15 7 *", DUE)


def test_sunday_matches_zero_and_seven() -> None:
    sunday = datetime(2024, 6, 16, 0, 0, tzinfo=UTC)
    monday = datetime(2024, 6, 17, 0, 0, tzinfo=UTC)
    assert due("0 0 * * 0", sunday)
    assert due("0 0 * * 7", sunday)
    assert not due("0 0 * * 0", monday)
    assert due("0 0 * * 1", monday)


def test_a_restricted_day_and_weekday_match_either() -> None:
    saturday = datetime(2024, 6, 15, 0, 0, tzinfo=UTC)
    monday = datetime(2024, 6, 17, 0, 0, tzinfo=UTC)
    sunday = datetime(2024, 6, 16, 0, 0, tzinfo=UTC)
    assert due("0 0 15 * 1", saturday)
    assert due("0 0 15 * 1", monday)
    assert not due("0 0 15 * 1", sunday)


def test_a_naive_clock_is_rejected() -> None:
    with pytest.raises(ScheduleError):
        due("* * * * *", datetime(2024, 6, 15, 3, 30))


def test_the_wait_lands_on_the_next_minute() -> None:
    assert seconds_until_next_minute(datetime(2024, 6, 15, 3, 30, 0)) == 60
    assert seconds_until_next_minute(datetime(2024, 6, 15, 3, 30, 15)) == 45


def test_a_due_heartbeat_starts_once_for_that_minute(tmp_path: Path) -> None:
    store = Store(tmp_path / "app.db")
    add_heartbeat(store, FAVLO_AGENT_ID, "30 3 15 6 *", "Check the garden.", "UTC")
    quiet = datetime(2024, 6, 15, 3, 29, tzinfo=UTC)
    assert store.dispatch_heartbeats(quiet) == []
    started = store.dispatch_heartbeats(DUE)
    assert len(started) == 1
    assert started[0][0]["text"] == "Check the garden."
    assert store.dispatch_heartbeats(DUE) == []
    assert "fired_minute" not in store.heartbeats()[0]
    store.close()


def test_a_busy_heartbeat_skips_that_minute(tmp_path: Path) -> None:
    store = Store(tmp_path / "app.db")
    add_heartbeat(store, FAVLO_AGENT_ID, "0,30 3 15 6 *", "Check the garden.", "UTC")
    row = store.heartbeats()[0]
    early = datetime(2024, 6, 15, 3, 0, tzinfo=UTC)
    _status, result, _events = store.begin_heartbeat_run("manual", row["id"])
    assert store.dispatch_heartbeats(early) == []
    store.finish_run(result["run_id"], "failed", "Stopped.")
    assert store.dispatch_heartbeats(early) == []
    started = store.dispatch_heartbeats(DUE)
    assert len(started) == 1
    assert started[0][0]["session_id"] == result["session_id"]
    store.close()


def test_one_check_starts_only_that_minute(tmp_path: Path) -> None:
    store = Store(tmp_path / "app.db")
    add_heartbeat(store, FAVLO_AGENT_ID, "0,30 3 15 6 *", "Check the garden.", "UTC")
    started = store.dispatch_heartbeats(DUE)
    assert len(started) == 1
    assert store.cancel_run(started[0][0]["run_id"])["status"] == "running"
    store.close()


def test_a_disabled_heartbeat_stays_quiet(tmp_path: Path) -> None:
    store = Store(tmp_path / "app.db")
    _status, created, _events = store.create_heartbeat(
        "quiet",
        {
            "agent_id": FAVLO_AGENT_ID,
            "cron_expression": "30 3 15 6 *",
            "timezone": "UTC",
            "instructions": "Check the garden.",
            "enabled": False,
        },
    )
    assert store.dispatch_heartbeats(DUE) == []
    store.update_heartbeat(
        created["id"],
        {
            "expected_revision": 1,
            "cron_expression": "30 3 15 6 *",
            "timezone": "UTC",
            "instructions": "Check the garden.",
            "enabled": True,
        },
    )
    assert len(store.dispatch_heartbeats(DUE)) == 1
    store.close()


def test_two_heartbeats_start_in_the_same_minute(tmp_path: Path) -> None:
    store = Store(tmp_path / "app.db")
    add_heartbeat(store, FAVLO_AGENT_ID, "30 3 15 6 *", "Check the garden.", "UTC")
    add_heartbeat(store, FAVLO_AGENT_ID, "30 3 15 6 6", "Water the garden.", "UTC")
    started = store.dispatch_heartbeats(DUE)
    texts = {item[0]["text"] for item in started}
    sessions = {item[0]["session_id"] for item in started}
    assert texts == {"Check the garden.", "Water the garden."}
    assert len(sessions) == 2
    store.close()


def test_the_clock_uses_the_heartbeat_timezone(tmp_path: Path) -> None:
    store = Store(tmp_path / "app.db")
    add_heartbeat(store, FAVLO_AGENT_ID, "0 9 * * *", "Shanghai morning.", "Asia/Shanghai")
    add_heartbeat(store, FAVLO_AGENT_ID, "0 9 * * *", "UTC morning.", "UTC")
    morning = datetime(2024, 6, 15, 1, 0, tzinfo=UTC)
    started = store.dispatch_heartbeats(morning)
    assert [item[0]["text"] for item in started] == ["Shanghai morning."]
    later = datetime(2024, 6, 15, 9, 0, tzinfo=UTC)
    started = store.dispatch_heartbeats(later)
    assert [item[0]["text"] for item in started] == ["UTC morning."]
    store.close()


def test_startup_fails_an_abandoned_heartbeat_run(tmp_path: Path) -> None:
    store = Store(tmp_path / "app.db")
    add_heartbeat(store, FAVLO_AGENT_ID, "30 3 15 6 *", "Check the garden.", "UTC")
    row = store.heartbeats()[0]
    _status, result, _events = store.begin_heartbeat_run("manual", row["id"])
    home = next(row["session_id"] for row in store.channels() if row["agent_id"] == FAVLO_AGENT_ID)
    _status, channel, _events = store.begin_message("chat", home, "Hello")
    assert set(store.abandoned_runs()) == {result["run_id"], channel["run_id"]}
    store.close()
    app = create_app(tmp_path, reply=_reply)
    with TestClient(app):
        live = app.state.store
        assert live.cancel_run(result["run_id"])["status"] == "failed"
        chat = live.run_context(channel["run_id"])["run"]
        assert chat["status"] == "failed"
        assert chat["error"] == "The server stopped during that run."
        assert app.state.heartbeat_stop.is_set() is False


def test_heartbeat_run_uses_the_owning_agent_model(tmp_path: Path) -> None:
    store = Store(tmp_path / "app.db")
    store.ensure_initial_model("example-model")
    settings = store.settings()
    global_row = settings["default_model_id"]
    assert global_row
    status, agent_model, _events = store.add_model(
        "hb-agent-model",
        {
            "provider_id": LOCAL_PROVIDER_ID,
            "model_id": "agent-only-model",
            "display_name": "Agent only",
        },
    )
    assert status == 201
    assert global_row != agent_model["id"]
    _status, created, _events = store.create_agent(
        "hb-agent",
        {
            "name": "River",
            "persona": "A short persona.",
            "default_model_id": agent_model["id"],
        },
    )
    agent = created["agent"]
    add_heartbeat(store, agent["id"], "30 3 15 6 *", "Check the garden.", "UTC")
    row = next(item for item in store.heartbeats() if item["agent_id"] == agent["id"])
    _status, result, _events = store.begin_heartbeat_run("hb-agent-run", row["id"])
    assert store.run_context(result["run_id"])["model"] == "agent-only-model"
    favlo_status, favlo_beat, _events = store.create_heartbeat(
        "favlo-hb",
        {
            "agent_id": FAVLO_AGENT_ID,
            "cron_expression": "31 3 15 6 *",
            "timezone": "UTC",
            "instructions": "Use the global model.",
            "enabled": True,
        },
    )
    assert favlo_status == 201
    _status, favlo_run, _events = store.begin_heartbeat_run("hb-favlo-run", favlo_beat["id"])
    assert store.run_context(favlo_run["run_id"])["model"] == "example-model"
    store.close()


def test_heartbeat_run_rereads_agent_model_after_a_change(tmp_path: Path) -> None:
    store = Store(tmp_path / "app.db")
    store.ensure_initial_model("example-model")
    status, first, _events = store.add_model(
        "hb-first",
        {
            "provider_id": LOCAL_PROVIDER_ID,
            "model_id": "first-model",
            "display_name": "First",
        },
    )
    assert status == 201
    status, second, _events = store.add_model(
        "hb-second",
        {
            "provider_id": LOCAL_PROVIDER_ID,
            "model_id": "second-model",
            "display_name": "Second",
        },
    )
    assert status == 201
    _status, created, _events = store.create_agent(
        "hb-switch",
        {
            "name": "Cedar",
            "persona": "A short persona.",
            "default_model_id": first["id"],
        },
    )
    agent = created["agent"]
    add_heartbeat(store, agent["id"], "30 3 15 6 *", "Check the garden.", "UTC")
    row = next(item for item in store.heartbeats() if item["agent_id"] == agent["id"])
    _status, first_run, _events = store.begin_heartbeat_run("hb-first-run", row["id"])
    assert store.run_context(first_run["run_id"])["model"] == "first-model"
    store.freeze_definition(
        first_run["session_id"],
        {
            "model": {"model_id": "first-model", "base_url": "http://127.0.0.1:8129/v1"},
            "hash": "stale",
        },
    )
    store.finish_run(first_run["run_id"], "failed", None)
    store.update_agent(
        agent["id"],
        agent["revision"],
        "Cedar",
        "A short persona.",
        second["id"],
        [],
    )
    _status, second_run, _events = store.begin_heartbeat_run("hb-second-run", row["id"])
    assert store.definition(second_run["session_id"]) is None
    assert store.run_context(second_run["run_id"])["model"] == "second-model"
    store.close()


def test_a_tick_starts_the_reply(tmp_path: Path) -> None:
    store = Store(tmp_path / "app.db")
    add_heartbeat(store, FAVLO_AGENT_ID, "30 3 15 6 *", "Check the garden.", "UTC")
    app = SimpleNamespace(
        state=SimpleNamespace(
            store=store,
            bus=Bus(),
            cancels={},
            tasks=set(),
            reply=_reply,
        )
    )

    async def once() -> None:
        await tick(app, DUE)
        pending = list(app.state.tasks)
        assert pending
        await asyncio.gather(*pending)

    asyncio.run(once())
    run_id = store.cancel_run(next(iter(app.state.cancels)))["id"]
    assert store.cancel_run(run_id)["status"] == "failed"
    store.close()
