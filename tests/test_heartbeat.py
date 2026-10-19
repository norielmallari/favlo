"""Idle check. No model and no socket."""

from __future__ import annotations

import os
import threading
from pathlib import Path

import pytest
from qwen38_27b.fetch import remaining_download, shortfall

from favlo.machine import (
    MIN_RAM_BYTES,
    download_state,
    machine_report,
    offer_qwen38_27b,
    set_suggestions,
    situation,
    suggestions_enabled,
)
from favlo.registry import save_registry
from favlo.talk import HEARTBEAT_QUIET, HEARTBEAT_SECONDS, Idle, _ask_again, heartbeat_loop


def test_reminders_start_on_and_can_be_turned_off(tmp_path: Path) -> None:
    assert suggestions_enabled(tmp_path) is True
    assert set_suggestions(False, tmp_path) == "download suggestions disabled"
    assert suggestions_enabled(tmp_path) is False
    assert set_suggestions(True, tmp_path) == "download suggestions enabled"
    assert suggestions_enabled(tmp_path) is True
    with pytest.raises(ValueError, match="true or false"):
        set_suggestions("no", tmp_path)  # type: ignore[arg-type]


def test_a_bad_suggestions_file_is_an_error(tmp_path: Path) -> None:
    path = tmp_path / "suggestions.json"
    path.write_text("{", encoding="utf-8")
    with pytest.raises(ValueError, match="cannot be read"):
        suggestions_enabled(tmp_path)


def test_the_suggestion_asks_to_be_reminded() -> None:
    text = _ask_again("I started the download.")
    assert "Should I remind you" in text
    assert _ask_again(text) == text


def test_a_suggestion_is_kept_for_the_reply() -> None:
    from agent_framework import AgentSession, InMemoryHistoryProvider

    from favlo.talk import _note

    session = AgentSession()
    _note(session, ["Should I remind you about other models again?"])
    messages = session.state[InMemoryHistoryProvider.DEFAULT_SOURCE_ID]["messages"]
    assert messages[-1].role == "assistant"
    assert "remind" in messages[-1].text


def test_the_idle_check_waits_thirty_minutes() -> None:
    assert HEARTBEAT_SECONDS == 30 * 60


def test_typing_starts_the_quiet_interval_over() -> None:
    clock = {"t": 0.0}
    idle = Idle(now=lambda: clock["t"])
    clock["t"] = 100.0
    assert idle.remaining(30) == 0
    idle.mark()
    assert idle.remaining(30) == 30


def test_a_busy_server_does_not_run_the_check() -> None:
    clock = {"t": 0.0}
    idle = Idle(now=lambda: clock["t"])
    idle.enter()
    clock["t"] = 10_000.0
    seen: list[float] = []

    def pause(seconds: float) -> bool:
        seen.append(seconds)
        return True

    fired: list[str] = []
    heartbeat_loop(
        idle, HEARTBEAT_SECONDS, threading.Event(), lambda: "no", fired.append, pause
    )
    assert fired == []
    assert seen == [1.0]


def test_one_quiet_interval_runs_once_and_prints_the_offer() -> None:
    clock = {"t": 0.0}
    idle = Idle(now=lambda: clock["t"])
    lines: list[str] = []

    def run() -> str:
        return "Qwen3.8-27B fits. I started the download."

    def pause(seconds: float) -> bool:
        if lines or clock["t"] >= 5:
            return bool(lines)
        clock["t"] += seconds
        return False

    heartbeat_loop(idle, 5, threading.Event(), run, lines.append, pause)
    assert lines == ["Qwen3.8-27B fits. I started the download."]


def test_a_quiet_reply_is_not_printed() -> None:
    clock = {"t": 0.0}
    idle = Idle(now=lambda: clock["t"])
    lines: list[str] = []

    def pause(seconds: float) -> bool:
        if clock["t"] >= 5:
            return True
        clock["t"] += seconds
        return False

    heartbeat_loop(idle, 5, threading.Event(), lambda: HEARTBEAT_QUIET, lines.append, pause)
    assert lines == []


def test_sixteen_gigabytes_is_the_floor(tmp_path: Path) -> None:
    home = tmp_path / "home"
    assert situation(tmp_path, home, ram=MIN_RAM_BYTES - 1, check=lambda: None) == "ram"
    assert situation(tmp_path, home, ram=MIN_RAM_BYTES, check=lambda: None) == "ready"
    assert situation(tmp_path, home, ram=MIN_RAM_BYTES, check=lambda: "short") == "disk"


def test_a_registered_row_is_not_offered(tmp_path: Path) -> None:
    save_registry(
        {
            "version": 1,
            "models": {
                "Qwen3.8-27B-fp4": {"pipeline": "qwen38-27b", "alias": "qwen38-27b"},
            },
        },
        tmp_path,
    )
    assert situation(tmp_path, tmp_path, ram=MIN_RAM_BYTES, check=lambda: None) == "registered"


def test_a_ready_machine_starts_the_download_once(tmp_path: Path) -> None:
    started: list[int] = []
    text = offer_qwen38_27b(
        tmp_path,
        tmp_path,
        ram=MIN_RAM_BYTES,
        free=50 << 30,
        check=lambda: None,
        spawn=lambda: started.append(1) or 1,
    )
    assert "started yes" in text
    assert "ram_bytes" in text
    assert "free_bytes" in text
    assert started == [1]
    report = machine_report(
        tmp_path,
        tmp_path,
        ram=MIN_RAM_BYTES - 1,
        free=1,
        check=lambda: None,
    )
    assert "status ram" in report
    refused = offer_qwen38_27b(
        tmp_path,
        tmp_path,
        ram=MIN_RAM_BYTES - 1,
        free=1,
        check=lambda: None,
        spawn=lambda: started.append(2) or 2,
    )
    assert "started no" in refused
    assert started == [1]


def test_a_live_pid_counts_as_downloading(tmp_path: Path) -> None:
    path = tmp_path / "downloads" / "qwen38-27b.pid"
    path.parent.mkdir()
    path.write_text(f"{os.getpid()}\n", encoding="utf-8")
    assert download_state(tmp_path, tmp_path) == "running"
    marker = tmp_path / "models" / "Qwen3.8-27B-bf16" / "fetch_complete.json"
    marker.parent.mkdir(parents=True)
    marker.write_text("{}\n", encoding="utf-8")
    assert download_state(tmp_path, tmp_path) == "complete"


def test_shortfall_names_the_missing_bytes(tmp_path: Path) -> None:
    dest = tmp_path / "checkout"
    dest.mkdir()
    (dest / "weights").write_bytes(b"abcd")
    assert remaining_download(dest, [("weights", 10), ("config.json", 2)]) == 8
    assert shortfall(0, dest) is None
    message = shortfall(1 << 60, dest)
    assert message is not None
    assert "checkout volume needs" in message
