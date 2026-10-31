"""Rolling server log. No device and no checkpoint."""

import io
import json
import sys

import pytest

from favlo.console import Console
from favlo.logs import (
    DEFAULT_LOG_MAX_BYTES,
    RollingLog,
    configured_thinking_budget,
    ensure_settings,
    heartbeat_seconds,
    install_rolling_stdio,
    load_settings,
    log_max_bytes,
    log_path,
    output_token_limit,
    set_heartbeat_seconds,
    settings_path,
)


def test_a_missing_settings_file_uses_one_mebibyte(tmp_path) -> None:
    assert log_max_bytes(ensure_settings(tmp_path)) == DEFAULT_LOG_MAX_BYTES
    stored = json.loads(settings_path(tmp_path).read_text(encoding="utf-8"))
    assert stored["log_max_bytes"] == 1024 * 1024
    stored["log_max_bytes"] = 4096
    settings_path(tmp_path).write_text(json.dumps(stored), encoding="utf-8")
    assert log_max_bytes(ensure_settings(tmp_path)) == 4096
    assert json.loads(settings_path(tmp_path).read_text(encoding="utf-8"))["log_max_bytes"] == 4096


def test_the_quiet_interval_defaults_to_five_minutes(tmp_path) -> None:
    assert heartbeat_seconds({}) == 300
    assert heartbeat_seconds({"heartbeat_seconds": 12}) == 300
    assert heartbeat_seconds({"heartbeat_seconds": "nope"}) == 300
    assert set_heartbeat_seconds(tmp_path, 120) == 120
    assert heartbeat_seconds(load_settings(tmp_path)) == 120
    with pytest.raises(ValueError):
        set_heartbeat_seconds(tmp_path, 30)


def test_a_bad_log_cap_falls_back(tmp_path) -> None:
    settings_path(tmp_path).write_text("{", encoding="utf-8")
    assert log_max_bytes(ensure_settings(tmp_path)) == DEFAULT_LOG_MAX_BYTES
    assert log_max_bytes({"log_max_bytes": 0}) == DEFAULT_LOG_MAX_BYTES
    assert log_max_bytes({"log_max_bytes": "nope"}) == DEFAULT_LOG_MAX_BYTES


def test_the_log_keeps_only_the_newest_bytes(tmp_path) -> None:
    path = log_path(tmp_path)
    rolling = RollingLog(path, 48)
    for index in range(20):
        rolling.write(f"line-{index:02d} pad")
    blob = path.read_bytes()
    assert len(blob) <= 48
    text = blob.decode("utf-8")
    assert text.startswith("line-")
    assert "line-19" in text
    assert "line-00" not in text


def test_stdio_stays_under_the_configured_cap(tmp_path, monkeypatch) -> None:
    stored = ensure_settings(tmp_path)
    stored["log_max_bytes"] = 64
    settings_path(tmp_path).write_text(json.dumps(stored), encoding="utf-8")
    path = tmp_path / "logs" / "serve-9.log"
    path.parent.mkdir()
    monkeypatch.setenv("FAVLO_LOG_FILE", str(path))
    original = sys.stdout, sys.stderr
    try:
        install_rolling_stdio()
        for index in range(20):
            sys.stdout.write(f"line-{index:02d} pad\n")
        sys.stdout.flush()
        blob = path.read_bytes()
        assert len(blob) <= 64
        assert b"line-19" in blob
        assert b"line-00" not in blob
        again = sys.stdout
        install_rolling_stdio()
        assert sys.stdout is again
        assert sys.stderr is again
    finally:
        sys.stdout, sys.stderr = original


def test_console_lines_land_in_the_rolling_log(tmp_path) -> None:
    stream = io.StringIO()
    ui = Console("pretty", pipeline="", stream=stream, start_thread=False)
    ui.attach_log(log_path(tmp_path), 256)
    ui.plate()
    progress = ui.begin("prefill", total=4, unit="tok")
    ui.log("listening on http://127.0.0.1:8129")
    ui.end(progress, "prefill 4 tok in 1.0s")
    ui.report("RuntimeError: boom\n")
    text = log_path(tmp_path).read_text(encoding="utf-8")
    assert "favlo 0.2.0" in text
    assert "listening on http://127.0.0.1:8129" in text
    assert "prefill 4 tok in 1.0s" in text
    assert "RuntimeError: boom" in text


def test_omitted_output_limit_and_thinking_budget_stay_unset(tmp_path) -> None:
    created = ensure_settings(tmp_path)
    assert "max_tokens" not in created
    assert "thinking_budget" not in created
    settings = {"log_max_bytes": 1024}
    assert output_token_limit({}, settings) is None
    assert output_token_limit({"max_tokens": None, "max_completion_tokens": 16}, settings) == 16
    assert output_token_limit({"max_tokens": 32}, {"max_tokens": 8}) == 32
    assert output_token_limit({}, {"max_tokens": 64}) == 64
    assert configured_thinking_budget(settings) is None
    assert configured_thinking_budget({"thinking_budget": 512}) == 512
