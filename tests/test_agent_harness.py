"""The file agent against a local Favlo server."""

import asyncio
import socket
import time
from pathlib import Path

import pytest
from harness.file_agent import EDITED, calls_in_order, run_file_task

from test_packing import pipeline

ROOT = Path(__file__).resolve().parents[1]


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def test_calls_follow_create_edit_confirm() -> None:
    assert calls_in_order(["create", "edit", "confirm"])
    assert calls_in_order(["create", "create", "edit", "confirm"])
    assert not calls_in_order(["edit", "create", "confirm"])
    assert not calls_in_order(["create", "confirm"])


def test_file_agent_creates_edits_and_confirms(tmp_path: Path) -> None:
    root = pipeline.DEFAULT_RECORD
    if not (root / "manifest.json").is_file():
        pytest.skip("quantized record is not installed")
    pytest.importorskip("agent_framework.openai")
    from favlo.daemon import DaemonError, alive, ensure, plain_text, stop_pid, telemetry_path

    port = _free_port()
    try:
        pid, _started = ensure("127.0.0.1", port, root=ROOT)
    except DaemonError as exc:
        raise RuntimeError(str(exc)) from exc
    try:
        deadline = time.monotonic() + 180
        logged = ""
        while time.monotonic() < deadline and "listening on" not in logged:
            path = telemetry_path(port)
            blob = path.read_text(encoding="utf-8", errors="replace") if path.is_file() else ""
            logged = plain_text(blob)
            if not alive(pid):
                raise RuntimeError(logged)
            time.sleep(0.2)
        if "listening on" not in logged:
            raise TimeoutError(logged)
        try:
            calls, text = asyncio.run(run_file_task(f"http://127.0.0.1:{port}/v1", tmp_path))
        except Exception as exc:
            message = plain_text(
                telemetry_path(port).read_text(encoding="utf-8", errors="replace")
            )
            raise RuntimeError(message) from exc
    finally:
        stop_pid(pid)
    assert calls_in_order(calls), (calls, text, logged)
    assert text == EDITED, (calls, text, logged)
