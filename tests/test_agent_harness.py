"""The file agent against a local MADMetal server."""

import asyncio
import socket
import subprocess
import sys
import threading
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


def _read_log(proc: subprocess.Popen[str], lines: list[str]) -> None:
    assert proc.stdout is not None
    for line in proc.stdout:
        lines.append(line)


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
    port = _free_port()
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "madmetal",
            "--qwen38-flash-next",
            "serve",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    lines: list[str] = []
    reader = threading.Thread(target=_read_log, args=(proc, lines), daemon=True)
    reader.start()
    try:
        deadline = time.monotonic() + 180
        while time.monotonic() < deadline and "listening on" not in "".join(lines):
            if proc.poll() is not None:
                raise RuntimeError("".join(lines))
            time.sleep(0.2)
        if "listening on" not in "".join(lines):
            raise TimeoutError("".join(lines))
        try:
            calls, text = asyncio.run(run_file_task(f"http://127.0.0.1:{port}/v1", tmp_path))
        except Exception as exc:
            message = "".join(lines[-80:])
            raise RuntimeError(message) from exc
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=30)
    reader.join(timeout=5)
    assert calls_in_order(calls), (calls, text, "".join(lines[-50:]))
    assert text == EDITED, (calls, text, "".join(lines[-50:]))
