"""One local server and one greedy chat completion at a time."""

from __future__ import annotations

import json
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]


class Log:
    def __init__(self) -> None:
        self._lines: list[str] = []
        self._lock = threading.Lock()

    def append(self, line: str) -> None:
        with self._lock:
            self._lines.append(line)

    def text(self) -> str:
        with self._lock:
            return "".join(self._lines)


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def start_server(port: int) -> tuple[subprocess.Popen[str], Log]:
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
        cwd=_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    log = Log()

    def _read() -> None:
        assert proc.stdout is not None
        for line in proc.stdout:
            log.append(line)

    threading.Thread(target=_read, daemon=True).start()
    deadline = time.monotonic() + 600
    while time.monotonic() < deadline:
        if "listening on" in log.text():
            return proc, log
        if proc.poll() is not None:
            raise RuntimeError(log.text()[-4000:])
        time.sleep(0.2)
    raise TimeoutError(log.text()[-4000:])


class Server:
    """The local serve process. A dropped connection starts a new one."""

    def __init__(self) -> None:
        self.port = 0
        self.proc: subprocess.Popen[str] | None = None
        self.log = Log()

    def start(self) -> None:
        self.port = free_port()
        self.proc, self.log = start_server(self.port)

    def restart(self) -> None:
        self.stop()
        self.start()

    def stop(self) -> None:
        if self.proc is None:
            return
        stop_server(self.proc)
        self.proc = None


def stop_server(proc: subprocess.Popen[str]) -> None:
    proc.terminate()
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()


def _has_code(text: str) -> bool:
    return "def " in text or "```" in text


def reply_text(content: str) -> str:
    """Drop one leading close tag. The prompt already closed the think block."""
    stripped = content.lstrip()
    if stripped.startswith("</think>"):
        return stripped[len("</think>") :].lstrip()
    return content


def complete(
    port: int,
    messages: list[dict[str, str]],
    max_tokens: int,
    *,
    thinking: bool = False,
    effort: str | None = None,
    presence: float | None = None,
    continue_final: bool = False,
) -> tuple[str, str, int]:
    """Greedy reply. Returns content, finish reason, and completion tokens."""
    payload: dict[str, object] = {
        "model": "qwen38-flash-next",
        "messages": messages,
        "stream": False,
        "max_tokens": max_tokens,
        "temperature": 0,
        "enable_thinking": thinking,
    }
    if effort:
        payload["reasoning_effort"] = effort
    if presence is not None:
        payload["presence_penalty"] = presence
    if continue_final:
        payload["continue_final_message"] = True
    body = json.dumps(payload).encode()
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=max(900, max_tokens)) as response:
        payload = json.loads(response.read().decode("utf-8"))
    choice = payload["choices"][0]
    message = choice.get("message") or {}
    content = reply_text(message.get("content") or "")
    reasoning = message.get("reasoning_content") or ""
    if reasoning and not _has_code(content) and _has_code(reasoning):
        content = reasoning
    elif not content.strip():
        content = reasoning
    usage = payload.get("usage") or {}
    return content, str(choice.get("finish_reason") or ""), int(usage.get("completion_tokens") or 0)
