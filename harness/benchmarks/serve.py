"""One local server and one chat completion at a time."""

from __future__ import annotations

import json
import socket
import threading
import time
import urllib.request
from dataclasses import dataclass


@dataclass(frozen=True)
class Request:
    """Fields the harness sends on ``POST /v1/chat/completions``.

    An omitted sampling field is left out of the body, so the server applies
    that model's card. ``max_tokens`` is per suite.
    """

    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    presence_penalty: float | None = None
    enable_thinking: bool = False
    reasoning_effort: str | None = None
    max_tokens: dict[str, int] | None = None

    def limit(self, suite: str) -> int:
        limits = self.max_tokens or {}
        if suite not in limits:
            message = f"{suite} has no max_tokens"
            raise KeyError(message)
        return int(limits[suite])


def card_model_id(model: str, *, thinking: bool) -> str:
    """The registry row whose ``sampling`` object fills omitted request fields."""
    stem, separator, named = model.partition(":")
    if separator and named:
        return model
    return f"{stem}:{'thinking' if thinking else 'instruct'}"


def apply_sampling(request: Request, sampling: dict[str, object]) -> Request:
    """Copy card numbers into fields the caller left unset."""

    def _number(name: str, current: float | int | None, cast):
        if current is not None:
            return current
        value = sampling.get(name)
        if value is None:
            return None
        return cast(value)

    return Request(
        temperature=_number("temperature", request.temperature, float),
        top_p=_number("top_p", request.top_p, float),
        top_k=_number("top_k", request.top_k, int),
        presence_penalty=_number("presence_penalty", request.presence_penalty, float),
        enable_thinking=request.enable_thinking,
        reasoning_effort=request.reasoning_effort,
        max_tokens=request.max_tokens,
    )


def select_sampling(rows: list[object], model_id: str) -> dict[str, object]:
    """The sampling card for ``model_id``.

    An id that is already listed wins. A pipeline name matches the row whose
    ``pipeline`` and ``variant`` are that name and the requested suffix.
    """
    stem, separator, variant = model_id.partition(":")
    for row in rows:
        if isinstance(row, dict) and row.get("id") == model_id:
            return _sampling_card(row, model_id)
    if separator and variant:
        for row in rows:
            if (
                isinstance(row, dict)
                and row.get("pipeline") == stem
                and row.get("variant") == variant
            ):
                return _sampling_card(row, model_id)
    message = f"{model_id} is not listed by GET /v1/models"
    raise RuntimeError(message)


def _sampling_card(row: dict[str, object], model_id: str) -> dict[str, object]:
    sampling = row.get("sampling")
    if isinstance(sampling, dict) and "temperature" in sampling:
        return sampling
    message = f"{model_id} has no sampling card"
    raise RuntimeError(message)


def fetch_sampling(port: int, model_id: str) -> dict[str, object]:
    """``sampling`` from ``GET /v1/models`` for one registry id."""
    http = urllib.request.Request(f"http://127.0.0.1:{port}/v1/models", method="GET")
    with urllib.request.urlopen(http, timeout=30) as response:
        payload = json.loads(response.read().decode("utf-8"))
    rows = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        message = "GET /v1/models did not return data"
        raise RuntimeError(message)
    return select_sampling(rows, model_id)


def completion_body(
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int | None,
    request: Request,
) -> dict[str, object]:
    """The JSON body for one completion. Unset sampling fields are omitted."""
    body: dict[str, object] = {
        "model": model,
        "messages": messages,
        "stream": False,
        "enable_thinking": request.enable_thinking,
    }
    if max_tokens is not None:
        body["max_tokens"] = max_tokens
    if request.temperature is not None:
        body["temperature"] = request.temperature
    if request.top_p is not None:
        body["top_p"] = request.top_p
    if request.top_k is not None:
        body["top_k"] = request.top_k
    if request.presence_penalty is not None:
        body["presence_penalty"] = request.presence_penalty
    if request.reasoning_effort:
        body["reasoning_effort"] = request.reasoning_effort
    return body


def completion_text(payload: dict[str, object]) -> tuple[str, str, int]:
    """``message.content``, the finish reason, and completion tokens.

    The checker scores that content. Reasoning stays on the server.
    """
    choices = payload["choices"]
    if not isinstance(choices, list) or not choices:
        message = "completion has no choices"
        raise RuntimeError(message)
    choice = choices[0]
    if not isinstance(choice, dict):
        message = "completion choice is not an object"
        raise RuntimeError(message)
    message_body = choice.get("message") or {}
    if not isinstance(message_body, dict):
        message = "completion message is not an object"
        raise RuntimeError(message)
    content = message_body.get("content") or ""
    if not isinstance(content, str):
        message = "completion content is not a string"
        raise RuntimeError(message)
    usage = payload.get("usage") or {}
    tokens = 0
    if isinstance(usage, dict):
        tokens = int(usage.get("completion_tokens") or 0)
    return content, str(choice.get("finish_reason") or ""), tokens


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


def start_server(port: int, max_memory: str | None = None) -> tuple[int, threading.Event, Log]:
    from favlo.cli import repo_root
    from favlo.daemon import alive, ensure, follow, telemetry_path

    log = Log()
    pid, _started = ensure(
        "127.0.0.1",
        port,
        max_memory=max_memory,
        root=repo_root(),
    )
    stop = threading.Event()

    def on_line(line: str) -> None:
        log.append(line + "\n")

    follow(telemetry_path(port), on_line, stop, offset=0)
    deadline = time.monotonic() + 600
    while time.monotonic() < deadline:
        if "listening on" in log.text():
            return pid, stop, log
        if not alive(pid):
            raise RuntimeError(log.text()[-4000:])
        time.sleep(0.2)
    raise TimeoutError(log.text()[-4000:])


class Server:
    """The local serve process. A dropped connection starts a new one."""

    def __init__(self, max_memory: str | None = None) -> None:
        self.port = 0
        self.max_memory = max_memory
        self.pid: int | None = None
        self.follow_stop: threading.Event | None = None
        self.log = Log()

    def start(self) -> None:
        self.port = free_port()
        self.pid, self.follow_stop, self.log = start_server(self.port, self.max_memory)

    def restart(self) -> None:
        self.stop()
        self.start()

    def stop(self) -> None:
        if self.pid is None:
            return
        stop_server(self.pid, self.follow_stop)
        self.pid = None
        self.follow_stop = None


def stop_server(pid: int, follow_stop: threading.Event | None = None) -> None:
    from favlo.daemon import stop_pid

    if follow_stop is not None:
        follow_stop.set()
    stop_pid(pid)


def complete(
    port: int,
    messages: list[dict[str, str]],
    max_tokens: int,
    request: Request,
    *,
    model: str,
) -> tuple[str, str, int]:
    """One completion. Returns content, finish reason, and completion tokens."""
    encoded = json.dumps(completion_body(model, messages, max_tokens, request)).encode()
    http = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=encoded,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(http, timeout=max(900, max_tokens)) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(payload, dict):
        message = "completion response is not an object"
        raise RuntimeError(message)
    return completion_text(payload)
