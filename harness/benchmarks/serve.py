"""One shared listener and one chat completion at a time."""

from __future__ import annotations

import json
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
    mtp: bool | None = None
    mtp_depth: int | None = None

    def limit(self, suite: str) -> int:
        limits = self.max_tokens or {}
        if suite not in limits:
            message = f"{suite} has no max_tokens"
            raise KeyError(message)
        return int(limits[suite])


def will_think(request: Request) -> bool:
    """Whether the completion opens a trace.

    A named effort wins. ``none`` leaves thinking off. An omitted effort
    follows ``enable_thinking``.
    """
    effort = request.reasoning_effort
    if isinstance(effort, str) and effort:
        return effort != "none"
    return request.enable_thinking


def probe_limit(request: Request) -> int:
    """Tokens for the two completions that run before the suites.

    A trace needs the smallest suite cap so it can close and the reply can
    appear. A request that does not think uses 32.
    """
    if not will_think(request):
        return 32
    caps = [int(value) for value in (request.max_tokens or {}).values()]
    if not caps:
        return 1024
    return min(caps)


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
        mtp=request.mtp,
        mtp_depth=request.mtp_depth,
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
    if request.mtp is not None:
        body["mtp"] = request.mtp
    if request.mtp_depth is not None:
        body["mtp_depth"] = request.mtp_depth
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

    def clear(self) -> None:
        with self._lock:
            self._lines.clear()

    def text(self) -> str:
        with self._lock:
            return "".join(self._lines)


class Server:
    """The shared listener on ``127.0.0.1:8129``.

    ``start`` reuses a daemon that is already listening and starts one only
    when the port is closed. ``restart`` unloads the resident model on that
    same listener. ``stop`` stops the daemon only when this session started it.
    """

    def __init__(self, max_memory: str | None = None, *, pipeline: str = "") -> None:
        self.port = 8129
        self.max_memory = max_memory
        self.pipeline = pipeline
        self.owned = False
        self.pid: int | None = None
        self.follow_stop: threading.Event | None = None
        self.log = Log()

    def start(self) -> None:
        from favlo.cli import repo_root
        from favlo.daemon import alive, follow, read_pid
        from favlo.logs import favlo_home, log_path
        from favlo.smoke_lock import open_attached

        path = log_path(favlo_home())
        before = path.stat().st_size if path.is_file() else 0
        attached = open_attached(
            root=repo_root(),
            pipeline=self.pipeline,
            max_memory=self.max_memory,
        )
        self.port = attached.port
        self.owned = attached.owned
        self.pid = attached.pid
        if not self.pid:
            found = read_pid(self.port)
            if found is not None and alive(found):
                self.pid = found
        if self.follow_stop is not None:
            self.follow_stop.set()
        self.follow_stop = threading.Event()
        path = log_path(favlo_home())
        size = path.stat().st_size if path.is_file() else 0
        # A rolled log keeps its newest bytes. Starting at 0 would replay them.
        offset = before if size >= before else size

        def on_line(line: str) -> None:
            self.log.append(line + "\n")

        follow(path, on_line, self.follow_stop, offset=offset)
        # A rolled favlo.log can re-deliver retained lines from earlier sessions.
        time.sleep(0.2)
        self.log.clear()

    def restart(self) -> None:
        """Unload the resident model. Keep the listener and the followed log."""
        from favlo.daemon import alive, port_open
        from favlo.smoke_lock import unload_model

        unload_model(port=self.port)
        daemon_up = port_open("127.0.0.1", self.port)
        if self.pid is not None and not alive(self.pid):
            daemon_up = False
        if daemon_up:
            return
        self.start()

    def stop(self) -> None:
        if self.follow_stop is not None:
            self.follow_stop.set()
            self.follow_stop = None
        if not self.owned or self.pid is None:
            self.pid = None
            return
        from favlo.daemon import stop_pid

        stop_pid(self.pid)
        print(f"stopped 127.0.0.1:{self.port}", flush=True)
        self.pid = None
        self.owned = False


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
