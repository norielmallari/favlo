"""What a pipeline writes. The server turns it into HTTP.

A pipeline does not see a socket, a status line, or chunked framing.
``json`` finishes one object. ``begin_events`` then ``event`` then ``done``
is one server-sent stream. ``stopped`` is true when that stream's client
has gone away.
"""

from __future__ import annotations

import json
import queue
import threading
from typing import Any


class MemoryReply:
    """Records one reply. Pipeline tests use this instead of a socket."""

    def __init__(self, model_id: str = "") -> None:
        self.model_id = model_id
        self.started = False
        self.status = 0
        self.body: dict[str, Any] = {}
        self.payload: dict[str, Any] = {}
        self.events: list[dict[str, Any]] = []
        self.sent: list[bytes] = []

    def json(self, status: int, payload: dict[str, Any]) -> None:
        self.started = True
        self.status = status
        self.body = payload
        self.payload = payload

    def begin_events(self) -> None:
        self.started = True

    def event(self, payload: dict[str, Any]) -> None:
        self.events.append(payload)

    def done(self) -> None:
        self.sent.append(b"data: [DONE]\n\n")

    def stopped(self) -> bool:
        return False


class LiveReply:
    """A reply the ASGI server is reading on another thread."""

    def __init__(self, disconnected: threading.Event) -> None:
        self.model_id = ""
        self.started = False
        self.disconnected = disconnected
        self._disconnected = disconnected
        self._ready = threading.Event()
        self._kind = ""
        self.status = 200
        self.payload: dict[str, Any] | None = None
        self._encoded = b""
        self._chunks: queue.Queue[bytes | None] = queue.Queue(maxsize=32)

    @property
    def ready(self) -> threading.Event:
        return self._ready

    @property
    def kind(self) -> str:
        return self._kind

    @property
    def encoded(self) -> bytes:
        """The JSON body captured when ``json`` returned. Later edits stay on ``payload``."""
        return self._encoded

    def json(self, status: int, payload: dict[str, Any]) -> None:
        if self._kind == "stream":
            return
        self.started = True
        self._kind = "json"
        self.status = status
        self.payload = payload
        self._encoded = json.dumps(payload).encode()
        self._ready.set()

    def begin_events(self) -> None:
        if self._disconnected.is_set():
            message = "client closed"
            raise OSError(message)
        self.started = True
        self._kind = "stream"
        self._ready.set()

    def event(self, payload: dict[str, Any]) -> None:
        self._write(b"data: " + json.dumps(payload).encode() + b"\n\n")

    def done(self) -> None:
        self._write(b"data: [DONE]\n\n")

    def stopped(self) -> bool:
        if self._kind == "json":
            return False
        return self._disconnected.is_set()

    def finish_stream(self) -> None:
        if self._kind != "stream":
            self._ready.set()
            return
        try:
            self._chunks.put(None, timeout=2.0)
        except queue.Full:
            return

    def _write(self, data: bytes) -> None:
        if self._disconnected.is_set():
            message = "client closed"
            raise OSError(message)
        try:
            self._chunks.put(data, timeout=2.0)
        except queue.Full:
            message = "client closed"
            raise OSError(message) from None

    def pull(self) -> bytes | None:
        try:
            return self._chunks.get(timeout=0.05)
        except queue.Empty:
            return b""
