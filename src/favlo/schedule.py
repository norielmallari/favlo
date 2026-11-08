"""GPU admission for the pipeline manager.

Two queues. The foreground queue is embeddings, then chats that have not
started. The media queue runs only when the foreground queue is empty.
A chat preempts media at its next safe point and leaves ``preempt`` set
until the chat queue has drained. After a chat releases, a short debounce
keeps media from admitting so the next model call (the reply to a tool
result) can reuse the resident language model instead of unloading it for
media and loading it again. An embedding does not set ``preempt`` and does
not preempt a chat that already holds the GPU. ``flush`` drops every
waiter. The request that holds admission stays until it releases.
"""

from __future__ import annotations

import threading
import time
from collections import deque

_FOREGROUND = frozenset({"embed", "chat", "responses"})
_KINDS = _FOREGROUND | {"media"}
# Gap between a chat release and the next /v1/responses that carries the
# tool result. Media must not win that race and unload the language model.
_CHAT_DEBOUNCE_SECONDS = 1.0


class ClientClosed(Exception):
    """The client left before this request was admitted."""


class _Ticket:
    def __init__(self, kind: str) -> None:
        self.kind = kind


def ensure(state: dict) -> Scheduler:
    """The pipeline manager's scheduler, created on the first use of ``state``."""
    found = state.get("schedule")
    if not isinstance(found, Scheduler):
        found = Scheduler()
        state["schedule"] = found
    state["preempt"] = found.preempt
    return found


class Scheduler:
    """Admission queues. The pipeline manager admits one request at a time."""

    def __init__(self, *, chat_debounce: float | None = None) -> None:
        self._cond = threading.Condition()
        self._embeds: deque[_Ticket] = deque()
        self._chats: deque[_Ticket] = deque()
        self._media: deque[_Ticket] = deque()
        self._running: _Ticket | None = None
        self._holding = 0
        self._holders: list[str] = []
        self._dropped: set[int] = set()
        self._chat_debounce_until = 0.0
        self._chat_debounce_seconds = (
            _CHAT_DEBOUNCE_SECONDS if chat_debounce is None else max(0.0, float(chat_debounce))
        )
        self.epoch = 0
        self.preempt = threading.Event()

    def acquire(self, kind: str, cancel: threading.Event | None = None) -> _Ticket:
        """Block until ``kind`` is the next request, then mark it running.

        ``cancel`` drops the ticket while it is still waiting. A chat whose
        client has gone does not take the GPU.
        """
        if kind not in _KINDS:
            message = f"unknown queue kind {kind}"
            raise ValueError(message)
        ticket = _Ticket(kind)
        noted = False
        self._cond.acquire()
        try:
            self._queue(kind).append(ticket)
            self._sync_preempt()
            while self._holding or not self._is_next(ticket):
                if id(ticket) in self._dropped or (cancel is not None and cancel.is_set()):
                    self._dropped.discard(id(ticket))
                    raise ClientClosed()
                if not noted:
                    noted = True
                    holding = self._holding > 0
                    holder = self._memory_holder() if holding else ""
                    self._cond.release()
                    try:
                        self._note_wait(holding, holder)
                    finally:
                        self._cond.acquire()
                    continue
                wait = self._wait_seconds(cancel is not None)
                self._cond.wait(wait)
            if cancel is not None and cancel.is_set():
                raise ClientClosed()
            self._queue(kind).popleft()
            self._running = ticket
            if ticket.kind in {"chat", "responses"}:
                self._chat_debounce_until = 0.0
            self._sync_preempt()
        except BaseException:
            self._abandon(ticket)
            raise
        finally:
            self._cond.release()
        return ticket

    def flush(self) -> None:
        """Drop every waiter. The request that holds admission stays until it releases."""
        with self._cond:
            self.epoch += 1
            self._chat_debounce_until = 0.0
            for queue in (self._embeds, self._chats, self._media):
                while queue:
                    self._dropped.add(id(queue.popleft()))
            self._sync_preempt()
            self._cond.notify_all()

    def release(self, ticket: _Ticket) -> None:
        """Drop a running request and wake whoever is next."""
        with self._cond:
            if self._running is ticket:
                self._running = None
                if ticket.kind in {"chat", "responses"} and self._chat_debounce_seconds > 0:
                    self._chat_debounce_until = time.monotonic() + self._chat_debounce_seconds
            self._sync_preempt()
            self._cond.notify_all()

    def waiting(self, kind: str) -> int:
        """How many requests of ``kind`` are queued and not running."""
        with self._cond:
            return len(self._queue(kind))

    def running(self) -> str | None:
        """The kind that holds admission, or none."""
        with self._cond:
            if self._running is None:
                return None
            return self._running.kind

    def hold_memory(self, pipeline: str | None = None) -> None:
        """Keep the next call queued until an allocation or a release finishes.

        ``pipeline`` is the runtime that owns the hold. Nested holds stack; the
        newest non-empty name is what a waiter prints.
        """
        with self._cond:
            self._holding += 1
            self._holders.append(str(pipeline or "").strip())

    def release_memory(self) -> None:
        """The ledger check passed. Admit whoever is next."""
        with self._cond:
            self._holding = max(0, self._holding - 1)
            if self._holders:
                self._holders.pop()
            self._cond.notify_all()

    def _memory_holder(self) -> str:
        for name in reversed(self._holders):
            if name:
                return name
        return ""

    def foreground(self) -> bool:
        """True when an embedding or a chat is queued or holds admission.

        Also true during the post-chat debounce so a resumed media job does
        not unload the language model before the tool-result reply arrives.
        """
        with self._cond:
            if self._embeds or self._chats:
                return True
            if self._running is not None and self._running.kind in _FOREGROUND:
                return True
            return self._debounce_active()

    def _queue(self, kind: str) -> deque[_Ticket]:
        if kind == "embed":
            return self._embeds
        if kind in {"chat", "responses"}:
            return self._chats
        if kind == "media":
            return self._media
        message = f"unknown queue kind {kind}"
        raise ValueError(message)

    def _is_next(self, ticket: _Ticket) -> bool:
        if self._running is not None:
            return False
        if self._embeds:
            return ticket is self._embeds[0]
        if self._chats:
            return ticket is self._chats[0]
        if self._media:
            if self._debounce_active():
                return False
            return ticket is self._media[0]
        return False

    def _debounce_active(self) -> bool:
        return time.monotonic() < self._chat_debounce_until

    def _wait_seconds(self, cancellable: bool) -> float | None:
        if cancellable:
            return 0.05
        remaining = self._chat_debounce_until - time.monotonic()
        if remaining > 0 and self._media and not self._embeds and not self._chats:
            return min(0.05, remaining)
        return None

    def _abandon(self, ticket: _Ticket) -> None:
        queue = self._queue(ticket.kind)
        try:
            queue.remove(ticket)
        except ValueError:
            pass
        if self._running is ticket:
            self._running = None
        self._sync_preempt()
        self._cond.notify_all()

    def _sync_preempt(self) -> None:
        media = self._running is not None and self._running.kind == "media"
        if media and self._chats:
            self.preempt.set()
            return
        self.preempt.clear()

    def _note_wait(self, holding: bool, pipeline: str = "") -> None:
        from favlo.listener import _log

        if holding:
            if pipeline:
                _log(f"waiting for memory for {pipeline}")
            else:
                _log("waiting for memory")
            return
        _log("waiting for the open request")
