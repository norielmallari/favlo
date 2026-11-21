"""GPU admission for the pipeline manager.

Four queues. Embeddings, then chats that have not started, then interactive
media, then bulk. Which kinds preempt, drop a disconnected waiter, debounce,
and wait on a memory hold live on ``LanguagePipeline``, ``EmbeddingsPipeline``,
``MediaPipeline``, and ``BulkMediaPipeline``. This module runs the queues. A
chat or an embedding preempts interactive media and bulk at the next safe
point and leaves ``preempt`` set until that higher deque has drained. An
interactive media waiter preempts bulk. After a chat releases, a short
debounce keeps media and bulk from admitting so the next model call can reuse
the resident language model. After bulk is preempted, bulk stays out for
``bulk_debounce_seconds`` and a later higher request restarts that wait.
``flush`` drops every waiter. The request that holds admission stays until
it releases.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import Any

_CHAT_DEBOUNCE_SECONDS = 1.0
_QUEUE_KINDS = frozenset({"embed", "chat", "responses", "media", "bulk"})


def seam_for(kind: str) -> type:
    """The admission class for an HTTP or queue kind."""
    from favlo.bulk_media_pipeline import BulkMediaPipeline
    from favlo.embeddings_pipeline import EmbeddingsPipeline
    from favlo.language_pipeline import LanguagePipeline
    from favlo.media_pipeline import MediaPipeline

    if kind in LanguagePipeline.admission:
        return LanguagePipeline
    if kind in EmbeddingsPipeline.admission:
        return EmbeddingsPipeline
    if kind in BulkMediaPipeline.admission:
        return BulkMediaPipeline
    if kind in MediaPipeline.admission:
        return MediaPipeline
    message = f"unknown queue kind {kind}"
    raise ValueError(message)


def disconnect_event(kind: str, flag: Any) -> Any:
    """The cancel passed to ``acquire`` when this seam drops a disconnected waiter."""
    if not isinstance(flag, threading.Event):
        return None
    try:
        seam = seam_for(kind)
    except ValueError:
        return None
    if not seam.drop_if_disconnected:
        return None
    return flag


def _foreground(kind: str) -> bool:
    """True for a language or embeddings kind. Media and bulk wait behind both."""
    try:
        seam = seam_for(kind)
    except ValueError:
        return False
    from favlo.bulk_media_pipeline import BulkMediaPipeline
    from favlo.media_pipeline import MediaPipeline

    return seam is not MediaPipeline and seam is not BulkMediaPipeline


def _queue_is_media(kind: str) -> bool:
    from favlo.media_pipeline import MediaPipeline

    try:
        return seam_for(kind) is MediaPipeline
    except ValueError:
        return False


def _queue_is_bulk(kind: str) -> bool:
    from favlo.bulk_media_pipeline import BulkMediaPipeline

    try:
        return seam_for(kind) is BulkMediaPipeline
    except ValueError:
        return False


class ClientClosed(Exception):
    """The client left before this request was admitted."""


class _Ticket:
    def __init__(self, kind: str) -> None:
        self.kind = kind


def ensure(state: dict) -> Scheduler:
    """The pipeline manager's scheduler, created on the first use of ``state``."""
    found = state.get("schedule")
    if not isinstance(found, Scheduler):
        from favlo.logs import bulk_debounce_seconds, favlo_home, load_settings

        found = Scheduler(bulk_debounce=bulk_debounce_seconds(load_settings(favlo_home())))
        state["schedule"] = found
    state["preempt"] = found.preempt
    return found


class Scheduler:
    """Admission queues. The pipeline manager admits one request at a time."""

    def __init__(
        self, *, chat_debounce: float | None = None, bulk_debounce: float | None = None
    ) -> None:
        self._cond = threading.Condition()
        self._embeds: deque[_Ticket] = deque()
        self._chats: deque[_Ticket] = deque()
        self._media: deque[_Ticket] = deque()
        self._bulk: deque[_Ticket] = deque()
        self._running: _Ticket | None = None
        self._holding = 0
        self._holders: list[str] = []
        self._dropped: set[int] = set()
        self._chat_debounce_until = 0.0
        self._chat_debounce_seconds = (
            _CHAT_DEBOUNCE_SECONDS if chat_debounce is None else max(0.0, float(chat_debounce))
        )
        self._bulk_debounce_until = 0.0
        self._bulk_preempted = False
        self._bulk_debounce_seconds = (
            0.0 if bulk_debounce is None else max(0.0, float(bulk_debounce))
        )
        self.epoch = 0
        self.preempt = threading.Event()

    def acquire(self, kind: str, cancel: threading.Event | None = None) -> _Ticket:
        """Block until ``kind`` is the next request, then mark it running.

        ``cancel`` drops the ticket while it is still waiting. A chat whose
        client has gone does not take the GPU.
        """
        if kind not in _QUEUE_KINDS:
            message = f"unknown queue kind {kind}"
            raise ValueError(message)
        ticket = _Ticket(kind)
        noted = False
        waits_for_memory = bool(getattr(seam_for(kind), "waits_for_memory", True))
        self._cond.acquire()
        try:
            self._queue(kind).append(ticket)
            self._sync_preempt()
            while (waits_for_memory and self._holding) or not self._is_next(ticket):
                if id(ticket) in self._dropped or (cancel is not None and cancel.is_set()):
                    self._dropped.discard(id(ticket))
                    raise ClientClosed()
                if not noted:
                    noted = True
                    holding = waits_for_memory and self._holding > 0
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
            if seam_for(ticket.kind).debounce_after_release:
                self._chat_debounce_until = 0.0
            if _queue_is_bulk(ticket.kind):
                self._bulk_preempted = False
                self._bulk_debounce_until = 0.0
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
            self._bulk_debounce_until = 0.0
            self._bulk_preempted = False
            for queue in (self._embeds, self._chats, self._media, self._bulk):
                while queue:
                    self._dropped.add(id(queue.popleft()))
            self._sync_preempt()
            self._cond.notify_all()

    def release(self, ticket: _Ticket) -> None:
        """Drop a running request and wake whoever is next."""
        with self._cond:
            if self._running is ticket:
                self._running = None
                if seam_for(ticket.kind).debounce_after_release and self._chat_debounce_seconds > 0:
                    self._chat_debounce_until = time.monotonic() + self._chat_debounce_seconds
                if self._bulk_preempted and not _queue_is_bulk(ticket.kind):
                    self._arm_bulk_debounce()
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
            if self._running is not None and _foreground(self._running.kind):
                return True
            return self._debounce_active()

    def media_busy(self) -> bool:
        """True when interactive media is queued or holds admission."""
        with self._cond:
            if self._media:
                return True
            return self._running is not None and _queue_is_media(self._running.kind)

    def _queue(self, kind: str) -> deque[_Ticket]:
        from favlo.bulk_media_pipeline import BulkMediaPipeline
        from favlo.embeddings_pipeline import EmbeddingsPipeline
        from favlo.language_pipeline import LanguagePipeline

        seam = seam_for(kind)
        if seam is EmbeddingsPipeline:
            return self._embeds
        if seam is LanguagePipeline:
            return self._chats
        if seam is BulkMediaPipeline:
            return self._bulk
        return self._media

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
        if self._bulk:
            if self._debounce_active() or self._bulk_debounce_active():
                return False
            return ticket is self._bulk[0]
        return False

    def _debounce_active(self) -> bool:
        return time.monotonic() < self._chat_debounce_until

    def _bulk_debounce_active(self) -> bool:
        return time.monotonic() < self._bulk_debounce_until

    def _arm_bulk_debounce(self) -> None:
        if self._bulk_debounce_seconds <= 0:
            return
        self._bulk_debounce_until = time.monotonic() + self._bulk_debounce_seconds

    def _mark_bulk_preempted(self) -> None:
        self._bulk_preempted = True
        self._arm_bulk_debounce()

    def _wait_seconds(self, cancellable: bool) -> float | None:
        if cancellable:
            return 0.05
        higher = self._embeds or self._chats
        waits: list[float] = []
        chat_left = self._chat_debounce_until - time.monotonic()
        if chat_left > 0 and (self._media or self._bulk) and not higher:
            waits.append(min(0.05, chat_left))
        bulk_left = self._bulk_debounce_until - time.monotonic()
        if bulk_left > 0 and self._bulk and not higher and not self._media:
            waits.append(min(0.05, bulk_left))
        if not waits:
            return None
        return min(waits)

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
        from favlo.embeddings_pipeline import EmbeddingsPipeline
        from favlo.language_pipeline import LanguagePipeline
        from favlo.media_pipeline import MediaPipeline

        running = self._running
        if running is None:
            self.preempt.clear()
            return
        media = _queue_is_media(running.kind)
        bulk = _queue_is_bulk(running.kind)
        waiting = False
        if media and self._chats and LanguagePipeline.preempts_media:
            waiting = True
        if media and self._embeds and EmbeddingsPipeline.preempts_media:
            waiting = True
        if bulk and self._chats and LanguagePipeline.preempts_bulk:
            waiting = True
        if bulk and self._embeds and EmbeddingsPipeline.preempts_bulk:
            waiting = True
        if bulk and self._media and MediaPipeline.preempts_bulk:
            waiting = True
        if waiting:
            if bulk:
                self._mark_bulk_preempted()
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
