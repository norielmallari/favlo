"""Listener shell for one Responses generation.

The subclass implements ``handle_responses``. Sampling, tools, and the token
loop stay there. Chat Completions is translated on the listener. Hooks are
called directly. A hook exception propagates.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Any

from favlo.pipeline import ClientClosed, Pipeline


class LanguagePipeline(Pipeline):
    """Disconnect handling around one pipeline's own Responses completion.

    ``admission`` is both HTTP kinds that share the chat deque. The scheduler
    reads the flags below. A package does not.
    """

    kind = "responses"
    admission = frozenset({"chat", "responses"})
    preempts_media = True
    drop_if_disconnected = True
    debounce_after_release = True
    resident = "exclusive"

    def handle(
        self,
        reply: Any,
        runtime: Any,
        body: dict[str, Any],
        selection: Any,
        args: Any,
        state: dict[str, Any],
    ) -> None:
        self.on_open(runtime, reply, selection)
        from favlo.response_chain import ChainError, finish_turn, present, recording_reply

        try:
            presented, turn = present(body, state)
        except ChainError as exc:
            reply.json(
                400,
                {"error": {"message": str(exc), "type": "invalid_request_error"}},
            )
            self.on_finish(runtime, selection, complete=False)
            return
        recorded = recording_reply(reply, previous=None if turn is None else turn.previous)
        completed = False
        try:
            self.prepare(recorded, runtime, presented, selection, args, state)
            try:
                self.handle_responses(recorded, runtime, presented, selection, args, state)
                completed = True
            except ClientClosed:
                self.on_disconnect(runtime)
            except OSError as exc:
                if not self.client_closed(exc):
                    raise
                self.on_disconnect(runtime)
        finally:
            finish_turn(turn, recorded, state)
            self.on_finish(runtime, selection, complete=completed)

    def on_open(self, runtime: Any, reply: Any, selection: Any) -> None:
        """Work that already sits outside the handler's try.

        Assigns ``runtime._stop`` from the reply. A subclass that adds work
        calls ``super``. Gemma 4 keeps a bool and does not call this.
        """
        del selection
        if runtime is not None:
            runtime._stop = getattr(reply, "stopped", None)

    def prepare(
        self,
        reply: Any,
        runtime: Any,
        body: dict[str, Any],
        selection: Any,
        args: Any,
        state: dict[str, Any],
    ) -> None:
        """Work inside the outer try and outside the disconnect try."""
        del reply, runtime, body, selection, args, state

    @abstractmethod
    def handle_responses(
        self,
        reply: Any,
        runtime: Any,
        body: dict[str, Any],
        selection: Any,
        args: Any,
        state: dict[str, Any],
    ) -> None:
        """The pipeline's Responses completion. Raise ``ClientClosed`` to stop."""

    def client_closed(self, exc: OSError) -> bool:
        """True when this ``OSError`` is the client going away."""
        del exc
        return False

    def on_disconnect(self, runtime: Any) -> None:
        status = getattr(runtime, "_status", None)
        if callable(status):
            status("client closed")

    def on_finish(self, runtime: Any, selection: Any, *, complete: bool) -> None:
        """Always runs after ``prepare``, including when ``prepare`` raises."""
        del selection, complete
        runtime._stop = None
