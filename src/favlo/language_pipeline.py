"""Listener shell for one chat completion.

The subclass implements ``complete``. Sampling, tools, and the token loop
stay there. Hooks are called directly. A hook exception propagates.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Any

from favlo.pipeline import ClientClosed, Pipeline


class LanguagePipeline(Pipeline):
    """Disconnect handling around one pipeline's own completion."""

    kind = "chat"

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
        completed = False
        try:
            self.prepare(reply, runtime, body, selection, args, state)
            try:
                self.complete(reply, runtime, body, selection, args, state)
                completed = True
            except ClientClosed:
                self.on_disconnect(runtime)
            except OSError as exc:
                if not self.client_closed(exc):
                    raise
                self.on_disconnect(runtime)
        finally:
            self.on_finish(runtime, selection, complete=completed)

    def on_open(self, runtime: Any, reply: Any, selection: Any) -> None:
        """Work that already sits outside the handler's try."""
        del runtime, reply, selection

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
    def complete(
        self,
        reply: Any,
        runtime: Any,
        body: dict[str, Any],
        selection: Any,
        args: Any,
        state: dict[str, Any],
    ) -> None:
        """The pipeline's completion. Raise ``ClientClosed`` to stop."""

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
