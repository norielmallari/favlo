"""Listener shell for one Responses generation.

The subclass implements ``handle_responses``. Sampling, tools, and the token
loop stay there. Chat Completions is translated on the listener. Hooks are
called directly. A hook exception propagates.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Any, ClassVar, Mapping

from favlo.pipeline import ClientClosed, Pipeline


class LanguagePipeline(Pipeline):
    """Disconnect handling around one pipeline's own Responses completion.

    ``admission`` is both HTTP kinds that share the chat deque. The scheduler
    reads the flags below. A package does not. Sampling defaults mirror the
    package ``sampling.py`` card; registration still path-loads that file so
    it does not import Metal.
    """

    kind = "responses"
    admission = frozenset({"chat", "responses"})
    preempts_media = True
    drop_if_disconnected = True
    debounce_after_release = True
    resident = "exclusive"
    waits_for_memory = True

    context_length: ClassVar[int | None] = None
    variants: ClassVar[Mapping[str, Mapping[str, Any]]] = {}
    default_variant: ClassVar[str] = ""
    default_thinking: ClassVar[bool] = True
    default_reasoning_effort: ClassVar[str | None] = None

    def has_vision(self) -> bool:
        """True when this pipeline embeds a still. The default does no vision work."""
        return False

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
        if not self.has_vision() and _request_has_image(body):
            reply.json(
                400,
                {
                    "error": {
                        "message": "this model does not accept images",
                        "type": "invalid_request_error",
                    }
                },
            )
            self.on_finish(runtime, selection, complete=False)
            return
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


def _request_has_image(body: dict[str, Any]) -> bool:
    """True when a Responses body carries a still."""
    raw = body.get("input")
    items = raw if isinstance(raw, list) else ([] if raw is None else [raw])
    for item in items:
        if not isinstance(item, dict):
            continue
        if item.get("type") == "input_image" or "image_url" in item:
            return True
        content = item.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict):
                continue
            if part.get("type") in ("input_image", "image_url") or "image_url" in part:
                return True
    return False
