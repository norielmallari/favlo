"""The daemon call shared by every served pipeline.

``LanguagePipeline``, ``EmbeddingsPipeline``, and ``MediaPipeline`` extend
this class. It does not run a forward, name a stage, or branch on the
pipeline. ``handle`` is abstract. This class catches nothing.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class ClientClosed(Exception):
    """The client went away, or the listener asked the forward to stop."""


class Pipeline(ABC):
    """What the daemon calls. ``__call__`` is ``handle``."""

    kind: str

    def __call__(
        self,
        reply: Any,
        runtime: Any,
        body: dict[str, Any],
        selection: Any,
        args: Any,
        state: dict[str, Any],
    ) -> None:
        self.handle(reply, runtime, body, selection, args, state)

    @abstractmethod
    def handle(
        self,
        reply: Any,
        runtime: Any,
        body: dict[str, Any],
        selection: Any,
        args: Any,
        state: dict[str, Any],
    ) -> None:
        """One request. The seam writes the reply its route already returns."""

    def reject(self, reply: Any, message: str, param: str | None) -> None:
        """A 400. ``param`` is ``None`` when the route does not name a field."""
        reply.json(
            400,
            {
                "error": {
                    "message": message,
                    "type": "invalid_request_error",
                    "param": param,
                    "code": None,
                }
            },
        )

    def model_id(self, body: dict[str, Any], selection: Any) -> str:
        """The requested id, then the body's model, then an empty string."""
        if selection is not None:
            requested = getattr(selection, "requested_id", "")
            if isinstance(requested, str) and requested:
                return requested
        model = body.get("model")
        if isinstance(model, str):
            return model
        return ""
