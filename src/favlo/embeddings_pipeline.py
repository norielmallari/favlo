"""Listener shell for one embedding request.

``embed`` returns the vector and the token count. The OpenAI list object
stays here. The class does not normalize the vector.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Any

from favlo.pipeline import Pipeline


class EmbeddingsRequestError(ValueError):
    """The body cannot be embedded. The reply is a 400."""

    def __init__(self, message: str, param: str | None = None) -> None:
        super().__init__(message)
        self.param = param


class EmbeddingsPipeline(Pipeline):
    """One string in, one dense vector out."""

    kind = "embed"

    def handle(
        self,
        reply: Any,
        runtime: Any,
        body: dict[str, Any],
        selection: Any,
        args: Any,
        state: dict[str, Any],
    ) -> None:
        del args, state
        try:
            text = self.parse(body)
        except EmbeddingsRequestError as exc:
            self.reject(reply, str(exc), exc.param)
            return
        try:
            vector, tokens = self.embed(runtime, text)
        except ValueError as exc:
            self.reject(reply, str(exc), None)
            return
        model = ""
        if selection is not None:
            requested = getattr(selection, "requested_id", "")
            if isinstance(requested, str):
                model = requested
        reply.json(
            200,
            {
                "object": "list",
                "model": model,
                "data": [
                    {
                        "object": "embedding",
                        "index": 0,
                        "embedding": vector,
                    }
                ],
                "usage": {"prompt_tokens": tokens, "total_tokens": tokens},
            },
        )

    def parse(self, body: dict[str, Any]) -> str:
        text = body.get("input")
        if not isinstance(text, str) or not text.strip():
            raise EmbeddingsRequestError("input must be a string", "input")
        return text

    @abstractmethod
    def embed(self, runtime: Any, text: str) -> tuple[list[float], int]:
        """The pipeline's vector and its token count."""
