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
            texts = self.parse(body)
        except EmbeddingsRequestError as exc:
            self.reject(reply, str(exc), exc.param)
            return
        model = ""
        if selection is not None:
            requested = getattr(selection, "requested_id", "")
            if isinstance(requested, str):
                model = requested
        data: list[dict[str, Any]] = []
        prompt_tokens = 0
        for index, text in enumerate(texts):
            try:
                vector, tokens = self.embed(runtime, text)
            except ValueError as exc:
                self.reject(reply, str(exc), None)
                return
            prompt_tokens += tokens
            data.append(
                {
                    "object": "embedding",
                    "index": index,
                    "embedding": vector,
                }
            )
        reply.json(
            200,
            {
                "object": "list",
                "model": model,
                "data": data,
                "usage": {"prompt_tokens": prompt_tokens, "total_tokens": prompt_tokens},
            },
        )

    def parse(self, body: dict[str, Any]) -> list[str]:
        raw = body.get("input")
        if isinstance(raw, str):
            if not raw.strip():
                raise EmbeddingsRequestError(
                    "input must be a non-empty string or array of strings",
                    "input",
                )
            return [raw]
        if isinstance(raw, list):
            if not raw:
                raise EmbeddingsRequestError(
                    "input must be a non-empty string or array of strings",
                    "input",
                )
            texts: list[str] = []
            for item in raw:
                if not isinstance(item, str) or not item.strip():
                    raise EmbeddingsRequestError(
                        "input must be a non-empty string or array of strings",
                        "input",
                    )
                texts.append(item)
            return texts
        raise EmbeddingsRequestError(
            "input must be a non-empty string or array of strings",
            "input",
        )

    @abstractmethod
    def embed(self, runtime: Any, text: str) -> tuple[list[float], int]:
        """The pipeline's vector and its token count."""
