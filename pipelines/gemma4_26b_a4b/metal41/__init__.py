"""Gemma 4 26B-A4B text pipeline.

Names load on first use so a stream process can import this package without
opening the Metal runtime — except ``pipeline``, which is the typed export.
"""

from __future__ import annotations

from typing import Any

from gemma4_26b_a4b.metal41.chat import pipeline

PIPELINE = "gemma4_26b_a4b"

handle_responses = pipeline
main = pipeline.main
open_runtime = pipeline.open_runtime

__all__ = [
    "pipeline",
    "CONTEXT_LENGTH",
    "DEFAULT_VARIANT",
    "VARIANTS",
    "handle_responses",
    "main",
    "open_runtime",
]


def __getattr__(name: str) -> Any:
    if name in {"CONTEXT_LENGTH", "DEFAULT_VARIANT", "VARIANTS"}:
        from gemma4_26b_a4b.metal41 import sampling

        return getattr(sampling, name)
    message = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(message)


def __dir__() -> list[str]:
    return sorted(__all__)
