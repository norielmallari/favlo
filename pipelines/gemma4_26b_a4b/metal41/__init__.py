"""Gemma 4 26B-A4B text pipeline.

Names load on first use so a stream process can import this package without
opening the Metal runtime.
"""

from __future__ import annotations


PIPELINE = "gemma4_26b_a4b"


from typing import Any

__all__ = [
    "CONTEXT_LENGTH",
    "DEFAULT_VARIANT",
    "VARIANTS",
    "handle_responses",
    "main",
    "open_runtime",
]


def __getattr__(name: str) -> Any:
    if name == "main":
        from gemma4_26b_a4b.metal41.commands import main

        return main
    if name in {"handle_responses", "open_runtime"}:
        from gemma4_26b_a4b.metal41.runtime import handle_responses, open_runtime

        return {
            "handle_responses": handle_responses,
            "open_runtime": open_runtime,
        }[name]
    if name in {"CONTEXT_LENGTH", "DEFAULT_VARIANT", "VARIANTS"}:
        from gemma4_26b_a4b.metal41 import sampling

        return getattr(sampling, name)
    message = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(message)


def __dir__() -> list[str]:
    return sorted(__all__)
