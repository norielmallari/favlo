"""Qwen3.5-4B pipeline."""

from __future__ import annotations


PIPELINE = "qwen35_4b"


from typing import Any

from qwen35_4b.metal41.runtime import handle_responses, open_runtime


def __getattr__(name: str) -> Any:
    if name == "main":
        from qwen35_4b.metal41.commands import main

        return main
    message = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(message)


__all__ = ["handle_responses", "main", "open_runtime"]
