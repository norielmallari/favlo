"""Spark-X2.6-1.7B pipeline."""

from __future__ import annotations

from typing import Any

from spark_x26_1_7b.runtime import handle_responses, open_runtime


def __getattr__(name: str) -> Any:
    if name == "main":
        from spark_x26_1_7b.commands import main

        return main
    message = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(message)


__all__ = ["handle_responses", "main", "open_runtime"]
