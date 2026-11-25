"""Inkling 2 Mini pipeline entry. Device packages open from ``open_runtime``."""

from __future__ import annotations

from typing import Any

from inkling_2_mini.portable.chat import pipeline

PIPELINE = "inkling_2_mini"


def idle_heartbeat(runtime: Any, selection: Any) -> None:
    """Write a finished request's expert hits after the reply closes."""
    from inkling_2_mini.metal41.firing import idle_heartbeat as _write

    _write(runtime, selection)


handle_responses = pipeline
main = pipeline.main
open_runtime = pipeline.open_runtime

__all__ = ["pipeline", "handle_responses", "idle_heartbeat", "main", "open_runtime"]
