"""Qwen3.6-35B-A3B pipeline entry. Device packages open from ``open_runtime``."""

from __future__ import annotations

from typing import Any

from qwen36_35b_a3b.portable.chat import pipeline

PIPELINE = "qwen36_35b_a3b"


def idle_heartbeat(runtime: Any, selection: Any = None) -> None:
    """Pull the finished route hist and write SQLite. Lazy to avoid metal41↔portable import cycles."""
    from qwen36_35b_a3b.metal41.firing import idle_heartbeat as pump

    pump(runtime, selection)


handle_responses = pipeline
main = pipeline.main
open_runtime = pipeline.open_runtime

__all__ = ["pipeline", "handle_responses", "idle_heartbeat", "main", "open_runtime"]
