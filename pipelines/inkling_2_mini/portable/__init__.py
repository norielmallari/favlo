"""Inkling 2 Mini pipeline entry. Device packages open from ``open_runtime``."""

from __future__ import annotations

from typing import Any

from inkling_2_mini.portable.chat import pipeline

PIPELINE = "inkling_2_mini"


def idle_heartbeat(runtime: Any, selection: Any) -> None:
    """The runtime has no idle promotion pass."""
    del runtime, selection


handle_responses = pipeline
main = pipeline.main
open_runtime = pipeline.open_runtime

__all__ = ["pipeline", "handle_responses", "idle_heartbeat", "main", "open_runtime"]
