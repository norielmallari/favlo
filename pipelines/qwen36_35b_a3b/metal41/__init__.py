"""Qwen3.6-35B-A3B pipeline. It does not import another pipeline package."""

from __future__ import annotations

from qwen36_35b_a3b.metal41.chat import pipeline
from qwen36_35b_a3b.metal41.firing import idle_heartbeat

PIPELINE = "qwen36_35b_a3b"

handle_responses = pipeline
main = pipeline.main
open_runtime = pipeline.open_runtime

__all__ = ["pipeline", "handle_responses", "idle_heartbeat", "main", "open_runtime"]
