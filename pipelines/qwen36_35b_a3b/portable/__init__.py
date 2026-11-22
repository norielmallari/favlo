"""Qwen3.6-35B-A3B pipeline entry. Device packages open from ``open_runtime``."""

from __future__ import annotations

from qwen36_35b_a3b.portable.chat import pipeline

PIPELINE = "qwen36_35b_a3b"

handle_responses = pipeline
main = pipeline.main
open_runtime = pipeline.open_runtime

__all__ = ["pipeline", "handle_responses", "main", "open_runtime"]
