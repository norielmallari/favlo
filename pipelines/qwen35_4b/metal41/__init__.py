"""Qwen3.5-4B pipeline."""

from __future__ import annotations

from qwen35_4b.metal41.chat import pipeline

PIPELINE = "qwen35_4b"

handle_responses = pipeline
main = pipeline.main
open_runtime = pipeline.open_runtime

__all__ = ["pipeline", "handle_responses", "main", "open_runtime"]
