"""Qwen3.8-27B pipeline. It does not import the Flash Next package."""

from __future__ import annotations

from qwen38_27b.metal41.chat import pipeline

PIPELINE = "qwen38_27b"

handle_responses = pipeline
main = pipeline.main
open_runtime = pipeline.open_runtime

__all__ = ["pipeline", "handle_responses", "main", "open_runtime"]
