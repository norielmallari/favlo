"""Qwen3.8-27B pipeline entry. Device packages open from ``open_runtime``."""

from __future__ import annotations

from qwen38_27b.portable.chat import pipeline

PIPELINE = "qwen38_27b"

handle_responses = pipeline
main = pipeline.main
open_runtime = pipeline.open_runtime

__all__ = ["pipeline", "handle_responses", "main", "open_runtime"]
