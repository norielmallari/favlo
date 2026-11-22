"""Gemma 4 26B-A4B pipeline entry. Device packages open from ``open_runtime``."""

from __future__ import annotations

from gemma4_26b_a4b.portable.chat import pipeline

PIPELINE = "gemma4_26b_a4b"

handle_responses = pipeline
main = pipeline.main
open_runtime = pipeline.open_runtime

__all__ = ["pipeline", "handle_responses", "main", "open_runtime"]
