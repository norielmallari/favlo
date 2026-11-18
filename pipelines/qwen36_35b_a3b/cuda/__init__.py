"""Qwen3.6-35B-A3B CUDA pipeline."""

from __future__ import annotations

from qwen36_35b_a3b.cuda.chat import pipeline

PIPELINE = "qwen36_35b_a3b"

handle_responses = pipeline
main = pipeline.main
open_runtime = pipeline.open_runtime

__all__ = ["pipeline", "handle_responses", "main", "open_runtime"]
