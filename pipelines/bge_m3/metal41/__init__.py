"""BGE-M3 dense embeddings from the int8 GPU record."""

from __future__ import annotations

from bge_m3.metal41.runtime import pipeline

PIPELINE = "bge_m3"

handle_embed = pipeline
main = pipeline.main
open_runtime = pipeline.open_runtime

__all__ = ["pipeline", "handle_embed", "main", "open_runtime"]
