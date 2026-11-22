"""BGE-M3 pipeline entry. Device packages open from ``open_runtime``."""

from __future__ import annotations

from bge_m3.portable.embed import pipeline

PIPELINE = "bge_m3"

handle_embed = pipeline
main = pipeline.main
open_runtime = pipeline.open_runtime

__all__ = ["pipeline", "handle_embed", "main", "open_runtime"]
