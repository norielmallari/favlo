"""Default chat model id for agent harnesses."""

from __future__ import annotations

from pathlib import Path

from favlo.registry import default_llm_id


def judge_model(root: Path | None = None) -> str:
    """The registry default language model, usually the instruct variant."""
    return default_llm_id(root)
