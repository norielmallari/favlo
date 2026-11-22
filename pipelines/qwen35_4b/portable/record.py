"""Installed record lookup and the carrot prompt. No device import."""

from __future__ import annotations

import json
from pathlib import Path

CARROT_PROMPT = (
    "Amanda and Celestia have 15 carrots. They collect more carrots for Favlo the Horse. "
    "After collecting, they have 21 carrots. How many carrots did they collect?"
)

_FOLDER = "Qwen3.5-4B-fp8"


def installed_record() -> Path | None:
    """The packed folder when its manifest says the record is complete."""
    from favlo.registry import models_dir

    path = models_dir() / _FOLDER
    manifest = path / "manifest.json"
    if not manifest.is_file():
        return None
    try:
        body = json.loads(manifest.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if body.get("complete") is not True or not (path / "weights.bin").is_file():
        return None
    return path
