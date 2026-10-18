"""Greedy carrot completion stored beside the quantized record."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from qwen38_flash_next.session import _log

CARROT_PROMPT = (
    "Amanda and Celestia have 15 carrots. They collect more carrots for Favlo the Horse. "
    "After collecting, they have 21 carrots. How many carrots did they collect?"
)
CARROT_MAX_TOKENS = 384
CARROT_FILE = "carrots.json"


def carrot_path(root: Path) -> Path:
    return root / CARROT_FILE


def load_carrot(root: Path) -> dict[str, Any]:
    path = carrot_path(root)
    if not path.is_file():
        message = f"carrot completion is missing: {path}"
        raise FileNotFoundError(message)
    data = json.loads(path.read_text())
    if data.get("prompt") != CARROT_PROMPT:
        message = f"carrot prompt does not match this pipeline: {path}"
        raise RuntimeError(message)
    if float(data.get("temperature", 1)) != 0.0:
        message = f"carrot completion is not greedy: {path}"
        raise RuntimeError(message)
    if "kv_cache" in data and data["kv_cache"] != "e4m3":
        message = f"carrot completion used kv cache {data['kv_cache']}: {path}"
        raise RuntimeError(message)
    return data


def write_carrot(root: Path, *, replace: bool = False, runtime: Any | None = None) -> None:
    """Store one greedy carrot completion. An existing file stays unless replace is set."""
    path = carrot_path(root)
    if path.is_file() and not replace:
        _log(f"carrot already stored: {path.name}")
        return
    if runtime is None:
        from qwen38_flash_next.runtime import Qwen38Runtime

        runtime = Qwen38Runtime(root)
    text = _greedy(runtime)
    payload = {
        "prompt": CARROT_PROMPT,
        "output": text,
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": 0,
        "max_tokens": CARROT_MAX_TOKENS,
        "thinking": False,
        "kv_cache": "e4m3",
    }
    _write_json(path, payload)
    _log(f"carrot stored: {path.name}")


def _greedy(runtime: Any) -> str:
    prompt = runtime.encode_chat([{"role": "user", "content": CARROT_PROMPT}], None, False)
    tokens = runtime.generate(prompt, CARROT_MAX_TOKENS, 0.0, 1.0, 0)
    if runtime._hit_limit:
        message = "carrot completion hit the token cap"
        raise RuntimeError(message)
    text = runtime.decode(tokens).rstrip()
    if not text:
        message = "carrot completion is empty"
        raise RuntimeError(message)
    return text


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n")
    fd = os.open(tmp, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    os.replace(tmp, path)
