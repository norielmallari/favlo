"""Greedy grove completion stored beside the quantized record."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from qwen38_flash_next.session import _log

GROVE_PROMPT = (
    "There are 15 trees in the grove. Grove workers will plant trees in the grove today. "
    "After they are done, there will be 21 trees. How many trees did the grove workers plant today?"
)
GROVE_MAX_TOKENS = 128
GROVE_FILE = "grove.json"


def grove_path(root: Path) -> Path:
    return root / GROVE_FILE


def load_grove(root: Path) -> dict[str, Any]:
    path = grove_path(root)
    if not path.is_file():
        message = f"grove completion is missing: {path}"
        raise FileNotFoundError(message)
    data = json.loads(path.read_text())
    if data.get("prompt") != GROVE_PROMPT:
        message = f"grove prompt does not match this pipeline: {path}"
        raise RuntimeError(message)
    if float(data.get("temperature", 1)) != 0.0:
        message = f"grove completion is not greedy: {path}"
        raise RuntimeError(message)
    return data


def write_grove(root: Path, *, replace: bool = False, runtime: Any | None = None) -> None:
    """Store one greedy grove completion. An existing file stays unless replace is set."""
    path = grove_path(root)
    if path.is_file() and not replace:
        _log(f"grove already stored: {path.name}")
        return
    if runtime is None:
        from qwen38_flash_next.runtime import Qwen38Runtime

        runtime = Qwen38Runtime(root)
    text = _greedy(runtime)
    payload = {
        "prompt": GROVE_PROMPT,
        "output": text,
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": 0,
        "max_tokens": GROVE_MAX_TOKENS,
        "thinking": False,
    }
    _write_json(path, payload)
    _log(f"grove stored: {path.name}")


def _greedy(runtime: Any) -> str:
    prompt = runtime.encode_chat([{"role": "user", "content": GROVE_PROMPT}], None, False)
    tokens = runtime.generate(prompt, GROVE_MAX_TOKENS, 0.0, 1.0, 0)
    if runtime._hit_limit:
        message = "grove completion hit the token cap"
        raise RuntimeError(message)
    text = runtime.decode(tokens).rstrip()
    if not text:
        message = "grove completion is empty"
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
