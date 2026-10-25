"""MCP and terminal memory tool wiring."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any

from favlo.memory_notes import DIM, EMBED_MODEL, MemoryStore, memory_db_path

REMEMBER_DESCRIPTION = (
    "Store one sentence that will still make sense next session: a lasting fact or a "
    "preference. Do not store a greeting, a refusal, a one-off question, the "
    "transcript, or a secret."
)

RECALL_DESCRIPTION = "Find stored sentences from an earlier session by meaning."

FORGET_DESCRIPTION = (
    "Drop one stored sentence when the person asks. The id comes from recall or remember."
)


def remember_text(
    store: MemoryStore,
    embed: Callable[[str], list[float] | str],
    text: str,
    client: str | None = None,
) -> str:
    vector = embed(text)
    if isinstance(vector, str):
        return vector
    try:
        return store.remember(text, vector, client=client)
    except ValueError as exc:
        return str(exc)


def recall_text(
    store: MemoryStore,
    embed: Callable[[str], list[float] | str],
    query: str,
) -> str | list[dict[str, Any]]:
    vector = embed(query)
    if isinstance(vector, str):
        return vector

    def reembed(line: str) -> list[float]:
        again = embed(line)
        if isinstance(again, str):
            message = again
            raise RuntimeError(message)
        return again

    try:
        rows = store.recall(query, vector, reembed=reembed)
    except (ValueError, RuntimeError) as exc:
        return str(exc)
    return rows


def forget_text(store: MemoryStore, mem_id: str) -> str:
    try:
        store.forget(mem_id)
    except KeyError as exc:
        return str(exc)
    return f"forgot {mem_id}"


def open_store(home: Any = None) -> MemoryStore:
    if home is None:
        return MemoryStore()
    if isinstance(home, Path):
        return MemoryStore(memory_db_path(home))
    return MemoryStore(memory_db_path(Path(home)))


def embed_http(base: str, text: str, model: str = EMBED_MODEL) -> list[float] | str:
    """POST one string to ``/v1/embeddings`` on the listener."""
    url = base.rstrip("/") + "/v1/embeddings"
    body = json.dumps({"model": model, "input": text}).encode()
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=None) as response:
            payload = json.loads(response.read().decode())
    except (OSError, urllib.error.URLError, json.JSONDecodeError):
        return "embeddings are not available"
    if not isinstance(payload, dict):
        return "embeddings are not available"
    data = payload.get("data")
    if not isinstance(data, list) or not data:
        return "embeddings are not available"
    row = data[0]
    if not isinstance(row, dict):
        return "embeddings are not available"
    vector = row.get("embedding")
    if not isinstance(vector, list) or len(vector) != DIM:
        return "embeddings are not available"
    return [float(value) for value in vector]
