"""Application event envelopes. A run delta is not given a sequence."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any


def now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def envelope(
    *,
    event_id: str,
    stream_id: str,
    event_type: str,
    payload: dict[str, Any],
    sequence: int | None = None,
    channel_id: str | None = None,
    session_id: str | None = None,
    run_id: str | None = None,
    actor: dict[str, str] | None = None,
    correlation_id: str | None = None,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "schema_version": 1,
        "event_id": event_id,
        "stream_id": stream_id,
        "type": event_type,
        "occurred_at": now(),
        "payload": payload,
    }
    if sequence is not None:
        body["sequence"] = sequence
    if channel_id is not None:
        body["channel_id"] = channel_id
    if session_id is not None:
        body["session_id"] = session_id
    if run_id is not None:
        body["run_id"] = run_id
    if actor is not None:
        body["actor"] = actor
    if correlation_id is not None:
        body["correlation_id"] = correlation_id
    return body
