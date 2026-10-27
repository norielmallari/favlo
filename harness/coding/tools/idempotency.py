"""Remember one outcome per key. A changed argument is a conflict."""

from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from harness.coding.tools.errors import ToolError
from harness.coding.tools.limits import IDEMPOTENCY_KEY_MAX


@dataclass(frozen=True)
class Reservation:
    """``replay`` carries the recorded JSON. ``pending`` has not finished."""

    kind: str
    outcome: str | None = None
    job_id: str | None = None


class IdempotencyStore:
    def __init__(self, root: Path) -> None:
        self.root = root
        self._lock = threading.Lock()
        self.root.mkdir(parents=True, exist_ok=True)

    def reserve(self, key: str, payload: dict[str, Any]) -> Reservation:
        token = _key(key)
        fingerprint = _fingerprint(payload)
        with self._lock:
            current = self._read(token)
            if current is None:
                self._write(token, {"fingerprint": fingerprint, "job_id": None, "outcome": None})
                return Reservation("fresh")
            if current.get("fingerprint") != fingerprint:
                raise ToolError("conflict", "idempotency key was used with different arguments")
            outcome = current.get("outcome")
            if isinstance(outcome, str):
                return Reservation("replay", outcome=outcome)
            job_id = current.get("job_id")
            return Reservation("pending", job_id=job_id if isinstance(job_id, str) else None)

    def note_job(self, key: str, job_id: str) -> None:
        token = _key(key)
        with self._lock:
            current = self._read(token)
            if current is None:
                return
            current["job_id"] = job_id
            self._write(token, current)

    def finish(self, key: str, outcome: str) -> None:
        token = _key(key)
        with self._lock:
            current = self._read(token)
            if current is None:
                return
            current["outcome"] = outcome
            self._write(token, current)

    def abandon(self, key: str) -> None:
        """Drop a reservation that did not publish an outcome."""
        token = _key(key)
        with self._lock:
            current = self._read(token)
            if current is None or isinstance(current.get("outcome"), str):
                return
            self._path(token).unlink(missing_ok=True)

    def _path(self, token: str) -> Path:
        return self.root / f"{token}.json"

    def _read(self, token: str) -> dict[str, Any] | None:
        path = self._path(token)
        if not path.is_file():
            return None
        try:
            body = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None
        if not isinstance(body, dict):
            return None
        return body

    def _write(self, token: str, body: dict[str, Any]) -> None:
        path = self._path(token)
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(body, sort_keys=True), encoding="utf-8")
        temporary.replace(path)


def check_key(key: str) -> str:
    if not isinstance(key, str) or key == "":
        raise ToolError("invalid_arguments", "idempotency_key is required")
    if len(key) > IDEMPOTENCY_KEY_MAX:
        raise ToolError("invalid_arguments", "idempotency_key is too long")
    return key


def _key(key: str) -> str:
    check_key(key)
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def _fingerprint(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()
