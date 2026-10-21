"""Expert firing log. The request path only adds integers.

The lull writes a finished request. It does not change an expert's width.
"""

from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

LAYERS = 40
EXPERTS = 256
WINDOW = 16384
QUIET_SECONDS = 0.25

_lock = threading.Lock()
_pending: dict[int, list[tuple[str, RouteScratch]]] = {}


def relative_score(hits: int, layer_hits: int, experts: int = EXPERTS) -> float:
    """Score against the experts in one layer. A uniform expert scores 1."""
    if layer_hits <= 0 or experts <= 0:
        return 0.0
    return hits / (layer_hits / experts)


class RouteScratch:
    """In-memory adds for one request. A lull is what writes them."""

    def __init__(self, layers: int, experts: int) -> None:
        self.layers = layers
        self.experts = experts
        self.prefill = np.zeros((layers, experts), dtype=np.uint32)
        self.decode = np.zeros((layers, experts), dtype=np.uint32)
        self.prefill_tokens = 0
        self.decode_tokens = 0
        self._held: np.ndarray | None = None
        self._clear_turn()

    def begin(self) -> None:
        self.prefill.fill(0)
        self.decode.fill(0)
        self.prefill_tokens = 0
        self.decode_tokens = 0
        self._held = None
        self._clear_turn()

    def note_turn(
        self,
        *,
        cached_tokens: int,
        finished_at: int,
        prefill_seconds: float,
        decode_seconds: float,
        fp8_experts: int,
        fp4_experts: int,
        int2_experts: int,
    ) -> None:
        """Cache length, finish time, durations, and the widths that served the request."""
        self.cached_tokens = int(cached_tokens)
        self.finished_at = int(finished_at)
        self.prefill_seconds = float(prefill_seconds)
        self.decode_seconds = float(decode_seconds)
        self.fp8_experts = int(fp8_experts)
        self.fp4_experts = int(fp4_experts)
        self.int2_experts = int(int2_experts)

    def _clear_turn(self) -> None:
        self.cached_tokens = 0
        self.finished_at = 0
        self.prefill_seconds = 0.0
        self.decode_seconds = 0.0
        self.fp8_experts = 0
        self.fp4_experts = 0
        self.int2_experts = 0

    def add_prefill(self, layer: int, rows: np.ndarray) -> None:
        count = int(rows.shape[0]) if rows.ndim else 0
        if count <= 0:
            return
        if layer == 0:
            self.prefill_tokens += count
        self._add(self.prefill, layer, rows)

    def note_decode(self, rows: np.ndarray) -> None:
        """Replace the uncommitted span. ``rows`` is layers, tokens, experts."""
        self._held = np.array(rows, dtype=np.int64, copy=True)

    def take_decode(self, tokens: int) -> None:
        self._take(self.decode, "decode_tokens", tokens)

    def take_prefill_span(self) -> None:
        """Count the uncommitted span as prefill. The last prompt token uses it."""
        held = self._held
        if held is None:
            return
        self._take(self.prefill, "prefill_tokens", int(held.shape[1]))

    def drop_decode(self) -> None:
        self._held = None

    def snapshot(self) -> RouteScratch:
        snap = RouteScratch(self.layers, self.experts)
        snap.prefill = self.prefill.copy()
        snap.decode = self.decode.copy()
        snap.prefill_tokens = self.prefill_tokens
        snap.decode_tokens = self.decode_tokens
        snap.cached_tokens = self.cached_tokens
        snap.finished_at = self.finished_at
        snap.prefill_seconds = self.prefill_seconds
        snap.decode_seconds = self.decode_seconds
        snap.fp8_experts = self.fp8_experts
        snap.fp4_experts = self.fp4_experts
        snap.int2_experts = self.int2_experts
        return snap

    def _take(self, table: np.ndarray, field: str, tokens: int) -> None:
        held = self._held
        self._held = None
        if held is None or tokens <= 0:
            return
        part = held[:, :tokens, :]
        for layer in range(self.layers):
            self._add(table, layer, part[layer])
        setattr(self, field, int(getattr(self, field)) + int(min(tokens, held.shape[1])))

    def _add(self, table: np.ndarray, layer: int, rows: np.ndarray) -> None:
        flat = np.asarray(rows, dtype=np.int64).reshape(-1)
        keep = flat[(flat >= 0) & (flat < self.experts)]
        if keep.size == 0:
            return
        table[layer] += np.bincount(keep, minlength=self.experts).astype(np.uint32)


class FiringStore:
    """SQLite log beside the telemetry directory."""

    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.executescript(
            """
            CREATE TABLE IF NOT EXISTS request (
                id INTEGER PRIMARY KEY,
                record TEXT NOT NULL,
                prefill_tokens INTEGER NOT NULL,
                decode_tokens INTEGER NOT NULL,
                complete INTEGER NOT NULL,
                cached_tokens INTEGER NOT NULL DEFAULT 0,
                finished_at INTEGER NOT NULL DEFAULT 0,
                prefill_seconds REAL NOT NULL DEFAULT 0,
                decode_seconds REAL NOT NULL DEFAULT 0,
                fp8_experts INTEGER NOT NULL DEFAULT 0,
                fp4_experts INTEGER NOT NULL DEFAULT 0,
                int2_experts INTEGER NOT NULL DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS layer_hits (
                request_id INTEGER NOT NULL,
                layer INTEGER NOT NULL,
                prefill BLOB NOT NULL,
                decode BLOB NOT NULL,
                PRIMARY KEY (request_id, layer)
            );
            """
        )

    def close(self) -> None:
        self._db.close()

    def commit_request(self, record: str, routes: RouteScratch) -> None:
        """One transaction for a finished request."""
        with self._db:
            cur = self._db.execute(
                "INSERT INTO request ("
                " record, prefill_tokens, decode_tokens, complete,"
                " cached_tokens, finished_at, prefill_seconds, decode_seconds,"
                " fp8_experts, fp4_experts, int2_experts"
                ") VALUES (?, ?, ?, 1, ?, ?, ?, ?, ?, ?, ?)",
                (
                    record,
                    int(routes.prefill_tokens),
                    int(routes.decode_tokens),
                    int(routes.cached_tokens),
                    int(routes.finished_at),
                    float(routes.prefill_seconds),
                    float(routes.decode_seconds),
                    int(routes.fp8_experts),
                    int(routes.fp4_experts),
                    int(routes.int2_experts),
                ),
            )
            request_id = int(cur.lastrowid or 0)
            for layer in range(routes.layers):
                self._db.execute(
                    "INSERT INTO layer_hits (request_id, layer, prefill, decode)"
                    " VALUES (?, ?, ?, ?)",
                    (
                        request_id,
                        layer,
                        routes.prefill[layer].tobytes(),
                        routes.decode[layer].tobytes(),
                    ),
                )

    def window(self, record: str, layers: int, experts: int) -> tuple[int, list[np.ndarray]]:
        """Most recent complete tokens, capped at the window. A partial request is floored."""
        rows = self._db.execute(
            "SELECT id, prefill_tokens, decode_tokens FROM request"
            " WHERE record = ? AND complete = 1 ORDER BY id DESC",
            (record,),
        ).fetchall()
        hits = [np.zeros(experts, dtype=np.uint64) for _ in range(layers)]
        eligible = 0
        for request_id, prefill_tokens, decode_tokens in rows:
            span = int(prefill_tokens) + int(decode_tokens)
            if span <= 0:
                continue
            room = WINDOW - eligible
            if room <= 0:
                break
            use = min(span, room)
            stored = self._db.execute(
                "SELECT layer, prefill, decode FROM layer_hits WHERE request_id = ?",
                (int(request_id),),
            ).fetchall()
            for layer, prefill, decode in stored:
                if layer < 0 or layer >= layers:
                    continue
                both = np.frombuffer(prefill, dtype=np.uint32).astype(np.uint64)
                both = both + np.frombuffer(decode, dtype=np.uint32).astype(np.uint64)
                if both.size != experts:
                    continue
                if use == span:
                    hits[layer] += both
                else:
                    hits[layer] += np.floor(both * (use / span)).astype(np.uint64)
            eligible += use
        return eligible, hits


def telemetry_dir(folder: str) -> Path:
    name = Path(folder).name
    if name in ("", ".", ".."):
        message = "firing folder is empty"
        raise ValueError(message)
    return Path.home() / ".favlo" / "telemetry" / name


def _store(folder: str) -> FiringStore:
    return FiringStore(telemetry_dir(folder) / "firing.sqlite")


def _width_counts(runtime: Any) -> tuple[int, int, int]:
    """fp8, fp4, and int2. This record is uniform fp4, so int2 stays 0."""
    manifest = getattr(runtime, "manifest", None) or {}
    layers = int(manifest.get("layers") or LAYERS)
    experts = int(manifest.get("experts") or EXPERTS)
    return 0, layers * experts, 0


def _stamp_turn(routes: RouteScratch, runtime: Any) -> None:
    fp8, fp4, int2 = _width_counts(runtime)
    routes.note_turn(
        cached_tokens=int(getattr(runtime, "_cached_tokens", 0) or 0),
        finished_at=int(time.time()),
        prefill_seconds=float(getattr(runtime, "_prefill_seconds", 0.0) or 0.0),
        decode_seconds=float(getattr(runtime, "_decode_seconds", 0.0) or 0.0),
        fp8_experts=fp8,
        fp4_experts=fp4,
        int2_experts=int2,
    )


def open_request(runtime: Any) -> None:
    routes = getattr(runtime, "routes", None)
    if routes is not None:
        routes.begin()
    runtime._firing_open = True


def finish_request(runtime: Any, selection: Any, *, complete: bool) -> None:
    """Queue this request. The listener runs the lull after the request closes."""
    runtime._firing_open = False
    routes = getattr(runtime, "routes", None)
    if routes is None or selection is None:
        return
    if not complete:
        routes.drop_decode()
        routes.begin()
        return
    _stamp_turn(routes, runtime)
    snap = routes.snapshot()
    routes.begin()
    record = str(getattr(runtime, "root", ""))
    token = id(runtime)
    with _lock:
        _pending.setdefault(token, []).append((record, snap))
    runtime._idle_heartbeat = (selection.folder, record, token)


def _lull(runtime: Any, folder: str, record: str, token: int) -> None:
    time.sleep(QUIET_SECONDS)
    with _lock:
        if getattr(runtime, "_firing_open", False):
            return
        batch = _pending.pop(token, [])
    if not batch:
        return
    store = _store(folder)
    try:
        for identity, routes in batch:
            if identity != record:
                continue
            store.commit_request(identity, routes)
    finally:
        store.close()


def idle_heartbeat(runtime: Any, selection: Any = None) -> None:
    """Write the queued routes. The listener calls this after the request returns."""
    del selection
    job = getattr(runtime, "_idle_heartbeat", None)
    if job is None:
        return
    runtime._idle_heartbeat = None
    _lull(runtime, *job)
