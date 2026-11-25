"""Expert firing log. The request path only adds integers.

The lull pulls the finished hist plane, stamps the turn, and writes SQLite.
It does not change an expert's width. A mid-forward stop is not stored.
Hits count every visit, so layers 6–17 are recorded three times. Token
counts are the prefill suffix and the decode steps, once each.
"""

from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

LAYERS = 24
EXPERTS = 128
QUIET_SECONDS = 0.25

_lock = threading.Lock()
_pending: dict[int, list[tuple[str, int | None, RouteScratch | None]]] = {}


class RouteScratch:
    """Host copy of one request. The device hist is the source of the counts."""

    def __init__(self, layers: int, experts: int) -> None:
        self.layers = layers
        self.experts = experts
        self.prefill = np.zeros((layers, experts), dtype=np.uint32)
        self.decode = np.zeros((layers, experts), dtype=np.uint32)
        self.prefill_tokens = 0
        self.decode_tokens = 0
        self._clear_turn()

    def begin(self) -> None:
        self.prefill.fill(0)
        self.decode.fill(0)
        self.prefill_tokens = 0
        self.decode_tokens = 0
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

    def drop_decode(self) -> None:
        self.decode.fill(0)
        self.decode_tokens = 0

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


def telemetry_dir(folder: str) -> Path:
    name = Path(folder).name
    if name in ("", ".", ".."):
        message = "firing folder is empty"
        raise ValueError(message)
    return Path.home() / ".favlo" / "telemetry" / name


def flat_hit_scores(
    path: Path,
    *,
    layers: int = LAYERS,
    experts: int = EXPERTS,
    since: int | None = None,
) -> dict[int, float]:
    """Flat index ``layer * experts + expert`` to total hits."""
    totals = np.zeros((layers, experts), dtype=np.float64)
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as db:
        if since is None:
            rows = db.execute("SELECT layer, prefill, decode FROM layer_hits")
        else:
            rows = db.execute(
                "SELECT layer_hits.layer, layer_hits.prefill, layer_hits.decode"
                " FROM layer_hits JOIN request ON request.id = layer_hits.request_id"
                " WHERE request.finished_at >= ?",
                (int(since),),
            )
        for layer, prefill, decode in rows:
            if layer < 0 or layer >= layers:
                continue
            pre = np.frombuffer(prefill, dtype=np.uint32)
            dec = np.frombuffer(decode, dtype=np.uint32)
            if pre.size != experts or dec.size != experts:
                message = f"layer {layer} hit vector length is not {experts}"
                raise ValueError(message)
            totals[layer] += pre
            totals[layer] += dec
    flat = totals.ravel()
    return {index: float(flat[index]) for index in range(flat.size)}


def _store(folder: str) -> FiringStore:
    return FiringStore(telemetry_dir(folder) / "firing.sqlite")


def _width_counts(runtime: Any) -> tuple[int, int, int]:
    """fp8, fp4, and int2. int2 stays 0. A mixed pack counts its width plane."""
    weights = getattr(runtime, "weights", None)
    layers = getattr(weights, "layers", None) if weights is not None else None
    if not layers:
        return 0, LAYERS * EXPERTS, 0
    fp8 = 0
    fp4 = 0
    for layer in layers:
        widths = getattr(layer, "widths", None)
        if widths is not None:
            fp8 += int(np.sum(widths == 8))
            fp4 += int(np.sum(widths == 4))
            continue
        experts = getattr(layer, "w13", None)
        if getattr(layer, "nvfp4", False) or getattr(experts, "fp4", False):
            fp4 += EXPERTS
        else:
            fp8 += EXPERTS
    return fp8, fp4, 0


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


def _clear_hist(runtime: Any, plane: int | None = None) -> None:
    clear = getattr(runtime, "_clear_route_hist", None)
    if callable(clear):
        clear(plane)


def open_request(runtime: Any) -> None:
    routes = getattr(runtime, "routes", None)
    if routes is not None:
        routes.begin()
    runtime._prefill_token_count = 0
    runtime._decode_token_count = 0
    plane = int(getattr(runtime, "_route_plane", 0) or 0)
    bind = getattr(runtime, "_bind_route_plane", None)
    if callable(bind):
        bind(plane)
    _clear_hist(runtime, plane)
    runtime._firing_open = True


def finish_request(runtime: Any, selection: Any, *, complete: bool) -> None:
    """Flip the hist plane and schedule the lull. The pull runs off this path."""
    runtime._firing_open = False
    routes = getattr(runtime, "routes", None)
    if routes is None or selection is None:
        return
    if not complete:
        routes.drop_decode()
        routes.begin()
        _clear_hist(runtime)
        return
    record = str(getattr(runtime, "root", ""))
    token = id(runtime)
    pull = getattr(runtime, "_pull_route_hist", None)
    planes = getattr(runtime, "_route_hist_planes", None)
    if callable(pull) and planes is not None:
        finished = int(getattr(runtime, "_route_plane", 0) or 0)
        nxt = 1 - (finished % 2)
        bind = getattr(runtime, "_bind_route_plane", None)
        if callable(bind):
            bind(nxt)
        else:
            runtime._route_plane = nxt
        _clear_hist(runtime, nxt)
        snap: RouteScratch | None = None
        plane: int | None = finished
    else:
        _stamp_turn(routes, runtime)
        snap = routes.snapshot()
        routes.begin()
        _clear_hist(runtime)
        plane = None
    with _lock:
        _pending.setdefault(token, []).append((record, plane, snap))
    runtime._idle_heartbeat = (selection.folder, record, token)


def _lull(runtime: Any, folder: str, record: str, token: int) -> None:
    time.sleep(QUIET_SECONDS)
    with _lock:
        if getattr(runtime, "_firing_open", False):
            return
        batch = _pending.pop(token, [])
    if not batch:
        return
    routes = getattr(runtime, "routes", None)
    pull = getattr(runtime, "_pull_route_hist", None)
    store = _store(folder)
    try:
        for identity, plane, snap in batch:
            if identity != record:
                continue
            if snap is None:
                if callable(pull) and plane is not None:
                    pull(plane)
                if routes is None:
                    continue
                _stamp_turn(routes, runtime)
                snap = routes.snapshot()
                routes.begin()
                _clear_hist(runtime, plane)
            store.commit_request(identity, snap)
    finally:
        store.close()


def idle_heartbeat(runtime: Any, selection: Any = None) -> None:
    """Pull the finished plane, stamp, and write. The listener calls this after close."""
    del selection
    job = getattr(runtime, "_idle_heartbeat", None)
    if job is None:
        return
    runtime._idle_heartbeat = None
    _lull(runtime, *job)
