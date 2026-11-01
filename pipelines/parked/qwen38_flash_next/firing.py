"""Expert firing log. The request path only adds integers."""

from __future__ import annotations

import random
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import numpy as np

WINDOW = 16384
FLOOR = 4096
EXPERTS = 512


def relative_score(hits: int, layer_hits: int, experts: int = EXPERTS) -> float:
    """Score against the experts in one layer. A uniform expert scores 1."""
    if layer_hits <= 0 or experts <= 0:
        return 0.0
    return hits / (layer_hits / experts)


def desire_width(served: int, score: float) -> int:
    """One step. fp8 does not become int2 in one pack."""
    if served == 4:
        if score >= 4:
            return 8
        if score <= 0.25:
            return 2
        return 4
    if served == 8:
        return 4 if score <= 2 else 8
    if served == 2:
        return 4 if score >= 0.5 else 2
    return served


@dataclass(frozen=True)
class MovePlan:
    """Movers that have kept the same ids, and whether that set has held."""

    movers: tuple[tuple[int, int, int], ...]
    ready: bool
    changed: bool


def mover_key(movers: tuple[tuple[int, int, int], ...]) -> str:
    return ",".join(f"{layer}:{expert}:{bits}" for layer, expert, bits in movers)


def plan_moves(
    served: list[list[int]],
    hits: list[np.ndarray],
    *,
    eligible: int,
    pins: list[set[int]],
    cooled: set[tuple[int, int]],
    previous_key: str,
    previous_eligible: int,
) -> MovePlan:
    """Desires from the window. A pin stays. A cooled expert stays until the floor."""
    chosen: list[tuple[int, int, int]] = []
    if eligible >= FLOOR:
        for layer, row in enumerate(served):
            layer_hits = int(hits[layer].sum())
            for expert, width in enumerate(row):
                if expert in pins[layer] or (layer, expert) in cooled:
                    continue
                score = relative_score(int(hits[layer][expert]), layer_hits, len(row))
                nxt = desire_width(int(width), score)
                if nxt != int(width):
                    chosen.append((layer, expert, nxt))
    movers = tuple(chosen)
    key = mover_key(movers)
    changed = key != previous_key
    stable_for = 0 if changed else eligible - previous_eligible
    ready = bool(movers) and not changed and stable_for >= FLOOR
    return MovePlan(movers=movers, ready=ready, changed=changed)


def heartbeat_picks(
    served: list[list[int]],
    hits: list[np.ndarray],
    *,
    eligible: int,
    pins: list[set[int]],
    cooled: set[tuple[int, int]],
    rng: random.Random,
) -> tuple[tuple[int, int, int] | None, tuple[int, int, int] | None]:
    """One promotion and one demotion. A tie is one random expert."""
    promote: list[tuple[int, int, int]] = []
    demote: list[tuple[int, int, int]] = []
    if eligible < FLOOR:
        return None, None
    for layer, row in enumerate(served):
        layer_hits = int(hits[layer].sum())
        for expert, width in enumerate(row):
            if expert in pins[layer] or (layer, expert) in cooled:
                continue
            nxt = desire_width(
                int(width), relative_score(int(hits[layer][expert]), layer_hits, len(row))
            )
            if nxt > int(width):
                promote.append((layer, expert, nxt))
            elif nxt < int(width):
                demote.append((layer, expert, nxt))
    up = rng.choice(promote) if promote else None
    down = rng.choice(demote) if demote else None
    return up, down


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
        held = self._held
        self._held = None
        if held is None or tokens <= 0:
            return
        part = held[:, :tokens, :]
        for layer in range(self.layers):
            self._add(self.decode, layer, part[layer])
        self.decode_tokens += int(min(tokens, held.shape[1]))

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
            CREATE TABLE IF NOT EXISTS packed (
                layer INTEGER NOT NULL,
                expert INTEGER NOT NULL,
                eligible INTEGER NOT NULL,
                PRIMARY KEY (layer, expert)
            );
            CREATE TABLE IF NOT EXISTS meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            """
        )
        self._ensure_request_columns()

    def _ensure_request_columns(self) -> None:
        have = {str(row[1]) for row in self._db.execute("PRAGMA table_info(request)")}
        adds = {
            "cached_tokens": "INTEGER NOT NULL DEFAULT 0",
            "finished_at": "INTEGER NOT NULL DEFAULT 0",
            "prefill_seconds": "REAL NOT NULL DEFAULT 0",
            "decode_seconds": "REAL NOT NULL DEFAULT 0",
            "fp8_experts": "INTEGER NOT NULL DEFAULT 0",
            "fp4_experts": "INTEGER NOT NULL DEFAULT 0",
            "int2_experts": "INTEGER NOT NULL DEFAULT 0",
        }
        missing = [(name, decl) for name, decl in adds.items() if name not in have]
        if not missing:
            return
        with self._db:
            for name, decl in missing:
                self._db.execute(f"ALTER TABLE request ADD COLUMN {name} {decl}")

    def close(self) -> None:
        self._db.close()

    def commit_request(
        self,
        record: str,
        routes: RouteScratch,
        *,
        complete: bool,
    ) -> None:
        """One transaction. An incomplete request is stored and is not evidence."""
        with self._db:
            cur = self._db.execute(
                "INSERT INTO request ("
                " record, prefill_tokens, decode_tokens, complete,"
                " cached_tokens, finished_at, prefill_seconds, decode_seconds,"
                " fp8_experts, fp4_experts, int2_experts"
                ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    record,
                    int(routes.prefill_tokens) if complete else 0,
                    int(routes.decode_tokens) if complete else 0,
                    1 if complete else 0,
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
            if not complete:
                return
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
        """Most recent complete tokens, capped at the window. Missing experts are zero."""
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
            scale = use / span
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
                    hits[layer] += np.floor(both * scale).astype(np.uint64)
            eligible += use
        return eligible, hits

    def eligible_total(self, record: str) -> int:
        row = self._db.execute(
            "SELECT COALESCE(SUM(prefill_tokens + decode_tokens), 0) FROM request"
            " WHERE record = ? AND complete = 1",
            (record,),
        ).fetchone()
        return int(row[0] if row else 0)

    def meta(self, key: str, default: str = "") -> str:
        row = self._db.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        if row is None:
            return default
        return str(row[0])

    def set_meta(self, key: str, value: str) -> None:
        with self._db:
            self._db.execute(
                "INSERT INTO meta (key, value) VALUES (?, ?)"
                " ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (key, value),
            )

    def cooled(self, eligible: int) -> set[tuple[int, int]]:
        rows = self._db.execute("SELECT layer, expert, eligible FROM packed").fetchall()
        return {
            (int(layer), int(expert))
            for layer, expert, packed_at in rows
            if eligible < int(packed_at) + FLOOR
        }

    def mark_packed(self, layer: int, expert: int, eligible: int) -> None:
        with self._db:
            self._db.execute(
                "INSERT INTO packed (layer, expert, eligible) VALUES (?, ?, ?)"
                " ON CONFLICT(layer, expert) DO UPDATE SET eligible = excluded.eligible",
                (layer, expert, eligible),
            )
