"""Deterministic firing capture report. Does not load a checkpoint."""

from __future__ import annotations

import sqlite3
from datetime import timezone
from pathlib import Path

import numpy as np
from harness.analyze_experts_firing import (
    EXPERTS,
    FP4,
    FP8,
    LAYERS,
    format_report,
    global_top_mask,
    load_widths,
    plane_totals,
    report,
    top_mask,
)


def _write_log(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as db:
        db.executescript(
            """
            CREATE TABLE request (
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
            CREATE TABLE layer_hits (
                request_id INTEGER NOT NULL,
                layer INTEGER NOT NULL,
                prefill BLOB NOT NULL,
                decode BLOB NOT NULL,
                PRIMARY KEY (request_id, layer)
            );
            """
        )
        db.execute(
            "INSERT INTO request ("
            " id, record, prefill_tokens, decode_tokens, complete,"
            " cached_tokens, finished_at, prefill_seconds, decode_seconds"
            ") VALUES (1, 'example-model', 2, 1, 1, 0, 200, 1.0, 0.5)"
        )
        db.execute(
            "INSERT INTO request ("
            " id, record, prefill_tokens, decode_tokens, complete,"
            " cached_tokens, finished_at, prefill_seconds, decode_seconds"
            ") VALUES (2, 'example-model', 2, 1, 1, 0, 100, 1.0, 0.5)"
        )
        zero = np.zeros(EXPERTS, dtype=np.uint32)
        for request_id, hot, cold in ((1, 10, 1), (2, 4, 2)):
            for layer in range(LAYERS):
                pre = zero.copy()
                dec = zero.copy()
                if layer == 0:
                    pre[0] = hot
                    pre[1] = cold
                    dec[0] = 1
                    dec[2] = 1
                db.execute(
                    "INSERT INTO layer_hits (request_id, layer, prefill, decode)"
                    " VALUES (?, ?, ?, ?)",
                    (request_id, layer, pre.tobytes(), dec.tobytes()),
                )


def _write_widths(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    widths = np.full((LAYERS, EXPERTS), FP4, dtype=np.uint8)
    widths[:, 0] = FP8
    widths[0, 2] = FP8
    widths.tofile(path)


def test_plane_totals_respect_since(tmp_path: Path) -> None:
    db = tmp_path / "firing.sqlite"
    _write_log(db)
    prefill, decode = plane_totals(db, since=150)
    assert int(prefill[0, 0]) == 10
    assert int(prefill[0, 1]) == 1
    assert int(decode[0, 0]) == 1
    assert int(decode[0, 2]) == 1
    older, _decode = plane_totals(db, since=50)
    assert int(older[0, 0]) == 14


def test_top_mask_keeps_the_lower_index_on_a_tie() -> None:
    hits = np.zeros((1, 4), dtype=np.uint64)
    hits[0] = [5, 5, 1, 0]
    mask = top_mask(hits, np.array([2]))
    assert list(mask[0]) == [True, True, False, False]
    flat = global_top_mask(hits, 1)
    assert list(flat[0]) == [True, False, False, False]


def test_report_text_is_stable(tmp_path: Path) -> None:
    db = tmp_path / "firing.sqlite"
    widths = tmp_path / "experts" / "widths.bin"
    _write_log(db)
    _write_widths(widths)
    summary = report(
        folder="example-model",
        db=db,
        widths_path=widths,
        since=150,
    )
    text = format_report(summary, tz=timezone.utc)
    assert summary["requests"] == 1
    assert summary["fp8_experts"] == LAYERS + 1
    assert "folder example-model" in text
    assert "capture_pct" in text
    assert "from 1970-01-01" in text
    loaded = load_widths(widths)
    assert loaded.shape == (LAYERS, EXPERTS)
    assert int(loaded[0, 0]) == FP8
