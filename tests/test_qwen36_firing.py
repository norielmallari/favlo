"""Firing counters for the 35B pipeline. They do not load a checkpoint."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from qwen36_35b_a3b import firing
from qwen36_35b_a3b.firing import (
    FiringStore,
    RouteScratch,
    finish_request,
    idle_heartbeat,
    open_request,
    relative_score,
)


def _clear_pending() -> None:
    firing._pending.clear()


def test_a_uniform_expert_scores_one() -> None:
    assert relative_score(4, 4 * 256) == 1
    assert relative_score(0, 0) == 0
    assert relative_score(8, 8, experts=256) == 256


def test_prefill_bins_router_ids_once() -> None:
    routes = RouteScratch(2, 8)
    rows = np.array([[1, 1, 2, 0xFFFFFFFF], [7, 8, 3, 3]], dtype=np.int64)
    routes.add_prefill(0, rows)
    routes.add_prefill(1, rows)
    assert routes.prefill_tokens == 2
    assert int(routes.prefill[0, 1]) == 2
    assert int(routes.prefill[0, 2]) == 1
    assert int(routes.prefill[0, 3]) == 2
    assert int(routes.prefill[0, 7]) == 1
    assert int(routes.prefill[0].sum()) == 6
    assert int(routes.prefill[1, 1]) == 2


def test_decode_counts_a_kept_token_once() -> None:
    routes = RouteScratch(2, 4)
    first = np.array([[[0, 1, 0, 1]], [[2, 2, 2, 2]]], dtype=np.int64)
    second = np.array([[[3, 3, 3, 3]], [[1, 1, 1, 1]]], dtype=np.int64)
    routes.note_decode(first)
    routes.note_decode(second)
    routes.take_decode(1)
    assert int(routes.decode[0, 3]) == 4
    assert int(routes.decode[0, 0]) == 0
    assert routes.decode_tokens == 1
    routes.take_decode(1)
    assert routes.decode_tokens == 1
    routes.note_decode(first)
    routes.drop_decode()
    routes.take_decode(1)
    assert int(routes.decode[0, 0]) == 0


def test_the_last_prompt_token_counts_as_prefill() -> None:
    routes = RouteScratch(1, 4)
    routes.note_decode(np.array([[[1, 2, 1, 2]]], dtype=np.int64))
    routes.take_prefill_span()
    assert routes.prefill_tokens == 1
    assert routes.decode_tokens == 0
    assert int(routes.prefill[0, 1]) == 2
    assert routes._held is None


def test_a_snapshot_keeps_the_turn_after_begin() -> None:
    routes = RouteScratch(1, 4)
    routes.add_prefill(0, np.array([[0, 1]], dtype=np.int64))
    routes.note_turn(
        cached_tokens=3,
        finished_at=10,
        prefill_seconds=1.25,
        decode_seconds=0.5,
        fp8_experts=0,
        fp4_experts=10240,
        int2_experts=0,
    )
    snap = routes.snapshot()
    routes.begin()
    assert routes.prefill_tokens == 0
    assert routes.cached_tokens == 0
    assert snap.prefill_tokens == 1
    assert snap.cached_tokens == 3
    assert int(snap.prefill[0, 0]) == 1


def test_the_window_floors_a_partial_request(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(firing, "WINDOW", 10)
    store = FiringStore(tmp_path / "firing.sqlite")
    try:
        older = RouteScratch(1, 2)
        older.prefill_tokens = 6
        older.prefill[0, 0] = 6
        newer = RouteScratch(1, 2)
        newer.decode_tokens = 6
        newer.decode[0, 1] = 6
        store.commit_request("record", older)
        store.commit_request("record", newer)
        eligible, hits = store.window("record", 1, 2)
        assert eligible == 10
        assert int(hits[0][1]) == 6
        assert int(hits[0][0]) == 4
        assert relative_score(int(hits[0][0]), int(hits[0].sum()), experts=2) == 0.8
    finally:
        store.close()


def test_an_unfinished_request_is_not_queued() -> None:
    _clear_pending()
    routes = RouteScratch(1, 2)
    routes.note_decode(np.array([[[0, 1]]], dtype=np.int64))
    routes.prefill_tokens = 4
    runtime = SimpleNamespace(routes=routes, root="record", _firing_open=True)
    selection = SimpleNamespace(folder="example-model")
    finish_request(runtime, selection, complete=False)
    assert runtime._firing_open is False
    assert getattr(runtime, "_idle_heartbeat", None) is None
    assert routes.prefill_tokens == 0
    assert routes._held is None
    assert firing._pending == {}


def test_the_lull_writes_after_the_request_closes(tmp_path, monkeypatch) -> None:
    _clear_pending()
    monkeypatch.setattr(firing, "telemetry_dir", lambda folder: tmp_path / folder)
    monkeypatch.setattr(firing.time, "sleep", lambda _seconds: None)
    routes = RouteScratch(1, 2)
    routes.add_prefill(0, np.array([[0, 1]], dtype=np.int64))
    runtime = SimpleNamespace(
        routes=routes,
        root=tmp_path / "record",
        manifest={"layers": 1, "experts": 2},
        _cached_tokens=5,
        _prefill_seconds=1.0,
        _decode_seconds=2.0,
        _firing_open=True,
    )
    selection = SimpleNamespace(folder="example-model")
    finish_request(runtime, selection, complete=True)
    assert runtime._firing_open is False
    idle_heartbeat(runtime, selection)
    assert runtime._idle_heartbeat is None
    store = FiringStore(tmp_path / "example-model" / "firing.sqlite")
    try:
        eligible, hits = store.window(str(tmp_path / "record"), 1, 2)
        assert eligible == 1
        assert int(hits[0][0]) == 1
        assert int(hits[0][1]) == 1
        row = store._db.execute(
            "SELECT fp8_experts, fp4_experts, int2_experts, cached_tokens FROM request"
        ).fetchone()
        assert row == (0, 2, 0, 5)
    finally:
        store.close()


def test_an_open_request_keeps_the_queued_counts(tmp_path, monkeypatch) -> None:
    _clear_pending()
    monkeypatch.setattr(firing, "telemetry_dir", lambda folder: tmp_path / folder)

    def reopen(_seconds: float) -> None:
        runtime._firing_open = True

    monkeypatch.setattr(firing.time, "sleep", reopen)
    routes = RouteScratch(1, 2)
    routes.prefill_tokens = 1
    routes.prefill[0, 0] = 1
    runtime = SimpleNamespace(
        routes=routes,
        root="record",
        manifest={"layers": 1, "experts": 2},
        _cached_tokens=0,
        _prefill_seconds=0.0,
        _decode_seconds=0.0,
        _firing_open=True,
    )
    finish_request(runtime, SimpleNamespace(folder="example-model"), complete=True)
    idle_heartbeat(runtime, None)
    assert not (tmp_path / "example-model" / "firing.sqlite").exists()
    monkeypatch.setattr(firing.time, "sleep", lambda _seconds: None)
    runtime._firing_open = False
    runtime._idle_heartbeat = ("example-model", "record", id(runtime))
    idle_heartbeat(runtime, None)
    store = FiringStore(tmp_path / "example-model" / "firing.sqlite")
    try:
        eligible, _hits = store.window("record", 1, 2)
        assert eligible == 1
    finally:
        store.close()


def test_open_request_zeros_the_scratch() -> None:
    routes = RouteScratch(1, 2)
    routes.prefill_tokens = 9
    runtime = SimpleNamespace(routes=routes)
    open_request(runtime)
    assert routes.prefill_tokens == 0
    assert runtime._firing_open is True
