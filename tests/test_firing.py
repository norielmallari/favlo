"""Score, hysteresis, and the firing log."""

import random
import sqlite3

import numpy as np
from qwen38_flash_next.firing import (
    FLOOR,
    FiringStore,
    RouteScratch,
    desire_width,
    heartbeat_picks,
    mover_key,
    plan_moves,
    relative_score,
)


def test_uniform_expert_scores_one() -> None:
    assert relative_score(10, 5120) == 1
    assert relative_score(0, 0) == 0


def test_desire_moves_one_step() -> None:
    assert desire_width(4, 4) == 8
    assert desire_width(4, 3.9) == 4
    assert desire_width(4, 0.25) == 2
    assert desire_width(4, 0.26) == 4
    assert desire_width(8, 2) == 4
    assert desire_width(8, 2.1) == 8
    assert desire_width(2, 0.5) == 4
    assert desire_width(2, 0.49) == 2


def test_pin_and_cooldown_stay() -> None:
    served = [[4, 4, 4, 4]]
    hits = [np.array([100, 0, 0, 0], dtype=np.uint64)]
    pinned = plan_moves(
        served,
        hits,
        eligible=FLOOR,
        pins=[{0}],
        cooled=set(),
        previous_key="",
        previous_eligible=0,
    )
    assert all(expert != 0 for _layer, expert, _bits in pinned.movers)
    cooled = plan_moves(
        served,
        hits,
        eligible=FLOOR,
        pins=[set()],
        cooled={(0, 0)},
        previous_key="",
        previous_eligible=0,
    )
    assert all(expert != 0 for _layer, expert, _bits in cooled.movers)


def test_movers_wait_until_the_set_holds() -> None:
    served = [[4, 4, 4, 4]]
    hits = [np.array([100, 0, 0, 0], dtype=np.uint64)]
    first = plan_moves(
        served,
        hits,
        eligible=FLOOR,
        pins=[set()],
        cooled=set(),
        previous_key="",
        previous_eligible=0,
    )
    assert (0, 0, 8) in first.movers
    assert (0, 1, 2) in first.movers
    assert first.changed
    assert not first.ready
    second = plan_moves(
        served,
        hits,
        eligible=FLOOR * 2,
        pins=[set()],
        cooled=set(),
        previous_key=mover_key(first.movers),
        previous_eligible=FLOOR,
    )
    assert second.ready
    assert second.movers == first.movers


def test_incomplete_request_is_not_a_window(tmp_path) -> None:
    store = FiringStore(tmp_path / "firing.sqlite")
    try:
        missed = RouteScratch(1, 4)
        missed.add_prefill(0, np.array([[0, 1]], dtype=np.int64))
        store.commit_request("record-a", missed, complete=False)
        kept = RouteScratch(1, 4)
        kept.add_prefill(0, np.array([[0]], dtype=np.int64))
        store.commit_request("record-a", kept, complete=True)
        other = RouteScratch(1, 4)
        other.add_prefill(0, np.array([[1]], dtype=np.int64))
        store.commit_request("record-b", other, complete=True)
        eligible, hits = store.window("record-a", 1, 4)
        assert eligible == 1
        assert int(hits[0][0]) == 1
        assert int(hits[0][1]) == 0
        assert store.eligible_total("record-a") == 1
    finally:
        store.close()


def test_heartbeat_chooses_one_promotion_and_one_demotion() -> None:
    served = [[4, 4, 4, 4]]
    hits = [np.array([400, 0, 0, 0], dtype=np.uint64)]
    up, down = heartbeat_picks(
        served,
        hits,
        eligible=FLOOR,
        pins=[{1}],
        cooled=set(),
        rng=random.Random(0),
    )
    assert up == (0, 0, 8)
    assert down in {(0, 2, 2), (0, 3, 2)}
    held = heartbeat_picks(
        served,
        hits,
        eligible=FLOOR - 1,
        pins=[set()],
        cooled=set(),
        rng=random.Random(0),
    )
    assert held == (None, None)


def test_finish_request_leaves_the_lull_for_the_daemon(monkeypatch) -> None:
    import threading
    from types import SimpleNamespace

    from qwen38_flash_next.adapt import finish_request, idle_heartbeat

    class Routes:
        def begin(self) -> None:
            return None

        def snapshot(self) -> str:
            return "snap"

    class Files:
        root = "/tmp/record"

    runtime = SimpleNamespace(
        routes=Routes(),
        files=Files(),
        _firing_open=True,
        _cached_tokens=0,
        _prefill_seconds=0.0,
        _decode_seconds=0.0,
    )
    selection = SimpleNamespace(folder="Qwen3.8-Flash-Next-fp4")
    called: list[tuple[object, ...]] = []
    monkeypatch.setattr("qwen38_flash_next.adapt._stamp_turn", lambda _routes, _runtime: None)
    monkeypatch.setattr(
        "qwen38_flash_next.adapt._lull",
        lambda *args: called.append(args),
    )
    before = {thread.name for thread in threading.enumerate()}
    finish_request(runtime, selection, complete=True)
    started = {thread.name for thread in threading.enumerate()} - before
    assert "favlo-firing" not in started
    assert "favlo-pipeline-heartbeat" not in started
    assert runtime._idle_heartbeat[0] == "Qwen3.8-Flash-Next-fp4"
    idle_heartbeat(runtime, selection)
    assert called
    assert runtime._idle_heartbeat is None


def test_width_counts_name_each_width() -> None:
    from qwen38_flash_next.adapt import _width_counts

    class Files:
        def __init__(self, grid: list[list[int]] | None, manifest: dict[str, int]) -> None:
            self._quant_grid = grid
            self.manifest = manifest

    mixed = Files([[8, 4, 2, 4]], {})
    assert _width_counts(mixed) == (1, 2, 1)
    uniform = Files(None, {"expert_quant": 4, "layers": 2, "experts": 3})
    assert _width_counts(uniform) == (0, 6, 0)
    wider = Files(None, {"expert_quant": 8, "layers": 2, "experts": 3})
    assert _width_counts(wider) == (6, 0, 0)
    unnamed = Files(None, {"layers": 2, "experts": 3})
    assert _width_counts(unnamed) == (0, 0, 0)


def test_request_row_keeps_the_turn(tmp_path) -> None:
    store = FiringStore(tmp_path / "firing.sqlite")
    try:
        routes = RouteScratch(1, 4)
        routes.add_prefill(0, np.array([[0]], dtype=np.int64))
        routes.note_turn(
            cached_tokens=12,
            finished_at=1_700_000_000,
            prefill_seconds=1.5,
            decode_seconds=0.25,
            fp8_experts=3,
            fp4_experts=24570,
            int2_experts=3,
        )
        snap = routes.snapshot()
        routes.begin()
        assert routes.cached_tokens == 0
        store.commit_request("record-a", snap, complete=True)
        row = store._db.execute(
            "SELECT cached_tokens, finished_at, prefill_seconds, decode_seconds,"
            " fp8_experts, fp4_experts, int2_experts FROM request"
        ).fetchone()
        assert row == (12, 1_700_000_000, 1.5, 0.25, 3, 24570, 3)
        eligible, _hits = store.window("record-a", 1, 4)
        assert eligible == 1
    finally:
        store.close()


def test_old_request_table_gains_the_turn_columns(tmp_path) -> None:
    path = tmp_path / "firing.sqlite"
    db = sqlite3.connect(path)
    db.execute(
        "CREATE TABLE request ("
        " id INTEGER PRIMARY KEY,"
        " record TEXT NOT NULL,"
        " prefill_tokens INTEGER NOT NULL,"
        " decode_tokens INTEGER NOT NULL,"
        " complete INTEGER NOT NULL)"
    )
    db.commit()
    db.close()
    store = FiringStore(path)
    try:
        names = {str(row[1]) for row in store._db.execute("PRAGMA table_info(request)")}
        assert {
            "cached_tokens",
            "finished_at",
            "prefill_seconds",
            "decode_seconds",
            "fp8_experts",
            "fp4_experts",
            "int2_experts",
        } <= names
    finally:
        store.close()


def test_decode_take_keeps_the_prefix() -> None:
    routes = RouteScratch(1, 4)
    routes.note_decode(np.array([[[0], [1], [2]]], dtype=np.int64))
    routes.take_decode(2)
    assert routes.decode_tokens == 2
    assert int(routes.decode[0][0]) == 1
    assert int(routes.decode[0][1]) == 1
    assert int(routes.decode[0][2]) == 0
    routes.take_decode(1)
    assert routes.decode_tokens == 2
