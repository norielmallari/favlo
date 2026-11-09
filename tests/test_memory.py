"""The process memory manager: bins, spans, and one Metal round trip."""

from __future__ import annotations

from typing import Any

import numpy as np

from favlo.memory import memory, mint_owner
from favlo.offset import OffsetMap, bin_size


def test_bins_waste_at_most_an_eighth() -> None:
    for size in (1, 16, 1000, 4097, 50_000, 1_000_003):
        rounded = bin_size(size, 256)
        assert rounded >= size
        assert rounded % 256 == 0
        assert rounded - size <= max(rounded // 8, 256)


def test_offset_map_reuses_a_freed_span() -> None:
    mapping = OffsetMap(1 << 20, 256)
    first = mapping.allocate(1000)
    second = mapping.allocate(1000)
    assert first is not None and second is not None
    mapping.free(first[0])
    third = mapping.allocate(1000)
    assert third is not None
    assert third[1] == first[1]


def test_shared_span_round_trip() -> None:
    owner = "test-memory"
    handle = memory.allocate(4096, owner=owner, zero=True)
    memory.store(handle, 0, np.arange(8, dtype=np.float32))
    raw = memory.load(handle, 0, 32)
    assert np.frombuffer(raw, dtype=np.float32).tolist() == list(range(8))
    memory.release(handle)
    memory.sweep(owner)
    try:
        memory.nbytes(handle)
    except KeyError:
        missing = True
    else:
        missing = False
    assert missing


def test_sweep_new_leaves_the_recorded_owner() -> None:
    kept = "test-memory-kept"
    fresh = "test-memory-fresh"
    held = memory.allocate(4096, owner=kept, zero=True)
    before = memory.owners()
    made = memory.allocate(4096, owner=fresh, zero=True)
    memory.sweep_new(before)
    assert kept in memory.owners()
    assert fresh not in memory.owners()
    assert memory.nbytes(held) >= 16
    try:
        memory.nbytes(made)
    except KeyError:
        missing = True
    else:
        missing = False
    assert missing
    memory.sweep(kept)


def _freeze_reclaimable(monkeypatch: Any, value: int) -> dict[str, int]:
    state = {"n": value}
    monkeypatch.setattr("favlo.memory.reclaimable_bytes", lambda: state["n"])
    with memory._lock:
        memory._reclaim_mark = value
    return state


def test_a_released_heap_stays_in_the_budget_until_the_kernel_reports_it(monkeypatch: Any) -> None:
    owner = mint_owner("returning")
    handle = memory.allocate(1 << 20, owner=owner, heap="scratch")
    with memory._lock:
        saved = (memory._reclaim_mark, memory._pending, memory._returning)
    state = _freeze_reclaimable(monkeypatch, 90_000_000_000)
    try:
        before = memory.budget()
        memory.release(handle)
        handle = None
        assert memory.budget() == before
        state["n"] += 1 << 20
        assert memory.budget() == before
    finally:
        if handle is not None:
            memory.release(handle)
        memory.sweep(owner)
        with memory._lock:
            memory._reclaim_mark, memory._pending, memory._returning = saved


def test_discard_then_release_counts_the_span_once(monkeypatch: Any) -> None:
    owner = mint_owner("returning")
    handle = memory.allocate(1 << 20, owner=owner, heap="scratch")
    with memory._lock:
        saved = (memory._reclaim_mark, memory._pending, memory._returning)
    state = _freeze_reclaimable(monkeypatch, 90_000_000_000)
    try:
        before = memory.budget()
        memory.discard(handle)
        memory.release(handle)
        handle = None
        assert memory.budget() == before
        state["n"] += 1 << 20
        assert memory.budget() == before
    finally:
        if handle is not None:
            memory.release(handle)
        memory.sweep(owner)
        with memory._lock:
            memory._reclaim_mark, memory._pending, memory._returning = saved


def test_confirm_released_blocks_until_the_pages_are_back(monkeypatch: Any) -> None:
    before = 4_000_000_000
    held = (64 << 20) + (32 << 20)
    state = {"calls": 0}

    def reclaimable() -> int:
        state["calls"] += 1
        if state["calls"] < 3:
            return before
        return before + held

    monkeypatch.setattr("favlo.memory.reclaimable_bytes", reclaimable)
    monkeypatch.setattr("favlo.memory._RECLAIM_PAUSE", 0.01)
    assert memory.confirm_released(before, held) is True
    assert state["calls"] >= 3


def test_confirm_released_treats_one_slab_as_already_free() -> None:
    assert memory.confirm_released(0, 64 << 20, timeout=0.0) is True


def test_confirm_released_waits_until_the_pages_return(monkeypatch: Any) -> None:
    before = 4_000_000_000
    held = (64 << 20) + (32 << 20)
    state = {"n": before, "calls": 0}

    def reclaimable() -> int:
        state["calls"] += 1
        if state["calls"] < 3:
            return state["n"]
        return state["n"] + held

    monkeypatch.setattr("favlo.memory.reclaimable_bytes", reclaimable)
    monkeypatch.setattr("favlo.memory._RECLAIM_PAUSE", 0.01)
    assert memory.confirm_released(before, held, timeout=1.0) is True
    assert state["calls"] >= 3


def test_a_short_reclaim_notes_only_the_pages_still_missing(monkeypatch: Any) -> None:
    before = 4_000_000_000
    held = (128 << 20)
    rose = 40 << 20
    saved = memory._returning
    monkeypatch.setattr("favlo.memory.reclaimable_bytes", lambda: before + rose)
    try:
        memory._returning = 0
        missing = memory.note_shortfall(before, held)
        assert missing == held - (64 << 20) - rose
        assert memory._returning == missing
        assert memory.note_shortfall(before, 64 << 20) == 0
    finally:
        memory._returning = saved


def test_confirm_released_ends_when_the_count_stops_rising(monkeypatch: Any) -> None:
    monkeypatch.setattr("favlo.memory.reclaimable_bytes", lambda: 4_000_000_000)
    monkeypatch.setattr("favlo.memory._RECLAIM_PAUSE", 0.01)
    monkeypatch.setattr("favlo.memory._RECLAIM_QUIET", 0.05)
    assert memory.confirm_released(4_000_000_000, (64 << 20) + (32 << 20)) is False


def test_confirm_released_times_out_while_the_pages_stay_wired(monkeypatch: Any) -> None:
    monkeypatch.setattr("favlo.memory.reclaimable_bytes", lambda: 4_000_000_000)
    monkeypatch.setattr("favlo.memory._RECLAIM_PAUSE", 0.01)
    assert memory.confirm_released(4_000_000_000, (64 << 20) + (32 << 20), timeout=0.05) is False


def test_mint_owner_is_the_stem_and_a_counter() -> None:
    first = mint_owner("z-image-turbo")
    second = mint_owner("z-image-turbo")
    assert first.startswith("z-image-turbo:")
    assert first != second


def test_a_child_total_stays_in_the_budget_until_it_is_replaced(monkeypatch: Any) -> None:
    _freeze_reclaimable(monkeypatch, 8 << 30)
    before = memory.budget()
    memory.account_child(41, {"model": 1_000_000}, 0, 0)
    try:
        assert memory.budget() == before + 1_000_000
        memory.account_child(41, {"model": 1_000_000}, 0, 0)
        assert memory.budget() == before + 1_000_000
        memory.account_child(41, {"model": 250_000}, 0, 0)
        assert memory.budget() == before + 250_000
    finally:
        memory.drop_child(41)


def test_a_later_budget_ask_sees_pages_that_became_free() -> None:
    answers = {"n": 10}
    memory.set_budget_query(lambda: answers["n"])
    try:
        assert memory.budget() == 10
        answers["n"] = 25
        assert memory.budget() == 25
    finally:
        memory.set_budget_query(None)


def test_a_reported_working_set_is_released_with_the_owner(monkeypatch: Any) -> None:
    from favlo.media_memory import begin_release, confirm_release

    owner = "ic-light:reported"
    memory.report({owner: 80 << 20})
    try:
        before, held = begin_release(owner)
        assert held == 80 << 20
        memory.report({owner: 0})
        seen: dict[str, int] = {}

        def confirm(start: int, nbytes: int, timeout: float | None = None) -> bool:
            del start, timeout
            seen["nbytes"] = nbytes
            return True

        monkeypatch.setattr(memory, "confirm_released", confirm)
        confirm_release(owner, before, held)
        assert seen["nbytes"] == 80 << 20
    finally:
        memory.forget((owner,))


def test_a_global_cap_keeps_the_smaller_budget(monkeypatch: Any) -> None:
    _freeze_reclaimable(monkeypatch, 8 << 30)
    uncapped = memory.budget()
    memory.set_max_memory("1GiB")
    try:
        assert memory.budget() == min(uncapped, 1 << 30)
    finally:
        memory.set_max_memory(None)
