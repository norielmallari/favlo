"""Atlas eviction strategies. No device and no checkpoint."""

from qwen38_flash_next.eviction import (
    ColdLfuEviction,
    LruEviction,
    S3FifoEviction,
    make_eviction,
)

Run = tuple[int, int, int, int]


def _admit(
    strategy: LruEviction | S3FifoEviction | ColdLfuEviction,
    store: dict[Run, int],
    members: dict[Run, tuple[int, ...]],
    key: Run,
    span: int,
    budget: int,
    hits: int = 0,
    protect: set[tuple[int, int]] | None = None,
) -> Run | None:
    """Insert one run, evicting until it fits. Returns the last victim."""
    held = protect or set()
    victim = None
    while sum(store.values()) + span > budget:
        victim = strategy.victim(held, members, span)
        assert victim is not None
        assert victim in store
        del store[victim]
        strategy.remove(victim)
    strategy.insert(key, span)
    store[key] = span
    members[key] = (key[2],)
    for _ in range(hits):
        strategy.hit(key)
    while True:
        gone = strategy.relieve(held, members)
        if gone is None:
            break
        store.pop(gone, None)
        strategy.remove(gone)
    return victim


def test_lru_drops_the_oldest_unprotected_run() -> None:
    strategy = LruEviction(30)
    store: dict[Run, int] = {}
    members: dict[Run, tuple[int, ...]] = {}
    first = (0, 0, 0, 0)
    second = (0, 0, 1, 1)
    third = (0, 0, 2, 2)
    _admit(strategy, store, members, first, 10, 30)
    _admit(strategy, store, members, second, 10, 30)
    _admit(strategy, store, members, third, 10, 30)
    strategy.hit(first)
    added = (0, 0, 3, 3)
    victim = _admit(strategy, store, members, added, 10, 30)
    assert victim == second
    assert first in store and third in store and added in store


def test_lru_skips_a_protected_run() -> None:
    strategy = LruEviction(20)
    store: dict[Run, int] = {}
    members: dict[Run, tuple[int, ...]] = {}
    oldest = (1, 0, 0, 0)
    newer = (1, 0, 1, 1)
    _admit(strategy, store, members, oldest, 10, 20)
    _admit(strategy, store, members, newer, 10, 20)
    added = (1, 0, 2, 2)
    victim = _admit(
        strategy,
        store,
        members,
        added,
        10,
        20,
        protect={(1, 0)},
    )
    assert victim == newer
    assert oldest in store


def test_s3fifo_drops_a_one_hit_and_keeps_a_reused_run() -> None:
    budget = 100
    strategy = S3FifoEviction(budget)
    store: dict[Run, int] = {}
    members: dict[Run, tuple[int, ...]] = {}
    reused = []
    for index in range(9):
        key = (0, 0, index, index)
        _admit(strategy, store, members, key, 10, budget, hits=1)
        reused.append(key)
    one_hit = (0, 1, 0, 0)
    _admit(strategy, store, members, one_hit, 10, budget)
    incoming = (0, 1, 1, 1)
    victim = _admit(strategy, store, members, incoming, 10, budget)
    assert victim == one_hit
    assert all(key in store for key in reused)
    assert incoming in store


def test_s3fifo_admits_a_ghost_hit_into_the_main_queue() -> None:
    budget = 100
    strategy = S3FifoEviction(budget)
    store: dict[Run, int] = {}
    members: dict[Run, tuple[int, ...]] = {}
    kept = (0, 0, 0, 0)
    _admit(strategy, store, members, kept, 10, budget, hits=2)
    for index in range(1, 20):
        key = (0, 1, index, index)
        _admit(strategy, store, members, key, 10, budget)
    assert kept in store


def test_cold_lfu_drops_a_cold_run_before_an_older_hot_run() -> None:
    strategy = ColdLfuEviction(20)
    store: dict[Run, int] = {}
    members: dict[Run, tuple[int, ...]] = {}
    hot = (0, 0, 0, 0)
    cold = (0, 1, 1, 1)
    _admit(strategy, store, members, hot, 10, 20, hits=1)
    _admit(strategy, store, members, cold, 10, 20)
    incoming = (0, 0, 2, 2)
    victim = _admit(strategy, store, members, incoming, 10, 20)
    assert victim == cold
    assert hot in store


def test_cold_lfu_drops_the_least_used_run_in_one_tier() -> None:
    strategy = ColdLfuEviction(20)
    store: dict[Run, int] = {}
    members: dict[Run, tuple[int, ...]] = {}
    older = (2, 1, 0, 0)
    newer = (2, 1, 1, 1)
    _admit(strategy, store, members, older, 10, 20, hits=3)
    _admit(strategy, store, members, newer, 10, 20, hits=1)
    incoming = (2, 1, 2, 2)
    victim = _admit(strategy, store, members, incoming, 10, 20)
    assert victim == newer
    assert older in store


def test_unknown_strategy_is_rejected() -> None:
    try:
        make_eviction("clock", 64)
    except ValueError as exc:
        assert "clock" in str(exc)
    else:
        raise AssertionError("expected ValueError")
