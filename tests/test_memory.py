"""The process memory manager: bins, spans, and one Metal round trip."""

from __future__ import annotations

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


def test_mint_owner_is_the_stem_and_a_counter() -> None:
    first = mint_owner("z-image-turbo")
    second = mint_owner("z-image-turbo")
    assert first.startswith("z-image-turbo:")
    assert first != second
