"""Prefix store layout and sequential restore."""

from __future__ import annotations

from pathlib import Path

from favlo.prefix import (
    PrefixRecord,
    PrefixStore,
    Region,
    read_header,
    read_regions,
)


def test_write_keeps_capture_order(tmp_path: Path) -> None:
    store = PrefixStore(
        tmp_path, model="example-model", digest=bytes(range(32)), tag="half", budget=1 << 30
    )
    # Lexicographic order would put k.10 before k.2.
    path = store.save(
        PrefixRecord(
            [1, 2, 3],
            [
                Region("k.0", b"a" * 64),
                Region("k.1", b"b" * 64),
                Region("k.2", b"c" * 64),
                Region("k.10", b"d" * 64),
            ],
        )
    )
    assert path is not None
    header = read_header(path)
    assert header is not None
    names = [ref.name for ref in header.regions]
    assert names == ["k.0", "k.1", "k.2", "k.10"]
    assert header.region("k.2").offset < header.region("k.10").offset


def test_read_regions_loads_a_forward_span(tmp_path: Path) -> None:
    store = PrefixStore(
        tmp_path, model="example-model", digest=bytes(range(32)), tag="half", budget=1 << 30
    )
    payload = [
        Region("history", b"h" * 128),
        Region("k.0", b"k" * 4096),
        Region("v.0", b"v" * 4096),
    ]
    path = store.save(PrefixRecord([7, 8, 9], payload))
    assert path is not None
    header = read_header(path)
    assert header is not None
    blobs = read_regions(header, ["v.0", "history", "k.0"])
    assert blobs is not None
    assert bytes(blobs["history"]) == b"h" * 128
    assert bytes(blobs["k.0"]) == b"k" * 4096
    assert bytes(blobs["v.0"]) == b"v" * 4096
