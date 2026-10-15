"""Prefix files: header hash, longest match, page-aligned regions, eviction."""

import os
import shutil
from pathlib import Path

from qwen38_flash_next.formats import PAGE, PREFERRED_IO
from qwen38_flash_next.prefix_store import (
    MODEL_ID,
    AttentionImage,
    FixedImage,
    PrefixState,
    PrefixStore,
    checkpoint_digest,
    object_name,
    read_header,
    read_tokens,
    stored_bytes,
)

DIGEST = bytes(range(32))


def _store(root: Path, *, tag: str = "e4m3", budget: int | None = None) -> PrefixStore:
    return PrefixStore(root, digest=DIGEST, tag=tag, budget=budget)


def _state(
    tokens: list[int],
    *,
    key: bytes = b"key",
    value: bytes = b"value",
    indexer: bytes = b"index",
    tail: bytes = b"tail",
    gdn: bytes = b"gdn",
    conv: bytes = b"conv",
    ple: bytes = b"ple",
    logits: bytes = b"logits",
    layer: int = 3,
) -> PrefixState:
    return PrefixState(
        tokens,
        [
            AttentionImage(layer, key, value, tail, tail, indexer),
        ],
        [FixedImage(layer, gdn)],
        [FixedImage(layer, conv)],
        ple,
        logits,
    )


def test_round_trip_and_longest_prefix(tmp_path: Path) -> None:
    store = _store(tmp_path)
    short = _state([1, 2], key=b"hello", indexer=b"xy", logits=b"one")
    longer = _state([1, 2, 3], key=b"hello!!!", indexer=b"xyZZ", logits=b"two")
    other = _state([9, 9], key=b"nope", logits=b"zz")
    assert store.save(short) is not None
    assert store.save(longer) is not None
    assert store.save(other) is not None
    assert store.save(_state([])) is None

    hit = store.longest([1, 2, 3, 4])
    assert hit is not None
    assert hit.tokens == [1, 2, 3]
    assert hit.attention[0].key == b"hello!!!"
    assert hit.attention[0].indexer == b"xyZZ"
    assert hit.logits == b"two"
    assert store.longest([1, 2]) is not None
    assert store.longest([1, 2]).tokens == [1, 2]
    assert store.longest([1, 9]) is None
    assert store.longest([9, 9, 1]).tokens == [9, 9]


def test_exact_hit_needs_logits(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.save(_state([4, 5, 6], logits=b""))
    assert store.longest([4, 5, 6]) is None
    extended = store.longest([4, 5, 6, 7])
    assert extended is not None
    assert extended.tokens == [4, 5, 6]
    store.save(_state([4, 5, 6], logits=b"ready"))
    assert store.longest([4, 5, 6]).logits == b"ready"


def test_extension_keeps_the_shorter_file(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.save(_state([1, 2], key=b"aaaa", value=b"bbbb", indexer=b"cc"))
    store.save(_state([1, 2, 3], key=b"aaaaW", value=b"bbbbX", indexer=b"ccY"))
    short = store.longest([1, 2])
    longer = store.longest([1, 2, 3])
    assert short is not None and longer is not None
    assert short.attention[0].key == b"aaaa"
    assert longer.attention[0].key == b"aaaaW"
    assert longer.attention[0].value == b"bbbbX"
    assert len(list(store.directory.glob("*.bin"))) == 2


def test_changed_prefix_is_written_whole(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.save(_state([1, 2], key=b"aaaa"))
    store.save(_state([1, 2, 3], key=b"ZZZZaaaa"))
    hit = store.longest([1, 2, 3])
    assert hit is not None
    assert hit.attention[0].key == b"ZZZZaaaa"


def test_page_aligned_regions_and_wide_blob(tmp_path: Path) -> None:
    store = _store(tmp_path)
    key = b"k" * (PREFERRED_IO + 10)
    path = store.save(_state(list(range(5000)), key=key, logits=b"L"))
    assert path is not None
    assert path.stat().st_size % PAGE == 0
    assert path.stem == object_name(MODEL_ID, DIGEST, "e4m3", list(range(5000)))
    header = read_header(path)
    assert header is not None
    assert len(header.tokens) == 5000
    region = header.attention[0].key
    assert region.offset % PAGE == 0
    assert region.length == len(key)
    assert stored_bytes(region.length) % PAGE == 0
    assert read_tokens(path) == list(range(5000))
    assert store.longest(list(range(5000)) + [9]).attention[0].key == key


def test_two_layers_keep_separate_regions(tmp_path: Path) -> None:
    store = _store(tmp_path)
    state = PrefixState(
        [7, 8],
        [
            AttentionImage(4, b"k4", b"v4", b"t", b"t", b"i4"),
            AttentionImage(1, b"k1", b"v1", b"u", b"u", b"i1"),
        ],
        [FixedImage(2, b"g2"), FixedImage(0, b"g0")],
        [FixedImage(2, b"c2")],
        b"ple",
        b"logits",
    )
    path = store.save(state)
    assert path is not None
    header = read_header(path)
    assert header is not None
    assert [image.layer for image in header.attention] == [1, 4]
    assert header.attention[0].key.offset < header.attention[1].key.offset
    assert [image.layer for image in header.gdn] == [0, 2]
    hit = store.longest([7, 8, 9])
    assert hit is not None
    assert [(image.layer, image.key) for image in hit.attention] == [(1, b"k1"), (4, b"k4")]
    assert [(image.layer, image.blob) for image in hit.gdn] == [(0, b"g0"), (2, b"g2")]


def test_memoryview_round_trip(tmp_path: Path) -> None:
    store = _store(tmp_path)
    state = _state([8], key=memoryview(b"from-buffer"), logits=memoryview(b"logit"))
    store.save(state)
    hit = store.longest([8, 1])
    assert hit is not None
    assert hit.attention[0].key == b"from-buffer"
    assert hit.logits == b"logit"
    store.save(
        _state(
            [8, 9],
            key=memoryview(b"from-buffer-more"),
            indexer=memoryview(b"indexQQ"),
            logits=memoryview(b"next"),
        )
    )
    extended = store.longest([8, 9])
    assert extended is not None
    assert extended.attention[0].key == b"from-buffer-more"
    assert extended.attention[0].indexer == b"indexQQ"
    assert store.longest([8]).attention[0].key == b"from-buffer"


def test_identity_and_filename(tmp_path: Path) -> None:
    store = _store(tmp_path)
    path = store.save(_state([1, 2, 3], logits=b"ok"))
    assert path is not None
    other = PrefixStore(tmp_path, digest=bytes([9]) * 32, tag="e4m3")
    other.directory.mkdir(parents=True)
    shutil.copy(path, other.directory / path.name)
    assert other.longest([1, 2, 3]) is None
    sibling = _store(tmp_path, tag="bf16")
    assert sibling.longest([1, 2, 3]) is None
    renamed = store.directory / f"{'ab' * 32}.bin"
    shutil.copy(path, renamed)
    assert store.longest([1, 2, 3]).tokens == [1, 2, 3]
    renamed.unlink()
    blob = bytearray(path.read_bytes())
    blob[20] ^= 0xFF
    path.write_bytes(blob)
    assert read_header(path) is None
    assert store.longest([1, 2, 3]) is None


def test_truncated_file_is_ignored(tmp_path: Path) -> None:
    store = _store(tmp_path)
    path = store.save(_state([1], logits=b"ok"))
    path.write_bytes(path.read_bytes()[:10])
    assert read_tokens(path) is None
    assert store.longest([1, 2]) is None


def test_second_save_keeps_one_file(tmp_path: Path) -> None:
    store = _store(tmp_path)
    state = _state([3, 4], logits=b"same")
    first = store.save(state)
    planted = store.directory / f".{first.stem}.{os.getpid()}.tmp"
    planted.write_bytes(b"partial")
    second = store.save(state)
    assert first == second
    assert list(store.directory.glob("*.bin")) == [first]
    assert list(store.directory.glob(".*.tmp")) == []


def test_eviction_keeps_the_file_just_written(tmp_path: Path) -> None:
    store = _store(tmp_path, budget=1)
    store.save(_state([1], logits=b"a"))
    store.save(_state([2], logits=b"b"))
    last = store.save(_state([3], logits=b"c"))
    files = list(store.directory.glob("*.bin"))
    assert files == [last]


def test_object_name_and_checkpoint_digest(tmp_path: Path) -> None:
    base = object_name(MODEL_ID, DIGEST, "e4m3", [1, 2, 3])
    assert object_name("other", DIGEST, "e4m3", [1, 2, 3]) != base
    assert object_name(MODEL_ID, bytes([7]) * 32, "e4m3", [1, 2, 3]) != base
    assert object_name(MODEL_ID, DIGEST, "bf16", [1, 2, 3]) != base
    assert object_name(MODEL_ID, DIGEST, "e4m3", [1, 2, 4]) != base
    root = tmp_path / "checkpoint"
    root.mkdir()
    (root / "config.json").write_bytes(b"{}")
    first = checkpoint_digest(root)
    (root / "manifest.json").write_bytes(b'{"version":5}')
    assert checkpoint_digest(root) != first
    assert len(first) == 32
