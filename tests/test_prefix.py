"""Shared SSD prefix store, the session, and the per-pipeline plane pack."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import numpy as np
from qwen36_35b_a3b.cache import kv_buffer_bytes
from qwen36_35b_a3b.prefix import pack_plane as pack_35
from qwen36_35b_a3b.prefix import unpack_plane as unpack_35
from qwen38_27b.cache import plane_bytes
from qwen38_27b.prefix import pack_plane as pack_27
from qwen38_27b.prefix import unpack_plane as unpack_27

from madmetal.prefix import (
    PAGE,
    PREFERRED_IO,
    PrefixHeader,
    PrefixRecord,
    PrefixSession,
    PrefixStore,
    Region,
    checkpoint_digest,
    object_name,
    read_header,
    read_regions,
    read_tokens,
    stored_bytes,
)

MODEL = "qwen38-flash-next"
DIGEST = bytes(range(32))


def _store(root: Path, *, tag: str = "e4m3", budget: int | None = None) -> PrefixStore:
    return PrefixStore(root, model=MODEL, digest=DIGEST, tag=tag, budget=budget)


def _record(tokens: list[int], **regions: bytes) -> PrefixRecord:
    return PrefixRecord(tokens, [Region(name, data) for name, data in regions.items()])


def _blob(record: PrefixRecord, name: str) -> bytes:
    for region in record.regions:
        if region.name == name:
            return bytes(region.data)
    message = name
    raise KeyError(message)


def test_round_trip_and_longest_prefix(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.save(_record([1, 2, 3], k=b"abc", logits=b"logit"))
    store.save(_record([1, 2, 3, 4], k=b"abcd", logits=b"next"))
    short = store.longest([1, 2, 3])
    assert short is not None
    assert short.tokens == [1, 2, 3]
    assert _blob(short, "k") == b"abc"
    longer = store.longest([1, 2, 3, 4, 9])
    assert longer is not None
    assert longer.tokens == [1, 2, 3, 4]
    assert _blob(longer, "logits") == b"next"
    assert store.longest([9]) is None
    assert store.longest([]) is None


def test_exact_file_is_returned(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.save(_record([4, 5, 6], k=b"abc", logits=b""))
    exact = store.longest([4, 5, 6])
    assert exact is not None
    assert exact.tokens == [4, 5, 6]
    assert _blob(exact, "logits") == b""


def test_extension_keeps_the_shorter_file(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.save(_record([1, 2], k=b"aaaa", v=b"bbbb"))
    store.save(_record([1, 2, 3], k=b"aaaaW", v=b"bbbbX"))
    short = store.longest([1, 2])
    longer = store.longest([1, 2, 3])
    assert short is not None and longer is not None
    assert _blob(short, "k") == b"aaaa"
    assert _blob(longer, "k") == b"aaaaW"
    assert _blob(longer, "v") == b"bbbbX"
    assert len(list(store.directory.glob("*.bin"))) == 2


def test_changed_prefix_is_written_whole(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.save(_record([1, 2], k=b"aaaa"))
    store.save(_record([1, 2, 3], k=b"ZZZZaaaa"))
    hit = store.longest([1, 2, 3])
    assert hit is not None
    assert _blob(hit, "k") == b"ZZZZaaaa"


def test_page_aligned_regions_and_wide_blob(tmp_path: Path) -> None:
    store = _store(tmp_path)
    key = b"k" * (PREFERRED_IO + 10)
    tokens = list(range(5000))
    path = store.save(_record(tokens, k=key, logits=b"L"))
    assert path is not None
    assert path.stat().st_size % PAGE == 0
    assert path.stem == object_name(MODEL, DIGEST, "e4m3", tokens)
    header = read_header(path)
    assert header is not None
    assert len(header.tokens) == 5000
    region = header.region("k")
    assert region is not None
    assert region.offset % PAGE == 0
    assert region.length == len(key)
    assert stored_bytes(region.length) % PAGE == 0
    assert read_tokens(path) == tokens
    loaded = store.longest(tokens + [9])
    assert loaded is not None
    assert _blob(loaded, "k") == key


def test_page_exact_region_is_written(tmp_path: Path) -> None:
    store = _store(tmp_path)
    exact = b"e" * PAGE
    tail = b"t" * (PREFERRED_IO + PAGE)
    path = store.save(_record([1, 2], history=exact, state=tail))
    assert path is not None
    hit = store.longest([1, 2, 3])
    assert hit is not None
    assert _blob(hit, "history") == exact
    assert _blob(hit, "state") == tail


def test_align_continues_when_the_prefix_write_fails() -> None:
    class _Broken:
        def save(self, record: PrefixRecord) -> None:
            raise OSError("prefix region ended short")

        def matches(self, wanted: list[int]) -> list[PrefixHeader]:
            return []

    memory = _Memory()
    session = PrefixSession(_Broken(), memory)  # type: ignore[arg-type]
    memory.regions = [Region("k", b"aaaa")]
    session.note([1, 2])
    assert session.align([9]) == 0
    assert memory.resets == 1
    assert session.dirty is False
    assert session.tokens == []


def test_region_names_stay_distinct(tmp_path: Path) -> None:
    store = _store(tmp_path)
    path = store.save(
        _record([7, 8], **{"attn.4.key": b"k4", "attn.1.key": b"k1", "gdn.0": b"g0"})
    )
    assert path is not None
    header = read_header(path)
    assert header is not None
    assert [ref.name for ref in header.regions] == ["attn.1.key", "attn.4.key", "gdn.0"]
    assert header.regions[0].offset < header.regions[1].offset
    hit = store.longest([7, 8, 9])
    assert hit is not None
    assert _blob(hit, "attn.1.key") == b"k1"
    assert _blob(hit, "gdn.0") == b"g0"


def test_memoryview_round_trip(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.save(_record([8], k=memoryview(b"from-buffer"), logits=memoryview(b"logit")))
    hit = store.longest([8, 1])
    assert hit is not None
    assert _blob(hit, "k") == b"from-buffer"
    assert _blob(hit, "logits") == b"logit"


def test_identity_and_filename(tmp_path: Path) -> None:
    store = _store(tmp_path)
    path = store.save(_record([1, 2, 3], logits=b"ok"))
    assert path is not None
    other = PrefixStore(tmp_path, model=MODEL, digest=bytes([9]) * 32, tag="e4m3")
    other.directory.mkdir(parents=True)
    shutil.copy(path, other.directory / path.name)
    assert other.longest([1, 2, 3]) is None
    sibling = _store(tmp_path, tag="bf16")
    assert sibling.longest([1, 2, 3]) is None
    renamed = store.directory / f"{'ab' * 32}.bin"
    shutil.copy(path, renamed)
    assert store.longest([1, 2, 3]) is not None
    renamed.unlink()
    blob = bytearray(path.read_bytes())
    blob[20] ^= 0xFF
    path.write_bytes(blob)
    assert read_header(path) is None
    assert store.longest([1, 2, 3]) is None


def test_truncated_file_is_ignored(tmp_path: Path) -> None:
    store = _store(tmp_path)
    path = store.save(_record([1], logits=b"ok"))
    assert path is not None
    path.write_bytes(path.read_bytes()[:10])
    assert read_tokens(path) is None
    assert store.longest([1, 2]) is None


def test_second_save_keeps_one_file(tmp_path: Path) -> None:
    store = _store(tmp_path)
    record = _record([3, 4], logits=b"same")
    first = store.save(record)
    assert first is not None
    planted = store.directory / f".{first.stem}.{os.getpid()}.tmp"
    planted.write_bytes(b"partial")
    second = store.save(record)
    assert first == second
    assert list(store.directory.glob("*.bin")) == [first]
    assert list(store.directory.glob(".*.tmp")) == []


def test_eviction_keeps_the_file_just_written(tmp_path: Path) -> None:
    store = _store(tmp_path, budget=1)
    store.save(_record([1], logits=b"a"))
    store.save(_record([2], logits=b"b"))
    last = store.save(_record([3], logits=b"c"))
    files = list(store.directory.glob("*.bin"))
    assert files == [last]


def test_object_name_and_checkpoint_digest(tmp_path: Path) -> None:
    base = object_name(MODEL, DIGEST, "e4m3", [1, 2, 3])
    assert object_name("other", DIGEST, "e4m3", [1, 2, 3]) != base
    assert object_name(MODEL, bytes([7]) * 32, "e4m3", [1, 2, 3]) != base
    assert object_name(MODEL, DIGEST, "bf16", [1, 2, 3]) != base
    assert object_name(MODEL, DIGEST, "e4m3", [1, 2, 4]) != base
    root = tmp_path / "checkpoint"
    root.mkdir()
    (root / "config.json").write_bytes(b"{}")
    first = checkpoint_digest(root)
    (root / "manifest.json").write_bytes(b'{"version":5}')
    assert checkpoint_digest(root) != first
    assert len(first) == 32


class _Memory:
    def __init__(self, *, exact: bool = False) -> None:
        self.regions: list[Region] = []
        self.device: dict[str, bytes] = {}
        self.resets = 0
        self.exact = exact
        self.fail = False

    def capture(self) -> list[Region]:
        return list(self.regions)

    def install(self, header: PrefixHeader) -> bool:
        if self.fail:
            return False
        blobs = read_regions(header, [ref.name for ref in header.regions])
        if blobs is None:
            return False
        self.device = blobs
        self.regions = [Region(name, data) for name, data in blobs.items()]
        return True

    def reset(self) -> None:
        self.resets += 1
        self.device = {}

    def accept(self, header: PrefixHeader, prompt: list[int]) -> bool:
        if self.exact:
            return True
        return len(header.tokens) < len(prompt)


def test_session_extends_flushes_and_reloads(tmp_path: Path) -> None:
    memory = _Memory()
    session = PrefixSession(_store(tmp_path), memory)
    memory.regions = [Region("k", b"aaaa")]
    session.note([1, 2])
    assert session.flush() is not None
    assert session.align([1, 2, 9]) == 2
    assert memory.resets == 0
    assert session.align([7, 8]) == 0
    assert memory.resets == 1
    memory.regions = [Region("k", b"bbbb")]
    session.note([7])
    session.flush()
    session.tokens = []
    session.dirty = False
    assert session.align([7, 1]) == 1
    assert session.loaded
    assert memory.device["k"] == b"bbbb"
    session.tokens = []
    session.dirty = False
    assert session.align([7]) == 0
    memory.exact = True
    assert session.align([7]) == 1
    assert memory.device["k"] == b"bbbb"


def test_session_skips_a_header_the_adapter_refuses(tmp_path: Path) -> None:
    memory = _Memory()
    session = PrefixSession(_store(tmp_path), memory)
    memory.regions = [Region("k", b"short")]
    session.note([1])
    session.flush()
    memory.regions = [Region("k", b"longer")]
    session.note([1, 2])
    session.flush()
    memory.fail = True
    session.tokens = []
    session.dirty = False
    assert session.align([1, 2, 3]) == 0

    class _SkipLong(_Memory):
        def install(self, header: PrefixHeader) -> bool:
            if len(header.tokens) > 1:
                return False
            return _Memory.install(self, header)

    session.adapter = _SkipLong()
    assert session.align([1, 2, 3]) == 1
    assert session.adapter.device["k"] == b"short"


def test_27b_plane_round_trip() -> None:
    max_pos = 64
    count = 5
    plane = np.arange(plane_bytes(max_pos), dtype=np.uint8)
    original = plane.copy()
    for kind in ("key", "value"):
        packed = pack_27(plane, max_pos, count, kind=kind)
        plane[:] = 0
        unpack_27(plane, max_pos, count, packed, kind=kind)
        again = pack_27(plane, max_pos, count, kind=kind)
        assert again == pack_27(original, max_pos, count, kind=kind)
        plane[:] = original


def test_35b_plane_round_trip() -> None:
    max_pos = 64
    count = 5
    for mode in ("e4m3", "e2m1", "f32"):
        plane = np.arange(kv_buffer_bytes(mode, max_pos), dtype=np.uint8)
        original = plane.copy()
        for kind in ("key", "value"):
            packed = pack_35(plane, max_pos, count, mode, kind)
            plane[:] = 0
            unpack_35(plane, max_pos, count, mode, kind, packed)
            assert pack_35(plane, max_pos, count, mode, kind) == pack_35(
                original, max_pos, count, mode, kind
            )
            plane[:] = original
