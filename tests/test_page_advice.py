"""Swap backing for streamed expert bytes. No device and no checkpoint."""

import mmap

import pytest
from qwen38_flash_next.experts import ExpertAtlas
from qwen38_flash_next.formats import PAGE
from qwen38_flash_next.session import (
    MADV_FREE_REUSABLE,
    advise_pages,
    commit_pages,
    release_pages,
)


class _View:
    def __init__(self, mapping: mmap.mmap) -> None:
        self._mapping = mapping

    def as_buffer(self, count: int) -> memoryview:
        return memoryview(self._mapping)[:count]


class _Buffer:
    def __init__(self, nbytes: int) -> None:
        self._mapping = mmap.mmap(-1, nbytes)

    def length(self) -> int:
        return len(self._mapping)

    def contents(self) -> _View:
        return _View(self._mapping)


class _Device:
    def __init__(self, nbytes: int) -> None:
        self.buffer = _Buffer(nbytes)

    def newBufferWithLength_options_(self, nbytes: int, options: int) -> _Buffer:
        del options
        if nbytes != self.buffer.length():
            return None
        return self.buffer


class _Gpu:
    def __init__(self, nbytes: int) -> None:
        self.device = _Device(nbytes)
        self.metal = type("Metal", (), {"MTLResourceStorageModeShared": 0})()

    def alloc(self, nbytes: int) -> _Buffer:
        return _Buffer(nbytes)

    def zeros(self, nbytes: int) -> _Buffer:
        return _Buffer(nbytes)


def test_release_then_commit_keeps_the_mapping_writable() -> None:
    buf = _Buffer(PAGE * 2)
    view = memoryview(buf.contents().as_buffer(PAGE * 2))
    view[0:4] = b"abcd"
    release_pages(buf, 0, PAGE * 2)
    commit_pages(buf, 0, PAGE)
    view[0:4] = b"wxyz"
    assert bytes(view[0:4]) == b"wxyz"


def test_page_advice_rejects_a_partial_page() -> None:
    buf = _Buffer(PAGE * 2)
    with pytest.raises(ValueError, match="multiple"):
        advise_pages(buf, 0, PAGE + 1, MADV_FREE_REUSABLE)


def test_atlas_unused_bytes_can_be_released_and_filled_again() -> None:
    gpu = _Gpu(PAGE * 4)
    atlas = ExpertAtlas(gpu, None, 1, 1, candidates=(PAGE * 4,))
    offset = atlas._alloc_span(PAGE, set())
    commit_pages(atlas.atlas, offset, PAGE)
    key = (0, 0, 0, 0)
    atlas.runs[key] = (offset, PAGE)
    atlas.run_members[key] = (0,)
    atlas.resident[(0, 0)] = offset
    atlas.expert_run[(0, 0)] = key
    atlas._free_run(key)
    assert atlas.resident == {}
    atlas.release_unused()
    again = atlas._alloc_span(PAGE, set())
    commit_pages(atlas.atlas, again, PAGE)
    view = memoryview(atlas.atlas.contents().as_buffer(PAGE))
    view[0:4] = b"img!"
    assert bytes(view[0:4]) == b"img!"
