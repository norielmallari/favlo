"""Host n-gram row ids stay aligned with the per-token scan."""

import numpy as np
from qwen38_flash_next.checks import _ngram_row, ngram_rows
from qwen38_flash_next.formats import PAGE
from qwen38_flash_next.session import _aligned_offset, trim_page_cache


def test_ngram_rows_match_scan_across_eos():
    eos = 1
    tokens = [4, 9, eos, 8, 3, eos, 6, 2]
    mult = np.array([3, -5, 9], dtype=np.int64)
    offsets = np.array([10, 20, 30, 40], dtype=np.int64)
    vocabs = np.array([7, 11, 13, 17], dtype=np.int64)
    kwargs = {"heads_per_ngram": 2, "ngram_size": 3, "eos": eos}
    rows = ngram_rows(tokens, mult, offsets, vocabs, start=0, **kwargs)
    heads = 4
    assert len(rows) == len(tokens) * heads
    for pos in range(len(tokens)):
        history = tokens[: pos + 1]
        for head in range(heads):
            expected = _ngram_row(history, head, mult, offsets, vocabs, **kwargs)
            assert rows[pos * heads + head] == expected
    tail = ngram_rows(tokens, mult, offsets, vocabs, start=3, **kwargs)
    assert tail == rows[3 * heads :]


def test_packed_ngram_destination_is_page_aligned() -> None:
    used = 3_471_360
    off = _aligned_offset(used, PAGE)
    assert off % PAGE == 0
    assert off >= used
    assert off - used < PAGE


def test_page_cache_drops_pages_after_a_protected_oldest() -> None:
    order = [1, 2, 3, 4]
    pages = {1: "a", 2: "b", 3: "c", 4: "d"}
    trim_page_cache(order, pages, {1}, 2)
    assert pages == {4: "d", 1: "a"}
    assert order[-1] == 1
