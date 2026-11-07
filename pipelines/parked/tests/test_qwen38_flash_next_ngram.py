"""Admission for the resident n-gram set."""

from qwen38_flash_next.ngram_set import SET_BYTES, SET_PAGES, NgramTable


def test_a_page_enters_on_its_second_lookup() -> None:
    table = NgramTable(8)
    once_admits, once = table.plan([0])
    assert once_admits == []
    assert once == [0]
    admits, cold = table.plan([0])
    assert cold == []
    assert admits == [(0, SET_PAGES - 1)]
    assert SET_PAGES == 65_536
    assert SET_BYTES == 1 << 30


def test_a_full_set_evicts_the_smaller_count() -> None:
    table = NgramTable(8)
    table._free = []
    table.page_at[:] = 1
    table.page_at[1] = 2
    table.slot_of = {1: 0, 2: 1}
    table.count[1] = 9
    table.count[2] = 2
    table.count[3] = 1
    row = 3 * 64
    admits, cold = table.plan([row, row])
    assert cold == []
    assert admits == [(3, 1)]
    assert 2 not in table.slot_of
    assert table.slot_of[1] == 0
