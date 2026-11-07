"""Resident n-gram pages.

One count per 16 KB page. A page enters the 1 GiB set on its second lookup.
A page seen once is staged for that forward and does not take a slot. The
resident page with the smaller count leaves when the set is full.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from qwen38_flash_next.formats import NGRAM_FP8_SLOT, PAGE

SET_PAGES = 65_536
SET_BYTES = SET_PAGES * PAGE
# One prefill chunk can miss this many cold pages before they repeat.
COLD_RESERVE = 512 << 20
ROWS_PER_PAGE = PAGE // NGRAM_FP8_SLOT
METAL_MARGIN = 1 << 30


class NgramTable:
    """Page counts and the slot each resident page occupies."""

    def __init__(self, pages: int) -> None:
        if pages < 1:
            message = "n-gram page count is empty"
            raise ValueError(message)
        self.count = np.zeros(int(pages), dtype=np.int32)
        self.slot_of: dict[int, int] = {}
        self.page_at = np.full(SET_PAGES, -1, dtype=np.int64)
        self._free = list(range(SET_PAGES))

    def plan(self, rows: Sequence[int]) -> tuple[list[tuple[int, int]], list[int]]:
        """Count every row, then split new pages into admits and one-shot pages.

        Admits are ``(page, slot)`` and still need a load into the set. One-shot
        pages need a load into the forward's staging buffer. A page already in
        the set is neither.
        """
        seen: list[int] = []
        marked: set[int] = set()
        limit = int(self.count.shape[0])
        for row in rows:
            page = int(row) // ROWS_PER_PAGE
            if page < 0 or page >= limit:
                message = f"n-gram page {page} is outside {limit}"
                raise ValueError(message)
            self.count[page] += 1
            if page not in marked:
                marked.add(page)
                seen.append(page)
        admits: list[tuple[int, int]] = []
        once: list[int] = []
        protect: set[int] = set()
        for page in seen:
            if page in self.slot_of:
                protect.add(page)
                continue
            if int(self.count[page]) >= 2:
                slot = self._admit(page, protect)
                protect.add(page)
                admits.append((page, slot))
            else:
                once.append(page)
        return admits, once

    def _admit(self, page: int, protect: set[int]) -> int:
        if self._free:
            slot = self._free.pop()
        else:
            slot = self._victim(protect)
            old = int(self.page_at[slot])
            if old >= 0:
                self.slot_of.pop(old, None)
        self.page_at[slot] = page
        self.slot_of[page] = slot
        return slot

    def _victim(self, protect: set[int]) -> int:
        counts = self.count[self.page_at]
        for page in protect:
            slot = self.slot_of.get(page)
            if slot is not None:
                counts[slot] = np.iinfo(np.int32).max
        return int(np.argmin(counts))
