"""Offset map for one heap segment.

Sizes round up to the OffsetAllocator bins: exact sizes below one page, and at
most 12.5% internal waste above that. Free ranges coalesce.
"""

from __future__ import annotations


def align(value: int, alignment: int) -> int:
    if alignment < 1:
        message = f"alignment {alignment} is not positive"
        raise ValueError(message)
    return (int(value) + alignment - 1) // alignment * alignment


def bin_size(nbytes: int, alignment: int) -> int:
    """Round ``nbytes`` up to a bin. The waste stays within 12.5%."""
    size = align(max(int(nbytes), 1), alignment)
    if size <= 4096:
        return size
    step = 1 << max(size.bit_length() - 4, 0)
    return align(size, step)


class OffsetMap:
    """Address-ordered free list over ``size`` bytes."""

    def __init__(self, size: int, alignment: int) -> None:
        if size < alignment:
            message = f"heap span {size} is shorter than alignment {alignment}"
            raise ValueError(message)
        self.size = int(size)
        self.alignment = int(alignment)
        self._free: list[tuple[int, int]] = [(0, self.size)]
        self._used: dict[int, tuple[int, int]] = {}
        self._next = 1

    def allocate(self, nbytes: int) -> tuple[int, int] | None:
        need = bin_size(nbytes, self.alignment)
        if need > self.size:
            return None
        choice: int | None = None
        best = self.size + 1
        for index, (_offset, length) in enumerate(self._free):
            if length >= need and length < best:
                choice = index
                best = length
        if choice is None:
            return None
        offset, length = self._free.pop(choice)
        spare = length - need
        if spare:
            self._free.append((offset + need, spare))
            self._free.sort()
        token = self._next
        self._next += 1
        self._used[token] = (offset, need)
        return token, offset

    def free(self, token: int) -> None:
        found = self._used.pop(token, None)
        if found is None:
            return
        self._free.append(found)
        self._free.sort()
        merged: list[tuple[int, int]] = []
        for offset, length in self._free:
            if merged and merged[-1][0] + merged[-1][1] == offset:
                previous = merged[-1]
                merged[-1] = (previous[0], previous[1] + length)
            else:
                merged.append((offset, length))
        self._free = merged

    def holds(self, token: int) -> bool:
        return token in self._used

    def placement(self, token: int) -> tuple[int, int]:
        offset, length = self._used[token]
        return offset, length


class BumpMap:
    """Scratch cursor. ``rewind`` drops every span allocated from this map."""

    def __init__(self, size: int, alignment: int) -> None:
        self.size = int(size)
        self.alignment = int(alignment)
        self._cursor = 0
        self._used: dict[int, tuple[int, int]] = {}
        self._next = 1

    def allocate(self, nbytes: int) -> tuple[int, int] | None:
        need = bin_size(nbytes, self.alignment)
        offset = align(self._cursor, self.alignment)
        if offset + need > self.size:
            return None
        self._cursor = offset + need
        token = self._next
        self._next += 1
        self._used[token] = (offset, need)
        return token, offset

    def free(self, token: int) -> None:
        self._used.pop(token, None)

    def rewind(self) -> None:
        self._used.clear()
        self._cursor = 0

    def holds(self, token: int) -> bool:
        return token in self._used

    def placement(self, token: int) -> tuple[int, int]:
        offset, length = self._used[token]
        return offset, length
