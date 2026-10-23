"""Expert-atlas eviction strategies.

The strategy chooses which resident run is the victim. ``s3fifo`` is the
default. ``lru`` and ``cold_lfu`` are selectable. LRU and ``cold_lfu`` drop a
run only when the next load does not fit. S3-FIFO also drops a one-hit once
the small FIFO exceeds one tenth of the atlas.

``cold_lfu`` is the LFU expert cache used by MoE-Infinity and fMoE, with
Diff-MoE's split: a cold run is the victim before any hot run. Nothing is
dropped while the incoming load still fits.

S3-FIFO is Yang, Zhang, Qiu, Yue, and Rashmi, SOSP 2023. The small FIFO is one
tenth of the atlas. A run requested once is evicted from that queue. A run
requested again moves to the main FIFO with its frequency cleared. The main
FIFO reinserts a run whose two-bit frequency is still positive and decrements
it. A run that is still in the ghost queue is admitted to the main FIFO.
"""

from __future__ import annotations

from collections import OrderedDict, deque
from collections.abc import Mapping
from typing import Protocol

RunKey = tuple[int, int, int, int]
Members = Mapping[RunKey, tuple[int, ...]]

DEFAULT_EVICTION = "s3fifo"
_FREQ_CAP = 3


def _protected(key: RunKey, protect: set[tuple[int, int]], members: Members) -> bool:
    if not protect:
        return False
    layer = key[0]
    return any((layer, expert) in protect for expert in members.get(key, ()))


class EvictionStrategy(Protocol):
    """Hit, insert, and victim selection for resident expert runs."""

    def hit(self, key: RunKey) -> None:
        """Record that a resident run was routed again."""

    def insert(self, key: RunKey, span: int) -> None:
        """Admit a run that just occupied ``span`` bytes."""

    def remove(self, key: RunKey) -> None:
        """Drop a run that the atlas has already freed."""

    def relieve(
        self, protect: set[tuple[int, int]], members: Members
    ) -> RunKey | None:
        """Free a run that sits outside this strategy's caps, or None."""

    def victim(
        self,
        protect: set[tuple[int, int]],
        members: Members,
        incoming: int = 0,
    ) -> RunKey | None:
        """Return one run whose bytes can be freed, or None when every run is protected.

        ``incoming`` is the byte size of the load that did not fit. S3-FIFO
        admits that load before it chooses a victim.
        """


class LruEviction:
    """Oldest unprotected run. A hit moves that run to the newest end."""

    def __init__(self, budget: int) -> None:
        del budget
        self._order: OrderedDict[RunKey, int] = OrderedDict()

    def hit(self, key: RunKey) -> None:
        if key in self._order:
            self._order.move_to_end(key)

    def insert(self, key: RunKey, span: int) -> None:
        self._order[key] = span
        self._order.move_to_end(key)

    def remove(self, key: RunKey) -> None:
        self._order.pop(key, None)

    def relieve(
        self, protect: set[tuple[int, int]], members: Members
    ) -> RunKey | None:
        del protect, members
        return None

    def victim(
        self,
        protect: set[tuple[int, int]],
        members: Members,
        incoming: int = 0,
    ) -> RunKey | None:
        del incoming
        for key in self._order:
            if not _protected(key, protect, members):
                return key
        return None


class ColdLfuEviction:
    """Least frequently used, and a cold run before any hot run.

    Eviction runs only when the caller cannot fit the next load. A hit
    increments the run. Age breaks a tie, oldest first.
    """

    def __init__(self, budget: int) -> None:
        del budget
        self._freq: dict[RunKey, int] = {}
        self._order: OrderedDict[RunKey, int] = OrderedDict()

    def hit(self, key: RunKey) -> None:
        if key in self._freq:
            self._freq[key] += 1

    def insert(self, key: RunKey, span: int) -> None:
        self._order[key] = span
        self._freq[key] = 0

    def remove(self, key: RunKey) -> None:
        self._order.pop(key, None)
        self._freq.pop(key, None)

    def relieve(
        self, protect: set[tuple[int, int]], members: Members
    ) -> RunKey | None:
        del protect, members
        return None

    def victim(
        self,
        protect: set[tuple[int, int]],
        members: Members,
        incoming: int = 0,
    ) -> RunKey | None:
        del incoming
        chosen: RunKey | None = None
        chosen_rank: tuple[int, int, int] | None = None
        for index, key in enumerate(self._order):
            if _protected(key, protect, members):
                continue
            cold = 0 if key[1] == 1 else 1
            rank = (cold, self._freq.get(key, 0), index)
            if chosen_rank is None or rank < chosen_rank:
                chosen = key
                chosen_rank = rank
        return chosen


class S3FifoEviction:
    """Small FIFO, main FIFO, and a metadata ghost queue."""

    def __init__(self, budget: int) -> None:
        self._small_cap = max(1, budget // 10)
        self._main_cap = max(1, budget - self._small_cap)
        self._small: deque[RunKey] = deque()
        self._main: deque[RunKey] = deque()
        self._where: dict[RunKey, str] = {}
        self._freq: dict[RunKey, int] = {}
        self._span: dict[RunKey, int] = {}
        self._small_bytes = 0
        self._main_bytes = 0
        self._ghost: deque[RunKey] = deque()
        self._ghost_set: set[RunKey] = set()
        # One ghost key per mebibyte of the main FIFO, at least a few entries.
        self._ghost_slots = max(8, (budget - self._small_cap) // (1 << 20))

    def hit(self, key: RunKey) -> None:
        freq = self._freq.get(key)
        if freq is None or freq >= _FREQ_CAP:
            return
        self._freq[key] = freq + 1

    def insert(self, key: RunKey, span: int) -> None:
        if key in self._ghost_set:
            self._ghost_set.remove(key)
            self._main.append(key)
            self._where[key] = "m"
            self._main_bytes += span
        else:
            self._small.append(key)
            self._where[key] = "s"
            self._small_bytes += span
        self._freq[key] = 0
        self._span[key] = span

    def remove(self, key: RunKey) -> None:
        where = self._where.pop(key, None)
        span = self._span.pop(key, 0)
        self._freq.pop(key, None)
        if where == "s":
            self._small_bytes -= span
            self._discard(self._small, key)
        elif where == "m":
            self._main_bytes -= span
            self._discard(self._main, key)

    def relieve(
        self, protect: set[tuple[int, int]], members: Members
    ) -> RunKey | None:
        """Drop a one-hit once the small FIFO exceeds one tenth of the atlas.

        A reused tail is promoted instead. Returns None once both queues fit,
        including when the only change was a promotion.
        """
        return self._drain_small(protect, members, 0)

    def victim(
        self,
        protect: set[tuple[int, int]],
        members: Members,
        incoming: int = 0,
    ) -> RunKey | None:
        """Free one run, as if ``incoming`` bytes had just entered the small FIFO."""
        freed = self._drain_small(protect, members, incoming)
        if freed is not None:
            return freed
        return self._evict_main(protect, members)

    def _drain_small(
        self,
        protect: set[tuple[int, int]],
        members: Members,
        pending: int,
    ) -> RunKey | None:
        pinned = 0
        while self._small and self._small_bytes + pending > self._small_cap:
            if pinned >= len(self._small):
                break
            key = self._small.popleft()
            if _protected(key, protect, members):
                self._small.append(key)
                pinned += 1
                continue
            pinned = 0
            span = self._span[key]
            self._small_bytes -= span
            if self._freq.get(key, 0) > 0:
                self._freq[key] = 0
                self._main.append(key)
                self._where[key] = "m"
                self._main_bytes += span
                if self._main_bytes > self._main_cap:
                    freed = self._evict_main(protect, members)
                    if freed is not None:
                        return freed
                continue
            self._forget(key)
            self._remember(key)
            return key
        if pending == 0 and self._main_bytes > self._main_cap:
            return self._evict_main(protect, members)
        return None

    def _evict_main(
        self, protect: set[tuple[int, int]], members: Members
    ) -> RunKey | None:
        pinned = 0
        steps = 0
        limit = 4 * (len(self._main) + 1)
        while self._main and pinned < len(self._main) and steps < limit:
            steps += 1
            key = self._main.popleft()
            if _protected(key, protect, members):
                self._main.append(key)
                pinned += 1
                continue
            pinned = 0
            freq = self._freq.get(key, 0)
            if freq > 0:
                self._freq[key] = freq - 1
                self._main.append(key)
                continue
            self._main_bytes -= self._span.pop(key, 0)
            self._where.pop(key, None)
            self._freq.pop(key, None)
            self._remember(key)
            return key
        return None

    def _forget(self, key: RunKey) -> None:
        self._where.pop(key, None)
        self._freq.pop(key, None)
        self._span.pop(key, None)

    def _remember(self, key: RunKey) -> None:
        self._ghost.append(key)
        self._ghost_set.add(key)
        while len(self._ghost_set) > self._ghost_slots and self._ghost:
            old = self._ghost.popleft()
            if old in self._ghost_set:
                self._ghost_set.remove(old)

    @staticmethod
    def _discard(queue: deque[RunKey], key: RunKey) -> None:
        try:
            queue.remove(key)
        except ValueError:
            return


def make_eviction(name: str, budget: int) -> EvictionStrategy:
    """Build the named strategy. Unknown names are an error."""
    if name == "lru":
        return LruEviction(budget)
    if name == "cold_lfu":
        return ColdLfuEviction(budget)
    if name == "s3fifo":
        return S3FifoEviction(budget)
    message = f"unknown eviction strategy {name!r}; expected lru, cold_lfu, or s3fifo"
    raise ValueError(message)
