"""Process-local cache of MCP tool definitions.

The database stores a server and which of its groups are on. Names,
descriptions, and input schemas are read from ``tools/list`` and kept here
until ``TTL_SECONDS`` passes. A second reader of the same connection waits
for the fetch already in flight. A failed fetch is not stored.
"""

from __future__ import annotations

import asyncio
import copy
import time
from collections import OrderedDict
from collections.abc import Awaitable, Callable
from typing import Any

from web_api.mcp_client import list_tools

TTL_SECONDS = 60.0
_MAX_ENTRIES = 64
Fetch = Callable[[dict[str, Any]], Awaitable[list[dict[str, Any]]]]


def cache_key(server: dict[str, Any]) -> str:
    """The connection, so two rows that point at one server share one catalog."""
    transport = str(server.get("transport") or "")
    url = str(server.get("url") or "")
    command = str(server.get("command") or "")
    return f"{transport}\n{url}\n{command}"


class CatalogCache:
    """A TTL cache with one in-flight fetch per connection."""

    def __init__(
        self,
        *,
        ttl: float = TTL_SECONDS,
        max_entries: int = _MAX_ENTRIES,
        fetch: Fetch = list_tools,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.ttl = ttl
        self.max_entries = max_entries
        self._fetch = fetch
        self._clock = clock
        self._items: OrderedDict[str, tuple[float, list[dict[str, Any]]]] = OrderedDict()
        self._flights: dict[str, asyncio.Task[list[dict[str, Any]]]] = {}
        self._lock = asyncio.Lock()

    async def get(self, server: dict[str, Any], *, refresh: bool = False) -> list[dict[str, Any]]:
        """The catalog for ``server``. ``refresh`` skips a stored entry."""
        key = cache_key(server)
        if refresh:
            tools = await self._fetch(server)
            async with self._lock:
                self._store(key, tools)
            return copy.deepcopy(tools)
        async with self._lock:
            found = self._fresh(key)
            if found is not None:
                return copy.deepcopy(found)
            flight = self._flights.get(key)
            owner = flight is None
            if owner:
                flight = asyncio.create_task(self._fetch(server))
                self._flights[key] = flight
        assert flight is not None
        try:
            tools = await flight
        except Exception:
            if owner:
                async with self._lock:
                    if self._flights.get(key) is flight:
                        del self._flights[key]
            raise
        if owner:
            async with self._lock:
                self._store(key, tools)
                if self._flights.get(key) is flight:
                    del self._flights[key]
        return copy.deepcopy(tools)

    def _fresh(self, key: str) -> list[dict[str, Any]] | None:
        item = self._items.get(key)
        if item is None:
            return None
        expires, tools = item
        if expires <= self._clock():
            del self._items[key]
            return None
        self._items.move_to_end(key)
        return tools

    def _store(self, key: str, tools: list[dict[str, Any]]) -> None:
        self._items[key] = (self._clock() + self.ttl, copy.deepcopy(tools))
        self._items.move_to_end(key)
        while len(self._items) > self.max_entries:
            self._items.popitem(last=False)


catalog = CatalogCache()
