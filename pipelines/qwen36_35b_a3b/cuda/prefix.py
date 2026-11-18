"""Prefix adapter for the int8 cache and the Gated DeltaNet state."""

from __future__ import annotations

from typing import Any

from favlo.prefix import (
    DEFAULT_BUDGET,
    PrefixSession,
    PrefixStore,
    Region,
    checkpoint_digest,
    default_prefix_dir,
    read_regions,
)


class CacheAdapter:
    """The live sequence: key, value, DeltaNet state, and convolution history."""

    def __init__(self, runtime: Any) -> None:
        self.runtime = runtime

    def capture(self) -> list[Region]:
        return [Region(name, data) for name, data in self.runtime.capture_regions()]

    def install(self, header: Any) -> bool:
        blobs = read_regions(header, ("k", "v", "state", "history"))
        if blobs is None:
            return False
        return bool(self.runtime.install_regions(blobs, len(header.tokens)))

    def reset(self) -> None:
        self.runtime._reset_sequence()

    def accept(self, header: Any, prompt: list[int]) -> bool:
        tokens = list(header.tokens)
        if len(tokens) >= len(prompt):
            return False
        return tokens == list(prompt[: len(tokens)])


def open_session(runtime: Any) -> PrefixSession:
    """The shared prefix store for this record's int8 cache."""
    store = PrefixStore(
        default_prefix_dir(),
        model="qwen36_35b_a3b",
        digest=checkpoint_digest(runtime.root),
        tag=runtime.kv_cache,
        budget=DEFAULT_BUDGET,
    )
    return PrefixSession(store, CacheAdapter(runtime))
