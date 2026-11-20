"""fp16 key and value prefix for the CUDA runtime."""

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
    """The live sequence. Regions are fp16 K and V for the applied tokens."""

    def __init__(self, runtime: Any) -> None:
        self.runtime = runtime

    def capture(self) -> list[Region]:
        keys, values = self.runtime.capture_cache()
        return [Region("k", keys), Region("v", values)]

    def install(self, header: Any) -> bool:
        blobs = read_regions(header, ("k", "v"))
        if blobs is None:
            return False
        return bool(self.runtime.install_cache(blobs["k"], blobs["v"], len(header.tokens)))

    def reset(self) -> None:
        self.runtime.filled = 0

    def accept(self, header: Any, prompt: list[int]) -> bool:
        tokens = list(header.tokens)
        if len(tokens) >= len(prompt):
            return False
        return tokens == list(prompt[: len(tokens)])


def open_session(runtime: Any) -> PrefixSession:
    """The shared prefix store. The tag is the fp16 cache."""
    store = PrefixStore(
        default_prefix_dir(),
        model="gemma4_26b_a4b",
        digest=checkpoint_digest(runtime.root),
        tag="half",
        budget=DEFAULT_BUDGET,
    )
    return PrefixSession(store, CacheAdapter(runtime))
