"""Layout and sampling card for qwen35-4b. No device."""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1] / "pipelines" / "metal41"
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from qwen35_4b.layout import (  # noqa: E402
    FULL,
    GDN_QKV,
    GDN_STATE_BYTES,
    HIDDEN,
    LAYERS,
    VOCAB,
    cache_bytes,
    fp8_bytes,
    is_full,
)
from qwen35_4b.sampling import DEFAULT_VARIANT, VARIANTS  # noqa: E402


def test_hybrid_layer_pattern() -> None:
    assert LAYERS == 32
    assert FULL == (3, 7, 11, 15, 19, 23, 27, 31)
    assert sum(1 for layer in range(LAYERS) if is_full(layer)) == 8
    assert sum(1 for layer in range(LAYERS) if not is_full(layer)) == 24


def test_gdn_projection_width() -> None:
    assert GDN_QKV == 8192
    assert GDN_STATE_BYTES == 50_331_648


def test_fp8_and_cache_sizes() -> None:
    assert fp8_bytes(VOCAB, HIDDEN) == VOCAB * HIDDEN + VOCAB * (HIDDEN // 32)
    assert cache_bytes(2048) == 2048 * 8 * 2 * 4 * (256 + 8)


def test_sampling_card() -> None:
    assert DEFAULT_VARIANT == "thinking"
    assert set(VARIANTS) == {"thinking", "instruct"}
    for row in VARIANTS.values():
        assert {"temperature", "top_p", "top_k", "presence_penalty"} <= set(row)
