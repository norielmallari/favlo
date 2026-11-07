"""umT5 encoder for the DiT text context.

The encoder is bidirectional. Each block is RMSNorm, attention with a relative
bucket bias, a residual, RMSNorm, then the gated GELU feed-forward. There is
no dropout and no linear bias. The final RMSNorm is the context the DiT reads.
Padding is not inside this pass: only the real tokens, including the end
token, are returned.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

from wan22_i2v.metal import Gpu
from wan22_i2v.source import TOKENIZER_FILE

_LAYERS = 24
_HEADS = 64
_WIDTH = 4096
_FFN = 10240
_BUCKETS = 32
_MAX_DISTANCE = 128
_EPS = np.float32(1e-6)


def tokenizer_path(root: Path | None = None, pack: Path | None = None) -> Path:
    """Tokenizer beside the shared int8 pack."""
    from wan22_i2v.umt5_pack import shared_pack_dir

    return (pack or shared_pack_dir(root)) / TOKENIZER_FILE


def relative_buckets(query: int, key: int) -> np.ndarray:
    """T5 bidirectional buckets. Rows are queries and columns are keys."""
    positions = np.arange(key, dtype=np.int64)[None, :] - np.arange(query, dtype=np.int64)[:, None]
    half = _BUCKETS // 2
    buckets = (positions > 0).astype(np.int64) * half
    distance = np.abs(positions)
    exact = half // 2
    small = distance < exact
    span = math.log(_MAX_DISTANCE / exact)
    safe = np.maximum(distance, 1)
    scaled = np.log(safe.astype(np.float64) / exact) / span * (half - exact)
    large = exact + scaled.astype(np.int64)
    large = np.minimum(large, half - 1)
    buckets += np.where(small, distance, large)
    return buckets.astype(np.intp)


def encode_prompt(
    prompt: str,
    weights: Path | None = None,
    tokenizer: Path | None = None,
    on_layer: Callable[[], None] | None = None,
    gpu: Gpu | None = None,
    pack: Path | None = None,
) -> np.ndarray:
    """Token vectors ``[tokens, 4096]`` from the packed encoder."""
    del weights
    vocab = Tokenizer.from_file(str(tokenizer or tokenizer_path()))
    ids = vocab.encode(prompt).ids
    if not ids:
        for _ in range(_LAYERS):
            if on_layer is not None:
                on_layer()
        return np.zeros((0, 4096), dtype=np.float32)
    if pack is None or not (pack / "manifest.json").is_file():
        message = "umT5 pack is missing"
        raise RuntimeError(message)
    started = time.perf_counter()
    own = gpu is None
    device = Gpu() if own else gpu
    try:
        from wan22_i2v.umt5_pack import PackedEncoder

        vectors = PackedEncoder(pack).run(np.asarray(ids, dtype=np.int64), on_layer, device)
    finally:
        if own:
            device.close()
    elapsed = time.perf_counter() - started
    print(f"umt5 {len(ids)} tokens {elapsed:.2f}s", flush=True)
    return vectors

