"""Sparse indexer kernels match the host radix. One GPU."""

import numpy as np
import pytest
from qwen38_flash_next.context import INDEX_STRIDE, block_score, select_query_tokens
from qwen38_flash_next.formats import pack_bf16
from qwen38_flash_next.session import Dev, MetalSession


def _select(
    session: MetalSession, scores: np.ndarray, base: int, count: int = 1
) -> list[list[int]]:
    n_blocks = int(scores.shape[-1])
    flat = np.ascontiguousarray(scores, dtype=np.float32).reshape(count, n_blocks)
    idx = session.zeros(count * INDEX_STRIDE * 4)
    counts = session.zeros(count * 4)
    session.indexer_select(
        session.dev_from(flat),
        Dev(idx, count * INDEX_STRIDE),
        Dev(counts, count),
        count,
        base,
        n_blocks,
        INDEX_STRIDE,
    )
    got = session.read_u32(Dev(counts, count))
    tokens = session.read_u32(Dev(idx, count * INDEX_STRIDE)).reshape(count, INDEX_STRIDE)
    return [tokens[query, : int(got[query])].tolist() for query in range(count)]


def test_parallel_select_matches_the_host_radix() -> None:
    session = MetalSession()
    rng = np.random.default_rng(7)
    scores = rng.random(2000).astype(np.float32)
    scores[4] = scores[9]
    scores[15] = -1.0
    scores[16] = np.nan
    pos = 2000 * 4 + 1
    assert _select(session, scores, pos) == [select_query_tokens(pos, scores)]
    tied = np.ones(1000, np.float32)
    tied_pos = 1000 * 4
    assert _select(session, tied, tied_pos) == [select_query_tokens(tied_pos, tied)]
    short = np.zeros(32, np.float32)
    assert _select(session, short, 100) == [select_query_tokens(100, short)]
    pair_base = 2000 * 4
    pair = np.stack([scores, scores])
    got = _select(session, pair, pair_base, count=2)
    assert got[0] == select_query_tokens(pair_base, scores)
    assert got[1] == select_query_tokens(pair_base + 1, scores)


def test_one_token_block_score_matches_the_host_dot() -> None:
    session = MetalSession()
    rng = np.random.default_rng(3)
    keys = rng.normal(size=(4, 128)).astype(np.float32)
    heads = rng.normal(size=(4, 128)).astype(np.float32)
    index_qk = np.zeros(640, np.float32)
    index_qk[:512] = heads.reshape(-1)
    cache = np.zeros((8, 128), np.float32)
    cache[:4] = keys
    cache_bytes, cache_f = pack_bf16(cache)
    scores = session.dev_from(np.zeros(3, np.float32))
    session.indexer_block_score(
        session.dev_from(index_qk),
        session.upload(cache_bytes),
        scores,
        1,
        8,
        3,
    )
    got = session.read_f32(scores)
    reference = block_score(heads.tolist(), cache_f[:4].tolist())
    assert float(got[0]) == pytest.approx(reference, rel=1e-5, abs=1e-5)
    assert got[2] == 0.0
