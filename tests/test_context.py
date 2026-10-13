"""Prompt limit and causal block selection. No GPU."""

import math

import pytest
from qwen38_flash_next.context import (
    BLOCK,
    DEFAULT_LADDER,
    DENSE_LIMIT,
    INDEX_STRIDE,
    KEEP_BLOCKS,
    block_score,
    prompt_token_limit,
    resident_candidates,
    score_bits,
    select_query_tokens,
    top_blocks,
)


def test_unset_percent_keeps_the_ladder() -> None:
    assert resident_candidates(50 << 30, None) == DEFAULT_LADDER


def test_percent_steps_down_to_the_floor() -> None:
    hot = 40 << 30
    steps = resident_candidates(hot, 60)
    assert steps[0] == 24 << 30
    assert steps == (24 << 30, 20 << 30, 16 << 30, 12 << 30, 8 << 30)


def test_percent_below_the_floor_is_one_candidate() -> None:
    budget = resident_candidates(40 << 30, 10)
    assert budget == (4 << 30,)


def test_percent_page_aligns_before_the_steps() -> None:
    hot = (10 << 30) + 100
    steps = resident_candidates(hot, 100)
    assert steps[0] == 10 << 30
    assert steps[-1] == 8 << 30
    assert all(size % 16384 == 0 for size in steps)


def test_percent_rejects_out_of_range() -> None:
    with pytest.raises(ValueError):
        resident_candidates(1 << 30, 0)
    with pytest.raises(ValueError):
        resident_candidates(1 << 30, 101)
    with pytest.raises(ValueError):
        resident_candidates(1 << 30, math.nan)


def test_prompt_limit_uses_the_tighter_peak() -> None:
    # Prefill room 900 / 15 = 60. Decode room 520 / 10 = 52.
    assert (
        prompt_token_limit(
            memory=1000,
            resident_hot=400,
            kv_per_token=10,
            residual_per_token=5,
            scratch=50,
            staging=20,
            fixed=30,
        )
        == 52
    )


def test_block_score_is_the_relu_sum_of_the_mean_key() -> None:
    queries = ((1.0, 0.0), (0.0, 1.0))
    keys = ((1.0, 0.0), (1.0, 0.0))
    assert block_score(queries, keys) == pytest.approx(1.0)


def test_top_blocks_matches_a_sort() -> None:
    scores = [0.2, 5.0, 5.0, 0.0, 9.0, 1.5, 3.0, 3.0, 8.0, 0.4]
    chosen = top_blocks(scores, keep=4)
    order = sorted(range(len(scores)), key=lambda index: (-score_bits(scores[index]), index))
    assert chosen == sorted(order[:4])


def test_top_blocks_tie_keeps_the_lower_index() -> None:
    assert top_blocks([1.0, 1.0, 1.0, 1.0], keep=2) == [0, 1]


def test_select_query_tokens_keeps_the_tail_and_drops_the_later_tie() -> None:
    n_complete = KEEP_BLOCKS + 1
    scores = [1.0] * n_complete
    pos = n_complete * BLOCK + 2
    chosen = select_query_tokens(pos, scores)
    assert chosen[-1] == pos
    assert all(token <= pos for token in chosen)
    assert (KEEP_BLOCKS * BLOCK) not in chosen
    assert 0 in chosen
    assert len(chosen) <= INDEX_STRIDE


def test_short_query_attends_every_token() -> None:
    pos = DENSE_LIMIT - 1
    chosen = select_query_tokens(pos, [0.0])
    assert chosen == list(range(pos + 1))
