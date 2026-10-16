"""Prompt limit and causal block selection. No GPU."""

import math

import pytest
from qwen38_flash_next.context import (
    BLOCK,
    DEFAULT_LADDER,
    DEFAULT_PROMPT_LIMIT,
    DENSE_LIMIT,
    HEADROOM,
    INDEX_STRIDE,
    KEEP_BLOCKS,
    PREFILL_CHUNK,
    atlas_budget,
    atlas_fallbacks,
    block_score,
    cache_growth_bytes,
    process_footprint,
    prompt_token_limit,
    resident_candidates,
    score_bits,
    select_query_tokens,
    top_blocks,
)


def test_process_footprint_is_positive() -> None:
    assert process_footprint() > 0


def test_cache_growth_reserves_the_live_copy_once_it_is_larger() -> None:
    per_token = 16
    assert cache_growth_bytes(10, per_token) == 2 * PREFILL_CHUNK * per_token
    assert cache_growth_bytes(100_000, per_token) == 100_000 * per_token


def test_short_sequence_leaves_a_larger_atlas() -> None:
    memory = 48 << 30
    content = 64 << 30
    footprint = 10 << 30
    short = atlas_budget(
        memory=memory, content_bytes=content, footprint=footprint, growth=1 << 20
    )
    long = atlas_budget(
        memory=memory,
        content_bytes=content,
        footprint=footprint + (1 << 30),
        growth=1 << 30,
    )
    assert short > long
    assert short - long == (2 << 30) - (1 << 20)


def test_atlas_budget_keeps_headroom_and_the_expert_files() -> None:
    memory = 48 << 30
    footprint = 8 << 30
    growth = 1 << 30
    capped = atlas_budget(
        memory=memory, content_bytes=16 << 30, footprint=footprint, growth=growth
    )
    assert capped == 16 << 30
    open_budget = atlas_budget(
        memory=memory, content_bytes=0, footprint=footprint, growth=growth
    )
    assert open_budget == memory - HEADROOM - footprint - growth


def test_atlas_fallbacks_start_at_the_budget() -> None:
    budget = (30 << 30) + (512 << 20)
    steps = atlas_fallbacks(budget)
    assert steps[0] == budget
    assert steps[1] == budget - (1 << 30)
    assert steps[2] == steps[1] - (4 << 30)
    assert steps[-1] == 8 << 30


def test_prompt_budget_leaves_six_gibibytes() -> None:
    assert HEADROOM == 6 << 30


def test_unset_percent_keeps_the_ladder_when_it_fits() -> None:
    assert resident_candidates(50 << 30, None, memory=64 << 30) == DEFAULT_LADDER


def test_unset_percent_skips_a_step_larger_than_the_hot_tier_or_ram() -> None:
    memory = 48 << 30
    steps = resident_candidates(50 << 30, None, memory=memory)
    assert steps[0] == 24 << 30
    assert (32 << 30) not in steps
    hot = 16 << 30
    fitted = resident_candidates(hot, None, memory=memory)
    assert fitted[0] == hot
    assert fitted == (16 << 30, 12 << 30, 8 << 30)
    # Cold experts share the atlas. Counting those files uses the RAM step.
    both = resident_candidates(hot, None, memory=memory, cold_bytes=48 << 30)
    assert both[0] == 24 << 30
    assert both == DEFAULT_LADDER


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


def test_prompt_limit_defaults_to_128k() -> None:
    fitted = prompt_token_limit(
        memory=1 << 40,
        resident_hot=0,
        kv_per_token=1,
        residual_per_token=0,
        scratch=0,
        staging=0,
        fixed=0,
    )
    assert fitted == DEFAULT_PROMPT_LIMIT
    assert DEFAULT_PROMPT_LIMIT == 128 << 10


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
