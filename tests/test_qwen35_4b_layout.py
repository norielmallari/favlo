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
    kv_mode,
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
    assert cache_bytes(2048) == 2048 * 8 * 2 * 4 * 256 * 2
    assert cache_bytes(2048, "fp8_e4m3") == 2048 * 8 * 2 * 4 * (256 + 8)
    assert cache_bytes(2048, "half") == cache_bytes(2048)
    assert kv_mode(None) == "half"
    assert kv_mode("half") == "half"
    assert kv_mode("fp8_e4m3") == "fp8_e4m3"


def test_vision_pack_list() -> None:
    from qwen35_4b.convert import _order

    rows = _order()
    visual = [row for row in rows if row[1].startswith("visual.")]
    assert len(visual) == 297
    by_name = {logical: (kind, shape) for _raw, logical, kind, shape in visual}
    assert by_name["visual.patch"] == ("fp8", (1024, 1536))
    assert by_name["visual.patch_bias"][0] == "half"
    assert by_name["visual.pos"] == ("fp8", (2304, 1024))
    assert by_name["visual.0.norm1"][0] == "half"
    assert by_name["visual.0.qkv"] == ("fp8", (3072, 1024))
    assert by_name["visual.merger_fc2"] == ("fp8", (2560, 4096))
    assert by_name["visual.merger_norm"][0] == "half"


def test_thinking_has_no_2048_cap() -> None:
    from qwen35_4b.chat import _output_limit

    assert _output_limit({}, True, 60, 262144) == 262144 - 60
    assert _output_limit({"max_output_tokens": 32}, True, 60, 262144) == 32
    assert _output_limit({}, False, 60, 262144) == 2048


def test_thinking_follows_request_or_registry_not_variant() -> None:
    from types import SimpleNamespace

    from qwen35_4b.chat import _thinking

    thinking = SimpleNamespace(variant="thinking", enable_thinking=None)
    instruct = SimpleNamespace(variant="instruct", enable_thinking=None)
    bare = SimpleNamespace(variant=None, enable_thinking=None)
    closed = SimpleNamespace(variant="thinking", enable_thinking=False)
    assert _thinking(thinking, {"reasoning": {"effort": "none"}}) is False
    assert _thinking(instruct, {"reasoning": {"effort": "high"}}) is True
    assert _thinking(thinking, {"reasoning": {"effort": "low"}}) is True
    assert _thinking(thinking, {}) is True
    assert _thinking(instruct, {}) is True
    assert _thinking(closed, {}) is False
    assert _thinking(bare, {"reasoning": {"effort": "none"}}) is False
    assert _thinking(bare, {}) is True


def test_unsupported_effort_steps_up_or_falls_back() -> None:
    from qwen35_4b.sampling import resolve_effort

    assert resolve_effort(None) is None
    assert resolve_effort("none") == "none"
    assert resolve_effort("high") == "high"
    assert resolve_effort("minimal") == "high"
    assert resolve_effort("low") == "high"
    assert resolve_effort("medium") == "high"
    assert resolve_effort("xhigh") == "high"
    assert resolve_effort("max") == "high"


def test_vision_still_packing() -> None:
    from qwen35_4b.layout import IMAGE_TOKEN
    from qwen35_4b.vision import expand_image_tokens, rope_positions, smart_size, vision_rotary

    assert smart_size(256, 256) == (256, 256)
    assert smart_size(100, 100)[0] % 32 == 0
    expanded, spans = expand_image_tokens([1, IMAGE_TOKEN, 2], [4], IMAGE_TOKEN)
    assert expanded == [1, IMAGE_TOKEN, IMAGE_TOKEN, IMAGE_TOKEN, IMAGE_TOKEN, 2]
    assert spans == [1]
    positions, delta = rope_positions(len(expanded), [(1, 2, 2)])
    assert positions.shape == (3, 6)
    assert int(positions[0, 1]) == int(positions[0, 0]) + 1
    assert delta == int(positions.max()) + 1 - len(expanded)
    cos, sin = vision_rotary(4, 4)
    assert cos.shape == (16, 64)
    assert sin.shape == cos.shape


def test_historical_reasoning_is_dropped() -> None:
    from qwen35_4b.chat import _without_historical_reasoning

    history = _without_historical_reasoning(
        [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "", "reasoning_content": "old trace"},
            {"role": "assistant", "content": "<think>\nold trace\n</think>\n\napple"},
            {"role": "user", "content": "second"},
            {"role": "assistant", "content": "", "reasoning_content": "current trace"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"function": {"name": "lookup", "arguments": {}}}],
            },
        ]
    )
    assert history[1]["content"] == "apple"
    assert "reasoning_content" not in history[1]
    assert "</think>" not in history[1]["content"]
    assert history[3]["reasoning_content"] == "current trace"
    assert history[3]["tool_calls"][0]["function"]["name"] == "lookup"


def test_sampling_card() -> None:
    assert DEFAULT_VARIANT == "thinking"
    assert set(VARIANTS) == {"thinking", "instruct"}
    for row in VARIANTS.values():
        assert {"temperature", "top_p", "top_k", "presence_penalty"} <= set(row)
