"""Layout and sampling card for qwen35_4b. No device."""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1] / "pipelines" / "metal41" / "qwen35_4b"
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from qwen35_4b.metal41.layout import (  # noqa: E402
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
from qwen35_4b.metal41.mtp import kept_drafts, verify  # noqa: E402
from qwen35_4b.metal41.sampling import DEFAULT_VARIANT, VARIANTS  # noqa: E402


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
    from qwen35_4b.metal41.convert import _order

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
    from qwen35_4b.metal41.chat import _output_limit

    assert _output_limit({}, True, 60, 262144) == 262144 - 60
    assert _output_limit({"max_output_tokens": 32}, True, 60, 262144) == 32
    assert _output_limit({}, False, 60, 262144) == 2048


def test_effort_opens_thinking_without_changing_the_card() -> None:
    from types import SimpleNamespace

    from qwen35_4b.metal41.chat import _request_effort
    from qwen35_4b.metal41.sampling import mode_card

    instruct = SimpleNamespace(
        variant="instruct",
        sampling={"temperature": 0.7, "top_p": 0.8, "top_k": 20, "presence_penalty": 1.5},
        variants=None,
        default_variant="thinking",
        enable_thinking=None,
        reasoning_effort="none",
    )
    thinking, card = mode_card(instruct, _request_effort({"reasoning": {"effort": "low"}}))
    assert thinking is True
    assert card["temperature"] == 0.7
    closed, still = mode_card(instruct, _request_effort({"reasoning": {"effort": "none"}}))
    assert closed is False
    assert still["temperature"] == 0.7
    omitted, instruct_card = mode_card(instruct, _request_effort({}))
    assert omitted is False
    assert instruct_card["temperature"] == 0.7

    stored = SimpleNamespace(
        variant="thinking",
        sampling=None,
        variants=None,
        default_variant="thinking",
        enable_thinking=True,
        reasoning_effort="high",
    )
    open_think, thinking_card = mode_card(stored, _request_effort({}))
    assert open_think is True
    assert thinking_card["temperature"] == 1.0

    bare = SimpleNamespace(
        variant=None,
        sampling=None,
        variants=None,
        default_variant="thinking",
        enable_thinking=None,
        reasoning_effort=None,
    )
    default_open, default_card = mode_card(bare, _request_effort({}))
    assert default_open is True
    assert default_card["temperature"] == 1.0

    row_closed = SimpleNamespace(
        variant="thinking",
        sampling=None,
        variants=None,
        default_variant="thinking",
        enable_thinking=False,
        reasoning_effort=None,
    )
    assert mode_card(row_closed, _request_effort({}))[0] is False


def test_unsupported_effort_steps_up_or_falls_back() -> None:
    from qwen35_4b.metal41.sampling import resolve_effort

    assert resolve_effort(None) is None
    assert resolve_effort("none") == "none"
    assert resolve_effort("high") == "high"
    assert resolve_effort("minimal") == "high"
    assert resolve_effort("low") == "high"
    assert resolve_effort("medium") == "high"
    assert resolve_effort("xhigh") == "high"
    assert resolve_effort("max") == "high"


def test_vision_still_packing() -> None:
    from qwen35_4b.metal41.layout import IMAGE_TOKEN
    from qwen35_4b.metal41.vision import expand_image_tokens, rope_positions, smart_size, vision_rotary

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
    from qwen35_4b.metal41.chat import _without_historical_reasoning

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


class _StopSteps:
    def __init__(self, predictions: list[int]) -> None:
        self._predictions = list(predictions)
        self.stepped: list[int] = []

    def _step(self, token: int, *, sample: bool = False) -> int:
        del sample
        self.stepped.append(int(token))
        return self._predictions.pop(0)


def test_verify_leaves_a_stop_out_of_the_cache() -> None:
    runtime = _StopSteps([11, 248046, 99])
    accepted, nxt = verify(runtime, 0, [7, 11], [11, 248046], {248044, 248046})
    assert accepted == 2
    assert nxt == 248046
    assert runtime.stepped == [7, 11]


def test_kept_drafts_drop_the_stop_and_everything_after_it() -> None:
    stop = {248044, 248046}
    kept, halted = kept_drafts([11, 248046, 12], 3, stop)
    assert kept == [11]
    assert halted is True
    kept, halted = kept_drafts([11, 12, 13], 2, stop)
    assert kept == [11, 12]
    assert halted is False


def test_sampling_card() -> None:
    assert DEFAULT_VARIANT == "thinking"
    assert set(VARIANTS) == {"thinking", "instruct"}
    assert VARIANTS["thinking"]["reasoning_effort"] == "high"
    assert VARIANTS["instruct"]["reasoning_effort"] == "none"
    for row in VARIANTS.values():
        assert {"temperature", "top_p", "top_k", "presence_penalty", "reasoning_effort"} <= set(
            row
        )
