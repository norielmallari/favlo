"""Chat usage timings and generate phase clocks. No device."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_ROOT = Path(__file__).resolve().parents[1] / "pipelines" / "metal41"
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from qwen35_4b.chat import _usage  # noqa: E402
from qwen35_4b.forward import Qwen35Runtime  # noqa: E402
from qwen35_4b.sampling import CONTEXT_LENGTH  # noqa: E402


def test_usage_reports_timings_and_context_length() -> None:
    runtime = SimpleNamespace(
        max_pos=8192,
        _cached_tokens=12,
        _prefill_seconds=2.0,
        _decode_seconds=4.0,
    )
    usage = _usage(runtime, 100, 4)
    assert usage["prompt_tokens"] == 100
    assert usage["completion_tokens"] == 4
    assert usage["total_tokens"] == 104
    assert usage["context_length"] == 8192
    assert usage["prompt_tokens_details"] == {"cached_tokens": 12}
    assert usage["timings"]["prompt_n"] == 88
    assert usage["timings"]["prompt_per_second"] == 44.0
    assert usage["timings"]["predicted_n"] == 4
    assert usage["timings"]["predicted_per_second"] == 1.0


def test_usage_caps_context_at_registered_window() -> None:
    runtime = SimpleNamespace(
        max_pos=CONTEXT_LENGTH * 2,
        _cached_tokens=0,
        _prefill_seconds=1.0,
        _decode_seconds=1.0,
    )
    usage = _usage(runtime, 8, 1)
    assert usage["context_length"] == CONTEXT_LENGTH


def test_usage_rate_is_zero_when_the_clock_is_zero() -> None:
    runtime = SimpleNamespace(
        max_pos=4096,
        _cached_tokens=8,
        _prefill_seconds=0.0,
        _decode_seconds=0.0,
    )
    usage = _usage(runtime, 8, 1)
    assert usage["timings"]["prompt_n"] == 0
    assert usage["timings"]["prompt_per_second"] == 0.0
    assert usage["timings"]["predicted_n"] == 1
    assert usage["timings"]["predicted_per_second"] == 0.0


def test_generate_clock_ends_prefill_at_the_first_sample(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = object.__new__(Qwen35Runtime)
    runtime.console = None
    runtime.max_pos = 8
    runtime.pos = 2
    runtime._halted = False
    runtime._stop = None
    runtime._request_id = None
    runtime._presence = 0.0
    runtime._frequency = 0.0
    runtime._output_ids = []
    runtime._prefill_progress = None
    clock = {"t": 100.0}

    def now() -> float:
        return clock["t"]

    class Prefix:
        loaded = False
        tokens: list[int] = []
        dirty = False

        def align(self, prompt: list[int]) -> int:
            del prompt
            return 1

        def note(self, tokens: list[int]) -> None:
            self.tokens = list(tokens)
            clock["t"] += 3.0

        def flush(self) -> None:
            return None

    class Session:
        def wait(self) -> None:
            return None

    runtime.prefix = Prefix()
    runtime.session = Session()
    runtime.refresh_budget = lambda: None

    def prefill_chunk(tokens: list[int], *, logits: bool) -> None:
        del logits
        clock["t"] += 2.0 * len(tokens)
        runtime.pos += len(tokens)

    samples = {"n": 0}

    def sample() -> int:
        samples["n"] += 1
        clock["t"] += 1.0
        return 7

    def step(token: int) -> None:
        del token
        clock["t"] += 2.0
        runtime.pos += 1

    runtime._prefill_chunk = prefill_chunk
    runtime._sample = sample
    runtime._step = step
    runtime._check = lambda: None
    runtime._store_prefix = lambda: None
    runtime._reset_state = lambda: None
    monkeypatch.setattr("qwen35_4b.forward.time.perf_counter", now)

    def on_token(_token: int) -> None:
        clock["t"] += 0.5

    out = runtime.generate([9, 10], 2, set(), on_token=on_token)
    assert out == [7, 7]
    # Prefill: one uncached token at +2, first sample at +1 → 3.0s.
    assert runtime._prefill_seconds == pytest.approx(3.0)
    # Decode: first sample instant to second sample. Between them: note(+3),
    # on_token(+0.5), step(+2), sample(+1) = 6.5s. Mid prefix store is after
    # the first sample stamp, so it sits inside that window.
    assert runtime._decode_seconds == pytest.approx(6.5)

    clock["t"] = 100.0
    runtime.pos = 2
    samples["n"] = 0
    one = runtime.generate([9, 10], 1, set(), on_token=on_token)
    assert one == [7]
    assert runtime._prefill_seconds == pytest.approx(3.0)
    assert runtime._decode_seconds == 0.0

    # A stop id after one kept token must not stretch the decode clock.
    clock["t"] = 100.0
    runtime.pos = 2
    draws = iter([7, 99])

    def sample_then_stop() -> int:
        clock["t"] += 1.0
        return next(draws)

    runtime._sample = sample_then_stop
    stopped = runtime.generate([9, 10], 8, {99}, on_token=on_token)
    assert stopped == [7]
    assert runtime._prefill_seconds == pytest.approx(3.0)
    assert runtime._decode_seconds == 0.0
