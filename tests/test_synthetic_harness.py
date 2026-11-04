"""The synthetic prompt is an exact length. These tests do not start serve."""

from __future__ import annotations

from harness.file_session import RequestRate
from harness.synthetic.prompt import user_text
from harness.synthetic.rate import end_to_end, measured_problems


def _overhead(text: str) -> int:
    return len(text.split()) + 7


def test_the_same_seed_writes_the_same_prompt() -> None:
    first = user_text(tokens=200, label="Passage 200.", encode=_overhead, seed=3)
    second = user_text(tokens=200, label="Passage 200.", encode=_overhead, seed=3)
    assert first == second
    assert _overhead(first) == 200
    assert first.startswith("Passage 200.\n")


def test_two_labels_do_not_share_an_opening_line() -> None:
    short = user_text(tokens=120, label="Passage 512.", encode=_overhead, seed=1)
    long = user_text(tokens=120, label="Passage 10000.", encode=_overhead, seed=1)
    assert short.splitlines()[0] != long.splitlines()[0]
    assert _overhead(short) == 120
    assert _overhead(long) == 120


def test_a_label_longer_than_the_target_is_rejected() -> None:
    try:
        user_text(tokens=2, label="Passage 10000.", encode=_overhead, seed=0)
    except ValueError as exc:
        assert "label" in str(exc)
    else:
        raise AssertionError("expected the label to be rejected")


def test_end_to_end_adds_the_hot_path_clocks() -> None:
    request = _request()
    assert end_to_end(request) == "end to end 83.33 tok/s (640 tok in 7.68s)"
    assert measured_problems(request, 512) == []


def test_a_prefix_hit_is_not_a_measured_prefill() -> None:
    request = _request(prefix_hit=400, missed=False)
    assert measured_problems(request, 512) == ["prefill reused a prefix"]


def _request(*, prefix_hit: int | None = None, missed: bool = True) -> RequestRate:
    return RequestRate(
        prompt_tokens=512,
        prefix_hit=prefix_hit,
        missed=missed,
        prefill="prefill 100.00 tok/s (512 tok in 5.12s)",
        prefill_gpu="",
        decode="decode 50.00 tok/s (128 tok in 2.56s)",
        decode_tokens=128,
        atlas="",
        steady=(),
    )
