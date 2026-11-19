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


def test_a_trim_that_skips_a_count_is_padded() -> None:
    text = user_text(tokens=51, label="Hi.", encode=_skips, seed=0)
    assert _skips(text) == 51
    assert text.startswith("Hi.\n")


def _skips(text: str) -> int:
    body = text[:-1] if text.endswith("\n") else text
    if body.endswith("0"):
        return 20 + (len(text) - 1) * 2 + 1
    return 20 + len(text) * 2


def test_a_new_line_that_steps_over_is_closed_on_the_previous_line() -> None:
    text = user_text(tokens=48, label="Hi.", encode=_boundary, seed=0)
    assert _boundary(text) == 48
    assert text.startswith("Hi.\n")


def _boundary(text: str) -> int:
    body = text[:-1] if text.endswith("\n") else text
    rows = body.split("\n")
    extra = 1 if rows and rows[-1] else 0
    if rows and (rows[-1].endswith("00") or rows[-1].endswith("0.")):
        extra = 0
    return 30 + len(text) + extra


def test_a_long_prompt_searches_the_line_count() -> None:
    calls = 0

    def encode(text: str) -> int:
        nonlocal calls
        calls += 1
        return len(text.split()) + 7

    text = user_text(tokens=2000, label="Passage 2000.", encode=encode, seed=1)
    assert encode(text) == 2000
    assert calls < 50


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


def test_a_short_decode_is_not_the_cap() -> None:
    request = _request(decode_tokens=113)
    assert measured_problems(request, 512, 128) == ["decode counted 113 expected 128"]
    assert measured_problems(request, 512) == []


def _request(
    *,
    prefix_hit: int | None = None,
    missed: bool = True,
    decode_tokens: int = 128,
) -> RequestRate:
    return RequestRate(
        prompt_tokens=512,
        prefix_hit=prefix_hit,
        missed=missed,
        prefill="prefill 100.00 tok/s (512 tok in 5.12s)",
        prefill_gpu="",
        decode=f"decode 50.00 tok/s ({decode_tokens} tok in 2.56s)",
        decode_tokens=decode_tokens,
        atlas="",
        steady=(),
    )
