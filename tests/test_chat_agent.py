"""Chat scenarios: a follow-up extends the stored prefix."""

import pytest
from harness.chat_agent import (
    SCENARIOS,
    log_mark,
    prefix_holds,
    requests_since,
    scenario_named,
    thinking_model,
)
from harness.file_session import RequestRate


def _request(
    prompt: int,
    decode: int,
    *,
    hit: int | None = None,
    missed: bool = False,
) -> RequestRate:
    return RequestRate(
        prompt_tokens=prompt,
        prefix_hit=hit,
        missed=missed,
        prefill="",
        prefill_gpu="",
        decode=f"decode 10.00 tok/s ({decode} tok in 1.00s)",
        decode_tokens=decode,
        atlas="",
        steady=(),
    )


def test_a_rolled_log_still_yields_the_new_requests() -> None:
    older = "prompt tokens 10\ncache miss\ndecode 1.00 tok/s (1 tok in 1.00s)\n"
    added = "prompt tokens 12\ncache hit 11\ndecode 1.00 tok/s (2 tok in 1.00s)\n"
    mark = log_mark(older)
    found = requests_since(mark, older + added)
    assert len(found) == 1
    assert found[0].prompt_tokens == 12
    assert found[0].prefix_hit == 11


def test_thinking_model_uses_the_thinking_row() -> None:
    assert thinking_model("example-model:instruct") == "example-model:thinking"
    assert thinking_model("example-model") == "example-model:thinking"


def test_a_follow_up_extends_the_stored_prefix() -> None:
    requests = [
        _request(100, 20, missed=True),
        _request(140, 8, hit=120),
    ]
    assert prefix_holds(requests) == []


def test_a_dropped_prefix_is_a_problem() -> None:
    requests = [
        _request(100, 20, missed=True),
        _request(90, 4, missed=True),
    ]
    problems = prefix_holds(requests)
    assert problems == ["turn 2 prefix miss expected 120", "turn 2 prompt did not grow past 120"]


def test_unknown_scenario_names_the_known_ones() -> None:
    with pytest.raises(SystemExit, match="thinking-follow-up"):
        scenario_named("missing")
    assert scenario_named(None) == SCENARIOS
    assert scenario_named("thinking-follow-up")[0].name == "thinking-follow-up"
