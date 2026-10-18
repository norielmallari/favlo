"""Published checkers for the three harness suites. No server."""

from __future__ import annotations

from harness.benchmarks import gsm8k, humaneval, ifeval
from harness.benchmarks.serve import Request, completion_body, completion_text


def test_completion_body_omits_unset_sampling_fields() -> None:
    body = completion_body(
        "example-model",
        [{"role": "user", "content": "2+2"}],
        32,
        Request(temperature=0.0, enable_thinking=False),
    )
    assert body["model"] == "example-model"
    assert body["temperature"] == 0.0
    assert body["enable_thinking"] is False
    assert body["max_tokens"] == 32
    assert "presence_penalty" not in body
    assert "reasoning_effort" not in body
    assert "top_p" not in body


def test_completion_text_is_the_server_content() -> None:
    text, reason, tokens = completion_text(
        {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "content": "#### 4",
                        "reasoning_content": "def f():\n    return 1\n",
                    },
                }
            ],
            "usage": {"completion_tokens": 3},
        }
    )
    assert text == "#### 4"
    assert reason == "stop"
    assert tokens == 3


def test_gsm8k_matches_the_published_number() -> None:
    rows = gsm8k.load()
    assert len(rows) == 1319
    assert gsm8k.extract_answer("working\n#### 18") == "18"
    assert gsm8k.extract_answer("#### 1,024") == "1024"
    assert gsm8k.extract_answer("#### The answer is $70,000.") == "70000"
    assert gsm8k.extract_answer("The final answer is 540.") == "540"
    assert gsm8k.correct(rows[0]["answer"], rows[0]) is True
    assert gsm8k.correct("#### 0", rows[0]) is False


def test_humaneval_canonical_solution_passes() -> None:
    rows = humaneval.load()
    assert len(rows) == 164
    assert humaneval.passed(rows[0], rows[0]["canonical_solution"]) is True
    fenced = "```python\n" + rows[0]["prompt"] + rows[0]["canonical_solution"] + "\n```"
    assert humaneval.passed(rows[0], fenced) is True
    strlen = next(row for row in rows if row["entry_point"] == "strlen")
    renamed = "```python\ndef strlen(s: str) -> int:\n    return len(s)\n```"
    assert humaneval.passed(strlen, renamed) is True


def test_ifeval_sentence_tokenizer_is_vendored() -> None:
    ifeval.prepare()
    from instruction_following_eval import instructions_util

    assert instructions_util.count_sentences("One ends. Two ends.") == 2


def test_ifeval_no_comma_checker() -> None:
    examples = ifeval.load()
    assert len(examples) == 541
    example = next(item for item in examples if item.key == 1001)
    bare = "I shall travel unto Japan and behold Kyoto then Nara."
    scored = ifeval.score(example, bare)
    assert scored["strict_prompt"] is True
    scored = ifeval.score(example, "I shall travel to Japan, then Kyoto.")
    assert scored["strict_prompt"] is False
