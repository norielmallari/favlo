"""Published checkers for the three harness suites. No server."""

from __future__ import annotations

from harness.benchmarks import gsm8k, humaneval, ifeval
from harness.benchmarks.serve import reply_text


def test_reply_text_drops_a_leading_close_tag() -> None:
    assert reply_text("</think>\n\nThe answer is 4.") == "The answer is 4."
    assert reply_text("The answer is 4.") == "The answer is 4."


def test_gsm8k_matches_the_published_number() -> None:
    rows = gsm8k.load()
    assert len(rows) == 1319
    assert gsm8k.extract_answer("working\n#### 18") == "18"
    assert gsm8k.extract_answer("#### 1,024") == "1024"
    assert gsm8k.correct(rows[0]["answer"], rows[0]) is True
    assert gsm8k.correct("#### 0", rows[0]) is False


def test_humaneval_canonical_solution_passes() -> None:
    rows = humaneval.load()
    assert len(rows) == 164
    assert humaneval.passed(rows[0], rows[0]["canonical_solution"]) is True
    fenced = "```python\n" + rows[0]["prompt"] + rows[0]["canonical_solution"] + "\n```"
    assert humaneval.passed(rows[0], fenced) is True


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
