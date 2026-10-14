"""Published checkers for the three harness suites. No server."""

from __future__ import annotations

from harness.benchmarks import gsm8k, humaneval, ifeval
from harness.benchmarks.serve import reply_text


def test_code_detection_prefers_a_function() -> None:
    from harness.benchmarks.serve import _has_code

    assert _has_code("def f():\n    return 1\n")
    assert not _has_code("Let me analyze the docstring.")


def test_reply_text_drops_a_leading_close_tag() -> None:
    assert reply_text("</think>\n\nThe answer is 4.") == "The answer is 4."
    assert reply_text("The answer is 4.") == "The answer is 4."


def test_scored_reply_keeps_the_answer_after_the_trace() -> None:
    from harness.benchmarks.serve import scored_reply

    raw = "Let me draft, with a comma.\n</think>\n\nThe reply has no comma."
    assert scored_reply(raw, "", reply_only=True) == "The reply has no comma."
    assert scored_reply("", "def f():\n    return 1\n", reply_only=False).startswith("def ")
    assert scored_reply("", "the plan", reply_only=True) == ""


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
