"""Published checkers for the three harness suites. No server."""

from __future__ import annotations

from harness.benchmarks import gsm8k, humaneval, ifeval


def test_gsm8k_matches_the_published_number() -> None:
    rows = gsm8k.load()
    assert len(rows) == 1319
    prompt = gsm8k.messages(rows[0])[0]["content"]
    assert prompt.count("\nAnswer:") == 6
    assert "End with a line" not in prompt
    assert prompt.endswith(f"Question: {rows[0]['question']}\nAnswer:")
    assert gsm8k.extract_answer("working\n#### 18") == "18"
    assert gsm8k.extract_answer("#### 1,024") == "1024"
    assert gsm8k.extract_answer("#### The answer is $70,000.") == "[invalid]"
    assert gsm8k.extract_answer("The final answer is 540.") == "[invalid]"
    assert gsm8k.extract_answer("#### 18\nQuestion: next\n#### 1") == "18"
    assert gsm8k.correct(rows[0]["answer"], rows[0]) is True
    assert gsm8k.correct("#### 0", rows[0]) is False


def test_humaneval_canonical_solution_passes() -> None:
    rows = humaneval.load()
    assert len(rows) == 164
    assert "passes the tests" in humaneval.messages(rows[0])[0]["content"]
    assert humaneval.passed(rows[0], rows[0]["canonical_solution"]) is True
    fenced = "```python\n" + rows[0]["prompt"] + rows[0]["canonical_solution"] + "\n```"
    assert humaneval.passed(rows[0], fenced) is True
    trailed = rows[0]["canonical_solution"] + "\ndef other():\n    return 0\n"
    assert humaneval.passed(rows[0], trailed) is True
    assert "def other" not in humaneval.function_body(
        trailed, rows[0]["prompt"], rows[0]["entry_point"]
    )
    strlen = next(row for row in rows if row["entry_point"] == "strlen")
    renamed = "```python\ndef strlen(s: str) -> int:\n    return len(s)\n```"
    assert humaneval.passed(strlen, renamed) is False


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
