"""Judge helpers for the thinking-mode persona harness."""

from harness.coherence import PERSONA_CHARS, judge_turn, persona, repeated_span


def test_persona_is_about_24k_and_names_the_facts() -> None:
    text = persona()
    assert abs(len(text) - PERSONA_CHARS) < 50
    assert "Mara Ellison Voss" in text
    assert "14 March 1987" in text
    assert "glass heron at dusk" in text
    assert text.count("Note 0001") == 1


def test_judge_accepts_a_direct_reply() -> None:
    reasoning = "The notes name Mara Ellison Voss, born in Halifax in 1987. The ninth buoy is blue."
    content = (
        "I am Mara Ellison Voss. I was born on 14 March 1987 in Halifax. The ninth buoy is blue."
    )
    assert judge_turn(content, reasoning, which=1) == []


def test_judge_flags_a_repeated_sentence() -> None:
    sentence = "The tide marker repeats the same station log without a new fact."
    text = " ".join([sentence] * 3)
    assert repeated_span(text) == sentence
    problems = judge_turn(text, "The notes list 128 estuaries and the Skeena.", which=2)
    assert any("repeats" in problem for problem in problems)
