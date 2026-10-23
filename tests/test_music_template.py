"""The smoke compares one request template with the inspect template."""

from __future__ import annotations

import pytest

from harness.music_template import compare_templates, input_template, output_template, smoke_body


def _note() -> dict[str, object]:
    return {
        "model": "example-model",
        "prompt": "electric guitar",
        "lyrics": "[Intro]",
        "seconds": 10,
        "seed": 0,
        "vocal_language": "en",
        "bpm": 168,
        "keyscale": "D major",
        "timesignature": "4/4",
    }


def _report(key: str = "D major", bpm: str = "165.1536", seconds: str = "10.0000") -> str:
    return (
        f"seconds {seconds}\n"
        f"key {key} correlation 0.8653 margin 0.0290 runner_up A major\n"
        f"bpm {bpm} peak 0.7391\n"
    )


def test_input_uses_the_canonical_key() -> None:
    text = input_template({**_note(), "keyscale": "Db minor"})
    assert text == "template_input\nseconds 10.0000\nkey C# minor\nbpm 168.0000\n"


def test_output_keeps_the_measured_fields() -> None:
    text = output_template(_report())
    assert text == "template_output\nseconds 10.0000\nkey D major\nbpm 165.1536\n"


def test_nearby_tempo_matches() -> None:
    assert compare_templates(input_template(_note()), output_template(_report())) == []


def test_wrong_key_and_a_far_tempo_are_named() -> None:
    measured = output_template(_report(key="A minor", bpm="120.0000"))
    assert compare_templates(input_template(_note()), measured) == [
        "key A minor is not D major",
        "bpm 120.0000 is 48.0000 from 168.0000",
    ]


def test_a_missing_pulse_fails() -> None:
    measured = output_template(_report(bpm="none"))
    assert compare_templates(input_template(_note()), measured) == ["bpm none"]


def test_an_unreadable_key_is_rejected() -> None:
    with pytest.raises(ValueError):
        input_template({**_note(), "keyscale": "happy"})


def test_the_request_body_carries_the_same_metadata() -> None:
    body = smoke_body(_note())
    assert body["keyscale"] == "D major"
    assert body["bpm"] == 168
    assert body["timesignature"] == "4/4"
    assert body["seconds"] == 10
