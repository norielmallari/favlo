"""The needle harness grades a generated file. These tests do not start serve."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from harness.needle.grade import covered, finished, last_window_after, problems
from harness.needle.haystack import FILE_NAME, generate
from harness.needle.loop import continue_offset, rounds_for
from harness.needle.read_tool import read_tool


def _words(text: str) -> int:
    return len(text.split())


def test_the_same_seed_writes_the_same_file() -> None:
    first = generate(tokens=400, depth=0.5, seed=7, encode=_words)
    second = generate(tokens=400, depth=0.5, seed=7, encode=_words)
    assert first == second
    assert first.tokens == 400
    assert len(first.number) == 7
    assert first.number.isdigit()


def test_the_file_encodes_to_the_requested_length() -> None:
    for tokens in (50, 300, 801):
        haystack = generate(tokens=tokens, depth=0.5, seed=9, encode=_words)
        assert haystack.tokens == tokens
        assert _words(haystack.text) == tokens


def test_the_needle_is_the_only_line_with_a_digit() -> None:
    haystack = generate(tokens=400, depth=0.25, seed=1, encode=_words)
    lines = haystack.text.splitlines()
    assert lines[haystack.line - 1] == f"The special magic number is {haystack.number}."
    for index, line in enumerate(lines, start=1):
        if index == haystack.line:
            continue
        assert not any(character.isdigit() for character in line)


def test_depth_selects_the_ends() -> None:
    opening = generate(tokens=300, depth=0.0, seed=2, encode=_words)
    closing = generate(tokens=300, depth=1.0, seed=2, encode=_words)
    assert opening.line == 1
    assert closing.line == len(closing.text.splitlines())


def test_a_reply_passes_only_after_the_line_was_read(tmp_path: Path) -> None:
    haystack = generate(tokens=8000, depth=1.0, seed=3, encode=_words)
    assert haystack.line > 200
    (tmp_path / FILE_NAME).write_text(haystack.text, encoding="utf-8")
    reads: list[dict[str, object]] = []
    tool = read_tool(tmp_path, reads)
    first = json.loads(
        asyncio.run(tool.invoke(arguments={"file_path": FILE_NAME}, skip_parsing=True))
    )
    assert "next_offset" in first
    assert not covered(haystack.line, reads)
    assert problems(haystack.number, haystack.number, haystack.line, reads) == [
        "no read covered the needle line"
    ]
    steps = 0
    while not covered(haystack.line, reads):
        offset = reads[-1].get("next_offset")
        assert offset is not None
        json.loads(
            asyncio.run(
                tool.invoke(
                    arguments={"file_path": FILE_NAME, "offset": offset},
                    skip_parsing=True,
                )
            )
        )
        steps += 1
        assert steps < 20
    assert problems(haystack.number, haystack.number, haystack.line, reads) == []
    assert problems("none", haystack.number, haystack.line, reads) == ["reply missed the number"]


def test_retrieve_requires_the_needle_behind_the_last_window() -> None:
    early = {"lines": [{"line": 10, "text": "n"}], "next_offset": 20}
    tail = {"lines": [{"line": 400, "text": "end"}]}
    reads = [early, tail]
    assert finished(reads)
    assert last_window_after(10, reads)
    assert problems("7463343", "7463343", 10, reads, retrieve=True) == []
    assert problems("7463343", "7463343", 10, [early], retrieve=True) == [
        "the file was not read to the end",
        "the last window still held the needle line",
    ]
    held = {"lines": [{"line": 10, "text": "n"}]}
    assert problems("7463343", "7463343", 10, [held], retrieve=True) == [
        "the last window still held the needle line",
    ]
    empty = {"lines": [], "offset": 400}
    assert problems("7463343", "7463343", 10, [early, tail, empty], retrieve=True) == []


def test_page_mode_continues_from_the_last_offset_once() -> None:
    window = {"lines": [{"line": 1, "text": "filler"}], "next_offset": 193}
    assert continue_offset([window], retrieve=False, seen=set()) == 193
    assert continue_offset([window], retrieve=False, seen={193}) is None
    assert continue_offset([window], retrieve=True, seen=set()) is None
    assert continue_offset([{"lines": [{"line": 1, "text": "end"}]}], retrieve=False, seen=set()) is None


def test_rounds_cover_one_walk_of_the_file() -> None:
    haystack = generate(tokens=8000, depth=0.5, seed=4, encode=_words)
    assert rounds_for(haystack) >= 2
