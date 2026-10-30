"""GSM8K test split. Five training shots, then the published ``####`` number."""

from __future__ import annotations

import json
import re
from pathlib import Path

_DATA = Path(__file__).resolve().parent / "data" / "gsm8k" / "test.jsonl"
_SHOTS = Path(__file__).resolve().parent / "data" / "gsm8k" / "train_shots.jsonl"
_ANSWER = re.compile(r"#### (\-?[0-9\.\,]+)")
_INVALID = "[invalid]"
_SHOT_COUNT = 5


def load() -> list[dict[str, str]]:
    return _read(_DATA)


def messages(row: dict[str, str]) -> list[dict[str, str]]:
    """Five train examples, then ``Question`` / ``Answer`` for this row.

    The shots are the first five lines of the published training split.
    The reply is the text after ``Answer:``.
    """
    parts = [
        f"Question: {shot['question']}\nAnswer: {shot['answer']}" for shot in _shots()
    ]
    parts.append(f"Question: {row['question']}\nAnswer:")
    return [{"role": "user", "content": "\n\n".join(parts)}]


def extract_answer(completion: str) -> str:
    """The first number on a ``####`` line, commas removed.

    Words between the marker and the number do not match. A reply with no
    such number is invalid. Text from the next ``Question:`` on is ignored.
    """
    text = _until_question(completion)
    match = _ANSWER.search(text)
    if match is None:
        return _INVALID
    return match.group(1).replace(",", "")


def correct(response: str, row: dict[str, str]) -> bool:
    gold = extract_answer(row["answer"])
    if gold == _INVALID:
        message = "the published answer has no final number"
        raise RuntimeError(message)
    return extract_answer(response) == gold


def _shots() -> list[dict[str, str]]:
    rows = _read(_SHOTS)
    if len(rows) != _SHOT_COUNT:
        message = "GSM8K shots are not the five training examples"
        raise RuntimeError(message)
    return rows


def _read(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open() as handle:
        for index, line in enumerate(handle):
            if not line.strip():
                continue
            item = json.loads(line)
            rows.append(
                {
                    "id": str(index),
                    "question": item["question"],
                    "answer": item["answer"],
                }
            )
    return rows


def _until_question(completion: str) -> str:
    if completion.startswith("Question:"):
        return ""
    at = completion.find("\nQuestion:")
    if at >= 0:
        return completion[:at]
    return completion
