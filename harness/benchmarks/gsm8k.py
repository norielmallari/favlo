"""GSM8K test split. The number match is the published `####` extractor."""

from __future__ import annotations

import json
import re
from pathlib import Path

_DATA = Path(__file__).resolve().parent / "data" / "gsm8k" / "test.jsonl"
_ANSWER = re.compile(r"#### (\-?[0-9\.\,]+)")
_INVALID = "[invalid]"


def load() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with _DATA.open() as handle:
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


def messages(row: dict[str, str]) -> list[dict[str, str]]:
    text = (
        f"{row['question']}\n\n"
        "Solve the problem. End with a line that starts with #### and the final number."
    )
    return [{"role": "user", "content": text}]


def extract_answer(completion: str) -> str:
    """The grade-school-math match: the first `####` number, commas removed."""
    match = _ANSWER.search(completion)
    if match is None:
        return _INVALID
    return match.group(1).strip().replace(",", "")


def correct(response: str, row: dict[str, str]) -> bool:
    gold = extract_answer(row["answer"])
    if gold == _INVALID:
        message = "the published answer has no final number"
        raise RuntimeError(message)
    return extract_answer(response) == gold
