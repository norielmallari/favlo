"""GSM8K test split. The number match is the published `####` extractor."""

from __future__ import annotations

import json
import re
from pathlib import Path

_DATA = Path(__file__).resolve().parent / "data" / "gsm8k" / "test.jsonl"
_NUMBER = re.compile(r"-?[0-9][0-9,]*(?:\.[0-9]+)?")
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
    """The final number, commas removed.

    The first number on the ``####`` line wins. Words and a currency sign may
    sit between the marker and that number. A reply with no such number uses
    the last number in the text.
    """
    for line in completion.splitlines():
        if "####" not in line:
            continue
        found = _NUMBER.findall(line.split("####", 1)[1])
        if found:
            return found[0].replace(",", "")
        break
    found = _NUMBER.findall(completion)
    if not found:
        return _INVALID
    return found[-1].replace(",", "")


def correct(response: str, row: dict[str, str]) -> bool:
    gold = extract_answer(row["answer"])
    if gold == _INVALID:
        message = "the published answer has no final number"
        raise RuntimeError(message)
    return extract_answer(response) == gold
