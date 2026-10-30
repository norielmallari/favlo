"""HumanEval pass@1. One argmax continuation, then the published tests."""

from __future__ import annotations

import gzip
import json
import os
import re
import sys
from pathlib import Path

_DATA = Path(__file__).resolve().parent / "data" / "humaneval" / "HumanEval.jsonl.gz"
_VENDOR = Path(__file__).resolve().parent / "vendor"
_FENCE = re.compile(r"```(?:python)?[^\n]*\n(.*?)(?:```|$)", re.DOTALL)
_STOPS = ("\nclass", "\ndef", "\n#", "\nif", "\nprint")
_INSTRUCTION = (
    "Write a solution to the following problem and make sure that it passes the tests:\n"
    "```python\n{prompt}\n```\n"
)


def load() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with gzip.open(_DATA, "rt") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            rows.append(
                {
                    "id": item["task_id"],
                    "prompt": item["prompt"],
                    "entry_point": item["entry_point"],
                    "canonical_solution": item["canonical_solution"],
                    "test": item["test"],
                }
            )
    return rows


def messages(row: dict[str, str]) -> list[dict[str, str]]:
    return [{"role": "user", "content": _INSTRUCTION.format(prompt=row["prompt"])}]


def function_body(completion: str, prompt: str, entry_point: str) -> str:
    """The text that follows the published signature.

    A python fence is unwrapped first. A copy of the published prompt is
    dropped, and so is a rewritten ``def`` line, so the tests keep the
    dataset signature. Generation ends at a new top-level statement, the
    same stops as the published completion task.
    """
    fenced = _FENCE.search(completion)
    text = fenced.group(1) if fenced is not None else completion
    published = prompt.strip()
    if published and published in text:
        text = text[text.find(published) + len(published) :]
    else:
        needle = f"def {entry_point}"
        at = text.find(needle)
        if at >= 0:
            rest = text[at:]
            newline = rest.find("\n")
            text = "" if newline < 0 else rest[newline + 1 :]
    cut = len(text)
    for stop in _STOPS:
        at = text.find(stop)
        if 0 <= at < cut:
            cut = at
    text = text[:cut]
    if text and not text.endswith("\n"):
        text += "\n"
    return text


def passed(row: dict[str, str], completion: str, *, timeout: float = 3.0) -> bool:
    os.environ["PYTHONPATH"] = str(_VENDOR) + os.pathsep + os.environ.get("PYTHONPATH", "")
    vendor = str(_VENDOR)
    if vendor not in sys.path:
        sys.path.insert(0, vendor)
    from human_eval.execution import check_correctness

    problem = {
        "task_id": row["id"],
        "prompt": row["prompt"],
        "entry_point": row["entry_point"],
        "test": row["test"],
    }
    body = function_body(completion, row["prompt"], row["entry_point"])
    result = check_correctness(problem, body, timeout)
    return bool(result["passed"])
