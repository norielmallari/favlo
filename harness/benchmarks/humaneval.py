"""HumanEval pass@1. One greedy sample, then the published test subprocess."""

from __future__ import annotations

import gzip
import json
import os
import re
import sys
from pathlib import Path

_DATA = Path(__file__).resolve().parent / "data" / "humaneval" / "HumanEval.jsonl.gz"
_VENDOR = Path(__file__).resolve().parent / "vendor"
_FENCE = re.compile(r"```python\n(.*?)```", re.DOTALL)
_INSTRUCTION = (
    "Read the following function signature and docstring, and fully implement "
    "the function described. Your response should only contain the code for "
    "this function.\n"
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
    return [{"role": "user", "content": _INSTRUCTION + row["prompt"]}]


def function_body(completion: str, entry_point: str) -> str:
    """Keep the function body. A python fence is unwrapped first.

    A reply that repeats the signature is cut after that signature. A reply
    that is already the body is left indented, because the published prompt
    already includes the signature and the docstring.
    """
    fenced = _FENCE.findall(completion)
    text = fenced[0] if fenced else completion
    needle = f"def {entry_point}"
    if needle in text:
        signature = text.find(":\n", text.find(needle))
        if signature >= 0:
            text = text[signature + 2 :]
    for stop in ("\ndef ", "\nclass ", "\nif __name__"):
        at = text.find(stop)
        if at >= 0:
            text = text[:at]
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
    body = function_body(completion, row["entry_point"])
    result = check_correctness(problem, body, timeout)
    return bool(result["passed"])
