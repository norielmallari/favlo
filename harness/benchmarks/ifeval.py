"""IFEval strict and loose scores from the published checker."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parent
_DATA = _ROOT / "data" / "ifeval" / "input_data.jsonl"
_VENDOR = _ROOT / "vendor"
_NLTK = _VENDOR / "nltk_data"


def prepare() -> None:
    os.environ["NLTK_DATA"] = str(_NLTK)
    vendor = str(_VENDOR)
    if vendor not in sys.path:
        sys.path.insert(0, vendor)


def load() -> list[Any]:
    prepare()
    from instruction_following_eval.evaluation_lib import read_prompt_list

    return read_prompt_list(str(_DATA))


def messages(example: Any) -> list[dict[str, str]]:
    return [{"role": "user", "content": example.prompt}]


def score(example: Any, response: str) -> dict[str, Any]:
    prepare()
    import langdetect
    from instruction_following_eval.evaluation_lib import (
        test_instruction_following_loose,
        test_instruction_following_strict,
    )

    langdetect.DetectorFactory.seed = 0
    mapping = {example.prompt: response}
    strict = test_instruction_following_strict(example, mapping)
    loose = test_instruction_following_loose(example, mapping)
    return {
        "id": str(example.key),
        "strict_prompt": bool(strict.follow_all_instructions),
        "strict_instructions": [bool(item) for item in strict.follow_instruction_list],
        "loose_prompt": bool(loose.follow_all_instructions),
        "loose_instructions": [bool(item) for item in loose.follow_instruction_list],
    }


def aggregate(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    count = len(rows)
    if count == 0:
        message = "IFEval has no scored rows"
        raise RuntimeError(message)
    strict_inst = [item for row in rows for item in row["strict_instructions"]]
    loose_inst = [item for row in rows for item in row["loose_instructions"]]
    return {
        "n": count,
        "prompt_strict": sum(bool(row["strict_prompt"]) for row in rows) / count,
        "prompt_loose": sum(bool(row["loose_prompt"]) for row in rows) / count,
        "instruction_strict": sum(strict_inst) / len(strict_inst),
        "instruction_loose": sum(loose_inst) / len(loose_inst),
    }
