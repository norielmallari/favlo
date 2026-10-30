"""The coding harness grades a workspace. These tests do not start serve."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from harness.benchmarks.serve import Request, apply_sampling, card_model_id
from harness.coding.grade import grade
from harness.coding.loop import chat_body
from harness.coding.session import REQUIRED_TOOLS
from harness.coding.workspace import PROMPT, dispatch, locate

_MODULE = "def collect(before, after):\n    return after - before\n"


def test_thinking_card_id() -> None:
    assert card_model_id("qwen38-27b", thinking=True) == "qwen38-27b:thinking"


def test_thinking_body_omits_sampling_fields() -> None:
    request = apply_sampling(
        Request(enable_thinking=True),
        {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "presence_penalty": 0.0},
    )
    body = chat_body(
        "qwen38-27b:thinking",
        [{"role": "user", "content": "write it"}],
        request,
        mtp=True,
    )
    assert body["model"] == "qwen38-27b"
    assert body["enable_thinking"] is True
    assert body["mtp"] is True
    assert body["mtp_depth"] == 3
    assert "temperature" not in body
    assert "top_p" not in body
    assert "top_k" not in body
    assert "presence_penalty" not in body
    assert "max_tokens" not in body
    assert body["tools"]
    names = [item["function"]["name"] for item in body["tools"]]
    assert names == list(REQUIRED_TOOLS)
    off = chat_body(
        "qwen38-27b:thinking",
        [{"role": "user", "content": "write it"}],
        request,
        mtp=False,
    )
    assert off["mtp"] is False
    assert off["enable_thinking"] is True
    four = chat_body(
        "qwen38-27b:thinking",
        [{"role": "user", "content": "write it"}],
        request,
        mtp=True,
        mtp_depth=4,
    )
    assert four["mtp"] is True
    assert four["mtp_depth"] == 4
    assert four["enable_thinking"] is True


def test_path_stays_inside_the_workspace(tmp_path: Path) -> None:
    locate(tmp_path, "ledger.py")
    with pytest.raises(ValueError):
        locate(tmp_path, "../ledger.py")
    outside = tmp_path.parent / "outside.txt"
    outside.write_text("nope", encoding="utf-8")
    (tmp_path / "escape").symlink_to(outside)
    with pytest.raises(ValueError):
        locate(tmp_path, "escape")


def test_prompt_names_every_tool() -> None:
    for name in REQUIRED_TOOLS:
        assert name in PROMPT


def test_hidden_suite_passes_a_correct_module(tmp_path: Path) -> None:
    body = json.loads(
        dispatch(
            tmp_path,
            "write_file",
            {
                "file_path": "ledger.py",
                "content": _MODULE,
                "idempotency_key": "create-ledger",
            },
        )
    )
    assert body["file_path"] == "ledger.py"
    assert body["content_hash"]
    result = grade(tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr


def test_hidden_suite_fails_an_empty_workspace(tmp_path: Path) -> None:
    result = grade(tmp_path)
    assert result.returncode != 0
