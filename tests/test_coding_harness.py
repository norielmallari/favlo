"""The coding harness grades a workspace. These tests do not start serve."""

from __future__ import annotations

from pathlib import Path

import pytest
from harness.benchmarks.serve import Request, apply_sampling, card_model_id
from harness.coding.grade import grade
from harness.coding.loop import chat_body
from harness.coding.workspace import dispatch, locate

_MODULE = "def collect(before, after):\n    return after - before\n"


def test_thinking_card_id() -> None:
    assert card_model_id("qwen38-27b", thinking=True) == "qwen38-27b:thinking"


def test_thinking_body_uses_the_card() -> None:
    request = apply_sampling(
        Request(enable_thinking=True),
        {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "presence_penalty": 0.0},
    )
    body = chat_body("qwen38-27b", [{"role": "user", "content": "write it"}], request)
    assert body["enable_thinking"] is True
    assert body["temperature"] == 1.0
    assert body["top_p"] == 0.95
    assert body["top_k"] == 20
    assert body["presence_penalty"] == 0.0
    assert "max_tokens" not in body
    assert body["tools"]


def test_path_stays_inside_the_workspace(tmp_path: Path) -> None:
    locate(tmp_path, "ledger.py")
    with pytest.raises(ValueError):
        locate(tmp_path, "../ledger.py")


def test_hidden_suite_passes_a_correct_module(tmp_path: Path) -> None:
    assert dispatch(tmp_path, "write_file", {"path": "ledger.py", "content": _MODULE}) == (
        "wrote ledger.py"
    )
    result = grade(tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr


def test_hidden_suite_fails_an_empty_workspace(tmp_path: Path) -> None:
    result = grade(tmp_path)
    assert result.returncode != 0
