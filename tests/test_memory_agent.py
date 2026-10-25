"""Harness checks for MCP memory tools. The listener is not started."""

from __future__ import annotations

import pytest

from harness.memory_agent import check_recall, recall_rows, tool_value


def test_tool_value_reads_structured_content() -> None:
    assert tool_value({"structuredContent": "mem_abcd"}) == "mem_abcd"


def test_tool_value_reads_json_text() -> None:
    body = {
        "content": [
            {
                "type": "text",
                "text": '[{"id": "mem_1", "text": "fact", "score": 0.9}]',
            }
        ]
    }
    assert recall_rows(body) == [{"id": "mem_1", "text": "fact", "score": 0.9}]


def test_check_recall_requires_score() -> None:
    rows = [{"id": "mem_a", "text": "a", "score": 0.8}]
    assert check_recall(rows, want="mem_a", absent="mem_b") == 0.8
    both = [
        {"id": "mem_a", "text": "a", "score": 0.9},
        {"id": "mem_b", "text": "b", "score": 0.1},
    ]
    assert check_recall(both, want="mem_a", absent="mem_b") == 0.9
    with pytest.raises(RuntimeError, match="outrank"):
        check_recall(
            [{"id": "mem_a", "score": 0.2}, {"id": "mem_b", "score": 0.8}],
            want="mem_a",
            absent="mem_b",
        )
    with pytest.raises(RuntimeError, match="mem_x"):
        check_recall(rows, want="mem_x", absent="mem_b")
