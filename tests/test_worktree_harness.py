"""Worktree agent: grader, detached checkout, and the summarization gate."""

import asyncio
import subprocess
from pathlib import Path

from agent_framework import ChatResponse, Message, SummarizationStrategy
from agent_framework._compaction import annotate_message_groups

from harness.worktree import (
    SummaryGate,
    SummarizePastTarget,
    add_worktree,
    grade,
    remove_worktree,
)


def _seed(repo: Path) -> None:
    subprocess.run(["git", "init", "-b", "main"], cwd=repo, check=True, capture_output=True)
    (repo / "README.md").write_text("seed\n", encoding="utf-8")
    subprocess.run(["git", "add", "README.md"], cwd=repo, check=True, capture_output=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.email=dev@example.com",
            "-c",
            "user.name=dev",
            "commit",
            "-m",
            "seed",
        ],
        cwd=repo,
        check=True,
        capture_output=True,
    )


def _notes(root: Path, *, label: str = 'f"cover={total}"') -> None:
    notes = root / "notes"
    notes.mkdir(parents=True, exist_ok=True)
    (notes / "__init__.py").write_text("", encoding="utf-8")
    (notes / "span.py").write_text(
        "def width(start, end):\n"
        "    return end - start\n"
        "def cover(spans):\n"
        "    return sum(width(start, end) for start, end in spans)\n"
        "def label(total):\n"
        f"    return {label}\n",
        encoding="utf-8",
    )
    (notes / "check.py").write_text(
        "from notes.span import cover, label\n"
        "def report():\n"
        "    return label(cover([(1, 4), (10, 12)]))\n",
        encoding="utf-8",
    )
    (notes / "README.md").write_text(
        "span.py defines width, cover, and label.\ncheck.py defines report.\n",
        encoding="utf-8",
    )


def test_grade_accepts_the_task_and_reports_a_wrong_label(tmp_path: Path) -> None:
    _notes(tmp_path)
    assert grade(tmp_path) is None
    _notes(tmp_path, label='"total"')
    assert grade(tmp_path) == "label(5) returned 'total'; report() returned 'total'"


def test_a_fenced_module_is_stored_without_the_fence() -> None:
    from harness.worktree import _unwrap

    text = _unwrap("```python\ndef width(start, end):\n    return end - start\n```")
    assert text.startswith("def width")
    assert "```" not in text
    assert text.endswith("\n")


def test_add_worktree_checks_out_main(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _seed(repo)
    dest = tmp_path / "wt"
    add_worktree(repo, dest, "main")
    assert (dest / "README.md").read_text(encoding="utf-8") == "seed\n"
    remove_worktree(repo, dest)
    assert not dest.exists()


def test_summarize_past_target_waits_for_the_peak() -> None:
    asyncio.run(_summarize())


async def _summarize() -> None:
    class Fake:
        def __init__(self) -> None:
            self.calls = 0

        async def get_response(self, messages, **kwargs):
            del messages, kwargs
            self.calls += 1
            return ChatResponse(messages=[Message("assistant", ["kept the span module"])])

    fake = Fake()
    messages = [
        Message("user", ["write span"]),
        Message("assistant", ["wrote width, cover, and label"]),
        Message("user", ["write check"]),
        Message("assistant", ["wrote report"]),
        Message("user", ["write readme"]),
        Message("assistant", ["named the three functions"]),
        Message("user", ["read a file"]),
        Message("assistant", ["the file routes experts"]),
    ]
    gate = SummaryGate(131072)
    strategy = SummarizePastTarget(
        SummarizationStrategy(client=fake, target_count=4, threshold=0, max_summary_input_tokens=None),
        gate,
    )
    annotate_message_groups(messages)
    assert await strategy(messages) is False
    assert fake.calls == 0
    gate.peak = 131072
    assert await strategy(messages) is True
    assert fake.calls == 1
    assert gate.fired
    assert gate.text == "kept the span module"
    assert any(message.text == "kept the span module" for message in messages)
