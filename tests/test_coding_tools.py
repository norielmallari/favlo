"""The six coding tools. These tests do not start serve."""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from pathlib import Path

import pytest
from harness.coding.grade import grade
from harness.coding.session import missing_tools, session_for
from harness.coding.tools.limits import GREP_RESPONSE_BUDGET
from harness.coding.workspace import (
    LEDGER_DRAFT,
    LEDGER_RETURN_NEW,
    LEDGER_RETURN_OLD,
    SHELL_CHECK,
    dispatch,
    prepare_workspace,
)

_TERMINAL = {"cancelled", "failed", "interrupted", "succeeded", "timed_out"}


@pytest.fixture
def workspace(tmp_path: Path):
    yield tmp_path
    session_for(tmp_path).close()


def _call(workspace: Path, name: str, arguments: dict[str, object]) -> dict[str, object]:
    return json.loads(dispatch(workspace, name, arguments))


def _write(workspace: Path, path: str, content: str, key: str) -> dict[str, object]:
    return _call(
        workspace,
        "write_file",
        {"file_path": path, "content": content, "idempotency_key": key},
    )


def _wait(workspace: Path, job_id: str) -> dict[str, object]:
    last: dict[str, object] = {}
    deadline = time.monotonic() + 8
    while time.monotonic() < deadline:
        last = _call(workspace, "run_shell_command", {"action": "status", "job_id": job_id})
        if last.get("status") in _TERMINAL:
            return last
        time.sleep(0.05)
    return last


def test_read_file_numbers_lines_and_hashes_the_whole_file(workspace: Path) -> None:
    text = "".join(f"line {index}\n" for index in range(6))
    _write(workspace, "notes.txt", text, "notes")
    page = _call(workspace, "read_file", {"file_path": "notes.txt", "offset": 2, "limit": 2})
    assert page["offset"] == 2
    assert page["lines"] == [
        {"line": 3, "text": "line 2"},
        {"line": 4, "text": "line 3"},
    ]
    assert page["next_offset"] == 4
    assert page["content_hash"] == hashlib.sha256(text.encode("utf-8")).hexdigest()
    end = _call(workspace, "read_file", {"file_path": "notes.txt", "offset": 6})
    assert end["lines"] == []
    assert "next_offset" not in end
    past = _call(workspace, "read_file", {"file_path": "notes.txt", "offset": 7})
    assert past["error"] == "invalid_range"


def test_read_file_rejects_binary_missing_and_unreadable(workspace: Path) -> None:
    missing = _call(workspace, "read_file", {"file_path": "missing.txt"})
    assert missing["error"] == "not_found"
    binary = workspace / "binary.txt"
    binary.write_bytes(b"a\0b")
    assert _call(workspace, "read_file", {"file_path": "binary.txt"})["error"] == "binary_content"
    secret = workspace / "secret.txt"
    secret.write_text("hidden", encoding="utf-8")
    secret.chmod(0)
    try:
        denied = _call(workspace, "read_file", {"file_path": "secret.txt"})
        assert denied["error"] == "permission_denied"
    finally:
        secret.chmod(0o644)


def test_read_file_rejects_a_symlink_that_leaves_the_workspace(workspace: Path) -> None:
    outside = workspace.parent / "outside.txt"
    outside.write_text("nope", encoding="utf-8")
    (workspace / "escape").symlink_to(outside)
    body = _call(workspace, "read_file", {"file_path": "escape"})
    assert body["error"] == "permission_denied"
    inside = workspace / "inside.txt"
    inside.write_text("yes\n", encoding="utf-8")
    absolute = _call(workspace, "read_file", {"file_path": str(inside)})
    assert absolute["file_path"] == "inside.txt"
    assert absolute["lines"] == [{"line": 1, "text": "yes"}]


def test_glob_orders_pages_and_stays_under_the_root(workspace: Path) -> None:
    (workspace / "b.txt").write_text("b", encoding="utf-8")
    (workspace / "a.txt").write_text("a", encoding="utf-8")
    notes = workspace / "notes"
    notes.mkdir()
    (notes / "brief.txt").write_text("collect", encoding="utf-8")
    (workspace / ".hidden.txt").write_text("h", encoding="utf-8")
    (workspace / ".gitignore").write_text("secret.txt\n", encoding="utf-8")
    (workspace / "secret.txt").write_text("no", encoding="utf-8")
    git = workspace / ".git"
    git.mkdir()
    (git / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
    outside = workspace.parent / "elsewhere.txt"
    outside.write_text("out", encoding="utf-8")
    (workspace / "escape").symlink_to(outside)
    page = _call(workspace, "glob", {"pattern": "**/*.txt", "limit": 1})
    assert page["paths"] == [".hidden.txt"]
    assert page["truncated"] is True
    assert page["next_cursor"] == "1"
    rest = _call(workspace, "glob", {"pattern": "**/*.txt", "cursor": page["next_cursor"]})
    assert ".git/HEAD" not in rest["paths"]
    assert "secret.txt" not in rest["paths"]
    assert "escape" not in rest["paths"]
    assert "notes/brief.txt" in rest["paths"]
    names = _call(workspace, "glob", {"pattern": "*.txt"})
    assert "notes/brief.txt" in names["paths"]
    assert "secret.txt" not in names["paths"]
    rooted = _call(workspace, "glob", {"pattern": "**/*.txt", "path": "notes"})
    assert rooted["paths"] == ["notes/brief.txt"]
    assert rooted["truncated"] is False


def test_grep_accepts_boolean_and_integer_text(workspace: Path) -> None:
    (workspace / "notes.txt").write_text("collect here\n", encoding="utf-8")
    page = _call(
        workspace,
        "grep_search",
        {
            "pattern": "collect",
            "path": "notes.txt",
            "literal": "true",
            "case_sensitive": "false",
            "limit": "1",
            "context_lines": "0",
        },
    )
    assert page["match_count"] == 1
    assert page["matches"][0]["text"] == "collect here"
    rejected = _call(workspace, "grep_search", {"pattern": "collect", "literal": "yes"})
    assert rejected["error"] == "invalid_arguments"


def test_grep_reports_no_match_context_and_a_bad_pattern(workspace: Path) -> None:
    (workspace / "notes.txt").write_text("alpha\ncollect here\nomega\n", encoding="utf-8")
    empty = _call(workspace, "grep_search", {"pattern": "absent", "literal": True})
    assert empty["matches"] == []
    assert empty["match_count"] == 0
    assert "error" not in empty
    literal = _call(
        workspace,
        "grep_search",
        {"pattern": "collect.", "path": "notes.txt", "literal": True},
    )
    assert literal["matches"] == []
    regex = _call(
        workspace,
        "grep_search",
        {"pattern": "collect.", "path": "notes.txt", "context_lines": 1},
    )
    match = regex["matches"][0]
    assert match["text"] == "collect here"
    assert match["context_before"] == [{"line_number": 1, "text": "alpha"}]
    assert match["context_after"] == [{"line_number": 3, "text": "omega"}]
    folded = _call(
        workspace,
        "grep_search",
        {"pattern": "COLLECT", "path": "notes.txt", "case_sensitive": False, "literal": True},
    )
    assert folded["match_count"] == 1
    invalid = _call(workspace, "grep_search", {"pattern": "("})
    assert invalid["error"] == "invalid_arguments"


def test_grep_stores_an_artifact_when_the_result_exceeds_the_budget(workspace: Path) -> None:
    lines = [f"hit {index}" for index in range(800)]
    (workspace / "big.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    page = _call(workspace, "grep_search", {"pattern": "hit", "literal": True, "limit": 5})
    assert page["match_count"] == 800
    assert len(page["matches"]) == 5
    assert page["truncated"] is True
    artifact = page["artifact"]
    assert isinstance(artifact, str)
    stored = json.loads((workspace / artifact).read_text(encoding="utf-8"))
    assert len(stored["matches"]) == 800
    assert len(json.dumps(stored).encode("utf-8")) > GREP_RESPONSE_BUDGET
    follow = _call(
        workspace,
        "grep_search",
        {"pattern": "hit", "literal": True, "limit": 5, "cursor": page["next_cursor"]},
    )
    assert follow["matches"][0]["line_number"] == 6


def test_write_file_is_exclusive_and_idempotent(workspace: Path) -> None:
    created = _write(workspace, "ledger.py", "one\n", "create")
    again = _write(workspace, "ledger.py", "one\n", "create")
    assert again == created
    assert (workspace / "ledger.py").read_text(encoding="utf-8") == "one\n"
    conflict = _write(workspace, "ledger.py", "two\n", "create")
    assert conflict["error"] == "conflict"
    assert (workspace / "ledger.py").read_text(encoding="utf-8") == "one\n"
    exists = _write(workspace, "ledger.py", "one\n", "other")
    assert exists["error"] == "conflict"
    missing_parent = _call(
        workspace,
        "write_file",
        {"file_path": "a/b.txt", "content": "x", "idempotency_key": "nested"},
    )
    assert missing_parent["error"] == "not_found"
    nested = _call(
        workspace,
        "write_file",
        {
            "file_path": "a/b.txt",
            "content": "x",
            "create_directories": True,
            "idempotency_key": "nested-ok",
        },
    )
    assert nested["file_path"] == "a/b.txt"
    (workspace / "not-dir").write_text("x", encoding="utf-8")
    blocked = _call(
        workspace,
        "write_file",
        {
            "file_path": "not-dir/file.txt",
            "content": "x",
            "create_directories": True,
            "idempotency_key": "blocked",
        },
    )
    assert blocked["error"] == "invalid_arguments"
    reserved = _call(
        workspace,
        "write_file",
        {"file_path": ".favlo/note.txt", "content": "x", "idempotency_key": "reserved"},
    )
    assert reserved["error"] == "permission_denied"


def test_edit_reports_match_errors_and_a_bounded_diff(workspace: Path) -> None:
    _write(workspace, "jobs.py", "keep = 1\nvalue = 1\nvalue = 1\n", "jobs")
    missing = _call(
        workspace,
        "edit",
        {
            "file_path": "jobs.py",
            "old_string": "absent",
            "new_string": "present",
            "idempotency_key": "missing",
        },
    )
    assert missing["error"] == "match_not_found"
    ambiguous = _call(
        workspace,
        "edit",
        {
            "file_path": "jobs.py",
            "old_string": "value = 1",
            "new_string": "value = 2",
            "idempotency_key": "ambiguous",
        },
    )
    assert ambiguous["error"] == "ambiguous_match"
    assert (workspace / "jobs.py").read_text(encoding="utf-8").count("value = 1") == 2
    digest = hashlib.sha256(b"nope").hexdigest()
    mismatch = _call(
        workspace,
        "edit",
        {
            "file_path": "jobs.py",
            "old_string": "keep = 1",
            "new_string": "keep = 2",
            "expected_hash": digest,
            "idempotency_key": "hash",
        },
    )
    assert mismatch["error"] == "conflict"
    replaced = _call(
        workspace,
        "edit",
        {
            "file_path": "jobs.py",
            "old_string": "value = 1",
            "new_string": "value = 2",
            "replace_all": True,
            "idempotency_key": "all",
        },
    )
    assert replaced["replacements"] == 2
    assert replaced["before_hash"] != replaced["after_hash"]
    assert "value = 2" in str(replaced["diff"])
    replay = _call(
        workspace,
        "edit",
        {
            "file_path": "jobs.py",
            "old_string": "value = 1",
            "new_string": "value = 2",
            "replace_all": True,
            "idempotency_key": "all",
        },
    )
    assert replay == replaced
    assert (workspace / "jobs.py").read_text(encoding="utf-8").count("value = 2") == 2


def test_shell_start_returns_before_the_command_finishes(workspace: Path) -> None:
    started_at = time.monotonic()
    started = _call(
        workspace,
        "run_shell_command",
        {"command": "sleep 5", "idempotency_key": "slow"},
    )
    assert time.monotonic() - started_at < 0.5
    assert started["status"] == "queued"
    status_at = time.monotonic()
    status = _call(
        workspace,
        "run_shell_command",
        {"action": "status", "job_id": started["job_id"]},
    )
    assert time.monotonic() - status_at < 0.5
    assert status["status"] in {"queued", "running"}
    assert status["has_more"] is True
    extra = _call(
        workspace,
        "run_shell_command",
        {"action": "status", "job_id": started["job_id"], "command": "printf hi"},
    )
    assert extra["error"] == "invalid_arguments"


def test_shell_runs_input_cancel_and_timeout(workspace: Path) -> None:
    started = _call(
        workspace,
        "run_shell_command",
        {"command": "read line; printf '%s' \"$line\"", "idempotency_key": "read"},
    )
    sent = _call(
        workspace,
        "run_shell_command",
        {
            "action": "input",
            "job_id": started["job_id"],
            "input": "hello\n",
            "close_stdin": True,
            "idempotency_key": "hello",
        },
    )
    replay = _call(
        workspace,
        "run_shell_command",
        {
            "action": "input",
            "job_id": started["job_id"],
            "input": "hello\n",
            "close_stdin": True,
            "idempotency_key": "hello",
        },
    )
    assert replay == sent
    finished = _wait(workspace, str(started["job_id"]))
    assert finished["status"] == "succeeded"
    assert finished["exit_code"] == 0
    assert finished["output"] == "hello"
    assert finished["has_more"] is False
    conflict = _call(
        workspace,
        "run_shell_command",
        {
            "action": "input",
            "job_id": started["job_id"],
            "input": "other\n",
            "idempotency_key": "hello",
        },
    )
    assert conflict["error"] == "conflict"

    sleeping = _call(
        workspace,
        "run_shell_command",
        {"command": "sleep 30", "idempotency_key": "cancel-me", "timeout_seconds": 30},
    )
    requested = _call(
        workspace,
        "run_shell_command",
        {"action": "cancel", "job_id": sleeping["job_id"], "grace_period_seconds": 0.2},
    )
    assert requested["status"] == "cancel_requested"
    cancelled = _wait(workspace, str(sleeping["job_id"]))
    assert cancelled["status"] == "cancelled"

    limited = _call(
        workspace,
        "run_shell_command",
        {"command": "sleep 30", "idempotency_key": "limit", "timeout_seconds": 0.3},
    )
    timed_out = _wait(workspace, str(limited["job_id"]))
    assert timed_out["status"] == "timed_out"


def test_shell_kills_descendants_and_does_not_replay(workspace: Path) -> None:
    started = _call(
        workspace,
        "run_shell_command",
        {
            "command": "sleep 30 & echo $! > child.pid; wait",
            "idempotency_key": "group",
        },
    )
    deadline = time.monotonic() + 3
    pid_path = workspace / "child.pid"
    while not pid_path.is_file() and time.monotonic() < deadline:
        time.sleep(0.05)
    child = int(pid_path.read_text(encoding="utf-8").strip())
    _call(
        workspace,
        "run_shell_command",
        {"action": "cancel", "job_id": started["job_id"], "grace_period_seconds": 0.2},
    )
    cancelled = _wait(workspace, str(started["job_id"]))
    assert cancelled["status"] == "cancelled"
    with pytest.raises(ProcessLookupError):
        os.kill(child, 0)

    same = _call(
        workspace,
        "run_shell_command",
        {"command": "printf x >> marks.txt", "idempotency_key": "once"},
    )
    replay = _call(
        workspace,
        "run_shell_command",
        {"command": "printf x >> marks.txt", "idempotency_key": "once"},
    )
    assert replay["job_id"] == same["job_id"]
    changed = _call(
        workspace,
        "run_shell_command",
        {"command": "printf y >> marks.txt", "idempotency_key": "once"},
    )
    assert changed["error"] == "conflict"
    finished = _wait(workspace, str(same["job_id"]))
    assert finished["status"] == "succeeded"
    assert (workspace / "marks.txt").read_text(encoding="utf-8") == "x"


def test_shell_blocks_file_edits_until_it_leaves_the_workspace(workspace: Path) -> None:
    _write(workspace, "hold.txt", "value = 1\n", "hold")
    started = _call(
        workspace,
        "run_shell_command",
        {"command": "sleep 30", "idempotency_key": "hold-shell"},
    )
    done = {"body": None}

    def edit() -> None:
        done["body"] = _call(
            workspace,
            "edit",
            {
                "file_path": "hold.txt",
                "old_string": "value = 1",
                "new_string": "value = 2",
                "idempotency_key": "hold-edit",
            },
        )

    thread = threading.Thread(target=edit)
    thread.start()
    time.sleep(0.2)
    assert done["body"] is None
    _call(
        workspace,
        "run_shell_command",
        {"action": "cancel", "job_id": started["job_id"], "grace_period_seconds": 0.2},
    )
    thread.join(timeout=5)
    assert done["body"] is not None
    assert done["body"]["replacements"] == 1


def test_a_restart_interrupts_a_recorded_job_without_running_it(workspace: Path) -> None:
    jobs = workspace / ".favlo" / "jobs"
    jobs.mkdir(parents=True)
    record = {
        "command": "printf z > replayed.txt",
        "directory": ".",
        "job_id": "shell_1",
        "status": "running",
    }
    (jobs / "shell_1.json").write_text(json.dumps(record), encoding="utf-8")
    status = _call(workspace, "run_shell_command", {"action": "status", "job_id": "shell_1"})
    assert status["status"] == "interrupted"
    assert "exit_code" not in status
    assert status["message"] == "the process did not survive restart"
    assert not (workspace / "replayed.txt").exists()


def test_journal_records_the_call_identity(workspace: Path) -> None:
    (workspace / "note.txt").write_text("hello\n", encoding="utf-8")
    _call(workspace, "read_file", {"file_path": "note.txt"})
    dispatch(workspace, "read_file", {"file_path": "note.txt"}, tool_call_id="call_7")
    journal = workspace / ".favlo" / "journal.jsonl"
    rows = [json.loads(line) for line in journal.read_text(encoding="utf-8").splitlines()]
    assert rows[-1]["tool"] == "read_file"
    assert rows[-1]["tool_call_id"] == "call_7"
    assert rows[-1]["error"] is None
    session = session_for(workspace)
    assert rows[-1]["session_id"] == session.session_id
    assert rows[-1]["run_id"] == session.run_id
    private = _call(workspace, "read_file", {"file_path": ".favlo/journal.jsonl"})
    assert private["error"] == "permission_denied"


def test_prompt_sequence_uses_every_tool_and_passes(workspace: Path) -> None:
    prepare_workspace(workspace)
    found = _call(workspace, "glob", {"pattern": "**/*.txt", "path": "notes"})
    assert found["paths"] == ["notes/brief.txt"]
    matches = _call(
        workspace,
        "grep_search",
        {"pattern": "collect", "path": "notes/brief.txt", "literal": True},
    )
    assert matches["match_count"] == 1
    page = _call(workspace, "read_file", {"file_path": "notes/brief.txt"})
    assert "after minus before" in page["lines"][0]["text"]
    _write(workspace, "ledger.py", LEDGER_DRAFT, "create-ledger")
    edited = _call(
        workspace,
        "edit",
        {
            "file_path": "ledger.py",
            "old_string": LEDGER_RETURN_OLD,
            "new_string": LEDGER_RETURN_NEW,
            "idempotency_key": "fix-ledger",
        },
    )
    assert edited["replacements"] == 1
    started = _call(
        workspace,
        "run_shell_command",
        {"command": SHELL_CHECK, "directory": ".", "idempotency_key": "check-ledger"},
    )
    finished = _wait(workspace, str(started["job_id"]))
    assert finished["status"] == "succeeded", finished
    assert "6" in str(finished["output"])
    assert missing_tools(session_for(workspace).calls) == []
    result = grade(workspace)
    assert result.returncode == 0, result.stdout + result.stderr
