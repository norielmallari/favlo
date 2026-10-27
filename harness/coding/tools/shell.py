"""Asynchronous shell jobs. A start call does not wait for the command."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

from harness.coding.tools.args import (
    cursor_offset,
    optional_bool,
    optional_int,
    optional_number,
    optional_str,
    reject_unknown,
    require_str,
    require_text,
)
from harness.coding.tools.context import ToolContext
from harness.coding.tools.errors import ToolError, dump
from harness.coding.tools.idempotency import check_key
from harness.coding.tools.limits import (
    SHELL_CANCEL_GRACE,
    SHELL_DEFAULT_MAX_BYTES,
    SHELL_INPUT_WAIT,
    SHELL_MAX_BYTES,
    SHELL_OUTPUT_CAP,
)
from harness.coding.tools.mutations import MutationCoordinator
from harness.coding.tools.paths import display_path, resolve_inside

_TERMINAL = {"cancelled", "failed", "interrupted", "succeeded", "timed_out"}
_ACTIONS = {
    "start": {
        "action",
        "command",
        "description",
        "directory",
        "idempotency_key",
        "timeout_seconds",
    },
    "status": {"action", "cursor", "job_id", "max_bytes"},
    "input": {"action", "close_stdin", "idempotency_key", "input", "job_id"},
    "cancel": {"action", "grace_period_seconds", "job_id"},
}
_SHELL = "/bin/sh"


class Job:
    def __init__(self, job_id: str, command: str, directory: str, cwd: Path) -> None:
        self.job_id = job_id
        self.command = command
        self.directory = directory
        self.cwd = cwd
        self.description = ""
        self.timeout_seconds: float | None = None
        self.status = "queued"
        self.exit_code: int | None = None
        self.message = ""
        self.cancel_requested = False
        self.grace = SHELL_CANCEL_GRACE
        self.output = bytearray()
        self.output_truncated = False
        self.overflow = False
        self.proc: subprocess.Popen[bytes] | None = None
        self.session_id = ""
        self.run_id = ""
        self.tool_call_id = ""
        self.lock = threading.Lock()

    def record(self) -> dict[str, Any]:
        return {
            "cancel_requested": self.cancel_requested,
            "command": self.command,
            "description": self.description,
            "directory": self.directory,
            "exit_code": self.exit_code,
            "job_id": self.job_id,
            "message": self.message,
            "output_truncated": self.output_truncated,
            "run_id": self.run_id,
            "session_id": self.session_id,
            "status": self.status,
            "timeout_seconds": self.timeout_seconds,
            "tool_call_id": self.tool_call_id,
        }

    def append(self, chunk: bytes, path: Path) -> bool:
        with self.lock:
            self.output.extend(chunk)
            with path.open("ab") as handle:
                handle.write(chunk)
            if len(self.output) > SHELL_OUTPUT_CAP:
                self.overflow = True
                return True
        return False


class ShellJobs:
    def __init__(self, workspace: Path, state_dir: Path, mutations: MutationCoordinator) -> None:
        self.workspace = workspace.resolve()
        self.mutations = mutations
        self.jobs_dir = state_dir / "jobs"
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self._jobs: dict[str, Job] = {}
        self._threads: list[threading.Thread] = []
        self._lock = threading.Lock()
        self._load()

    def perform(self, arguments: dict[str, Any], ctx: ToolContext) -> str:
        action = _action(arguments)
        reject_unknown(arguments, _ACTIONS[action])
        if action == "start":
            return self._start(arguments, ctx)
        if action == "status":
            return self._status(arguments)
        if action == "input":
            return self._input(arguments, ctx)
        return self._cancel(arguments)

    def shutdown(self) -> None:
        for job in list(self._jobs.values()):
            with job.lock:
                if job.status in {"queued", "running"}:
                    job.cancel_requested = True
                    job.grace = 0.2
        for thread in list(self._threads):
            thread.join(timeout=3)

    def _start(self, arguments: dict[str, Any], ctx: ToolContext) -> str:
        command = require_str(arguments, "command")
        key = check_key(require_text(arguments, "idempotency_key", allow_empty=False))
        description = optional_str(arguments, "description") or ""
        timeout = optional_number(arguments, "timeout_seconds")
        if timeout is not None and timeout <= 0:
            raise ToolError("invalid_arguments", "timeout_seconds must be greater than zero")
        directory = optional_str(arguments, "directory") or "."
        cwd = _directory(ctx.workspace, directory)
        relative = display_path(ctx.workspace, cwd)
        payload = {
            "command": command,
            "description": description,
            "directory": relative,
            "timeout_seconds": timeout,
        }
        reserved = ctx.idempotency.reserve(key, payload)
        if reserved.kind == "replay" and reserved.outcome is not None:
            return reserved.outcome
        if reserved.kind == "pending":
            if reserved.job_id is None:
                raise ToolError("conflict", "attempt did not finish")
            outcome = dump({"job_id": reserved.job_id, "next_cursor": "0", "status": "queued"})
            ctx.idempotency.finish(key, outcome)
            return outcome
        self.mutations.acquire_shell()
        job: Job | None = None
        handed = False
        try:
            job = self._create(command, relative, cwd)
            job.description = description
            job.timeout_seconds = timeout
            job.session_id = ctx.session_id
            job.run_id = ctx.run_id
            job.tool_call_id = ctx.tool_call_id
            self._save(job)
            ctx.idempotency.note_job(key, job.job_id)
            thread = threading.Thread(target=self._worker, args=(job,), daemon=True)
            thread.start()
            handed = True
            with self._lock:
                self._threads.append(thread)
            outcome = dump({"job_id": job.job_id, "next_cursor": "0", "status": "queued"})
            ctx.idempotency.finish(key, outcome)
            return outcome
        except Exception:
            if job is not None and not handed:
                job.status = "failed"
                job.message = "the command could not be started"
                self._save(job)
            if not handed:
                ctx.idempotency.abandon(key)
            raise
        finally:
            if not handed:
                self.mutations.release_shell()

    def _status(self, arguments: dict[str, Any]) -> str:
        job = self._require(require_str(arguments, "job_id"))
        cursor = cursor_offset(arguments)
        max_bytes = optional_int(arguments, "max_bytes")
        if max_bytes is None:
            max_bytes = SHELL_DEFAULT_MAX_BYTES
        if max_bytes < 1:
            raise ToolError("invalid_arguments", "max_bytes must be at least 1")
        max_bytes = min(max_bytes, SHELL_MAX_BYTES)
        with job.lock:
            data = bytes(job.output)
            status = job.status
            exit_code = job.exit_code
            message = job.message
            cancel_requested = job.cancel_requested
            truncated = job.output_truncated
        if cursor > len(data):
            raise ToolError("invalid_range", "cursor is past the end of the output")
        text, end = _utf8_window(data, cursor, max_bytes)
        body: dict[str, Any] = {
            "has_more": end < len(data) or status not in _TERMINAL,
            "job_id": job.job_id,
            "next_cursor": str(end),
            "output": text,
            "status": status,
        }
        if exit_code is not None:
            body["exit_code"] = exit_code
        if message:
            body["message"] = message
        if cancel_requested and status not in _TERMINAL:
            body["cancel_requested"] = True
        if truncated:
            body["output_truncated"] = True
        return dump(body)

    def _input(self, arguments: dict[str, Any], ctx: ToolContext) -> str:
        job = self._require(require_str(arguments, "job_id"))
        text = require_text(arguments, "input", allow_empty=True)
        close_stdin = optional_bool(arguments, "close_stdin", default=False)
        key = check_key(require_text(arguments, "idempotency_key", allow_empty=False))
        payload = {"close_stdin": close_stdin, "input": text, "job_id": job.job_id}
        reserved = ctx.idempotency.reserve(key, payload)
        if reserved.kind == "replay" and reserved.outcome is not None:
            return reserved.outcome
        if reserved.kind == "pending":
            raise ToolError("conflict", "attempt did not finish")
        proc = self._wait_running(job)
        if proc is None or proc.stdin is None:
            ctx.idempotency.abandon(key)
            raise ToolError("conflict", "job is not running")
        try:
            if text:
                proc.stdin.write(text.encode("utf-8"))
                proc.stdin.flush()
            if close_stdin:
                proc.stdin.close()
        except (BrokenPipeError, OSError) as exc:
            ctx.idempotency.abandon(key)
            raise ToolError("execution_failed", "stdin is closed") from exc
        body = dump({"accepted": True, "job_id": job.job_id, "status": "running"})
        ctx.idempotency.finish(key, body)
        return body

    def _cancel(self, arguments: dict[str, Any]) -> str:
        job = self._require(require_str(arguments, "job_id"))
        grace = optional_number(arguments, "grace_period_seconds")
        if grace is None:
            grace = SHELL_CANCEL_GRACE
        if grace < 0:
            raise ToolError("invalid_arguments", "grace_period_seconds must be zero or greater")
        with job.lock:
            if job.status in _TERMINAL:
                status = job.status
                exit_code = job.exit_code
            else:
                job.grace = grace
                job.cancel_requested = True
                status = ""
                exit_code = None
        if status == "":
            return dump({"job_id": job.job_id, "status": "cancel_requested"})
        body: dict[str, Any] = {"job_id": job.job_id, "status": status}
        if exit_code is not None:
            body["exit_code"] = exit_code
        return dump(body)

    def _create(self, command: str, directory: str, cwd: Path) -> Job:
        with self._lock:
            number = 1
            while f"shell_{number}" in self._jobs:
                number += 1
            job = Job(f"shell_{number}", command, directory, cwd)
            self._jobs[job.job_id] = job
            return job

    def _require(self, job_id: str) -> Job:
        job = self._jobs.get(job_id)
        if job is None:
            raise ToolError("not_found", "job does not exist")
        return job

    def _worker(self, job: Job) -> None:
        try:
            self._run(job)
        except Exception:
            with job.lock:
                if job.status not in _TERMINAL:
                    job.status = "failed"
                    job.message = "the command could not be started"
            self._save(job)
        finally:
            # start() acquired the workspace lease. This thread releases it.
            self.mutations.release_shell()

    def _run(self, job: Job) -> None:
        with job.lock:
            if job.cancel_requested:
                job.status = "cancelled"
                self._save(job)
                return
        proc = subprocess.Popen(
            [_SHELL, "-c", job.command],
            cwd=str(job.cwd),
            env=_environment(self.workspace),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        with job.lock:
            job.proc = proc
            job.status = "running"
            self._save(job)
            cancel_requested = job.cancel_requested
            grace = job.grace
        if cancel_requested:
            self._signal_until(job, proc, "cancelled", grace)
            return
        reader = threading.Thread(target=self._drain, args=(job, proc), daemon=True)
        reader.start()
        deadline = None if job.timeout_seconds is None else time.monotonic() + job.timeout_seconds
        while proc.poll() is None:
            with job.lock:
                overflow = job.overflow
                cancel_requested = job.cancel_requested
                grace = job.grace
            if overflow:
                self._signal_until(job, proc, "failed", SHELL_CANCEL_GRACE)
                with job.lock:
                    job.message = "output exceeded the harness cap"
                    job.output_truncated = True
                    self._save(job)
                break
            if cancel_requested:
                self._signal_until(job, proc, "cancelled", grace)
                break
            if deadline is not None and time.monotonic() >= deadline:
                self._signal_until(job, proc, "timed_out", SHELL_CANCEL_GRACE)
                break
            time.sleep(0.02)
        else:
            with job.lock:
                if job.overflow:
                    job.status = "failed"
                    job.message = "output exceeded the harness cap"
                    job.output_truncated = True
                else:
                    job.status = "succeeded" if proc.returncode == 0 else "failed"
                job.exit_code = proc.returncode
                self._save(job)
        reader.join(timeout=2)
        with job.lock:
            job.proc = None
        self._close_stdin(proc)
        self._save(job)

    def _drain(self, job: Job, proc: subprocess.Popen[bytes]) -> None:
        stream = proc.stdout
        if stream is None:
            return
        path = self._output_path(job)
        while True:
            chunk = stream.read(4096)
            if not chunk:
                break
            if job.append(chunk, path):
                _signal(proc, signal.SIGKILL)
                break

    def _signal_until(
        self,
        job: Job,
        proc: subprocess.Popen[bytes],
        status: str,
        grace: float,
    ) -> None:
        _signal(proc, signal.SIGTERM)
        end = time.monotonic() + max(0.0, grace)
        while proc.poll() is None and time.monotonic() < end:
            time.sleep(0.02)
        if proc.poll() is None:
            _signal(proc, signal.SIGKILL)
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
        with job.lock:
            if proc.returncode is not None:
                job.exit_code = proc.returncode
            job.status = status
            job.proc = None
            self._save(job)

    def _wait_running(self, job: Job) -> subprocess.Popen[bytes] | None:
        deadline = time.monotonic() + SHELL_INPUT_WAIT
        while time.monotonic() < deadline:
            with job.lock:
                if job.status in _TERMINAL:
                    return None
                if job.status == "running" and job.proc is not None:
                    return job.proc
            time.sleep(0.02)
        return None

    def _output_path(self, job: Job) -> Path:
        return self.jobs_dir / f"{job.job_id}.out"

    def _save(self, job: Job) -> None:
        path = self.jobs_dir / f"{job.job_id}.json"
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(job.record(), sort_keys=True), encoding="utf-8")
        temporary.replace(path)

    def _close_stdin(self, proc: subprocess.Popen[bytes]) -> None:
        if proc.stdin is None:
            return
        try:
            proc.stdin.close()
        except OSError:
            pass

    def _load(self) -> None:
        for path in sorted(self.jobs_dir.glob("shell_*.json")):
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(raw, dict) or not isinstance(raw.get("job_id"), str):
                continue
            job = _job_from(self.workspace, raw)
            output = self.jobs_dir / f"{job.job_id}.out"
            if output.is_file():
                job.output = bytearray(output.read_bytes())
            if job.status in {"queued", "running"}:
                job.status = "interrupted"
                job.message = "the process did not survive restart"
                job.exit_code = None
                job.cancel_requested = False
                self._save(job)
            self._jobs[job.job_id] = job


def _action(arguments: dict[str, Any]) -> str:
    if "action" not in arguments or arguments["action"] is None:
        return "start"
    action = arguments["action"]
    if not isinstance(action, str) or action not in _ACTIONS:
        raise ToolError("invalid_arguments", "unknown action")
    return action


def _directory(workspace: Path, raw: str) -> Path:
    path = resolve_inside(workspace, raw)
    if not path.exists():
        raise ToolError("not_found", "directory does not exist")
    if not path.is_dir():
        raise ToolError("invalid_arguments", "directory is not a directory")
    return path


def _environment(workspace: Path) -> dict[str, str]:
    return {
        "HOME": str(workspace),
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONUNBUFFERED": "1",
    }


def _signal(proc: subprocess.Popen[bytes], sig: signal.Signals) -> None:
    if proc.pid is None or proc.pid <= 0:
        return
    try:
        os.killpg(proc.pid, sig)
    except ProcessLookupError:
        return
    except PermissionError:
        try:
            proc.send_signal(sig)
        except (ProcessLookupError, OSError):
            return


def _utf8_window(data: bytes, cursor: int, max_bytes: int) -> tuple[str, int]:
    end = min(len(data), cursor + max_bytes)
    while end < len(data) and end < cursor + max_bytes + 3:
        try:
            data[cursor:end].decode("utf-8")
            break
        except UnicodeDecodeError:
            end += 1
    chunk = data[cursor:end]
    while chunk:
        try:
            return chunk.decode("utf-8"), cursor + len(chunk)
        except UnicodeDecodeError:
            chunk = chunk[:-1]
    return "", cursor


def _job_from(workspace: Path, raw: dict[str, Any]) -> Job:
    job_id = str(raw["job_id"])
    command = raw.get("command") if isinstance(raw.get("command"), str) else ""
    directory = raw.get("directory") if isinstance(raw.get("directory"), str) else "."
    try:
        cwd = resolve_inside(workspace, directory)
    except ToolError:
        cwd = workspace
    job = Job(job_id, command, directory, cwd)
    description = raw.get("description")
    job.description = description if isinstance(description, str) else ""
    timeout = raw.get("timeout_seconds")
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
        job.timeout_seconds = None
    else:
        job.timeout_seconds = float(timeout)
    status = raw.get("status")
    job.status = status if isinstance(status, str) else "interrupted"
    exit_code = raw.get("exit_code")
    if isinstance(exit_code, bool) or not isinstance(exit_code, int):
        job.exit_code = None
    else:
        job.exit_code = exit_code
    message = raw.get("message")
    job.message = message if isinstance(message, str) else ""
    job.cancel_requested = raw.get("cancel_requested") is True
    job.output_truncated = raw.get("output_truncated") is True
    for field in ("session_id", "run_id", "tool_call_id"):
        value = raw.get(field)
        if isinstance(value, str):
            setattr(job, field, value)
    return job
