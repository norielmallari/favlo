"""Agent session on an isolated worktree of ``main``.

The model writes three files through tools. A grader outside the worktree
checks them. The same Microsoft Agent Framework session then reads source
until a prompt reaches 131072 tokens. The next turn runs
``SummarizationStrategy`` before the request is sent. The model id is the
pipeline name. Write turns set ``enable_thinking`` true. Read turns and the
summary turn set it false and leave every other sampling field unset.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

from agent_framework import (
    Agent,
    CompactionProvider,
    InMemoryHistoryProvider,
    Message,
    SummarizationStrategy,
    tool,
)
from agent_framework.openai import OpenAIChatCompletionClient
from openai import AsyncOpenAI
from pydantic import Field

from harness.coding.workspace import locate
from harness.file_session import (
    Log,
    Server,
    _completions,
    _free_port,
    _spawn,
    _stop_server,
    _wait_listening,
    _wait_requests,
    format_request,
    format_summary,
    model_stem,
    parse_requests,
    read_repository_file,
    source_files,
)

TARGET_TOKENS = 131072
_FIX_TURNS = 4
_SUMMARY_INPUT_TOKENS = 80000
_INSTRUCTIONS = (
    "You edit one git worktree, then read source files from that worktree. "
    "Use only the tool the user names. Paths stay inside the worktree."
)
_SPAN = """\
Write notes/span.py in this worktree.

The module defines width(start, end). start and end are integers. width returns end minus start.
The module defines cover(spans). spans is a list of pairs of integers. cover returns the sum of width for each pair.
The module defines label(total). total is an integer. label returns the string cover= followed by that integer, with no spaces.

Call write_file once. path is notes/span.py. content is the whole module.
After that call, stop. Do not call another tool.
"""
_CHECK = """\
Write notes/check.py in this worktree.

The module imports cover and label from notes.span.
The module defines report(). report calls cover with the pairs (1, 4) and (10, 12), then returns label of that result.

Call write_file once. path is notes/check.py. content is the whole module.
After that call, stop. Do not call another tool.
"""
_README = """\
Write notes/README.md in this worktree.

Name span.py and check.py.
Name the functions width, cover, and label, one sentence each.

Call write_file once. path is notes/README.md. content is the whole file.
After that call, stop. Do not call another tool.
"""
_FIX = """\
The check failed:
{error}

Replace the file that is wrong. Call write_file once with the whole file.
path is notes/span.py, notes/check.py, or notes/README.md.
After that call, stop. Do not call another tool.
"""
_READ = """\
Call read_file once. path is {path}.
After the file is returned, reply with at most 12 lines on what the file is for.
Do not call another tool.
"""
_AFTER = """\
Reply with one sentence that names span.py.
Do not call a tool.
"""


@dataclass
class SummaryGate:
    """Turns summarization on after a prompt has reached ``target`` tokens."""

    target: int
    peak: int = 0
    fired: bool = False
    text: str = ""


class _InstructSummarizer:
    """Summary calls use the instruct card."""

    def __init__(self, client: OpenAIChatCompletionClient) -> None:
        self._client = client

    async def get_response(self, messages, *, stream: bool = False, options=None, **kwargs):
        merged = dict(options or {})
        extra = dict(merged.get("extra_body") or {})
        extra["enable_thinking"] = False
        merged["extra_body"] = extra
        merged["max_tokens"] = 512
        return await self._client.get_response(messages, stream=stream, options=merged, **kwargs)


class SummarizePastTarget:
    """Run ``SummarizationStrategy`` once the recorded peak reaches the target."""

    def __init__(self, inner: SummarizationStrategy, gate: SummaryGate) -> None:
        self.inner = inner
        self.gate = gate
        self.tokenizer = inner.tokenizer

    async def __call__(self, messages: list[Message]) -> bool:
        if self.gate.peak < self.gate.target or self.gate.fired:
            return False
        changed = await self.inner(messages)
        if not changed:
            return False
        self.gate.fired = True
        self.gate.text = _summary_text(messages)
        return True


def _summary_text(messages: list[Message]) -> str:
    for message in messages:
        group = message.additional_properties.get("_group")
        if not isinstance(group, dict):
            continue
        covered = group.get("_summary_of_message_ids")
        if isinstance(covered, list) and covered:
            return message.text or ""
    return ""


def add_worktree(repo: Path, dest: Path, commit: str = "main") -> None:
    """Check out ``commit`` into ``dest`` as a detached worktree."""
    subprocess.run(
        ["git", "worktree", "add", "--detach", os.fspath(dest), commit],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )


def remove_worktree(repo: Path, dest: Path) -> None:
    """Remove a worktree. A leftover directory under ``artifacts`` is deleted."""
    subprocess.run(
        ["git", "worktree", "remove", "--force", os.fspath(dest)],
        cwd=repo,
        check=False,
        capture_output=True,
        text=True,
    )
    resolved = dest.resolve()
    artifacts = (repo.resolve() / "artifacts").resolve()
    if resolved == artifacts or artifacts not in resolved.parents:
        return
    if resolved.exists():
        shutil.rmtree(resolved)


def open_agent_worktree(repo: Path) -> Path:
    """Detached worktree of ``main`` at ``artifacts/agent-worktree``."""
    root = repo.resolve()
    dest = root / "artifacts" / "agent-worktree"
    if dest.parent.resolve() != (root / "artifacts").resolve():
        message = "worktree path is not under artifacts"
        raise RuntimeError(message)
    dest.parent.mkdir(parents=True, exist_ok=True)
    remove_worktree(root, dest)
    add_worktree(root, dest, "main")
    notes = dest / "notes"
    notes.mkdir()
    (notes / "__init__.py").write_text("", encoding="utf-8")
    return dest


def grade(root: Path) -> str | None:
    """``None`` when the worktree change matches the task. Otherwise the failure."""
    span_path = root / "notes" / "span.py"
    check_path = root / "notes" / "check.py"
    readme_path = root / "notes" / "README.md"
    missing = [path.name for path in (span_path, check_path, readme_path) if not path.is_file()]
    if missing:
        return "missing " + ", ".join(missing)
    root_text = str(root.resolve())
    sys.path.insert(0, root_text)
    for name in list(sys.modules):
        if name == "notes" or name.startswith("notes."):
            del sys.modules[name]
    try:
        span = importlib.import_module("notes.span")
        check = importlib.import_module("notes.check")
        width = span.width(1, 4)
        cover = span.cover([(1, 4), (10, 12)])
        label = span.label(5)
        report = check.report()
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"
    finally:
        if root_text in sys.path:
            sys.path.remove(root_text)
        for name in list(sys.modules):
            if name == "notes" or name.startswith("notes."):
                del sys.modules[name]
    errors: list[str] = []
    if width != 3:
        errors.append(f"width(1, 4) returned {width!r}")
    if cover != 5:
        errors.append(f"cover([(1, 4), (10, 12)]) returned {cover!r}")
    if label != "cover=5":
        errors.append(f"label(5) returned {label!r}")
    readme = readme_path.read_text(encoding="utf-8")
    if "span.py" not in readme or "check.py" not in readme:
        errors.append("README.md must name span.py and check.py")
    if "width" not in readme or "cover" not in readme or "label" not in readme:
        errors.append("README.md must name width, cover, and label")
    if report != "cover=5":
        errors.append(f"report() returned {report!r}")
    if errors:
        return "; ".join(errors)
    return None


def _snippet(root: Path) -> str:
    """The files the grader reads, shortened so a failure can be inspected."""
    parts: list[str] = []
    for name in ("span.py", "check.py", "README.md"):
        path = root / "notes" / name
        if not path.is_file():
            parts.append(f"{name}: missing")
            continue
        text = path.read_text(encoding="utf-8")
        parts.append(f"{name}:\n{text[:1200]}")
    return "\n".join(parts)


def _unwrap(content: str) -> str:
    """Drop one surrounding markdown fence so the file is the module itself."""
    text = content.strip()
    lines = text.splitlines()
    if len(lines) >= 2 and lines[0].startswith("```") and lines[-1].startswith("```"):
        text = "\n".join(lines[1:-1]).strip()
    if text == "":
        return ""
    if not text.endswith("\n"):
        text += "\n"
    return text


def _write_tool(root: Path):
    @tool(approval_mode="never_require")
    def write_file(
        path: Annotated[str, Field(description="Path inside the worktree.")],
        content: Annotated[str, Field(description="Full file contents.")],
    ) -> str:
        """Create or replace one file inside the worktree."""
        try:
            if ".git" in Path(path).parts or Path(path).name == "__init__.py":
                message = "path stays inside the worktree"
                raise ValueError(message)
            dest = locate(root, path)
            text = _unwrap(content)
            if text == "":
                message = "content is required"
                raise ValueError(message)
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(text, encoding="utf-8")
        except (OSError, ValueError) as exc:
            return str(exc)
        return f"wrote {path}"

    return write_file


def _read_tool(root: Path):
    @tool(approval_mode="never_require")
    def read_file(
        path: Annotated[str, Field(description="Worktree-relative path of the source file.")],
    ) -> str:
        """Return one source file from the worktree."""
        try:
            allowed = {item.relative_to(root).as_posix() for item in source_files(root)}
            text = read_repository_file(root, path, allowed)
        except ValueError as exc:
            return str(exc)
        return text + "\n\nThe file is above. Reply with at most 12 lines on what it is for."

    return read_file


def _open_agent(
    base_url: str,
    model: str,
    tools: list[object],
    gate: SummaryGate,
    http: AsyncOpenAI,
) -> tuple[OpenAIChatCompletionClient, Agent]:
    client = OpenAIChatCompletionClient(
        model=model,
        api_key="favlo",
        base_url=base_url,
        async_client=http,
    )
    client.function_invocation_configuration["max_iterations"] = 1
    summary_client = OpenAIChatCompletionClient(
        model=model,
        api_key="favlo",
        base_url=base_url,
        async_client=http,
    )
    strategy = SummarizationStrategy(
        client=_InstructSummarizer(summary_client),
        target_count=4,
        threshold=0,
        max_summary_input_tokens=_SUMMARY_INPUT_TOKENS,
    )
    history = InMemoryHistoryProvider(skip_excluded=True)
    compaction = CompactionProvider(
        before_strategy=SummarizePastTarget(strategy, gate),
        history_source_id=history.source_id,
    )
    agent = Agent(
        client=client,
        name="worktree-agent",
        instructions=_INSTRUCTIONS,
        tools=tools,
        context_providers=[history, compaction],
    )
    return client, agent


def _options(thinking: bool, tool_name: str | None) -> dict[str, object]:
    choice: object = "none" if tool_name is None else {"mode": "required", "required_function_name": tool_name}
    return {
        "tool_choice": choice,
        "extra_body": {"enable_thinking": thinking},
        "max_tokens": 2048 if thinking else 512,
    }


async def _turn(
    agent: Agent,
    session: object,
    prompt: str,
    log: Log,
    server: Server,
    seen: int,
    *,
    thinking: bool,
    tool_name: str | None,
) -> tuple[list, int]:
    before = len(parse_requests(log.text()))
    await agent.run(prompt, session=session, options=_options(thinking, tool_name))
    requests: list[RequestRate] = _wait_requests(log, before, server)
    for row, request in enumerate(requests[seen:], start=seen + 1):
        print(format_request(row, request), flush=True)
    return requests, len(requests)


async def run_worktree_agent(
    repo: Path,
    log: Log,
    server: Server,
    *,
    model: str,
    target: int = TARGET_TOKENS,
) -> tuple[list, SummaryGate, str]:
    """Edit the worktree, read until ``target``, then summarize."""
    root = open_agent_worktree(repo)
    print("worktree ready", flush=True)
    tools = [_write_tool(root), _read_tool(root)]
    gate = SummaryGate(target)
    base_url = f"http://127.0.0.1:{server.port}/v1"
    http = AsyncOpenAI(api_key="favlo", base_url=base_url, timeout=3600.0)
    _client, agent = _open_agent(base_url, model, tools, gate, http)
    session = agent.create_session()
    requests: list = []
    seen = 0
    try:
        for prompt in (_SPAN, _CHECK, _README):
            requests, seen = await _turn(
                agent, session, prompt, log, server, seen, thinking=True, tool_name="write_file"
            )
            gate.peak = max(request.prompt_tokens for request in requests)
        error = grade(root)
        for _ in range(_FIX_TURNS):
            if error is None:
                break
            print(f"grade {error}", flush=True)
            requests, seen = await _turn(
                agent,
                session,
                _FIX.format(error=error),
                log,
                server,
                seen,
                thinking=True,
                tool_name="write_file",
            )
            gate.peak = max(request.prompt_tokens for request in requests)
            error = grade(root)
        if error is not None:
            return requests, gate, error + "\n" + _snippet(root)
        print("grade passed", flush=True)
        for path in source_files(root):
            if requests and max(request.prompt_tokens for request in requests) >= target:
                break
            rel = path.relative_to(root).as_posix()
            requests, seen = await _turn(
                agent,
                session,
                _READ.format(path=rel),
                log,
                server,
                seen,
                thinking=False,
                tool_name="read_file",
            )
            gate.peak = max(request.prompt_tokens for request in requests)
            print(f"read {rel}", flush=True)
        if gate.peak < target:
            return requests, gate, f"source files ended at {gate.peak} tokens"
        requests, seen = await _turn(
            agent, session, _AFTER, log, server, seen, thinking=False, tool_name=None
        )
    finally:
        await http.close()
        remove_worktree(repo, root)
    if not requests:
        return requests, gate, "no requests"
    peak = max(request.prompt_tokens for request in requests)
    if peak < target:
        return requests, gate, f"stopped at {peak} tokens"
    if not gate.fired:
        return requests, gate, "summarization did not run\n" + _completions(log)
    peak_at = max(index for index, request in enumerate(requests) if request.prompt_tokens == peak)
    if not any(request.prompt_tokens < peak for request in requests[peak_at + 1 :]):
        return requests, gate, "summarized prompt was not shorter\n" + _completions(log)
    print(f"summarized {gate.text[:160]}", flush=True)
    return requests, gate, ""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Edit a worktree of main, then summarize a 131072-token session."
    )
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokens", type=int, default=TARGET_TOKENS)
    parser.add_argument("--max-memory", default=None, help="serve memory cap, such as 16GiB")
    args = parser.parse_args(argv)
    if args.tokens < 1:
        print("token target must be positive")
        return 1
    stem = model_stem(args.model)
    from favlo.cli import repo_root
    from favlo.smoke_lock import hold, release

    root = repo_root()
    created = hold(root=root, pipeline=stem, command="worktree")
    log = Log()
    server: Server | None = None
    try:
        port = args.port or _free_port()
        server = _spawn(port, log, args.max_memory)
        _wait_listening(server, log, 0)
        requests, _gate, note = asyncio.run(
            run_worktree_agent(args.repo, log, server, model=stem, target=args.tokens)
        )
    except Exception as exc:
        print(exc)
        print(log.text()[-3000:])
        return 1
    finally:
        if server is not None:
            _stop_server(server)
        if created:
            release(root=root)
    print(format_summary(requests, args.tokens))
    if note:
        print(note)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
