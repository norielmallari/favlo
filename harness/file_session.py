"""Read repository files until the prompt reaches a token target.

The file agent calls ``read_file`` on the OpenAI chat endpoint, then analyzes
the file it got back. The model id is the pipeline name. The request sets
``enable_thinking`` false and leaves every other sampling field unset, so the
listener fills the instruct card. The first completion of each task is required
to call ``read_file``. The server stops after the first task and starts again,
so the next task loads the prefix from disk. Later tasks keep that server. A
request that does not extend the stored token ids stops the run. Each request
prints the server's prefill and decode lines. Those lines are the instruct
series.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import socket
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

from agent_framework import Agent, Content, tool
from agent_framework.openai import OpenAIChatCompletionClient
from openai import AsyncOpenAI
from pydantic import Field

TARGET_TOKENS = 65536
INSTRUCTIONS = (
    "You are a repository agent. Call read_file with the path the user names. "
    "Think through the files already returned before that call. "
    "After the file comes back, think through what it does and how it relates "
    "to those files, then reply with that analysis."
)
SKIP_PARTS = frozenset(
    {
        ".git",
        ".venv",
        "models",
        "artifacts",
        "__pycache__",
        ".cursor",
        ".ruff_cache",
        ".pytest_cache",
    }
)
TEXT_SUFFIXES = frozenset({".py", ".metal", ".md", ".toml"})
_TOKENS = re.compile(r"\((\d+) tok in ")
_SPAN = re.compile(r"\((\d+) tok in ([0-9.]+)s")


def tool_call_text() -> str:
    """The checkpoint template's rendering of one ``read_file`` call with no arguments."""
    return "<tool_call>\n<function=read_file>\n</function>\n</tool_call>"


def read_prompt(path: str) -> str:
    return f"Call read_file with path {path}."


def model_stem(model: str) -> str:
    """The listener selects the instruct card when the id has no variant suffix."""
    return model.partition(":")[0]


def source_files(root: Path) -> list[Path]:
    """UTF-8 source files, largest first. Models, artifacts, and caches are omitted."""
    base = root.resolve()
    found: list[tuple[int, Path]] = []
    for dirpath, dirnames, filenames in os.walk(base):
        dirnames[:] = [name for name in dirnames if name not in SKIP_PARTS]
        for name in filenames:
            path = Path(dirpath) / name
            if path.suffix not in TEXT_SUFFIXES:
                continue
            try:
                size = path.stat().st_size
            except OSError:
                continue
            if size == 0:
                continue
            found.append((size, path))
    found.sort(key=lambda item: (-item[0], item[1].relative_to(base).as_posix()))
    return [path for _size, path in found]


def read_repository_file(
    root: Path, path: str, allowed: set[str], task: str | None = None
) -> str:
    """Return one allowed source file. Any other path is refused.

    A path that matches one allowed file regardless of case is that file.
    ``task`` is the file this turn asked for. A call whose path does not
    resolve still returns that file, so the tool result is the named source.
    """
    chosen = path
    if path not in allowed:
        folded = path.casefold()
        matches = [item for item in allowed if item.casefold() == folded]
        if len(matches) == 1:
            chosen = matches[0]
        elif task in allowed:
            chosen = task
        else:
            message = "path is not a repository source file"
            raise ValueError(message)
    base = root.resolve()
    target = (base / chosen).resolve()
    if not target.is_relative_to(base) or not target.is_file():
        message = "path is not a repository source file"
        raise ValueError(message)
    return target.read_text(encoding="utf-8", errors="replace")


@dataclass(frozen=True)
class RequestRate:
    prompt_tokens: int
    prefix_hit: int | None
    missed: bool
    prefill: str
    prefill_gpu: str
    decode: str
    decode_tokens: int
    atlas: str
    steady: tuple[str, ...]
    prefix_load: int | None = None


def token_count(line: str) -> int:
    if line.endswith(" 0 tok") or line.endswith(" 0 tok/s"):
        return 0
    match = _TOKENS.search(line)
    if match is None:
        return 0
    return int(match.group(1))


def _load_count(line: str) -> int | None:
    parts = line.split()
    if len(parts) >= 3 and parts[2].isdigit():
        return int(parts[2])
    return None


def parse_requests(text: str) -> list[RequestRate]:
    """Group server log lines into one record per chat request.

    A ``prefix load`` line before ``prompt tokens`` belongs to that request.
    One after ``prompt tokens`` and before decode belongs to the open request.
    """
    requests: list[RequestRate] = []
    current: dict[str, object] | None = None
    pending_load: int | None = None

    def finish() -> None:
        nonlocal current
        if current is None:
            return
        steady = current["steady"]
        assert isinstance(steady, list)
        hit = current["prefix_hit"]
        loaded = current["prefix_load"]
        requests.append(
            RequestRate(
                prompt_tokens=int(current["prompt_tokens"]),
                prefix_hit=hit if isinstance(hit, int) else None,
                missed=bool(current["missed"]),
                prefill=str(current["prefill"]),
                prefill_gpu=str(current["prefill_gpu"]),
                decode=str(current["decode"]),
                decode_tokens=int(current["decode_tokens"]),
                atlas=str(current["atlas"]),
                steady=tuple(steady),
                prefix_load=loaded if isinstance(loaded, int) else None,
            )
        )
        current = None

    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith("prefix load "):
            loaded = _load_count(line)
            if loaded is None:
                continue
            if current is not None and current["decode"] == "":
                current["prefix_load"] = loaded
            else:
                pending_load = loaded
            continue
        if line.startswith("prompt tokens "):
            parts = line.split()
            if len(parts) != 3 or not parts[2].isdigit():
                continue
            finish()
            current = {
                "prompt_tokens": int(parts[2]),
                "prefix_hit": None,
                "missed": False,
                "prefill": "",
                "prefill_gpu": "",
                "decode": "",
                "decode_tokens": 0,
                "atlas": "",
                "steady": [],
                "prefix_load": pending_load,
            }
            pending_load = None
            continue
        if line.startswith("incoming tokens"):
            continue
        if current is None:
            continue
        steady = current["steady"]
        assert isinstance(steady, list)
        if line in {"prefix miss", "cache miss"}:
            current["missed"] = True
            current["prefix_hit"] = None
        elif line.startswith("cache hit ") or line.startswith("prefix hit "):
            hit = line.split()[-1]
            if hit.isdigit():
                current["prefix_hit"] = int(hit)
        elif line.startswith("prefill after load "):
            current["prefill_gpu"] = line
        elif line.startswith("prefill "):
            current["prefill"] = line
        elif line.startswith("decode after atlas load"):
            steady.append(line)
        elif line.startswith("decode atlas load "):
            current["atlas"] = line
        elif line.startswith("decode "):
            current["decode"] = line
            current["decode_tokens"] = token_count(line)
    finish()
    return requests


def _cold_start(request: RequestRate) -> bool:
    if request.prefix_load is not None:
        return False
    if request.prefix_hit not in (None, 0):
        return False
    return request.prefix_hit is not None or request.missed


def _applied(request: RequestRate) -> int:
    return request.prompt_tokens + request.decode_tokens


def _extends(previous: RequestRate, current: RequestRate) -> bool:
    return (
        not current.missed
        and current.prefix_hit == _applied(previous)
        and current.prefix_load is None
    )


def _stored_load(request: RequestRate) -> bool:
    return (
        request.prefix_load is not None
        and not request.missed
        and request.prefix_hit == request.prefix_load
    )


def prefix_so_far(requests: list[RequestRate]) -> bool:
    """True while these requests can still finish as one restart load and in-memory extensions.

    The first request misses, or loads a prefix that was already on disk. A tool
    follow-up on that same server extends it in memory. After the server restarts,
    one request loads the applied prefix from disk. Every other later request
    extends the previous one in memory.
    """
    if not requests:
        return False
    first = requests[0]
    if not (_cold_start(first) or _stored_load(first)):
        return False
    loaded = False
    for previous, current in zip(requests, requests[1:], strict=False):
        if current.prefix_load is not None:
            if loaded or not _stored_load(current) or current.prefix_hit != _applied(previous):
                return False
            loaded = True
            continue
        if not _extends(previous, current):
            return False
    return True


def extensions_hold(requests: list[RequestRate]) -> bool:
    """True when one later request loaded the prefix from disk and the others extended it."""
    return (
        len(requests) >= 2
        and prefix_so_far(requests)
        and any(request.prefix_load is not None for request in requests[1:])
    )


def summed_rate(lines: list[str]) -> tuple[int, float] | None:
    tokens = 0
    seconds = 0.0
    for line in lines:
        match = _SPAN.search(line)
        if match is None:
            continue
        tokens += int(match.group(1))
        seconds += float(match.group(2))
    if tokens <= 0 or seconds <= 0:
        return None
    return tokens, seconds


def format_request(index: int, request: RequestRate) -> str:
    prefix = "cache miss" if request.missed else f"cache hit {request.prefix_hit}"
    parts = [f"{index} prompt {request.prompt_tokens}"]
    if request.prefix_load is not None:
        parts.append(f"prefix load {request.prefix_load}")
    parts.extend(
        [
            prefix,
            request.prefill or "prefill pending",
            request.decode or "decode pending",
        ]
    )
    if request.prefill_gpu:
        parts.append(request.prefill_gpu)
    if request.atlas:
        parts.append(request.atlas)
    parts.extend(request.steady)
    return "  ".join(parts)


def format_summary(requests: list[RequestRate], target: int) -> str:
    if not requests:
        return "no requests"
    context = requests[-1].prompt_tokens
    lines = [f"context {context} tokens over {len(requests)} requests"]
    prefill = summed_rate([request.prefill for request in requests])
    decode = summed_rate([request.decode for request in requests])
    if prefill is not None:
        tokens, seconds = prefill
        lines.append(f"prefill {tokens / seconds:.2f} tok/s ({tokens} tok in {seconds:.2f}s)")
    if decode is not None:
        tokens, seconds = decode
        lines.append(f"decode {tokens / seconds:.2f} tok/s ({tokens} tok in {seconds:.2f}s)")
    lines.append(f"last {format_request(len(requests), requests[-1])}")
    if context < target:
        lines.append(f"stopped before {target} tokens")
    return "\n".join(lines)


class Log:
    def __init__(self) -> None:
        self._lines: list[str] = []
        self._lock = threading.Lock()

    def append(self, line: str) -> None:
        with self._lock:
            self._lines.append(line)

    def text(self) -> str:
        with self._lock:
            return "".join(self._lines)


@dataclass(frozen=True)
class RunResult:
    requests: list[RequestRate]
    reached: bool
    extended: bool
    note: str


def _read_tool(root: Path, allowed: set[str], loaded: list[str], task: dict[str, str]):
    @tool(approval_mode="never_require")
    def read_file(
        path: Annotated[str, Field(description="Repository-relative path of the source file.")],
    ) -> str:
        """Return one repository source file."""
        text = read_repository_file(root, path, allowed, task.get("path"))
        loaded.append(path)
        return text + "\n\nThe file is above. Reply with the analysis of this file."

    return read_file


@dataclass
class Server:
    pid: int
    port: int
    max_memory: str | None = None
    follow_stop: threading.Event | None = None


def _wait_requests(log: Log, previous: int, server: Server) -> list[RequestRate]:
    from favlo.daemon import alive

    deadline = time.monotonic() + 1800
    stable_at: float | None = None
    last = previous
    while time.monotonic() < deadline:
        if not alive(server.pid):
            message = "server exited\n" + log.text()[-4000:]
            raise RuntimeError(message)
        requests = parse_requests(log.text())
        count = len(requests)
        complete = count > previous and requests[-1].decode != ""
        if count != last:
            last = count
            stable_at = time.monotonic()
        elif complete and stable_at is not None and time.monotonic() - stable_at >= 0.3:
            return requests
        time.sleep(0.05)
    requests = parse_requests(log.text())
    if len(requests) > previous and requests[-1].decode != "":
        return requests
    message = "server log did not record the request\n" + log.text()[-2000:]
    raise TimeoutError(message)


def _completions(log: Log) -> str:
    lines = log.text().splitlines()
    blocks: list[str] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        if line.startswith("output tokens"):
            blocks.append(line[len("output tokens") :].strip()[:800])
            index += 1
            continue
        if not line.startswith("completion:"):
            index += 1
            continue
        block = [line[len("completion:") :].lstrip()]
        index += 1
        while index < len(lines) and not lines[index].startswith(
            (
                "incoming tokens",
                "prompt tokens ",
                "cache ",
                "prefix ",
                "prefill ",
                "decode ",
                "output tokens",
                "listening ",
            )
        ):
            block.append(lines[index])
            index += 1
        blocks.append("\n".join(block)[:800])
    return "\n---\n".join(blocks[-2:])


def _content_text(content: object) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for item in content:
        if isinstance(item, str):
            parts.append(item)
        elif isinstance(item, dict) and item.get("type") == "text":
            parts.append(str(item.get("text") or ""))
    return "".join(parts)


def read_reasoning(message: object, contents: list[Content]) -> list[Content]:
    """Keep a ``reasoning_content`` delta as reasoning, not as the reply."""
    extra = getattr(message, "model_extra", None) or {}
    text = extra.get("reasoning_content") if isinstance(extra, dict) else None
    if not isinstance(text, str) or text == "":
        found = getattr(message, "reasoning_content", None)
        text = found if isinstance(found, str) else ""
    if text == "":
        return contents
    return [*contents, Content.from_text_reasoning(text=text)]


def echo_reasoning(message: object, built: list[dict[str, object]]) -> list[dict[str, object]]:
    """Put reasoning back on ``reasoning_content`` for the chat template.

    The chat client copies reasoning into ``content``. The template then writes
    an empty think block. A reply that already contains the call XML is left
    as that text, so the template does not write the call a second time.
    """
    if getattr(message, "role", None) != "assistant":
        return built
    reasoning = "".join(
        part.text or ""
        for part in getattr(message, "contents", ())
        if getattr(part, "type", None) == "text_reasoning"
        and not getattr(part, "protected_data", None)
    )
    if not reasoning:
        return built
    kept: list[dict[str, object]] = []
    placed = False
    for item in built:
        if item.get("role") != "assistant":
            kept.append(item)
            continue
        text = _content_text(item.get("content"))
        if text == reasoning and "tool_calls" not in item:
            continue
        copied = dict(item)
        if not placed:
            copied["reasoning_content"] = reasoning
            placed = True
        if "<tool_call>" in text:
            copied.pop("tool_calls", None)
        kept.append(copied)
    if not placed:
        kept.append({"role": "assistant", "reasoning_content": reasoning, "content": ""})
    return kept


def _open_agent(base_url: str, model: str, read: object) -> tuple[AsyncOpenAI, Agent]:
    http = AsyncOpenAI(api_key="favlo", base_url=base_url, timeout=3600.0)
    client = OpenAIChatCompletionClient(
        model=model_stem(model),
        api_key="favlo",
        base_url=base_url,
        async_client=http,
        response_parser=read_reasoning,
        message_preparer=echo_reasoning,
    )
    # One tool call, then the framework asks for a reply with tools disabled.
    client.function_invocation_configuration["max_iterations"] = 1
    agent = Agent(
        client=client,
        name="file-agent",
        instructions=INSTRUCTIONS,
        tools=[read],
        default_options={
            "tool_choice": {"mode": "required", "required_function_name": "read_file"},
            "extra_body": {"enable_thinking": False},
        },
    )
    return http, agent


async def run_file_agent(
    root: Path,
    target: int,
    log: Log,
    server: Server,
    *,
    model: str,
) -> RunResult:
    """Read source files until ``prompt tokens`` reaches ``target``.

    Each task is one agent run: the model calls ``read_file``, the framework
    returns the file, and the model analyzes it. The server is stopped after
    the first task and replaced. Later tasks use that new process and the
    same session.
    """
    root = root.resolve()
    files = source_files(root)
    if not files:
        return RunResult([], False, False, "no source files")
    allowed = {path.relative_to(root).as_posix() for path in files}
    loaded: list[str] = []
    task = {"path": ""}
    read = _read_tool(root, allowed, loaded, task)
    base_url = f"http://127.0.0.1:{server.port}/v1"
    http, agent = _open_agent(base_url, model, read)
    session = agent.create_session()
    seen = 0
    turn = 0
    requests: list[RequestRate] = []
    try:
        for index, path in enumerate(files):
            if turn >= 2 and requests and requests[-1].prompt_tokens >= target:
                break
            rel = path.relative_to(root).as_posix()
            task["path"] = rel
            before_requests = len(parse_requests(log.text()))
            await agent.run(read_prompt(rel), session=session)
            requests = _wait_requests(log, before_requests, server)
            for row, request in enumerate(requests[seen:], start=seen + 1):
                print(format_request(row, request), flush=True)
            seen = len(requests)
            if not prefix_so_far(requests):
                note = "prefix did not extend\n" + _completions(log)
                return RunResult(requests, False, False, note)
            print(f"read {rel}", flush=True)
            turn += 1
            if turn == 1 and index + 1 < len(files):
                print("restarting server", flush=True)
                await http.close()
                http = None
                _restart(server, log)
                http, agent = _open_agent(f"http://127.0.0.1:{server.port}/v1", model, read)
    finally:
        if http is not None:
            await http.close()
    context = requests[-1].prompt_tokens if requests else 0
    reached = context >= target
    extended = extensions_hold(requests)
    if not reached:
        note = f"source files ended at {context} tokens"
    elif not extended:
        note = "prefix was not loaded from disk"
    else:
        note = ""
    return RunResult(requests, reached, extended, note)


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_listening(server: Server, log: Log, seen: int) -> None:
    from favlo.daemon import alive

    deadline = time.monotonic() + 300
    while time.monotonic() < deadline:
        if log.text().count("listening on") > seen:
            return
        if not alive(server.pid):
            message = log.text()[-4000:]
            raise RuntimeError(message)
        time.sleep(0.2)
    message = log.text()[-4000:]
    raise TimeoutError(message)


def _spawn(port: int, log: Log, max_memory: str | None = None) -> Server:
    from favlo.cli import repo_root
    from favlo.daemon import ensure, follow, telemetry_path

    pid, _started = ensure(
        "127.0.0.1",
        port,
        max_memory=max_memory,
        root=repo_root(),
    )
    stop = threading.Event()

    def on_line(line: str) -> None:
        log.append(line + "\n")

    follow(telemetry_path(port), on_line, stop, offset=0)
    return Server(pid, port, max_memory, stop)


def _stop_server(server: Server) -> None:
    """SIGTERM so the daemon can write the live prefix, then wait for that exit."""
    from favlo.daemon import stop_pid

    if server.follow_stop is not None:
        server.follow_stop.set()
    stop_pid(server.pid)


def _restart(server: Server, log: Log) -> None:
    _stop_server(server)
    seen = log.text().count("listening on")
    server.port = _free_port()
    started = _spawn(server.port, log, server.max_memory)
    server.pid = started.pid
    server.follow_stop = started.follow_stop
    _wait_listening(server, log, seen)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Read repository files until the prompt reaches a token target."
    )
    parser.add_argument("--tokens", type=int, default=TARGET_TOKENS)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--model", required=True)
    parser.add_argument("--max-memory", default=None, help="serve memory cap, such as 16GiB")
    args = parser.parse_args(argv)
    if args.tokens < 1:
        print("token target must be positive")
        return 1
    stem = model_stem(args.model)
    from favlo.cli import repo_root
    from favlo.smoke_lock import hold, release

    root = repo_root()
    created = hold(root=root, pipeline=stem, command="file-agent")
    log = Log()
    server: Server | None = None
    try:
        port = args.port or _free_port()
        server = _spawn(port, log, args.max_memory)
        _wait_listening(server, log, 0)
        result = asyncio.run(
            run_file_agent(args.repo, args.tokens, log, server, model=stem)
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
    print(format_summary(result.requests, args.tokens))
    if result.note:
        print(result.note)
    if result.reached and result.extended:
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
