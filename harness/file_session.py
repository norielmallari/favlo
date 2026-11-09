"""Read shared package sources until the prompt reaches a token target.

A Microsoft Agent Framework agent calls ``read_file``, then analyzes the file
it got back. The corpus is ``src/favlo`` only, so the ladder does not depend on
any pipeline package. The model id is the pipeline name under test. The request
sets ``reasoning.effort`` to ``none`` and opens the ``:instruct`` sampling card.
``--mtp on`` or ``--mtp off`` sends ``favlo:mtp``; omitted, the registry row
decides. The first completion of each task is required to call ``read_file``.
The harness reuses ``127.0.0.1:8129``. After the first task it unloads the
resident model, so the next task loads the prefix from disk. Later tasks keep
that listener. A request that does not extend the stored token ids stops the
run. Each model call streams its reply and tool calls. Each request also
prints the server's prefill and decode lines. Those lines are the instruct series.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import re
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

from agent_framework import Agent, tool
from pydantic import Field

from favlo.chat_client import close_chat_client, open_chat_client
from harness.turn_printer import TurnPrinter

TARGET_TOKENS = 65536
SHARED_PACKAGE = Path("src") / "favlo"
INSTRUCTIONS = (
    "You are a repository agent. Call read_file with the path the user names. "
    "Use the files already returned before that call. "
    "After the file comes back, say what it does and how it relates to those "
    "files, then reply with that analysis."
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
TEXT_SUFFIXES = frozenset({".py", ".md", ".toml"})
_TOKENS = re.compile(r"\((\d+) tok in ")
_SPAN = re.compile(r"\((\d+) tok in ([0-9.]+)s")


def tool_call_text() -> str:
    """The checkpoint template's rendering of one ``read_file`` call with no arguments."""
    return "<tool_call>\n<function=read_file>\n</function>\n</tool_call>"


def read_prompt(path: str) -> str:
    return f"Call read_file with path {path}."


def model_stem(model: str) -> str:
    """Bare folder id. Sampling follows that row; thinking follows the request."""
    return model.partition(":")[0]


def instruct_model(model: str) -> str:
    """Instruct variant of the pipeline. The file agent never opens thinking."""
    return f"{model_stem(model)}:instruct"


def shared_package(root: Path) -> Path:
    """The shared runtime package the ladder reads. Independent of any pipeline."""
    return (root.resolve() / SHARED_PACKAGE).resolve()


def source_files(root: Path) -> list[Path]:
    """UTF-8 files under ``src/favlo``, largest first. Caches are omitted."""
    package = shared_package(root)
    if not package.is_dir():
        return []
    found: list[tuple[int, Path]] = []
    for dirpath, dirnames, filenames in os.walk(package):
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
    found.sort(key=lambda item: (-item[0], item[1].relative_to(package).as_posix()))
    return [path for _size, path in found]


def read_repository_file(
    root: Path, path: str, allowed: set[str], task: str | None = None
) -> str:
    """Return one allowed shared-package file. Any other path is refused.

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
            message = "path is not a shared package source file"
            raise ValueError(message)
    package = shared_package(root)
    target = (package / chosen).resolve()
    if not target.is_relative_to(package) or not target.is_file():
        message = "path is not a shared package source file"
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


def pipeline_log(text: str, model: str) -> str:
    """Log lines that belong to ``model`` on a shared listener.

    ``request open`` names the model. Lines stay with that model until the next
    open, including a decode line that lands after ``request closed``. Another
    pipeline's prompt and vision notes stay out.
    """
    stem = model.split(":", 1)[0]
    kept: list[str] = []
    owner = ""
    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith("request open"):
            owner = line.removeprefix("request open").strip()
            continue
        if line.startswith("request closed"):
            continue
        if owner == stem or owner.startswith(stem + ":"):
            kept.append(line)
    return "\n".join(kept)


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


def printed_requests(requests: list[RequestRate]) -> list[RequestRate]:
    """One printed row per prompt length.

    A rolled serve log delivers the same ``prompt tokens`` line again, and the
    lines that follow it can be a different request's prefill. That repeat is
    the same chat request. The later row replaces the earlier one when it has
    a prefill or a decode line.
    """
    order: list[int] = []
    by_prompt: dict[int, RequestRate] = {}
    for request in requests:
        prompt = request.prompt_tokens
        previous = by_prompt.get(prompt)
        if previous is None:
            order.append(prompt)
            by_prompt[prompt] = request
            continue
        if request.prefill or request.decode:
            by_prompt[prompt] = request
    return [by_prompt[prompt] for prompt in order]


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
        path: Annotated[
            str, Field(description="Path of the shared package source file under src/favlo.")
        ],
    ) -> str:
        """Return one shared package source file."""
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
    owned: bool = False


def _wait_requests(log: Log, previous: int, server: Server) -> list[RequestRate]:
    from favlo.daemon import alive

    deadline = time.monotonic() + 1800
    stable_at: float | None = None
    last = previous
    while time.monotonic() < deadline:
        if server.pid > 0 and not alive(server.pid):
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


def _open_agent(
    base_url: str,
    model: str,
    read: object,
    *,
    mtp: bool | None,
    mtp_depth: int,
) -> tuple[object, Agent]:
    client = open_chat_client(instruct_model(model), base_url)
    # One tool call, then the framework asks for a reply with tools disabled.
    client.function_invocation_configuration["max_iterations"] = 1
    options: dict[str, object] = {
        "tool_choice": {"mode": "required", "required_function_name": "read_file"},
        "reasoning": {"effort": "none"},
    }
    if mtp is not None:
        options["extra_body"] = {
            "favlo:mtp": mtp,
            "favlo:mtp_depth": mtp_depth,
        }
    agent = Agent(
        client=client,
        name="file-agent",
        instructions=INSTRUCTIONS,
        tools=[read],
        default_options=options,
    )
    return client, agent


async def run_file_agent(
    root: Path,
    target: int,
    log: Log,
    server: Server,
    *,
    model: str,
    mtp: bool | None = None,
    mtp_depth: int = 3,
) -> RunResult:
    """Read shared package files until ``prompt tokens`` reaches ``target``.

    Each task is one Agent Framework run: the model calls ``read_file``, the
    framework returns the file, and the model analyzes it. The stream printer
    writes each model call as it arrives. The resident model is unloaded after
    the first task. Later tasks use the same listener and the same session.
    """
    root = root.resolve()
    package = shared_package(root)
    files = source_files(root)
    if not files:
        return RunResult([], False, False, f"no source files under {SHARED_PACKAGE.as_posix()}")
    allowed = {path.relative_to(package).as_posix() for path in files}
    loaded: list[str] = []
    task = {"path": ""}
    read = _read_tool(root, allowed, loaded, task)
    base_url = f"http://127.0.0.1:{server.port}/v1"
    chat, agent = _open_agent(base_url, model, read, mtp=mtp, mtp_depth=mtp_depth)
    session = agent.create_session()
    printer = TurnPrinter(sys.stdout)
    seen = 0
    turn = 0
    requests: list[RequestRate] = []
    try:
        for index, path in enumerate(files):
            if turn >= 2 and requests and requests[-1].prompt_tokens >= target:
                break
            rel = path.relative_to(package).as_posix()
            task["path"] = rel
            print(f"turn {turn + 1} read {SHARED_PACKAGE.as_posix()}/{rel}", flush=True)
            before_requests = len(parse_requests(log.text()))
            stream = agent.run(read_prompt(rel), stream=True, session=session)
            async for update in stream:
                printer.take(update)
            requests = _wait_requests(log, before_requests, server)
            for row, request in enumerate(requests[seen:], start=seen + 1):
                print(format_request(row, request), flush=True)
            seen = len(requests)
            if not prefix_so_far(requests):
                note = "prefix did not extend\n" + _completions(log)
                return RunResult(requests, False, False, note)
            print(f"read {SHARED_PACKAGE.as_posix()}/{rel}", flush=True)
            turn += 1
            if turn == 1 and index + 1 < len(files):
                _restart(server, log)
    finally:
        printer.close()
        if chat is not None:
            await close_chat_client(chat)
    context = requests[-1].prompt_tokens if requests else 0
    reached = context >= target
    extended = extensions_hold(requests)
    if not reached:
        note = f"shared package files ended at {context} tokens"
    elif not extended:
        note = "prefix was not loaded from disk"
    else:
        note = ""
    return RunResult(requests, reached, extended, note)


def _spawn(
    log: Log,
    pipeline: str,
    max_memory: str | None = None,
    port: int = 8129,
) -> Server:
    from favlo.cli import repo_root
    from favlo.daemon import alive, follow, read_pid, telemetry_path
    from favlo.smoke_lock import open_attached

    path = telemetry_path(port)
    before = path.stat().st_size if path.is_file() else 0
    attached = open_attached(
        root=repo_root(),
        pipeline=pipeline,
        port=port,
        max_memory=max_memory,
    )
    pid = attached.pid or 0
    if pid <= 0:
        found = read_pid(attached.port)
        if found is not None and alive(found):
            pid = found
    stop = threading.Event()

    def on_line(line: str) -> None:
        log.append(line + "\n")

    path = telemetry_path(attached.port)
    size = path.stat().st_size if path.is_file() else 0
    # A rolled log keeps its newest bytes. Starting at 0 would replay them.
    offset = before if size >= before else size
    follow(path, on_line, stop, offset=offset)
    return Server(pid, attached.port, max_memory, stop, attached.owned)


def _stop_server(server: Server) -> None:
    """Stop the log follower. SIGTERM the daemon only when this session started it."""
    from favlo.daemon import stop_pid

    if server.follow_stop is not None:
        server.follow_stop.set()
        server.follow_stop = None
    if not server.owned:
        return
    stop_pid(server.pid)
    print(f"stopped 127.0.0.1:{server.port}", flush=True)


def _restart(server: Server, log: Log) -> None:
    """Unload the resident model. The listener and the followed log stay."""
    del log
    from favlo.smoke_lock import unload_model

    print("unloading model", flush=True)
    unload_model(port=server.port)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Instruct file agent: read src/favlo until the prompt reaches a token target."
        )
    )
    parser.add_argument("--tokens", type=int, default=TARGET_TOKENS)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--port", type=int, default=8129)
    parser.add_argument("--model", required=True)
    parser.add_argument("--max-memory", default=None, help="serve memory cap, such as 16GiB")
    parser.add_argument("--mtp", choices=("on", "off"), default=None)
    parser.add_argument("--mtp-depth", type=int, default=3)
    args = parser.parse_args(argv)
    if args.tokens < 1:
        print("token target must be positive")
        return 1
    if args.mtp_depth < 1:
        print("--mtp-depth must be a positive integer")
        return 1
    stem = model_stem(args.model)
    log = Log()
    server: Server | None = None
    mtp = None if args.mtp is None else args.mtp == "on"
    try:
        server = _spawn(log, stem, args.max_memory, args.port or 8129)
        if args.mtp is not None:
            print(f"mtp {args.mtp} depth {args.mtp_depth}", flush=True)
        result = asyncio.run(
            run_file_agent(
                args.repo,
                args.tokens,
                log,
                server,
                model=stem,
                mtp=mtp,
                mtp_depth=args.mtp_depth,
            )
        )
    except Exception as exc:
        print(exc)
        print(log.text()[-3000:])
        return 1
    finally:
        if server is not None:
            _stop_server(server)
    print(format_summary(result.requests, args.tokens))
    if result.note:
        print(result.note)
    if result.reached and result.extended:
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
