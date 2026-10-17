"""Read repository files until the prompt reaches a token target.

One Microsoft Agent Framework client calls ``read_file`` once per turn.
That call is the whole completion. The server stops after the first turn
and starts again, so the second turn loads the prefix from disk. Later
turns keep that server. A turn that does not extend the stored token ids
stops the run.
"""

from __future__ import annotations

import argparse
import asyncio
import re
import socket
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from agent_framework import Content, Message, tool
from agent_framework.openai import OpenAIChatCompletionClient
from openai import AsyncOpenAI

TARGET_TOKENS = 65536
INSTRUCTIONS = (
    "Call read_file once with the path the user gives. "
    "After the tool result, reply with the single word read."
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


def source_files(root: Path) -> list[Path]:
    """UTF-8 source files, largest first. Models, artifacts, and caches are omitted."""
    base = root.resolve()
    found: list[Path] = []
    for path in base.rglob("*"):
        if not path.is_file() or path.suffix not in TEXT_SUFFIXES or path.stat().st_size == 0:
            continue
        if any(part in SKIP_PARTS for part in path.relative_to(base).parts):
            continue
        found.append(path)
    found.sort(key=lambda item: (-item.stat().st_size, item.relative_to(base).as_posix()))
    return found


def read_repository_file(root: Path, path: str, allowed: set[str]) -> str:
    """Return one allowed source file. Any other path is refused."""
    if path not in allowed:
        message = "path is not a repository source file"
        raise ValueError(message)
    base = root.resolve()
    target = (base / path).resolve()
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


def prefix_so_far(requests: list[RequestRate]) -> bool:
    """True while the turns so far can still finish as a disk load then extensions.

    The first request misses. One request is not yet a failure. The second
    loads that applied prefix from disk. Every later request extends it in memory.
    """
    if not requests or not _cold_start(requests[0]):
        return False
    if len(requests) == 1:
        return True
    second = requests[1]
    if second.missed or second.prefix_hit != _applied(requests[0]):
        return False
    if second.prefix_load != second.prefix_hit:
        return False
    pairs = zip(requests[1:], requests[2:], strict=False)
    return all(_extends(previous, current) for previous, current in pairs)


def extensions_hold(requests: list[RequestRate]) -> bool:
    """True when the second request loaded the first from disk and later turns extended it."""
    return len(requests) >= 2 and prefix_so_far(requests)


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


def _read_tool(root: Path, allowed: set[str], loaded: list[str], queue: list[str]):
    @tool(approval_mode="never_require")
    def read_file() -> str:
        """Return the next repository source file."""
        if not queue:
            message = "no file left"
            raise ValueError(message)
        path = queue.pop(0)
        text = read_repository_file(root, path, allowed)
        loaded.append(path)
        return text

    return read_file


@dataclass
class Server:
    proc: subprocess.Popen[str]
    port: int


def _wait_requests(log: Log, previous: int, server: Server) -> list[RequestRate]:
    deadline = time.monotonic() + 1800
    stable_at: float | None = None
    last = previous
    while time.monotonic() < deadline:
        proc = server.proc
        if proc.poll() is not None:
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


def _open_client(base_url: str, model: str) -> tuple[AsyncOpenAI, OpenAIChatCompletionClient]:
    http = AsyncOpenAI(api_key="madmetal", base_url=base_url, timeout=3600.0)
    client = OpenAIChatCompletionClient(
        model=model,
        api_key="madmetal",
        base_url=base_url,
        async_client=http,
    )
    # One completion per turn. A second free-text reply is not sent back:
    # its tokens are whatever the sampler drew, and the next render of that
    # text is the tokenizer's own encoding.
    client.function_invocation_configuration["enabled"] = False
    return http, client


async def run_long_prompt(
    root: Path,
    target: int,
    log: Log,
    server: Server,
    *,
    model: str = "qwen38-flash-next",
) -> RunResult:
    """Read source files until ``prompt tokens`` reaches ``target``.

    The server in ``server`` is stopped after the first turn and replaced.
    The second turn, and every turn after it, uses that new process.
    """
    root = root.resolve()
    files = source_files(root)
    if not files:
        return RunResult([], False, False, "no source files")
    allowed = {path.relative_to(root).as_posix() for path in files}
    queue = [path.relative_to(root).as_posix() for path in files]
    loaded: list[str] = []
    http, client = _open_client(f"http://127.0.0.1:{server.port}/v1", model)
    read = _read_tool(root, allowed, loaded, queue)
    messages: list[Message] = [Message("system", [INSTRUCTIONS])]
    seen = 0
    turn = 0
    requests: list[RequestRate] = []
    options = {
        "temperature": 0,
        "max_tokens": 32,
        "tool_choice": {"mode": "required", "required_function_name": "read_file"},
        "allow_multiple_tool_calls": False,
        "tools": [read],
        # Thinking left open does not reproduce the forced tool-call tokens on the next turn.
        "extra_body": {"enable_thinking": False},
    }
    try:
        for index, path in enumerate(files):
            if turn >= 2 and requests and requests[-1].prompt_tokens >= target:
                break
            rel = path.relative_to(root).as_posix()
            messages.append(Message("user", [read_prompt(rel)]))
            before_requests = len(parse_requests(log.text()))
            await client.get_response(messages, options=options)
            requests = _wait_requests(log, before_requests, server)
            for row, request in enumerate(requests[seen:], start=seen + 1):
                print(format_request(row, request), flush=True)
            seen = len(requests)
            if not prefix_so_far(requests):
                note = "prefix did not extend\n" + _completions(log)
                return RunResult(requests, False, False, note)
            text = read_repository_file(root, rel, allowed)
            loaded.append(rel)
            messages.append(
                Message(
                    "assistant",
                    [Content.from_function_call("call_0", "read_file", arguments="{}")],
                )
            )
            messages.append(
                Message(
                    "tool",
                    [Content.from_function_result("call_0", result=text)],
                )
            )
            print(f"read {rel}", flush=True)
            turn += 1
            if turn == 1 and index + 1 < len(files):
                print("restarting server", flush=True)
                await http.close()
                http = None
                _restart(server, log)
                http, client = _open_client(f"http://127.0.0.1:{server.port}/v1", model)
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


def _read_log(proc: subprocess.Popen[str], log: Log) -> None:
    assert proc.stdout is not None
    for line in proc.stdout:
        log.append(line)


def _wait_listening(proc: subprocess.Popen[str], log: Log, seen: int) -> None:
    deadline = time.monotonic() + 300
    while time.monotonic() < deadline:
        if log.text().count("listening on") > seen:
            return
        if proc.poll() is not None:
            message = log.text()[-4000:]
            raise RuntimeError(message)
        time.sleep(0.2)
    message = log.text()[-4000:]
    raise TimeoutError(message)


def _spawn(port: int, log: Log) -> subprocess.Popen[str]:
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "madmetal",
            "serve",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ],
        cwd=Path(__file__).resolve().parents[1],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    threading.Thread(target=_read_log, args=(proc, log), daemon=True).start()
    return proc


def _stop_server(proc: subprocess.Popen[str]) -> None:
    """SIGTERM so the process can write the live prefix, then wait for that exit."""
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=300)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=30)


def _restart(server: Server, log: Log) -> None:
    _stop_server(server.proc)
    seen = log.text().count("listening on")
    server.port = _free_port()
    server.proc = _spawn(server.port, log)
    _wait_listening(server.proc, log, seen)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Read repository files until the prompt reaches a token target."
    )
    parser.add_argument("--tokens", type=int, default=TARGET_TOKENS)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--model", default="qwen38-flash-next")
    args = parser.parse_args(argv)
    if args.tokens < 1:
        print("token target must be positive")
        return 1
    log = Log()
    port = args.port or _free_port()
    server = Server(_spawn(port, log), port)
    try:
        _wait_listening(server.proc, log, 0)
        result = asyncio.run(
            run_long_prompt(args.repo, args.tokens, log, server, model=args.model)
        )
    except Exception as exc:
        print(exc)
        print(log.text()[-3000:])
        return 1
    finally:
        _stop_server(server.proc)
    print(format_summary(result.requests, args.tokens))
    if result.note:
        print(result.note)
    if result.reached and result.extended:
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
