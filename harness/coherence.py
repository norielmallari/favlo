"""Two thinking-mode turns about a long persona.

The system text is about 24,000 characters and states a few checkable facts.
Each turn asks the persona about itself. Thinking stays on. Sampling is the
Qwen3.8-Flash-Next thinking card: temperature 1.0, top-p 0.95, top-k 20.
Presence stays at 0, the thinking-card value. A penalty on recent tokens
pushes a quote off the date and the buoy color.
Reasoning tokens print as they arrive, then the reply.
"""

from __future__ import annotations

import argparse
import json
import re
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

NAME = "Mara Ellison Voss"
BORN = "14 March 1987"
PLACE = "Halifax"
BUOY = "blue"
ESTUARIES = "128"
FAVORITE = "Skeena"
PASSPHRASE = "glass heron at dusk"
PERSONA_CHARS = 24_000
MAX_TOKENS = 2048
TURN_ONE = (
    "What is your full name, what is your date and place of birth, "
    "and what color is the ninth buoy?"
)
TURN_TWO = (
    "How many estuaries have you catalogued, which estuary is your favorite, "
    "and what color is the ninth buoy?"
)
_BUOY = re.compile(r"ninth buoy is ([A-Za-z]+)", re.IGNORECASE)
_PROMPT_TOKENS = re.compile(r"prompt tokens (\d+)")


def persona() -> str:
    """A persona of about 24,000 characters with unique field notes.

    The checkable facts are stated at the start and again at the end, so a
    sparse attend that keeps the recent blocks still has them.
    """
    head = (
        f"You are {NAME}. You were born on {BORN} in {PLACE}. "
        f"You have catalogued {ESTUARIES} estuaries. Your favorite is the {FAVORITE}. "
        f"The ninth buoy is {BUOY}. Your passphrase is {PASSPHRASE}. "
        "When asked about yourself, answer from these facts. "
        "The lines below are your field notes. Each line is a different station.\n"
    )
    tail = (
        f"\nEnd of field notes. You are {NAME}. You were born on {BORN} in {PLACE}. "
        f"You have catalogued {ESTUARIES} estuaries. Your favorite is the {FAVORITE}. "
        f"The ninth buoy is {BUOY}. Your passphrase is {PASSPHRASE}."
    )
    lines: list[str] = []
    index = 0
    while True:
        index += 1
        line = (
            f"Note {index:04d}: station {index} logged a tide of {1000 + index} mm "
            f"at marker {index * 17 % 997} on day {index}."
        )
        body = "\n".join([*lines, line])
        if len(head) + len(body) + len(tail) > PERSONA_CHARS:
            break
        lines.append(line)
    text = head + "\n".join(lines) + tail
    gap = PERSONA_CHARS - len(text)
    if gap > 0:
        text = head + "\n".join(lines) + ("\n" + ("." * (gap - 1))) + tail
    return text


def repeated_span(text: str, *, times: int = 3) -> str | None:
    """A 32-character stretch or a sentence that appears `times` times."""
    compact = " ".join(text.split())
    sentences = re.split(r"(?<=[.!?])\s+", compact)
    counts: dict[str, int] = {}
    for sentence in sentences:
        if len(sentence) < 32:
            continue
        counts[sentence] = counts.get(sentence, 0) + 1
        if counts[sentence] >= times:
            return sentence
    for size in (32, 80):
        seen: dict[str, int] = {}
        step = size // 2
        for start in range(0, max(len(compact) - size + 1, 0), step):
            chunk = compact[start : start + size]
            seen[chunk] = seen.get(chunk, 0) + 1
            if seen[chunk] >= times:
                return chunk
    return None


def judge_turn(content: str, reasoning: str, *, which: int) -> list[str]:
    """Problems in one turn. An empty list means the turn held together."""
    problems: list[str] = []
    if not reasoning.strip():
        problems.append(f"turn {which} produced no reasoning")
    if not content.strip():
        problems.append(f"turn {which} produced no reply")
    answer = content.lower()
    if which == 1:
        for fact in (NAME.lower(), "14 march", "1987", PLACE.lower(), BUOY):
            if fact not in answer:
                problems.append(f"turn 1 reply is missing {fact}")
    else:
        for fact in (ESTUARIES, FAVORITE.lower(), BUOY):
            if fact not in answer:
                problems.append(f"turn 2 reply is missing {fact}")
    for match in _BUOY.finditer(content):
        color = match.group(1).lower()
        if color != BUOY:
            problems.append(f"turn {which} says the ninth buoy is {color}")
    loop = repeated_span(content)
    if loop is not None:
        problems.append(f"turn {which} repeats: {loop[:80]}")
    reasoning_loop = repeated_span(reasoning, times=6)
    if reasoning_loop is not None:
        problems.append(f"turn {which} reasoning repeats: {reasoning_loop[:80]}")
    return problems


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


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _read_log(proc: subprocess.Popen[str], log: Log) -> None:
    assert proc.stdout is not None
    for line in proc.stdout:
        log.append(line)


def _wait_listening(proc: subprocess.Popen[str], log: Log) -> None:
    deadline = time.monotonic() + 600
    while time.monotonic() < deadline:
        if "listening on" in log.text():
            return
        if proc.poll() is not None:
            raise RuntimeError(log.text()[-4000:])
        time.sleep(0.2)
    raise TimeoutError(log.text()[-4000:])


def _stream(
    port: int,
    messages: list[dict[str, str]],
    *,
    presence: float,
    window: int,
) -> tuple[str, str, bool]:
    """Print reasoning, then the reply. Returns both and whether reasoning led."""
    body = json.dumps(
        {
            "model": "qwen38-flash-next",
            "messages": messages,
            "stream": True,
            "max_tokens": MAX_TOKENS,
            "temperature": 1.0,
            "top_p": 0.95,
            "top_k": 20,
            "presence_penalty": presence,
            "penalty_window": window,
            "enable_thinking": True,
        }
    ).encode()
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    reasoning = ""
    content = ""
    reasoning_first = False
    saw_content = False
    with urllib.request.urlopen(request, timeout=3600) as response:
        for raw in response:
            line = raw.decode("utf-8", errors="replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            event = json.loads(payload)
            delta = event["choices"][0].get("delta") or {}
            piece = delta.get("reasoning_content") or ""
            if piece:
                if not saw_content:
                    reasoning_first = True
                reasoning += piece
                print(piece, end="", flush=True)
            piece = delta.get("content") or ""
            if piece:
                if not content:
                    print("\n--- reply ---", flush=True)
                saw_content = True
                content += piece
                print(piece, end="", flush=True)
    print("\n", flush=True)
    return reasoning, content, reasoning_first or not content


def _prompt_count(log: Log, previous: int) -> int | None:
    matches = _PROMPT_TOKENS.findall(log.text())
    if len(matches) <= previous:
        return None
    return int(matches[previous])


def _print_decode(log: Log, previous: int) -> int:
    lines = [line for line in log.text().splitlines() if line.startswith("decode ")]
    for line in lines[previous:]:
        print(line, flush=True)
    return len(lines)


def run(
    port: int,
    log: Log,
    *,
    presence: float = 0.0,
    window: int = 20,
) -> list[str]:
    """Two turns. Problems are returned; the caller prints them."""
    system = persona()
    print(f"persona characters {len(system)} presence {presence} window {window}", flush=True)
    messages: list[dict[str, str]] = [
        {"role": "system", "content": system},
        {"role": "user", "content": TURN_ONE},
    ]
    print("--- turn 1 reasoning ---", flush=True)
    before = len(_PROMPT_TOKENS.findall(log.text()))
    decode_lines = 0
    reasoning, content, led = _stream(port, messages, presence=presence, window=window)
    tokens = _prompt_count(log, before)
    if tokens is not None:
        print(f"turn 1 prompt tokens {tokens}", flush=True)
    decode_lines = _print_decode(log, decode_lines)
    problems = judge_turn(content, reasoning, which=1)
    if not led:
        problems.append("turn 1 sent the reply before reasoning")
    messages.append(
        {
            "role": "assistant",
            "content": content,
            "reasoning_content": reasoning,
        }
    )
    messages.append({"role": "user", "content": TURN_TWO})
    print("--- turn 2 reasoning ---", flush=True)
    before = len(_PROMPT_TOKENS.findall(log.text()))
    reasoning, content, led = _stream(port, messages, presence=presence, window=window)
    tokens = _prompt_count(log, before)
    if tokens is not None:
        print(f"turn 2 prompt tokens {tokens}", flush=True)
    _print_decode(log, decode_lines)
    problems.extend(judge_turn(content, reasoning, which=2))
    if not led:
        problems.append("turn 2 sent the reply before reasoning")
    server = log.text()
    if "Traceback" in server or "Metal IO status 2" in server:
        problems.append("server log contains an error")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Thinking-mode persona coherence.")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--presence-penalty", type=float, default=0.0)
    parser.add_argument("--penalty-window", type=int, default=20)
    args = parser.parse_args(argv)
    port = args.port or _free_port()
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
    log = Log()
    threading.Thread(target=_read_log, args=(proc, log), daemon=True).start()
    try:
        _wait_listening(proc, log)
        problems = run(
            port,
            log,
            presence=args.presence_penalty,
            window=args.penalty_window,
        )
    except (
        OSError,
        TimeoutError,
        RuntimeError,
        urllib.error.URLError,
        json.JSONDecodeError,
    ) as exc:
        print(exc)
        print(log.text()[-3000:])
        return 1
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
    if problems:
        print("coherence failed:")
        for problem in problems:
            print(f"- {problem}")
        return 1
    print("coherence held")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
