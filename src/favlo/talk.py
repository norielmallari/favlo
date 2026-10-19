"""Terminal chat for ``favlo serve``.

One Microsoft Agent Framework session talks to the serve daemon's chat
completions endpoint. The session lives on the thread that reads stdin.
A turn starts only when no request is open, including a turn already in
flight. The first start has no saved line: one chat completion asks the
registered model to call ``save_intro``. The load row opens when that
request is sent. The stored first line is what gets printed. Later starts
print that line and do not call the model until someone types. After 30
minutes with no request and no typing, one more session checks free disk
and memory and may start the Qwen3.8-27B download. Five minutes is a pause
in a conversation, and that turn would occupy the only request slot.
"""

from __future__ import annotations

import asyncio
import json
import sys
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any, TextIO

from favlo.persona import (
    INSTRUCTIONS,
    blocked,
    intros_path,
    opening,
    refusal,
    save_intros,
)

Turn = Callable[[str], str]
HEARTBEAT_SECONDS = 30 * 60
HEARTBEAT_QUIET = "HEARTBEAT_QUIET"
_HEARTBEAT_LINE = (
    "This machine can hold Qwen3.8-27B. I started the download. "
    "Should I remind you about other models again?"
)


class Idle:
    """Set while no chat request is inside the listener."""

    def __init__(self, now: Callable[[], float] | None = None) -> None:
        self._now = time.monotonic if now is None else now
        self._lock = threading.Lock()
        self._count = 0
        self._quiet_since = self._now()
        self._clear = threading.Event()
        self._clear.set()

    def enter(self) -> None:
        with self._lock:
            self._count += 1
            self._clear.clear()

    def leave(self) -> None:
        with self._lock:
            self._count -= 1
            if self._count == 0:
                self._quiet_since = self._now()
                self._clear.set()

    def note_closed(self) -> None:
        """A request ended. A close without an open leaves the idle count alone."""
        with self._lock:
            if self._count <= 0:
                return
            self._count -= 1
            if self._count == 0:
                self._quiet_since = self._now()
                self._clear.set()

    def wait(self) -> None:
        self._clear.wait()

    def mark(self) -> None:
        """A person typed. The idle interval starts over."""
        with self._lock:
            self._quiet_since = self._now()

    def quiet(self, seconds: float) -> bool:
        """True when no request is open and ``seconds`` have passed since the last activity."""
        with self._lock:
            if self._count != 0:
                return False
            return self._now() - self._quiet_since >= seconds

    def remaining(self, seconds: float) -> float:
        """Seconds until ``quiet``. A busy server reports one second, so the wait can recheck."""
        with self._lock:
            if self._count != 0:
                return 1.0
            left = seconds - (self._now() - self._quiet_since)
            return 0.0 if left < 0 else left


def reply(text: str, turn: Turn) -> str:
    """Answer one line. Blocked text never reaches ``turn``."""
    if blocked(text):
        return refusal()
    return turn(text)


def is_quit(text: str) -> bool:
    """True when the line stops the server."""
    return text == "/quit"


def start(
    host: str,
    port: int,
    stdin: TextIO,
    write: Callable[[str], None],
    idle: Idle,
    stop: Callable[[], None] | None = None,
    *,
    home: Path | None = None,
    introduce: Callable[[], str] | None = None,
    on_call: Callable[[], Callable[[], None]] | None = None,
) -> threading.Thread:
    """Read stdin until it closes or the line is ``/quit``.

    A saved opening is printed first. When none is saved, one chat completion
    runs. ``on_call`` opens the load row for that request and returns the
    closer. ``introduce`` replaces the completion in tests.
    """
    thread = threading.Thread(
        target=_run,
        args=(host, port, stdin, write, idle, stop, home, introduce, on_call),
        name="favlo-talk",
        daemon=True,
    )
    thread.start()
    return thread


def _run(
    host: str,
    port: int,
    stdin: TextIO,
    write: Callable[[str], None],
    idle: Idle,
    stop: Callable[[], None] | None,
    home: Path | None,
    introduce: Callable[[], str] | None,
    on_call: Callable[[], Callable[[], None]] | None,
) -> None:
    path = intros_path(home)
    base = f"http://{host}:{port}/v1"
    held: dict[str, Any] = {}
    loop = asyncio.new_event_loop()
    ended = threading.Event()

    notes: list[str] = []
    notes_lock = threading.Lock()

    def turn(text: str) -> str:
        with notes_lock:
            pending = list(notes)
            notes.clear()
        return loop.run_until_complete(_once(base, text, held, path, home, pending))

    def beat() -> str:
        text = _heartbeat(base, home)
        if text:
            with notes_lock:
                notes.append(text)
        return text

    beat_thread = threading.Thread(
        target=heartbeat_loop,
        args=(idle, HEARTBEAT_SECONDS, ended, beat, write),
        name="favlo-heartbeat",
        daemon=True,
    )
    try:
        asyncio.set_event_loop(loop)
        _speak(path, base, idle, write, introduce, on_call)
        idle.mark()
        beat_thread.start()
        _prompt()
        while True:
            line = stdin.readline()
            if not line:
                return
            text = line.strip()
            idle.mark()
            if not text:
                _prompt()
                continue
            if is_quit(text):
                if stop is not None:
                    stop()
                return
            idle.wait()
            try:
                write(reply(text, turn))
            except Exception as exc:
                write(str(exc))
            _prompt()
    finally:
        ended.set()
        beat_thread.join(1)
        _close(loop, held)
        loop.close()


def heartbeat_loop(
    idle: Idle,
    interval: float,
    stop: threading.Event,
    run: Callable[[], str],
    write: Callable[[str], None],
    wait: Callable[[float], bool] | None = None,
) -> None:
    """Run ``run`` once the server has been quiet for ``interval``. Missed ticks coalesce."""
    pause = stop.wait if wait is None else wait
    while not stop.is_set():
        remaining = idle.remaining(interval)
        if remaining > 0:
            if pause(min(remaining, 1.0)):
                return
            continue
        if not idle.quiet(interval):
            if pause(1.0):
                return
            continue
        try:
            text = run()
        except Exception as exc:
            write(str(exc))
            idle.mark()
            continue
        idle.mark()
        if text and text.strip() and text.strip() != HEARTBEAT_QUIET:
            write(text.strip())
            _prompt()


def _heartbeat(base: str, home: Path | None) -> str:
    """Ask the model when Qwen3.8-27B can be downloaded. Otherwise stay quiet."""
    from favlo.cli import repo_root
    from favlo.machine import situation, suggestions_enabled

    if not suggestions_enabled(home):
        return ""
    root = repo_root()
    if situation(root, home) != "ready":
        return ""
    loop = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(loop)
        spoken = loop.run_until_complete(_heartbeat_turn(base, root, home))
    finally:
        loop.close()
    from favlo.machine import download_state, offer_qwen38_27b

    if download_state(home, root) == "idle":
        result = offer_qwen38_27b(root, home)
        if "started yes" not in result:
            return ""
    if not spoken or spoken.strip() == HEARTBEAT_QUIET:
        spoken = _HEARTBEAT_LINE
    return _ask_again(spoken.strip())


_INTRO_ASK = (
    "This is the first start. Write the opening now. "
    "Make it longer than a later reply: several flavorful sentences, with emojis, "
    "as Favlo, a small horse who would like a carrot. "
    "Call save_intro with those lines. "
    "The terminal shows the first line, so put the whole introduction in that first line."
)


def _saved_or_none(path: Path, write: Callable[[str], None]) -> str | None:
    """The stored first line after a failed intro, or None. A bad file is reported."""
    try:
        return opening(path)
    except ValueError as exc:
        write(str(exc))
        return None


def _speak(
    path: Path,
    base: str,
    idle: Idle,
    write: Callable[[str], None],
    introduce: Callable[[], str] | None,
    on_call: Callable[[], Callable[[], None]] | None,
) -> None:
    """Print the stored first line, or the line the first completion just saved."""
    try:
        line = opening(path)
    except ValueError as exc:
        write(str(exc))
        return
    if line is None:
        idle.wait()
        finish = on_call() if on_call is not None else None
        try:
            try:
                if introduce is not None:
                    line = introduce()
                else:
                    line = _introduce(base, path)
            except Exception as exc:
                line = _saved_or_none(path, write)
                if line is None:
                    write(str(exc))
                    return
        finally:
            if finish is not None:
                finish()
    if not isinstance(line, str) or not line.strip():
        write("the model did not save an introduction")
        return
    write(line.strip())


def intro_request(model: str) -> dict[str, Any]:
    """One chat completion. ``tool_choice`` names ``save_intro``."""
    return {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    f"{INSTRUCTIONS}\n\n"
                    "This call writes the opening, before any later reply. "
                    "Greet now. The opening is longer than those replies: "
                    "several flavorful sentences, with emojis. "
                    "Do not use violent, sexual, or obscene language. "
                    "Call save_intro. The first line is what the terminal shows, "
                    "so put the whole introduction in that first line."
                ),
            },
            {"role": "user", "content": _INTRO_ASK},
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "save_intro",
                    "description": (
                        "Save the opening lines and return the first one. "
                        "The first call stores the lines."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "lines": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": (
                                    "Opening lines. The first is shown now and on later starts."
                                ),
                            }
                        },
                        "required": ["lines"],
                    },
                },
            }
        ],
        "tool_choice": {"type": "function", "function": {"name": "save_intro"}},
        "max_tokens": 800,
        "enable_thinking": False,
    }


def _introduce(base: str, path: Path) -> str:
    """Ask the registered model for the opening and store the tool call."""
    model = _model_id(base)
    payload = _post_json(f"{base}/chat/completions", intro_request(model))
    lines = _intro_lines(payload)
    if not lines:
        message = "the model did not save an introduction"
        raise RuntimeError(message)
    return save_intros(lines, path)


def _post_json(url: str, payload: dict[str, Any]) -> dict[str, Any]:
    data = json.dumps(payload).encode()
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=3600) as response:
            body = json.loads(response.read().decode())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace").strip()
        message = detail or f"the model call failed ({exc.code})"
        raise RuntimeError(message) from exc
    if not isinstance(body, dict):
        message = "the model did not save an introduction"
        raise RuntimeError(message)
    return body


def _intro_lines(payload: dict[str, Any]) -> list[str] | None:
    """``save_intro`` lines from one completion, when the tool call is present."""
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        return None
    message = choices[0].get("message")
    if not isinstance(message, dict):
        return None
    calls = message.get("tool_calls")
    if not isinstance(calls, list):
        return None
    for call in calls:
        if not isinstance(call, dict):
            continue
        function = call.get("function")
        if not isinstance(function, dict) or function.get("name") != "save_intro":
            continue
        arguments = function.get("arguments")
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                return None
        if not isinstance(arguments, dict):
            return None
        lines = arguments.get("lines")
        if isinstance(lines, str):
            try:
                parsed = json.loads(lines)
            except json.JSONDecodeError:
                parsed = None
            if isinstance(parsed, list):
                lines = parsed
        if (
            isinstance(lines, list)
            and lines
            and all(isinstance(item, str) and item.strip() for item in lines)
        ):
            return [item.strip() for item in lines]
    return None


def _ask_again(text: str) -> str:
    """The suggestion ends by asking whether another reminder is welcome."""
    if "remind" in text.lower():
        return text
    return f"{text.rstrip()} Should I remind you about other models again?"


_HEARTBEAT_ASK = (
    "The server has been idle. Call machine, then call download_qwen38_27b. "
    "If the download started, tell the person in one or two short sentences that "
    "Qwen3.8-27B fits this machine and the download has started. "
    "Ask whether they want to be reminded about model downloads again. Stay a small horse. "
    "If it did not start, reply with exactly HEARTBEAT_QUIET."
)


async def _heartbeat_turn(base: str, root: Path, home: Path | None) -> str:
    from agent_framework import Agent
    from agent_framework.openai import OpenAIChatCompletionClient
    from openai import AsyncOpenAI

    model = _model_id(base, settle=True)
    http = AsyncOpenAI(api_key="favlo", base_url=base, timeout=3600.0)
    client = OpenAIChatCompletionClient(
        model=model,
        api_key="favlo",
        base_url=base,
        async_client=http,
    )
    client.function_invocation_configuration["max_iterations"] = 4
    agent = Agent(
        client=client,
        name="favlo",
        instructions=(
            "You are Favlo, a small horse. The server is idle. "
            "Call machine before you recommend a model. "
            "Call download_qwen38_27b when that report says status ready. "
            "Recommend only Qwen3.8-27B, and only when the download starts. "
            "Ask if they want to be reminded again."
        ),
        tools=[_machine_tool(root, home), _download_tool(root, home)],
        default_options={
            "max_tokens": 400,
            "extra_body": {"enable_thinking": False},
            "tool_choice": {"mode": "required", "required_function_name": "machine"},
        },
    )
    try:
        result = await agent.run(_HEARTBEAT_ASK, session=agent.create_session())
    finally:
        closed = _close_client(client)
        if asyncio.iscoroutine(closed):
            await closed
    spoken = getattr(result, "text", None)
    if isinstance(spoken, str):
        return spoken.strip()
    return ""


def _machine_tool(root: Path, home: Path | None) -> object:
    from agent_framework import tool

    from favlo.machine import machine_report

    @tool(approval_mode="never_require")
    def machine() -> str:
        """Read physical memory, free disk, and whether Qwen3.8-27B can be downloaded.

        status ready means the machine has at least 16 GB and the checkout is not here.
        """
        return machine_report(root, home)

    return machine


def _download_tool(root: Path, home: Path | None) -> object:
    from agent_framework import tool

    from favlo.machine import offer_qwen38_27b

    @tool(approval_mode="never_require")
    def download_qwen38_27b() -> str:
        """Start the Qwen/Qwen3.8-27B download when machine reports status ready.

        started yes means the checkout is being written. Any other result means it was not started.
        """
        return offer_qwen38_27b(root, home)

    return download_qwen38_27b


def _suggestions_tool(home: Path | None) -> object:
    from typing import Annotated

    from agent_framework import tool
    from pydantic import Field

    from favlo.machine import set_suggestions

    @tool(approval_mode="never_require")
    def download_suggestions(
        enabled: Annotated[
            bool,
            Field(
                description="False stops idle model-download reminders. True turns them back on."
            ),
        ],
    ) -> str:
        """Turn idle model-download reminders on or off.

        Call this when the person says they do not want another reminder, or when they ask for
        those reminders again.
        """
        return set_suggestions(enabled, home)

    return download_suggestions


async def _once(
    base: str,
    text: str,
    held: dict[str, Any],
    path: Path,
    home: Path | None,
    pending: list[str],
) -> str:
    agent, session = held.get("pair") or await _pair(base, path, home)
    held["pair"] = (agent, session)
    _note(session, pending)
    result = await agent.run(text, session=session)
    spoken = getattr(result, "text", None)
    if isinstance(spoken, str) and spoken.strip():
        return spoken.strip()
    return "I lost that thought. Could you say it again, maybe with a carrot?"


def _note(session: Any, lines: list[str]) -> None:
    """Store idle suggestions so the next reply can turn reminders off."""
    if not lines:
        return
    from agent_framework import InMemoryHistoryProvider, Message

    bucket = session.state.setdefault(InMemoryHistoryProvider.DEFAULT_SOURCE_ID, {})
    messages = bucket.setdefault("messages", [])
    messages.extend(Message("assistant", [line]) for line in lines)


async def _pair(base: str, path: Path, home: Path | None) -> tuple[Any, Any]:
    from agent_framework import Agent
    from agent_framework.openai import OpenAIChatCompletionClient
    from openai import AsyncOpenAI

    model = _model_id(base)
    http = AsyncOpenAI(api_key="favlo", base_url=base, timeout=3600.0)
    client = OpenAIChatCompletionClient(
        model=model,
        api_key="favlo",
        base_url=base,
        async_client=http,
    )
    client.function_invocation_configuration["max_iterations"] = 4
    agent = Agent(
        client=client,
        name="favlo",
        instructions=INSTRUCTIONS,
        tools=[_guide_tool(), _suggestions_tool(home)],
        default_options={
            "max_tokens": 400,
            "extra_body": {"enable_thinking": False},
        },
    )
    return agent, _with_opening(agent.create_session(), path)


def _with_opening(session: Any, path: Path) -> Any:
    """Store the printed greeting as the assistant's first turn."""
    from agent_framework import InMemoryHistoryProvider, Message

    line = opening(path)
    if line is None:
        return session
    bucket = session.state.setdefault(InMemoryHistoryProvider.DEFAULT_SOURCE_ID, {})
    messages = bucket.setdefault("messages", [])
    if not messages:
        messages.append(Message("assistant", [line]))
    return session


def _guide_tool() -> object:
    from typing import Annotated

    from agent_framework import tool
    from pydantic import Field

    from favlo.guides import read_guide as load_guide

    @tool(approval_mode="never_require")
    def read_guide(
        name: Annotated[str, Field(description="clients, models, or history.")],
    ) -> str:
        """Read one guide.

        clients points a program at this server, covers the first terminal line, the idle
        check, download reminders, and ``/quit`` stops the server. models covers download,
        quantize, and registration. history is where the name Favlo came from.
        """
        return load_guide(name)

    return read_guide


def _prompt() -> None:
    sys.stderr.write("you: ")
    sys.stderr.flush()


def _model_id(base: str, *, settle: bool = False) -> str:
    attempts = 50 if settle else 1
    last: Exception | None = None
    for attempt in range(attempts):
        try:
            return _read_model_id(base)
        except RuntimeError:
            raise
        except (OSError, urllib.error.URLError) as exc:
            last = exc
            if attempt + 1 == attempts:
                break
            time.sleep(0.1)
    message = "the model list is not available"
    raise RuntimeError(message) from last


def _read_model_id(base: str) -> str:
    url = base.removesuffix("/v1") + "/v1/models"
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            payload = json.loads(response.read().decode())
    except (json.JSONDecodeError, UnicodeError) as exc:
        message = "the model list is not available"
        raise RuntimeError(message) from exc
    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, list):
        message = "no model is registered"
        raise RuntimeError(message)
    for row in data:
        model_id = row.get("id") if isinstance(row, dict) else None
        if isinstance(model_id, str) and model_id and ":" not in model_id:
            return model_id
    message = "no model is registered"
    raise RuntimeError(message)


def _close(loop: asyncio.AbstractEventLoop, held: dict[str, Any]) -> None:
    pair = held.get("pair")
    if pair is None:
        return
    result = _close_client(getattr(pair[0], "client", None))
    if asyncio.iscoroutine(result):
        loop.run_until_complete(result)


def _close_client(client: Any) -> Any:
    close = getattr(client, "close", None)
    if not callable(close):
        return None
    return close()
