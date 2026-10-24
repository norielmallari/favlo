"""Terminal chat for ``favlo serve``.

One Microsoft Agent Framework session talks to the serve daemon's chat
completions endpoint. The session lives on the thread that reads stdin.
A typed line is sent immediately. The model field is the first folder in
the registry. This terminal's own completion keeps the idle check waiting.
The first start has no saved line: one chat completion asks the
registered model to call ``save_intro``. This start prints the first saved
line, as ``favlo:``. Each later start prints the next line, then the first
again. A reply streams as the tokens arrive, and the terminal wraps it.
The daemon's prefill and decode rows stay on the line under that reply.
The model is not called again until someone types. After 30
minutes with no completion from this terminal and no typing, one more
session checks free disk and memory and may start the Qwen3.8-27B download.
Five minutes is a pause in a conversation, and that turn would occupy the
only request slot.
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

from favlo.persona import INSTRUCTIONS, intros_path, opening, rotate_opening, save_intros

Turn = Callable[[str], str]
HEARTBEAT_SECONDS = 30 * 60
HEARTBEAT_QUIET = "HEARTBEAT_QUIET"
_HEARTBEAT_LINE = (
    "This machine can hold Qwen3.8-27B. I started the download. "
    "Should I remind you about other models again?"
)


class Idle:
    """Set while this terminal has no completion in flight."""

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
        """True when this terminal is idle and ``seconds`` have passed."""
        with self._lock:
            if self._count != 0:
                return False
            return self._now() - self._quiet_since >= seconds

    def remaining(self, seconds: float) -> float:
        """Seconds until ``quiet``. A completion in flight reports one second."""
        with self._lock:
            if self._count != 0:
                return 1.0
            left = seconds - (self._now() - self._quiet_since)
            return 0.0 if left < 0 else left


def reply(text: str, turn: Turn) -> str:
    """Answer one line. The model applies its own guardrails."""
    return turn(text)


def is_quit(text: str) -> bool:
    """True when the line stops the server."""
    return text == "/quit"


def is_reload(text: str) -> bool:
    """True when the line reloads every pipeline and stays in the chat."""
    return text == "/reload"


def pipeline_command(host: str, port: int, action: str) -> dict[str, Any]:
    """Ask the daemon to close or reload. ``action`` is ``close`` or ``reload``."""
    url = f"http://{host}:{port}/v1/pipelines/{action}"
    request = urllib.request.Request(
        url,
        data=b"{}",
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=3600) as response:
            body = json.loads(response.read().decode())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace").strip()
        message = detail or f"the pipeline command failed ({exc.code})"
        raise RuntimeError(message) from exc
    if not isinstance(body, dict):
        message = "the pipeline command failed"
        raise RuntimeError(message)
    return body


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
    draft: Callable[[str], None] | None = None,
    prompt: Callable[[bool], None] | None = None,
    hold: Callable[[bool], None] | None = None,
) -> threading.Thread:
    """Read stdin until it closes or the line is ``/quit``.

    A saved opening is printed first. When none is saved, one chat completion
    runs. ``introduce`` replaces the completion in tests.
    """
    thread = threading.Thread(
        target=_run,
        args=(host, port, stdin, write, idle, stop, home, introduce, draft, prompt, hold),
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
    draft: Callable[[str], None] | None,
    prompt: Callable[[bool], None] | None,
    hold: Callable[[bool], None] | None,
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
        return loop.run_until_complete(_once(base, text, held, path, home, pending, draft))

    def beat() -> str:
        text = _heartbeat(base, home)
        if text:
            with notes_lock:
                notes.append(text)
        return text

    def ask(after_output: bool = False) -> None:
        _prompt(after_output, prompt)

    beat_thread = threading.Thread(
        target=heartbeat_loop,
        args=(idle, HEARTBEAT_SECONDS, ended, beat, write, None, ask),
        name="favlo-heartbeat",
        daemon=True,
    )
    try:
        asyncio.set_event_loop(loop)
        shown = _speak(path, base, idle, write, introduce, hold)
        if isinstance(shown, str):
            held["greeting"] = shown
        idle.mark()
        beat_thread.start()
        ask(True)
        while True:
            line = stdin.readline()
            if not line:
                return
            text = line.strip()
            idle.mark()
            if not text:
                ask(False)
                continue
            if is_reload(text):
                try:
                    pipeline_command(host, port, "reload")
                    write(_voice("reloaded"))
                except (OSError, RuntimeError, ValueError) as exc:
                    write(str(exc))
                ask(True)
                continue
            if is_quit(text):
                if stop is not None:
                    stop()
                return
            idle.enter()
            _hold(True, hold)
            try:
                try:
                    write(_voice(reply(text, turn)))
                except Exception as exc:
                    write(str(exc))
            finally:
                idle.leave()
                _hold(False, hold)
            ask(True)
    finally:
        ended.set()
        beat_thread.join(1)
        _finish_loop(loop, held)


def heartbeat_loop(
    idle: Idle,
    interval: float,
    stop: threading.Event,
    run: Callable[[], str],
    write: Callable[[str], None],
    wait: Callable[[float], bool] | None = None,
    ask: Callable[[bool], None] | None = None,
) -> None:
    """Run ``run`` once this terminal has been quiet for ``interval``. Missed ticks coalesce."""
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
            write(_voice(text))
            if ask is not None:
                ask(True)
            else:
                _prompt(True)


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


_INTRO_KEEP = 4

_INTRO_ASK = (
    "This is the first start. Write six opening lines. "
    "Each line is a different introduction: longer than a later reply, flavorful, with emojis, "
    "as Favlo, a small horse who would like a carrot. "
    "No two lines say the same thing. "
    "Call save_intro with those lines. "
    "The next start shows the next line, then the first line again."
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
    hold: Callable[[bool], None] | None = None,
) -> str | None:
    """Print this start's line and leave the next line first for the following start."""
    try:
        line = opening(path)
    except ValueError as exc:
        write(str(exc))
        return None
    if line is None:
        idle.enter()
        _hold(True, hold)
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
                    _hold(False, hold)
                    return None
        finally:
            idle.leave()
    try:
        if not isinstance(line, str) or not line.strip():
            write("the model did not save an introduction")
            return None
        write(_voice(line))
        rotate_opening(path)
        return line.strip()
    finally:
        _hold(False, hold)


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
                    "Greet now. Write six different lines. Each line is its own introduction, "
                    "longer than a later reply, flavorful, and with emojis. "
                    "No two lines say the same thing. "
                    "Do not use violent, sexual, NSFW, erotic, or obscene language, "
                    "and do not describe harm. "
                    "Call save_intro. A later start shows the next line."
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
                                    "Six different opening lines. "
                                    "The first is shown now. A later start shows the next."
                                ),
                                "minItems": _INTRO_KEEP,
                            }
                        },
                        "required": ["lines"],
                    },
                },
            }
        ],
        "tool_choice": {"type": "function", "function": {"name": "save_intro"}},
        "max_tokens": 1200,
        "enable_thinking": False,
    }


def _introduce(base: str, path: Path, root: Path | None = None) -> str:
    """Ask the registered model for the opening and store the tool call."""
    model = chat_model_id(root)
    payload = _post_json(f"{base}/chat/completions", intro_request(model))
    lines = accepted_openings(_intro_lines(payload))
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


def accepted_openings(lines: list[str] | None) -> list[str] | None:
    """Distinct opening lines. One line would repeat on every later start."""
    if not lines:
        return None
    kept: list[str] = []
    seen: set[str] = set()
    for line in lines:
        cleaned = " ".join(line.split())
        key = cleaned.casefold()
        if not key or key in seen:
            continue
        seen.add(key)
        kept.append(cleaned)
    if len(kept) < _INTRO_KEEP:
        return None
    return kept


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

    model = chat_model_id(root)
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
    draft: Callable[[str], None] | None,
) -> str:
    greeting = held.get("greeting")
    agent, session = held.get("pair") or await _pair(
        base,
        path,
        home,
        greeting if isinstance(greeting, str) else None,
    )
    held["pair"] = (agent, session)
    _note(session, pending)
    stream = agent.run(text, session=session, stream=True)
    try:
        parts: list[str] = []
        async for update in stream:
            piece = assistant_piece(update)
            if not piece:
                continue
            parts.append(piece)
            shown = spoken_text("".join(parts))
            if draft is not None and shown:
                draft(_voice(shown))
        final = await stream.get_final_response()
        spoken = spoken_text("".join(parts))
        if spoken:
            return spoken
        text_out = getattr(final, "text", None)
        if isinstance(text_out, str) and text_out.strip():
            return text_out.strip()
        return "I lost that thought. Could you say it again, maybe with a carrot?"
    finally:
        await _close_stream(stream)


_TOOL_OPEN = "<tool_call>"
_TOOL_CLOSE = "</tool_call>"


def spoken_text(text: str) -> str:
    """Reply text with tool-call markup removed. An unfinished call stays hidden."""
    kept: list[str] = []
    rest = text
    while rest:
        start = rest.find(_TOOL_OPEN)
        if start < 0:
            keep = _marker_prefix(rest, _TOOL_OPEN)
            kept.append(rest[:-keep] if keep else rest)
            break
        kept.append(rest[:start])
        end = rest.find(_TOOL_CLOSE, start)
        if end < 0:
            break
        rest = rest[end + len(_TOOL_CLOSE) :]
    return "".join(kept).strip()


def _marker_prefix(text: str, marker: str) -> int:
    limit = min(len(text), len(marker) - 1)
    for size in range(limit, 0, -1):
        if marker.startswith(text[-size:]):
            return size
    return 0


def assistant_piece(update: Any) -> str:
    """Assistant text from one stream update. Tool results stay off the screen."""
    role = getattr(update, "role", None)
    if role not in (None, "assistant"):
        return ""
    text = getattr(update, "text", "")
    return text if isinstance(text, str) else ""


def _voice(text: str) -> str:
    """The line the terminal shows for something Favlo said."""
    body = text.strip()
    if body.startswith("favlo:"):
        return body
    return f"favlo: {body}"


def _note(session: Any, lines: list[str]) -> None:
    """Store idle suggestions so the next reply can turn reminders off."""
    if not lines:
        return
    from agent_framework import InMemoryHistoryProvider, Message

    bucket = session.state.setdefault(InMemoryHistoryProvider.DEFAULT_SOURCE_ID, {})
    messages = bucket.setdefault("messages", [])
    messages.extend(Message("assistant", [line]) for line in lines)


async def _pair(
    base: str, path: Path, home: Path | None, greeting: str | None = None
) -> tuple[Any, Any]:
    from agent_framework import Agent
    from agent_framework.openai import OpenAIChatCompletionClient
    from openai import AsyncOpenAI

    model = chat_model_id()
    http = AsyncOpenAI(api_key="favlo", base_url=base, timeout=3600.0)
    client = OpenAIChatCompletionClient(
        model=model,
        api_key="favlo",
        base_url=base,
        async_client=http,
    )
    client.function_invocation_configuration["max_iterations"] = 6
    agent = Agent(
        client=client,
        name="favlo",
        instructions=INSTRUCTIONS,
        tools=[
            _guide_tool(),
            _models_tool(),
            _suggestions_tool(home),
            _setup_guide_tool(),
            _setup_run_tool(base),
        ],
        default_options={
            "max_tokens": 400,
            "extra_body": {"enable_thinking": False},
        },
    )
    return agent, _with_opening(agent.create_session(), path, greeting)


def _with_opening(session: Any, path: Path, shown: str | None = None) -> Any:
    """Store the printed greeting as the assistant's first turn."""
    from agent_framework import InMemoryHistoryProvider, Message

    line = shown.strip() if isinstance(shown, str) and shown.strip() else opening(path)
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
        check, download reminders, ``/reload``, and ``/quit`` stops the server. models covers
        download, quantize, and registration. history is where the name Favlo came from.
        """
        return load_guide(name)

    return read_guide


def chat_model_id(root: Path | None = None) -> str:
    """The folder the terminal sends as ``model``."""
    from favlo.registry import load_registry

    models = load_registry(root).get("models")
    if isinstance(models, dict):
        for folder in models:
            if isinstance(folder, str) and folder and ":" not in folder:
                return folder
    message = "no model is registered"
    raise RuntimeError(message)


def registered_models(root: Path | None = None) -> str:
    """The registry listing. An empty registry says so."""
    from favlo.registry import format_listing

    listing = format_listing(root).strip()
    if not listing:
        return "no models are registered"
    return listing


def _setup_guide_tool() -> object:
    from typing import Annotated

    from agent_framework import tool
    from pydantic import Field

    from favlo.guides import read_setup as load_setup

    @tool(approval_mode="never_require")
    def read_setup(
        pipeline: Annotated[
            str,
            Field(description="Pipeline id, such as qwen36-35b-a3b, qwen38-27b, or z-image-turbo."),
        ],
    ) -> str:
        """Read one pipeline's setup guide.

        Call this when someone wants to set up, download, quantize, or delete a model.
        qwen36-35b-a3b is Qwen3.6-35B-A3B. qwen38-27b is Qwen3.8-27B.
        z-image-turbo is Z-Image-Turbo.
        The result is that pipeline's setup page.
        A pipeline with no setup page names the ones that have one.
        """
        return load_setup(pipeline)

    return read_setup


def _setup_run_tool(base: str) -> object:
    from typing import Annotated

    from agent_framework import tool
    from pydantic import Field

    from favlo.setup_run import accepted_argv, frees_model, run_setup_command

    @tool(approval_mode="never_require")
    def run_setup(
        command: Annotated[
            str,
            Field(description="One command copied from the setup guide."),
        ],
    ) -> str:
        """Run one setup-guide command in this install.

        The working directory is the folder where Favlo is installed. A path cannot
        leave that folder. This is not a shell: pipes, cd, and other programs are refused.
        """
        try:
            argv = accepted_argv(command, _install())
        except ValueError as exc:
            return str(exc)
        if frees_model(argv):
            closed = _close_resident(base)
            if closed is not None:
                return closed
        try:
            return run_setup_command(command)
        except ValueError as exc:
            return str(exc)

    return run_setup


def _install() -> Path:
    from favlo.cli import repo_root

    return repo_root()


def _close_resident(base: str) -> str | None:
    request = urllib.request.Request(base + "/pipelines/close", data=b"", method="POST")
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            response.read()
    except (OSError, urllib.error.URLError) as exc:
        return f"could not close the open model: {exc}"
    return None


def _models_tool() -> object:
    from agent_framework import tool

    @tool(approval_mode="never_require")
    def list_models() -> str:
        """List the registered models.

        Each line is one record: folder, pipeline, alias, and key/value mode.
        An unregistered complete record is named too. This is the live registry,
        not the download guide.
        """
        return registered_models()

    return list_models


def _hold(active: bool, hold: Callable[[bool], None] | None) -> None:
    if hold is not None:
        hold(active)


def _prompt(after_output: bool = False, show: Callable[[bool], None] | None = None) -> None:
    if show is not None:
        show(after_output)
        return
    sys.stderr.write("you: ")
    sys.stderr.flush()


async def _close_stream(stream: Any) -> None:
    """Release the chat stream. An open generator warns when the loop closes."""
    close = getattr(stream, "close", None)
    if not callable(close):
        return
    result = close()
    if asyncio.iscoroutine(result):
        await result


def _finish_loop(loop: asyncio.AbstractEventLoop, held: dict[str, Any]) -> None:
    """Close the client, then any stream the turn left open, then the loop."""
    try:
        _close(loop, held)
        loop.run_until_complete(loop.shutdown_asyncgens())
    finally:
        loop.close()


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
