"""Operator console for ``favlo serve``.

The console reads stdin and does not call a model. ``quit`` stops the daemon.
``reload`` closes every open pipeline and stays attached. ``/quit`` and
``/reload`` are the same commands. Any other line is ignored.
"""

from __future__ import annotations

import json
import sys
import threading
import urllib.error
import urllib.request
from collections.abc import Callable
from typing import Any, TextIO


def is_quit(text: str) -> bool:
    """True when the line stops the server."""
    return text in {"quit", "/quit"}


def is_reload(text: str) -> bool:
    """True when the line reloads every pipeline and stays attached."""
    return text in {"reload", "/reload"}


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
    stop: Callable[[], None] | None = None,
    *,
    prompt: Callable[[bool], None] | None = None,
) -> threading.Thread:
    """Read stdin until it closes or the line is ``quit``."""
    thread = threading.Thread(
        target=_run,
        args=(host, port, stdin, write, stop, prompt),
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
    stop: Callable[[], None] | None,
    prompt: Callable[[bool], None] | None,
) -> None:
    def ask(after_output: bool = False) -> None:
        _prompt(after_output, prompt)

    ask(False)
    while True:
        line = stdin.readline()
        if not line:
            return
        text = line.strip()
        if not text:
            ask(False)
            continue
        if is_reload(text):
            try:
                pipeline_command(host, port, "reload")
                write("reloaded")
            except (OSError, RuntimeError, ValueError) as exc:
                write(str(exc))
            ask(True)
            continue
        if is_quit(text):
            if stop is not None:
                stop()
            return
        ask(False)


def _prompt(after_output: bool = False, show: Callable[[bool], None] | None = None) -> None:
    if show is not None:
        show(after_output)
        return
    if after_output:
        sys.stderr.write("\n")
        sys.stderr.flush()
