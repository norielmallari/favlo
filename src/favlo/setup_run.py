"""Run one setup-guide command inside the Favlo install.

The working directory is the checkout. The command is not a shell.
"""

from __future__ import annotations

import os
import re
import shlex
import subprocess
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from favlo.cli import pipeline_names, repo_root

_TIMEOUT = 7200
_TAIL = 8000
_SUBCOMMANDS = frozenset({"setup", "quantize", "delete", "download", "models"})
_HEAVY = frozenset({"setup", "quantize", "delete", "download"})
_REPO = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$")
_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_CONSOLE = frozenset({"auto", "pretty", "plain", "json"})
_HOSTS = frozenset({"huggingface.co", "www.huggingface.co", "hf.co"})
_SETUP_FLAGS = {
    "--repo": "repo",
    "--name": "name",
    "--expert-fp8": "percent",
    "--keep-bf16": "bool",
    "--console": "console",
}
_QUANTIZE_FLAGS = {
    "--src": "path",
    "--source": "path",
    "--output": "path",
    "--name": "name",
    "--alias": "name",
    "--expert-fp8": "percent",
    "--ffn-fp8": "percent",
    "--keep-bf16": "bool",
}
_DOWNLOAD_FLAGS = {"--check": "bool", "--console": "console"}


def frees_model(argv: Sequence[str]) -> bool:
    """True when the command should close a resident model before it starts."""
    return len(argv) >= 5 and argv[4] in _HEAVY


def accepted_argv(command: str, root: Path) -> list[str]:
    """The argv to run, or ``ValueError`` when the command is not a setup step."""
    try:
        tokens = shlex.split(command, posix=True)
    except ValueError as exc:
        message = "command could not be read"
        raise ValueError(message) from exc
    if not tokens or any(_messy(token) for token in tokens):
        message = "refused: command is not a setup step"
        raise ValueError(message)
    if tokens == ["uv", "sync", "--extra", "dev"]:
        return tokens
    if len(tokens) >= 4 and tokens[:3] == ["uv", "run", "favlo"]:
        return _favlo(tokens[3:], root)
    if tokens and tokens[0] == "favlo":
        return _favlo(tokens[1:], root)
    message = "refused: command is not a setup step"
    raise ValueError(message)


def run_setup_command(
    command: str,
    *,
    root: Path | None = None,
    run: Callable[..., Any] = subprocess.run,
) -> str:
    """Run ``command`` in the install. A refused command raises ``ValueError``."""
    base = (root or repo_root()).resolve()
    argv = accepted_argv(command, base)
    env = os.environ.copy()
    env["FAVLO_ROOT"] = str(base)
    env["FAVLO_SKIP_PLATE"] = "1"
    try:
        completed = run(
            argv,
            cwd=base,
            env=env,
            capture_output=True,
            text=True,
            timeout=_TIMEOUT,
            shell=False,
        )
    except subprocess.TimeoutExpired:
        return "the command timed out"
    except OSError as exc:
        return f"the command did not start: {exc}"
    return _report(completed)


def _favlo(argv: Sequence[str], root: Path) -> list[str]:
    if len(argv) < 2 or not argv[0].startswith("--"):
        message = "refused: command is not a setup step"
        raise ValueError(message)
    name = argv[0][2:]
    if name not in pipeline_names(root):
        message = f"refused: unknown pipeline {name}"
        raise ValueError(message)
    sub = argv[1]
    if sub not in _SUBCOMMANDS:
        message = f"refused: {sub} is not a setup step"
        raise ValueError(message)
    rest = list(argv[2:])
    if sub == "setup":
        _flags(rest, _SETUP_FLAGS, root)
    elif sub == "quantize":
        _flags(rest, _QUANTIZE_FLAGS, root)
    elif sub == "download":
        _flags(rest, _DOWNLOAD_FLAGS, root)
    elif sub == "delete":
        _delete(rest)
    else:
        _models(rest, root)
    return ["uv", "run", "favlo", f"--{name}", sub, *rest]


def _flags(rest: Sequence[str], spec: Mapping[str, str], root: Path) -> None:
    index = 0
    while index < len(rest):
        flag = rest[index]
        kind = spec.get(flag)
        if kind is None:
            message = f"refused: {flag} is not a setup flag"
            raise ValueError(message)
        if kind == "bool":
            index += 1
            continue
        if index + 1 >= len(rest):
            message = f"refused: {flag} needs a value"
            raise ValueError(message)
        value = rest[index + 1]
        if kind == "repo":
            _repo(value)
        elif kind == "name":
            _word(value)
        elif kind == "percent":
            _percent(value)
        elif kind == "console":
            if value not in _CONSOLE:
                message = f"refused: console {value}"
                raise ValueError(message)
        else:
            _inside(root, value)
        index += 2


def _delete(rest: Sequence[str]) -> None:
    if len(rest) != 1:
        message = "refused: delete takes the folder name or the alias"
        raise ValueError(message)
    text = rest[0]
    base, sep, variant = text.partition(":")
    if not _NAME.fullmatch(base) or (sep and not _NAME.fullmatch(variant)):
        message = "refused: delete takes the folder name or the alias"
        raise ValueError(message)


def _models(rest: Sequence[str], root: Path) -> None:
    if not rest:
        return
    command = rest[0]
    if command == "add":
        if len(rest) < 2 or not _NAME.fullmatch(rest[1]):
            message = "refused: models add takes a folder name"
            raise ValueError(message)
        _flags(rest[2:], {"--alias": "name", "--source": "path"}, root)
        return
    if (
        command == "alias"
        and len(rest) == 3
        and _NAME.fullmatch(rest[1])
        and _NAME.fullmatch(rest[2])
    ):
        return
    message = "refused: models lists, adds, or renames an alias"
    raise ValueError(message)


def _repo(text: str) -> None:
    raw = text.strip()
    if "://" in raw or raw.startswith(("huggingface.co/", "www.huggingface.co/", "hf.co/")):
        if "://" not in raw:
            raw = "https://" + raw
        parsed = urlparse(raw)
        host = (parsed.hostname or "").lower()
        if parsed.username or parsed.password or host not in _HOSTS:
            message = "refused: repository link is a huggingface.co model"
            raise ValueError(message)
        parts = [part for part in parsed.path.split("/") if part]
        if len(parts) < 2:
            message = "refused: repository is org/name or a huggingface.co link"
            raise ValueError(message)
        raw = f"{parts[0]}/{parts[1]}"
    if _REPO.fullmatch(raw) is None:
        message = "refused: repository is org/name or a huggingface.co link"
        raise ValueError(message)


def _word(text: str) -> None:
    if _NAME.fullmatch(text) is None:
        message = f"refused: {text} is not a single name"
        raise ValueError(message)


def _percent(text: str) -> None:
    try:
        value = float(text)
    except ValueError as exc:
        message = "refused: expert percentage is not a number"
        raise ValueError(message) from exc
    if value < 0 or value > 100:
        message = "refused: expert percentage is outside 0..100"
        raise ValueError(message)


def _inside(root: Path, text: str) -> None:
    path = Path(text)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        message = "refused: path leaves this install"
        raise ValueError(message)
    resolved = (root / path).resolve()
    base = root.resolve()
    if resolved != base and base not in resolved.parents:
        message = "refused: path leaves this install"
        raise ValueError(message)


def _messy(token: str) -> bool:
    return any(char in token for char in "\n\r;&|$`<>") or token in {"&&", "||"}


def _report(completed: Any) -> str:
    code = int(getattr(completed, "returncode", 1))
    out = _tail(str(getattr(completed, "stdout", "") or ""))
    err = _tail(str(getattr(completed, "stderr", "") or ""))
    lines = [f"exit {code}"]
    if out:
        lines.append(out)
    if err:
        lines.append(err)
    return "\n".join(lines)


def _tail(text: str) -> str:
    if len(text) <= _TAIL:
        return text.rstrip()
    return text[-_TAIL:].rstrip()
