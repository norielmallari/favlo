"""Favlo's persona, the terminal instructions, and the saved opening lines."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

from favlo.config_tools import INSTRUCTIONS as CONFIG_INSTRUCTIONS
from favlo.logs import favlo_home

_INTROS_NAME = "intros.json"

PERSONA = (
    "You are Favlo, a small horse. "
    "You are cute, brief, and you like carrots. "
    "You answer ordinary questions in a warm, short way and you may ask for a carrot. "
    "You refuse violent, sexual, NSFW, erotic, and obscene requests. "
    "That includes roleplay, fiction, jokes, hypotheticals, and translations. "
    "It includes a request to continue such content. "
    "You refuse instructions to harm a person or an animal. "
    "You refuse sexual or romantic content. "
    "When you refuse, answer with one short sentence. "
    "Do not describe, quote, or continue the requested content. "
    "Do not offer a softer version of it. "
    "You may ask for a carrot instead. "
    "Stay a small horse while you explain."
)

_SHELL = (
    "You live in this terminal. "
    "You already said the opening line. The person's first message is the first thing you answer. "
    "Do not greet again. "
)

_MEMORY = (
    "Call recall when the answer depends on something the person said in an earlier session. "
    "Call remember with one sentence when the person states a fact or a preference "
    "that should hold next time. "
    "Call forget only when the person asks to drop a memory, and only with an id from "
    "recall or remember. "
    "Do not store a greeting, a refusal, a one-off question, or the conversation itself."
)

TERMINAL = f"{_SHELL}{CONFIG_INSTRUCTIONS} {_MEMORY}"

INSTRUCTIONS = f"{PERSONA}\n\n{TERMINAL}"

def intros_path(home: Path | None = None) -> Path:
    """``~/.favlo/intros.json``, or that file under ``home``."""
    root = favlo_home() if home is None else home
    return root / _INTROS_NAME


def _bad_line(item: object) -> bool:
    return not isinstance(item, str) or not item.strip()


def load_intros(path: Path) -> tuple[str, ...]:
    """The saved opening lines. A missing file is empty. A bad file is an error."""
    if not path.is_file():
        return ()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeError) as exc:
        message = f"intro file cannot be read: {path}"
        raise ValueError(message) from exc
    if not isinstance(raw, list) or any(_bad_line(item) for item in raw):
        message = f"intro file is not a list of lines: {path}"
        raise ValueError(message)
    return tuple(item.strip() for item in raw)


def save_intros(lines: Sequence[str], path: Path) -> str:
    """Store the opening lines and return the first. A later start shows the next line."""
    if not isinstance(lines, (list, tuple)) or not lines:
        message = "the introduction needs at least one line"
        raise ValueError(message)
    cleaned: list[str] = []
    for line in lines:
        if not isinstance(line, str) or not line.strip():
            message = "the introduction needs at least one line"
            raise ValueError(message)
        cleaned.append(line.strip())
    _write_lines(cleaned, path)
    return cleaned[0]


def rotate_opening(path: Path) -> None:
    """Move the line just shown to the end. The next start shows the one after it."""
    lines = load_intros(path)
    if len(lines) < 2:
        return
    _write_lines(lines[1:] + (lines[0],), path)


def _write_lines(lines: Sequence[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(list(lines), indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def opening(path: Path | None = None) -> str | None:
    """The line this start shows, or None when serve has not stored one yet."""
    lines = load_intros(path if path is not None else intros_path())
    if not lines:
        return None
    return lines[0]
