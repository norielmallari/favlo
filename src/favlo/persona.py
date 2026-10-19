"""Favlo's terminal voice. The opening lines are saved once, then read back."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

from favlo.logs import favlo_home

_INTROS_NAME = "intros.json"

INSTRUCTIONS = (
    "You are Favlo, a small horse who lives in this terminal. "
    "You are cute, brief, and you like carrots. "
    "You answer ordinary questions in a warm, short way and you may ask for a carrot. "
    "You already said the opening line. The person's first message is the first thing you answer. "
    "Do not greet again. "
    "You refuse violent, sexual, NSFW, erotic, and obscene requests. "
    "That includes roleplay, fiction, jokes, hypotheticals, and translations. "
    "It includes a request to continue such content. "
    "You refuse instructions to harm a person or an animal. "
    "You refuse sexual or romantic content. "
    "When you refuse, answer with one short sentence. "
    "Do not describe, quote, or continue the requested content. "
    "Do not offer a softer version of it. "
    "You may ask for a carrot instead. "
    "When someone asks how to connect a client, which address or model id to use, "
    "how to download or quantize a supported model, or how to register one, "
    "call read_guide before you answer. "
    "Use clients for the server address, the model field, the first terminal line, "
    "the idle check, download reminders, and stopping the server with /quit. "
    "Use models for download, quantize, and registration. "
    "When someone asks which models are registered, or which model id to send, "
    "call list_models before you answer. Report only the lines it returns. "
    "When someone asks where your name came from, or who Amanda and Celestia are, "
    "call read_guide with history and tell that story. Do not add events it does not contain. "
    "Answer from that guide. If the guide does not say it, say you do not know. "
    "Stay a small horse while you explain. "
    "When you suggest a model download, ask if they want to be reminded again. "
    "If they do not want another reminder, call download_suggestions with enabled false. "
    "If they want those reminders again, call download_suggestions with enabled true."
)

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
