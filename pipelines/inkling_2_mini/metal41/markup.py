"""Inkling 2 Mini completion markup.

A model message is ``<|message_model|>``, an optional name, one content marker,
a body, and ``<|end_message|>``. ``<|content_thinking|>`` is reasoning,
``<|content_text|>`` is the reply, and ``<|content_invoke_tool_json|>`` is one
call ``{"name": ..., "args": {...}}`` whose name also follows the message
marker. ``<|content_model_end_sampling|>`` ends the turn. A body cut by the
length limit has no end marker and still counts.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

HEADER = "<|message_model|>"
THINK = "<|content_thinking|>"
TEXT = "<|content_text|>"
TOOL = "<|content_invoke_tool_json|>"
END = "<|end_message|>"
DONE = "<|content_model_end_sampling|>"
KINDS = {THINK: "reasoning", TEXT: "content", TOOL: "tool"}
_MARK = re.compile(r"<\|[a-z0-9_]+\|>")
_BAD_ESCAPE = re.compile(r'\\(?!["\\/bfnrtu])')
_CALL_OPEN = re.compile(r'\s*\{\s*"name"\s*:\s*"((?:[^"\\]|\\.)*)"\s*,\s*"args"\s*:\s*')


@dataclass
class Segment:
    kind: str
    name: str
    body: str


def segments(text: str) -> list[Segment]:
    """Every content segment in order. Text outside a segment is dropped."""
    found: list[Segment] = []
    kind = ""
    name = ""
    header = False
    body: list[str] = []
    pos = 0
    for match in [*_MARK.finditer(text), None]:
        chunk = text[pos : match.start()] if match else text[pos:]
        if header:
            name += chunk
        elif kind:
            body.append(chunk)
        if match is None:
            break
        pos = match.end()
        mark = match.group()
        if mark in KINDS:
            if kind:
                found.append(Segment(kind, name.strip(), "".join(body)))
            kind = KINDS[mark]
            header = False
            body = []
        elif mark in (HEADER, END, DONE):
            if kind:
                found.append(Segment(kind, name.strip(), "".join(body)))
            kind = ""
            body = []
            header = mark == HEADER
            if header:
                name = ""
    if kind:
        found.append(Segment(kind, name.strip(), "".join(body)))
    return found


def parse_call(body: str, name: str = "") -> tuple[str, dict[str, Any]] | None:
    """One tool body as a name and an argument object, or ``None``."""
    start = body.find("{")
    if start < 0:
        return None
    decoder = json.JSONDecoder(strict=False)
    try:
        payload, _end = decoder.raw_decode(body, start)
    except json.JSONDecodeError:
        try:
            payload, _end = decoder.raw_decode(_BAD_ESCAPE.sub(r"\\\\", body), start)
        except json.JSONDecodeError:
            return None
    if not isinstance(payload, dict):
        return None
    called = payload.get("name")
    if not isinstance(called, str) or not called:
        called = name
    if not called:
        return None
    args = payload.get("args", payload.get("arguments", {}))
    if isinstance(args, str):
        try:
            args = json.loads(args, strict=False) if args.strip() else {}
        except json.JSONDecodeError:
            return None
    if not isinstance(args, dict):
        return None
    return called, args


def parse_completion(text: str) -> tuple[str, str | None, list[tuple[str, dict[str, Any]]]]:
    """Reasoning, the visible reply, and the calls of one completion.

    A completion with no markup is the reply.
    """
    found = segments(text)
    if not found and not _MARK.search(text):
        return "", text or None, []
    reasoning = "".join(item.body for item in found if item.kind == "reasoning")
    visible = "".join(item.body for item in found if item.kind == "content")
    calls = []
    for item in found:
        if item.kind != "tool":
            continue
        call = parse_call(item.body, item.name)
        if call is not None:
            calls.append(call)
    return reasoning, visible or None, calls


def _object_end(text: str) -> int:
    """Index just past the first complete JSON object in ``text``, or -1."""
    depth = 0
    quoted = False
    escaped = False
    for index, char in enumerate(text):
        if quoted:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quoted = False
            continue
        if char == '"':
            quoted = True
        elif char in "{[":
            depth += 1
        elif char in "}]":
            depth -= 1
            if depth == 0:
                return index + 1
        elif depth == 0 and not char.isspace():
            return -1
    return -1


class StreamGate:
    """Turns completion text, fed in pieces, into stream events.

    ``push`` and ``finish`` yield ``(channel, value)``. ``reasoning`` and
    ``content`` carry text. ``tool_start`` carries ``{"index", "name"}`` and
    ``tool_args`` carries ``{"index", "arguments"}``: the argument object as
    the model writes it, one fragment at a time. A call whose JSON does not
    open with ``name`` and then ``args`` streams its arguments at the end.
    """

    def __init__(self) -> None:
        self.text = ""
        self.pos = 0
        self.kind = ""
        self.name = ""
        self.header = False
        self.calls = 0
        self.tool = ""
        self.args_at = -1
        self.sent = 0
        self.started = False
        self.closed = False

    def push(self, piece: str) -> Iterator[tuple[str, Any]]:
        self.text += piece
        yield from self._drain(final=False)

    def finish(self) -> Iterator[tuple[str, Any]]:
        yield from self._drain(final=True)
        if self.kind == "tool":
            yield from self._close_tool()
        self.kind = ""

    def _drain(self, *, final: bool) -> Iterator[tuple[str, Any]]:
        while self.pos < len(self.text):
            rest = self.text[self.pos :]
            match = _MARK.search(rest)
            if match is None:
                cut = len(rest)
                tail = rest.rfind("<")
                if not final and tail >= 0 and "|>" not in rest[tail:]:
                    cut = tail
                if cut == 0:
                    return
                yield from self._body(rest[:cut])
                self.pos += cut
                if cut < len(rest):
                    return
                continue
            if match.start():
                yield from self._body(rest[: match.start()])
            self.pos += match.end()
            yield from self._mark(match.group())

    def _body(self, chunk: str) -> Iterator[tuple[str, Any]]:
        if self.header:
            self.name += chunk
        elif self.kind in ("reasoning", "content"):
            yield self.kind, chunk
        elif self.kind == "tool":
            self.tool += chunk
            yield from self._tool_args()

    def _mark(self, mark: str) -> Iterator[tuple[str, Any]]:
        if mark in KINDS:
            if self.kind == "tool":
                yield from self._close_tool()
            self.kind = KINDS[mark]
            self.header = False
            if self.kind == "tool":
                self.tool = ""
                self.args_at = -1
                self.sent = 0
                self.started = False
                self.closed = False
        elif mark in (HEADER, END, DONE):
            if self.kind == "tool":
                yield from self._close_tool()
            self.kind = ""
            self.header = mark == HEADER
            if self.header:
                self.name = ""

    def _tool_args(self) -> Iterator[tuple[str, Any]]:
        if self.closed:
            return
        if self.args_at < 0:
            match = _CALL_OPEN.match(self.tool)
            if match is None:
                return
            self.args_at = match.end()
            name = json.loads(f'"{match.group(1)}"') or self.name.strip()
            self.started = True
            yield "tool_start", {"index": self.calls, "name": name}
        args = self.tool[self.args_at :]
        end = _object_end(args)
        upto = end if end >= 0 else len(args)
        if upto > self.sent:
            yield "tool_args", {"index": self.calls, "arguments": args[self.sent : upto]}
            self.sent = upto
        if end >= 0:
            self.closed = True

    def _close_tool(self) -> Iterator[tuple[str, Any]]:
        call = parse_call(self.tool, self.name.strip())
        if call is not None:
            if not self.started:
                yield "tool_start", {"index": self.calls, "name": call[0]}
                arguments = json.dumps(call[1], separators=(",", ":"), sort_keys=True)
                yield "tool_args", {"index": self.calls, "arguments": arguments}
            self.calls += 1
        self.kind = ""
        self.tool = ""
