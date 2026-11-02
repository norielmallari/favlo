"""Stream Agent Framework updates as labeled request sections."""

from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Any, TextIO


class TurnPrinter:
    """Print reasoning, the reply, and tool calls for each model call."""

    def __init__(self, sink: TextIO) -> None:
        self.sink = sink
        self.turn = 0
        self.response_id = ""
        self.ended = True
        self.saw_reply = False
        self.section = ""
        self.calls: list[dict[str, str]] = []

    def take(self, update: Any) -> None:
        response_id = getattr(update, "response_id", None) or ""
        if self.ended and _has_body(update):
            self._start(response_id)
        else:
            self._roll(response_id)
        role = getattr(update, "role", None)
        if role in (None, "assistant"):
            reason = _reasoning(update)
            if reason:
                self._reasoning(reason)
            text = getattr(update, "text", "")
            if isinstance(text, str) and text:
                self._reply(text)
        for item in getattr(update, "contents", None) or []:
            if getattr(item, "type", None) == "function_call":
                self._call(item)
        if getattr(update, "finish_reason", None):
            self._end()

    def close(self) -> None:
        self._end()

    def _roll(self, response_id: str) -> None:
        if self.turn == 0:
            self._start(response_id)
            return
        if response_id and self.response_id and response_id != self.response_id:
            self._end()
            self._start(response_id)

    def _start(self, response_id: str) -> None:
        self.turn += 1
        self.response_id = response_id
        self.ended = False
        self.saw_reply = False
        self.section = "reasoning"
        self.calls = []
        print(f"--- request {self.turn} reasoning ---", file=self.sink, flush=True)

    def _reasoning(self, text: str) -> None:
        if self.section != "reasoning":
            print(f"\n--- request {self.turn} reasoning ---", file=self.sink, flush=True)
            self.section = "reasoning"
        print(text, end="", file=self.sink, flush=True)

    def _reply(self, text: str) -> None:
        if not self.saw_reply:
            print(f"\n--- request {self.turn} reply ---", file=self.sink, flush=True)
            self.saw_reply = True
            self.section = "reply"
        print(text, end="", file=self.sink, flush=True)

    def _call(self, item: Any) -> None:
        call_id = getattr(item, "call_id", None) or ""
        if not isinstance(call_id, str):
            call_id = str(call_id)
        found = next((row for row in self.calls if row["id"] == call_id), None) if call_id else None
        if found is None:
            found = {"id": call_id or f"call_{len(self.calls)}", "name": "", "arguments": ""}
            self.calls.append(found)
        name = getattr(item, "name", None) or ""
        if isinstance(name, str) and name:
            found["name"] += name
        arguments = getattr(item, "arguments", None)
        if isinstance(arguments, dict):
            found["arguments"] = json.dumps(arguments)
        elif isinstance(arguments, str) and arguments:
            found["arguments"] += arguments

    def _end(self) -> None:
        if self.ended:
            return
        self.ended = True
        if not self.saw_reply:
            print(f"\n--- request {self.turn} reply ---", file=self.sink, flush=True)
        named = [row for row in self.calls if row["name"]]
        if named:
            print(f"\n--- request {self.turn} tool_calls ---", file=self.sink, flush=True)
            for row in named:
                print(f"{row['name']} {row['arguments']}", file=self.sink, flush=True)
        print(file=self.sink, flush=True)


def print_updates(updates: Iterable[Any], sink: TextIO) -> None:
    """Print one stream of framework updates. Tests use this without a listener."""
    printer = TurnPrinter(sink)
    for update in updates:
        printer.take(update)
    printer.close()


def _has_body(update: Any) -> bool:
    if _reasoning(update):
        return True
    text = getattr(update, "text", "")
    if isinstance(text, str) and text:
        return True
    return any(
        getattr(item, "type", None) == "function_call"
        for item in getattr(update, "contents", None) or []
    )


def _reasoning(update: Any) -> str:
    parts = [
        item.text
        for item in getattr(update, "contents", None) or []
        if getattr(item, "type", None) == "text_reasoning"
        and isinstance(getattr(item, "text", None), str)
        and item.text
    ]
    return "".join(parts)
