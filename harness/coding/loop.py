"""Tool loop against ``serve``. The score is the hidden suite, not this loop."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from harness.benchmarks.serve import Request, completion_body
from harness.coding.workspace import PROMPT, TOOLS, arguments_of, dispatch

# A stuck model ends here. The suite does not score this bound.
SESSION_ROUNDS = 8


def chat_body(model: str, messages: list[dict[str, Any]], request: Request) -> dict[str, object]:
    """One completion body. Sampling comes from the thinking card."""
    body = completion_body(model, messages, None, request)
    body["tools"] = TOOLS
    return body


def run_session(
    port: int,
    model: str,
    request: Request,
    workspace: Path,
) -> None:
    """Let the model write files, then return. The caller grades the workspace."""
    messages: list[dict[str, Any]] = [{"role": "user", "content": PROMPT}]
    for _ in range(SESSION_ROUNDS):
        payload = _post(port, chat_body(model, messages, request))
        message = _assistant(payload)
        messages.append(message)
        calls = message.get("tool_calls")
        if not isinstance(calls, list) or not calls:
            return
        for index, call in enumerate(calls):
            if not isinstance(call, dict):
                continue
            name, arguments = arguments_of(call)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": str(call.get("id") or f"call_{index}"),
                    "content": dispatch(workspace, name, arguments),
                }
            )


def _post(port: int, body: dict[str, object]) -> dict[str, Any]:
    encoded = json.dumps(body).encode()
    http = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=encoded,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(http, timeout=1800) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        message = f"chat completion failed: {exc.code} {detail}"
        raise RuntimeError(message) from exc
    if not isinstance(payload, dict):
        message = "completion response is not an object"
        raise RuntimeError(message)
    return payload


def _assistant(payload: dict[str, Any]) -> dict[str, Any]:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        message = "completion has no choices"
        raise RuntimeError(message)
    message_body = choices[0].get("message")
    if not isinstance(message_body, dict):
        message = "completion message is not an object"
        raise RuntimeError(message)
    item = {"role": "assistant", "content": message_body.get("content") or ""}
    reasoning = message_body.get("reasoning_content")
    if isinstance(reasoning, str) and reasoning:
        item["reasoning_content"] = reasoning
    calls = message_body.get("tool_calls")
    if isinstance(calls, list) and calls:
        item["tool_calls"] = calls
    return item
