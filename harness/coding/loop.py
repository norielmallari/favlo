"""Tool loop against ``serve``. The score is the hidden suite, not this loop."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from harness.benchmarks.serve import Request, completion_body
from harness.coding.session import session_for
from harness.coding.workspace import PROMPT, TOOLS, arguments_of, dispatch, prepare_workspace

# Six tools, then status polls, then a stop. The suite does not score this bound.
SESSION_ROUNDS = 16


def chat_body(
    model: str,
    messages: list[dict[str, Any]],
    request: Request,
    *,
    mtp: bool,
) -> dict[str, object]:
    """Bare model id, ``enable_thinking``, and the MTP flag. The listener fills the thinking card."""
    del request
    body = completion_body(
        model.partition(":")[0],
        messages,
        None,
        Request(enable_thinking=True),
    )
    body["tools"] = TOOLS
    body["mtp"] = bool(mtp)
    return body


def run_session(
    port: int,
    model: str,
    request: Request,
    workspace: Path,
    *,
    mtp: bool,
) -> list[str]:
    """Let the model use the tools, then return the ones that succeeded."""
    prepare_workspace(workspace)
    session = session_for(workspace)
    messages: list[dict[str, Any]] = [{"role": "user", "content": PROMPT}]
    try:
        for _ in range(SESSION_ROUNDS):
            payload = _post(port, chat_body(model, messages, request, mtp=mtp))
            message = _assistant(payload)
            messages.append(message)
            calls = message.get("tool_calls")
            if not isinstance(calls, list) or not calls:
                return list(session.calls)
            for index, call in enumerate(calls):
                if not isinstance(call, dict):
                    continue
                name, arguments = arguments_of(call)
                tool_call_id = str(call.get("id") or f"call_{index}")
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "content": dispatch(
                            workspace,
                            name,
                            arguments,
                            tool_call_id=tool_call_id,
                        ),
                    }
                )
        return list(session.calls)
    finally:
        session.close()


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
