"""MCP client for the media harnesses.

``server/discover`` is the 2026-07-28 probe. A generate call advertises
``io.modelcontextprotocol/tasks``, then polls ``tasks/get`` and reads the
``favlo://`` link. A dropped connection is one ``RuntimeError``. Judge turns
stay on ``POST /v1/chat/completions``.
"""

from __future__ import annotations

import base64
import json
import time
import urllib.error
import urllib.request
from typing import Any

_HOST = "127.0.0.1"


def call_tool(port: int, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Run one immediate MCP tool and return the tool result object."""
    _post(port, "server/discover", {})
    return _post(
        port,
        "tools/call",
        {"name": name, "arguments": arguments},
        tasks=False,
        header_name=name,
    )


def generate(port: int, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Run one generate tool and return the completed tool result."""
    _post(port, "server/discover", {})
    created = _post(
        port,
        "tools/call",
        {"name": name, "arguments": arguments},
        tasks=True,
        header_name=name,
    )
    task_id = str(created["taskId"])
    deadline = time.monotonic() + 3600
    while time.monotonic() < deadline:
        current = _post(port, "tasks/get", {"taskId": task_id}, tasks=True)
        status = current.get("status")
        if status == "completed":
            result = current.get("result")
            if isinstance(result, dict) and result.get("isError"):
                message = json.dumps(result)
                raise RuntimeError(message)
            if not isinstance(result, dict):
                message = "the task completed without a result"
                raise RuntimeError(message)
            return result
        if status in {"failed", "cancelled"}:
            message = json.dumps(current)
            raise RuntimeError(message)
        time.sleep(2)
    message = f"task {task_id} did not finish"
    raise RuntimeError(message)


def read_link(port: int, result: dict[str, Any], mime: str) -> bytes:
    """Read the first ``resource_link`` of ``mime``."""
    content = result.get("content")
    if not isinstance(content, list):
        message = "the task result has no content"
        raise RuntimeError(message)
    for block in content:
        if not isinstance(block, dict) or block.get("type") != "resource_link":
            continue
        if block.get("mimeType") != mime:
            continue
        uri = block.get("uri")
        if isinstance(uri, str):
            return read_uri(port, uri)
    message = f"the task result has no {mime} link"
    raise RuntimeError(message)


def read_uri(port: int, uri: str) -> bytes:
    body = _post(port, "resources/read", {"uri": uri}, header_name=uri)
    contents = body.get("contents")
    if not isinstance(contents, list) or not contents:
        message = "resources/read returned no contents"
        raise RuntimeError(message)
    blob = contents[0].get("blob") if isinstance(contents[0], dict) else None
    if not isinstance(blob, str):
        message = "resources/read returned no blob"
        raise RuntimeError(message)
    return base64.b64decode(blob)


def _post(
    port: int,
    method: str,
    params: dict[str, Any],
    *,
    tasks: bool = False,
    header_name: str | None = None,
) -> dict[str, Any]:
    extensions: dict[str, Any] = {}
    if tasks:
        extensions["io.modelcontextprotocol/tasks"] = {}
    body = dict(params)
    body["_meta"] = {
        "io.modelcontextprotocol/protocolVersion": "2026-07-28",
        "io.modelcontextprotocol/clientCapabilities": {"extensions": extensions},
        "io.modelcontextprotocol/clientInfo": {"name": "favlo-harness", "version": "0"},
    }
    payload = {"jsonrpc": "2.0", "id": 1, "method": method, "params": body}
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
        "mcp-protocol-version": "2026-07-28",
        "mcp-method": method,
    }
    if header_name is not None:
        headers["mcp-name"] = header_name
    request = urllib.request.Request(
        f"http://{_HOST}:{port}/mcp",
        data=json.dumps(payload).encode(),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=None) as response:
            parsed = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(detail) from exc
    except urllib.error.URLError as exc:
        reason = getattr(exc, "reason", None)
        message = str(reason) if reason else "the listener closed the connection"
        raise RuntimeError(message) from exc
    except (ConnectionError, TimeoutError) as exc:
        raise RuntimeError(str(exc)) from exc
    if not isinstance(parsed, dict):
        message = "MCP response is not an object"
        raise RuntimeError(message)
    if "error" in parsed:
        raise RuntimeError(json.dumps(parsed["error"]))
    result = parsed.get("result")
    if not isinstance(result, dict):
        message = "MCP response has no result"
        raise RuntimeError(message)
    return result
