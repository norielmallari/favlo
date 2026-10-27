"""MCP 2026-07-28 client for one HTTP or stdio server.

The client advertises ``io.modelcontextprotocol/tasks``. A generate tool then
returns a task handle. ``tasks/get`` is the poll.
"""

from __future__ import annotations

import json
import shlex
import subprocess
import threading
from typing import Any

import httpx

PROTOCOL = "2026-07-28"
_CLIENT = {"name": "favlo-web", "version": "0"}
_PROCS: dict[str, _Stdio] = {}
_PROCS_LOCK = threading.Lock()


class McpError(Exception):
    pass


def meta() -> dict[str, Any]:
    return {
        "io.modelcontextprotocol/protocolVersion": PROTOCOL,
        "io.modelcontextprotocol/clientCapabilities": {
            "extensions": {"io.modelcontextprotocol/tasks": {}}
        },
        "io.modelcontextprotocol/clientInfo": _CLIENT,
    }


async def rpc(
    server: dict[str, Any],
    method: str,
    params: dict[str, Any] | None = None,
    *,
    name: str | None = None,
    timeout: float = 15.0,
) -> dict[str, Any]:
    body = dict(params or {})
    body["_meta"] = meta()
    if server.get("transport") == "stdio":
        return await _stdio_rpc(server, method, body, timeout)
    return await _http_rpc(server, method, body, name, timeout)


async def list_tools(server: dict[str, Any]) -> list[dict[str, Any]]:
    result = await rpc(server, "tools/list", {}, timeout=15.0)
    tools = result.get("tools")
    if not isinstance(tools, list):
        raise McpError("Connection failed")
    found = []
    for tool in tools:
        if not isinstance(tool, dict) or not isinstance(tool.get("name"), str):
            continue
        schema = tool.get("inputSchema")
        if not isinstance(schema, dict):
            schema = {"type": "object", "properties": {}}
        description = tool.get("description")
        found.append(
            {
                "name": tool["name"],
                "description": description if isinstance(description, str) else "",
                "schema": schema,
            }
        )
    return found


def task_handle(result: dict[str, Any]) -> tuple[str, int] | None:
    task_id = result.get("taskId")
    if not isinstance(task_id, str) or not task_id:
        return None
    if result.get("resultType") != "task" and result.get("status") != "working":
        return None
    interval = result.get("pollIntervalMs")
    milliseconds = int(interval) if isinstance(interval, int) and interval > 0 else 2000
    return task_id, milliseconds


def render_result(payload: dict[str, Any]) -> str:
    nested = payload.get("result")
    if isinstance(nested, dict) and "content" not in payload:
        return render_result(nested)
    content = payload.get("content")
    lines: list[str] = []
    if isinstance(content, list):
        for block in content:
            if not isinstance(block, dict):
                continue
            kind = block.get("type")
            if kind == "text" and isinstance(block.get("text"), str):
                lines.append(block["text"])
            elif kind == "resource_link" and isinstance(block.get("uri"), str):
                lines.append(block["uri"])
    if lines:
        return "\n".join(lines)
    if payload.get("isError") or (isinstance(nested, dict) and nested.get("isError")):
        return "The tool failed."
    error = payload.get("error")
    if isinstance(error, dict) and isinstance(error.get("message"), str):
        return error["message"]
    return "The tool finished."


async def _http_rpc(
    server: dict[str, Any],
    method: str,
    params: dict[str, Any],
    name: str | None,
    timeout: float,
) -> dict[str, Any]:
    url = str(server.get("url") or "").strip()
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
        "mcp-protocol-version": PROTOCOL,
        "mcp-method": method,
    }
    if name:
        headers["mcp-name"] = name
    payload = {"jsonrpc": "2.0", "id": 1, "method": method, "params": params}
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(url, headers=headers, json=payload)
    except httpx.HTTPError as exc:
        raise McpError("Connection failed") from exc
    if response.status_code >= 400:
        raise McpError("Connection failed")
    try:
        body = response.json()
    except json.JSONDecodeError as exc:
        raise McpError("Connection failed") from exc
    return _result(body)


async def _stdio_rpc(
    server: dict[str, Any],
    method: str,
    params: dict[str, Any],
    timeout: float,
) -> dict[str, Any]:
    command = str(server.get("command") or "").strip()
    key = str(server.get("id") or command)
    proc = _process(key, command)

    def run() -> dict[str, Any]:
        return proc.request(method, params, timeout)

    try:
        return await _thread(run)
    except McpError:
        _drop(key)
        raise


async def _thread(fn: Any) -> Any:
    import asyncio

    return await asyncio.to_thread(fn)


def _process(key: str, command: str) -> _Stdio:
    with _PROCS_LOCK:
        current = _PROCS.get(key)
        if current is not None and current.alive():
            return current
        opened = _Stdio(command)
        _PROCS[key] = opened
        return opened


def _drop(key: str) -> None:
    with _PROCS_LOCK:
        current = _PROCS.pop(key, None)
    if current is not None:
        current.close()


def _result(body: Any) -> dict[str, Any]:
    if not isinstance(body, dict):
        raise McpError("Connection failed")
    if "error" in body:
        message = body["error"].get("message") if isinstance(body["error"], dict) else None
        raise McpError(message or "Connection failed")
    result = body.get("result")
    if not isinstance(result, dict):
        raise McpError("Connection failed")
    return result


class _Stdio:
    def __init__(self, command: str) -> None:
        argv = shlex.split(command)
        if not argv:
            raise McpError("A command is required.")
        try:
            self._proc = subprocess.Popen(
                argv,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                bufsize=1,
            )
        except OSError as exc:
            raise McpError("Connection failed") from exc
        self._lock = threading.Lock()
        self._next = 1

    def alive(self) -> bool:
        return self._proc.poll() is None

    def close(self) -> None:
        if self._proc.poll() is None:
            self._proc.kill()

    def request(self, method: str, params: dict[str, Any], timeout: float) -> dict[str, Any]:
        with self._lock:
            if not self.alive() or self._proc.stdin is None or self._proc.stdout is None:
                raise McpError("Connection failed")
            rpc_id = self._next
            self._next += 1
            line = json.dumps(
                {"jsonrpc": "2.0", "id": rpc_id, "method": method, "params": params}
            )
            try:
                self._proc.stdin.write(line + "\n")
                self._proc.stdin.flush()
            except OSError as exc:
                raise McpError("Connection failed") from exc
            holder: dict[str, Any] = {}

            def read() -> None:
                stdout = self._proc.stdout
                if stdout is None:
                    holder["error"] = McpError("Connection failed")
                    return
                while True:
                    raw = stdout.readline()
                    if not raw:
                        holder["error"] = McpError("Connection failed")
                        return
                    try:
                        parsed = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(parsed, dict) and parsed.get("id") == rpc_id:
                        holder["body"] = parsed
                        return

            reader = threading.Thread(target=read, daemon=True)
            reader.start()
            reader.join(timeout)
            if reader.is_alive():
                self.close()
                raise McpError("Connection failed")
            if "error" in holder:
                raise holder["error"]
            return _result(holder.get("body"))
