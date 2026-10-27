"""One ASGI application for chat and MCP.

Uvicorn serves it. A pipeline receives a :class:`favlo.reply.LiveReply` and
never a socket. ``GET /v1/models`` does not take the request lock. The daemon
admits the GPU through :class:`favlo.schedule.Scheduler`: embeddings, then
chats, then media. An embedding waits until the open chat returns. A chat
that has not started waits while an embedding is queued or running. A chat
preempts media and leaves ``preempt`` set until the chat queue is empty.
An embedding does not set ``preempt``. It opens beside the resident model
and closes when the request returns. Image, video, music, and voice run
on the MCP queue.
"""

from __future__ import annotations

import asyncio
import json
import threading
from typing import Any

import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import Response, StreamingResponse
from starlette.routing import Route

from favlo.memory_notes import DIM, EMBED_MODEL
from favlo.mcp_server import build_mcp, media_kinds
from favlo.registry import load_registry, model_objects, one_model
from favlo.reply import LiveReply
from favlo.schedule import ClientClosed, Scheduler, ensure

_MAX_BODY = 64 * 1024 * 1024


class ServeHandle:
    """The object tests stop. It matches the old server's close methods."""

    def __init__(self) -> None:
        self.server_address = ("127.0.0.1", 0)
        self._server: uvicorn.Server | None = None
        self._sockets: list[Any] = []

    def shutdown(self) -> None:
        server = self._server
        if server is not None:
            server.should_exit = True

    def server_close(self) -> None:
        server = self._server
        if server is not None:
            server.should_exit = True
            server.force_exit = True
        for sock in self._sockets:
            close = getattr(sock, "close", None)
            if not callable(close):
                continue
            try:
                close()
            except OSError:
                pass


def serve(args: Any, state: dict[str, Any]) -> int:
    """Listen until the process is stopped. The port is ``args.port``."""
    from favlo.listener import _REQUEST_READ_TIMEOUT, _log

    handle = ServeHandle()
    app = build_app(args, state)
    config = uvicorn.Config(
        app,
        host=args.host,
        port=args.port,
        log_config=None,
        access_log=False,
        server_header=False,
        proxy_headers=False,
        timeout_keep_alive=_REQUEST_READ_TIMEOUT,
        timeout_graceful_shutdown=2,
        h11_max_incomplete_event_size=_MAX_BODY,
    )
    server = uvicorn.Server(config)
    server.install_signal_handlers = False
    handle._server = server

    async def run() -> None:
        task = asyncio.create_task(server.serve())
        while not server.started and not task.done():
            await asyncio.sleep(0.01)
        if task.done() and task.exception() is not None:
            raise task.exception()
        sockets = [sock for item in server.servers or [] for sock in (item.sockets or [])]
        handle._sockets = sockets
        if sockets:
            handle.server_address = sockets[0].getsockname()
        state["server"] = handle
        _log(f"listening on http://{args.host}:{handle.server_address[1]}")
        await task

    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        handle.server_close()
        _log("shutting down")
        return 0
    return 0


def _reject_legacy_initialize(mcp_app: Any) -> None:
    """`initialize` for a pre-2026-07-28 client returns -32022 so the client discovers."""
    route = mcp_app.routes[0]
    route.app = _LegacyInitialize(route.app)
    route.endpoint = route.app


class _LegacyInitialize:
    """Reject a legacy handshake. `server/discover` remains the negotiation."""

    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if scope.get("type") != "http" or scope.get("method") != "POST":
            await self.app(scope, receive, send)
            return
        body = b""
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            if message["type"] != "http.request":
                continue
            body += message.get("body", b"")
            if not message.get("more_body", False):
                break
        rejected = _legacy_initialize_error(body)
        if rejected is None:
            sent = False

            async def replay() -> dict[str, Any]:
                nonlocal sent
                if sent:
                    return {"type": "http.disconnect"}
                sent = True
                return {"type": "http.request", "body": body, "more_body": False}

            await self.app(scope, replay, send)
            return
        payload = json.dumps(rejected).encode()
        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [
                    [b"content-type", b"application/json"],
                    [b"content-length", str(len(payload)).encode()],
                ],
            }
        )
        await send({"type": "http.response.body", "body": payload})


def _legacy_initialize_error(body: bytes) -> dict[str, Any] | None:
    try:
        message = json.loads(body)
    except json.JSONDecodeError:
        return None
    if not isinstance(message, dict) or message.get("method") != "initialize":
        return None
    params = message.get("params")
    requested = params.get("protocolVersion") if isinstance(params, dict) else None
    meta = params.get("_meta") if isinstance(params, dict) else None
    version = None
    if isinstance(meta, dict):
        version = meta.get("io.modelcontextprotocol/protocolVersion")
    if requested == "2026-07-28" and version == "2026-07-28":
        return None
    return {
        "jsonrpc": "2.0",
        "id": message.get("id"),
        "error": {
            "code": -32022,
            "message": "Unsupported protocol version",
            "data": {
                "supported": ["2026-07-28"],
                "requested": requested if isinstance(requested, str) else "",
            },
        },
    }


def embed_input(
    args: Any,
    state: dict[str, Any],
    text: str,
    model: str = EMBED_MODEL,
) -> list[float] | str:
    """Run one embeddings request on the listener queue. Returns a vector or a sentence."""
    body = {"model": model, "input": text}
    reply = LiveReply(threading.Event())
    _run_locked(args, state, "embed", "application/json", json.dumps(body).encode(), reply)
    payload = reply.payload
    if reply.status >= 400 or not isinstance(payload, dict):
        return "embeddings are not available"
    data = payload.get("data")
    if not isinstance(data, list) or not data:
        return "embeddings are not available"
    row = data[0]
    if not isinstance(row, dict):
        return "embeddings are not available"
    vector = row.get("embedding")
    if not isinstance(vector, list) or len(vector) != DIM:
        return "embeddings are not available"
    return [float(value) for value in vector]


def build_app(args: Any, state: dict[str, Any]) -> Starlette:
    _prepare_listener(state)

    def dispatch(kind: str, body: dict[str, Any], task_id: str = "") -> Any:
        return _mcp_dispatch(args, state, kind, body, task_id)

    def embed(text: str) -> list[float] | str:
        return embed_input(args, state, text)

    mcp = build_mcp(dispatch, state, embed)
    mcp_app = mcp.http_app(path="/mcp", stateless_http=True, json_response=True)
    _reject_legacy_initialize(mcp_app)
    async def models(request: Request) -> Response:
        return _models(request, state)

    async def model(request: Request) -> Response:
        return _one_model(request, state)

    async def release(request: Request) -> Response:
        return await _release(request, args, state)

    async def pipelines_close(_request: Request) -> Response:
        return await _pipelines_close(args, state)

    async def pipelines_reload(_request: Request) -> Response:
        return await _pipelines_reload(args, state)

    async def resume(request: Request) -> Response:
        return await _resume(request, state)

    async def chat(request: Request) -> Response:
        return await _locked(request, args, state, "chat")

    async def embeddings(request: Request) -> Response:
        return await _locked(request, args, state, "embed")

    routes = [
        Route("/v1/models", models, methods=["GET"]),
        Route("/v1/models/{model_id:path}", model, methods=["GET"]),
        Route("/v1/release", release, methods=["POST"]),
        Route("/v1/resume", resume, methods=["POST"]),
        Route("/v1/pipelines/close", pipelines_close, methods=["POST"]),
        Route("/v1/pipelines/reload", pipelines_reload, methods=["POST"]),
        Route("/v1/chat/completions", chat, methods=["POST"]),
        Route("/v1/embeddings", embeddings, methods=["POST"]),
        *mcp_app.routes,
    ]
    return Starlette(routes=routes, lifespan=mcp_app.lifespan)


def _language_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    hidden = _hidden_pipelines()
    return [row for row in rows if row.get("pipeline") not in hidden]


def _models(_request: Request, state: dict[str, Any]) -> Response:
    rows = _language_rows(model_objects(resident=state.get("record")))
    return _json_response({"object": "list", "data": rows})


def _one_model(request: Request, state: dict[str, Any]) -> Response:
    try:
        row = one_model(request.path_params["model_id"], resident=state.get("record"))
    except LookupError as exc:
        return _error(404, str(exc), "not_found")
    if row.get("pipeline") in _hidden_pipelines():
        return _error(404, "not found", "not_found")
    return _json_response(row)


async def _release(request: Request, args: Any, state: dict[str, Any]) -> Response:
    from favlo.listener import _drop_runtime, _smoke_held

    pid, error = await _pid(request)
    if error is not None:
        return error
    lock: threading.Lock = state.setdefault("lock", threading.Lock())
    if not lock.acquire(blocking=False):
        return _error(409, "a request is open", "busy")
    try:
        holder = state.get("smoke_hold")
        if _smoke_held(state) and holder != pid:
            return _error(409, "a smoke hold is active", "busy")
        try:
            closed = _drop_runtime(state, args)
        except Exception as exc:
            return _error(500, str(exc), type(exc).__name__)
        state["smoke_hold"] = pid
        return _json_response({"released": True, "closed": closed})
    finally:
        lock.release()


async def _resume(request: Request, state: dict[str, Any]) -> Response:
    from favlo.listener import _smoke_held

    pid, error = await _pid(request)
    if error is not None:
        return error
    lock: threading.Lock = state.setdefault("lock", threading.Lock())
    if not lock.acquire(blocking=False):
        return _error(409, "a request is open", "busy")
    try:
        holder = state.get("smoke_hold")
        if _smoke_held(state) and holder != pid:
            return _error(409, "a smoke hold is active", "busy")
        state["smoke_hold"] = None
        return _json_response({"resumed": True})
    finally:
        lock.release()


async def _pipelines_close(args: Any, state: dict[str, Any]) -> Response:
    """Close every runtime and leave ``POST /mcp`` up. This does not set the smoke hold."""
    from favlo.pipelines import pipelines

    _prepare_listener(state)
    try:
        closed = await asyncio.to_thread(pipelines.close, state, args)
    except Exception as exc:
        return _error(500, str(exc), type(exc).__name__)
    return _json_response({"unloaded": True, "closed": closed})


async def _pipelines_reload(args: Any, state: dict[str, Any]) -> Response:
    """Close every runtime, drop the queues, and drop imported pipeline modules."""
    from favlo.pipelines import pipelines

    _prepare_listener(state)
    try:
        closed, modules = await asyncio.to_thread(pipelines.reload, state, args)
    except Exception as exc:
        return _error(500, str(exc), type(exc).__name__)
    return _json_response({"reloaded": True, "closed": closed, "modules": modules})


async def _locked(request: Request, args: Any, state: dict[str, Any], kind: str) -> Response:
    raw, error = await _read_body(request)
    if error is not None:
        return error
    disconnected = threading.Event()
    finished = threading.Event()
    reply = LiveReply(disconnected)

    async def watch() -> None:
        while not finished.is_set():
            message = await request.receive()
            if message["type"] == "http.disconnect":
                disconnected.set()
                return

    watcher = asyncio.create_task(watch())

    def work() -> None:
        try:
            _run_locked(args, state, kind, request.headers.get("content-type", ""), raw, reply)
        finally:
            reply.finish_stream()
            finished.set()

    asyncio.create_task(asyncio.to_thread(work))
    await asyncio.to_thread(reply.ready.wait)
    if reply.kind == "json":
        watcher.cancel()
        return Response(reply.encoded, status_code=reply.status, media_type="application/json")

    async def chunks():
        try:
            while True:
                item = await asyncio.to_thread(reply.pull)
                if item is None:
                    break
                if item == b"":
                    continue
                yield item
        finally:
            watcher.cancel()

    return StreamingResponse(
        chunks(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


_MEDIA_KINDS = frozenset({"image", "video", "music", "voice"})


def _prepare_listener(state: dict[str, Any]) -> None:
    ensure(state)
    state.setdefault("lock", threading.Lock())
    state.setdefault("cancel", threading.Event())
    state.setdefault("media_task", None)
    state.setdefault("media_yield", False)
    state.setdefault("media_tasks", {})
    state.setdefault("media_order", [])


def _run_locked(
    args: Any,
    state: dict[str, Any],
    kind: str,
    content_type: str,
    raw: bytes,
    reply: LiveReply,
) -> None:
    _prepare_listener(state)
    schedule: Scheduler = state["schedule"]
    ticket = None
    cancel = getattr(reply, "disconnected", None)
    if kind != "chat" or not isinstance(cancel, threading.Event):
        cancel = None
    try:
        if kind in {"embed", "chat"}:
            try:
                ticket = schedule.acquire(kind, cancel=cancel)
            except ClientClosed:
                _leave_cleared(reply)
                return
        if cancel is not None and cancel.is_set():
            return
        if kind in _MEDIA_KINDS:
            _run_media(args, state, kind, content_type, raw, reply)
        else:
            _run_once(args, state, kind, content_type, raw, reply)
    finally:
        if ticket is not None:
            schedule.release(ticket)


def _run_media(
    args: Any,
    state: dict[str, Any],
    kind: str,
    content_type: str,
    raw: bytes,
    reply: LiveReply,
) -> None:
    """Run one media call. Resume it only after the foreground queue drains."""
    schedule: Scheduler = state["schedule"]
    epoch = schedule.epoch
    while True:
        try:
            ticket = schedule.acquire("media")
        except ClientClosed:
            _leave_cleared(reply)
            return
        try:
            if schedule.epoch != epoch:
                _leave_cleared(reply)
                return
            state["media_yield"] = False
            _run_once(args, state, kind, content_type, raw, reply)
            if schedule.epoch != epoch:
                _discard_output(state, reply.payload)
                _leave_cleared(reply)
                return
            if state["cancel"].is_set():
                _discard_output(state, reply.payload)
                return
            if state.get("media_yield") or _paused(state, reply):
                _mark_paused(state)
                continue
            return
        finally:
            schedule.release(ticket)


def _run_once(
    args: Any,
    state: dict[str, Any],
    kind: str,
    content_type: str,
    raw: bytes,
    reply: LiveReply,
) -> None:
    from favlo.listener import (
        _bind_request,
        _close_embed,
        _drop_runtime,
        _embed_selection,
        _log,
        _open_embed,
        _report_failure,
        _schedule_pipeline_heartbeat,
        _smoke_held,
        _video_body,
    )

    lock: threading.Lock = state["lock"]
    holds = False
    if not lock.acquire(blocking=False):
        _log("waiting for the open request")
        lock.acquire()
    holds = True
    if state.get("unload"):
        lock.release()
        holds = False
        if not reply.started:
            reply.json(409, {"error": {"message": "the pipeline is closing", "type": "busy"}})
        return
    opened = False
    selection = None
    embed_child = None
    drop_embed = False
    try:
        schedule: Scheduler = state["schedule"]
        if kind in _MEDIA_KINDS and schedule.foreground():
            state["media_yield"] = True
            return
        if kind in _MEDIA_KINDS:
            _clear_paused(state)
        if _smoke_held(state):
            reply.json(
                503,
                {"error": {"message": "the gpu is held for a pipeline smoke", "type": "busy"}},
            )
            return
        _log("request open")
        opened = True
        try:
            if kind in {"video", "music"}:
                body = _video_body(content_type, raw)
            else:
                parsed = json.loads(raw or b"{}")
                if not isinstance(parsed, dict):
                    message = "request body must be an object"
                    raise ValueError(message)
                body = parsed
        except (json.JSONDecodeError, ValueError) as exc:
            reply.json(400, {"error": {"message": str(exc), "type": "invalid_request_error"}})
            return
        try:
            if kind == "embed" and getattr(args, "lazy", False):
                selection = _embed_selection(body, state)
                same = (
                    state.get("runtime") is not None
                    and state.get("record") == selection.folder
                    and state.get("pipeline") == selection.pipeline
                )
                if same:
                    drop_embed = True
                    runtime = state.get("runtime")
                    module = state.get("module")
                else:
                    embed_child = _open_embed(state, args, selection)
                    runtime = embed_child
                    module = embed_child.proxy
                state["requests"] = int(state.get("requests") or 0) + 1
                if runtime is not None:
                    runtime._request_id = state["requests"]
            else:
                selection = _bind_request(body, state, args)
                runtime = state.get("runtime")
                module = state.get("module")
        except LookupError as exc:
            reply.json(404, {"error": {"message": str(exc), "type": "not_found"}})
            return
        if selection is not None:
            reply.model_id = selection.requested_id
        if runtime is None or module is None:
            reply.json(404, {"error": {"message": "no model is loaded", "type": "not_found"}})
            return
        handler_name = {
            "chat": "handle_chat",
            "embed": "handle_embed",
            "image": "handle_image",
            "video": "handle_video",
            "music": "handle_music",
            "voice": "handle_voice",
        }[kind]
        call = getattr(module, handler_name, None)
        if not callable(call):
            if kind == "chat":
                name = state.get("pipeline") or "pipeline"
                reply.json(
                    500,
                    {
                        "error": {
                            "message": f"{name} does not complete a chat",
                            "type": "RuntimeError",
                        }
                    },
                )
            else:
                reply.json(404, {"error": {"message": "not found"}})
            return
        try:
            call(reply, runtime, body, selection, args, state)
        except KeyError as exc:
            if not reply.started:
                reply.json(404, {"error": {"message": str(exc), "type": "not_found"}})
        except LookupError as exc:
            if not reply.started:
                reply.json(404, {"error": {"message": str(exc), "type": "not_found"}})
        except Exception as exc:
            from favlo.pipeline_worker import ChildExited

            if isinstance(exc, ChildExited) and embed_child is None:
                _abandon_child(state, args)
            else:
                console = getattr(embed_child, "console", None) if embed_child is not None else None
                if console is None:
                    console = getattr(state.get("runtime"), "console", None)
                _report_failure(console)
            if not reply.started:
                reply.json(500, {"error": {"message": str(exc), "type": type(exc).__name__}})
    finally:
        try:
            if embed_child is not None:
                _close_embed(embed_child, args)
            elif drop_embed:
                _drop_runtime(state, args)
            if opened:
                _log("request closed")
            module = state.get("module")
            runtime = state.get("runtime")
            _touch_job(state, reply)
        finally:
            if holds:
                lock.release()
        if opened and kind != "embed":
            _schedule_pipeline_heartbeat(module, runtime, selection)
        if not reply.started and not state.get("media_yield"):
            reply.json(
                500,
                {"error": {"message": "the pipeline wrote no reply", "type": "RuntimeError"}},
            )


def _abandon_child(state: dict[str, Any], args: Any) -> None:
    """Drop a pipeline process that exited on its own and forget its owners."""
    from favlo.listener import _drop_runtime
    from favlo.pipeline_worker import PipelineChild

    runtime = state.get("runtime")
    if isinstance(runtime, PipelineChild):
        _drop_runtime(state, args)


def _leave_cleared(reply: LiveReply) -> None:
    """A flushed waiter leaves. A call that already started reports cancelled."""
    if reply.started:
        reply.json(200, {"status": "cancelled"})
        return
    reply.json(409, {"error": {"message": "the queue was cleared", "type": "busy"}})


def _paused(state: dict[str, Any], reply: LiveReply) -> bool:
    if not state["preempt"].is_set():
        return False
    payload = reply.payload
    return isinstance(payload, dict) and payload.get("status") == "in_progress"


def _mark_paused(state: dict[str, Any]) -> None:
    record = _task_record(state)
    if record is not None:
        record["paused"] = True


def _clear_paused(state: dict[str, Any]) -> None:
    record = _task_record(state)
    if record is not None:
        record["paused"] = False


def _touch_job(state: dict[str, Any], reply: LiveReply) -> None:
    record = _task_record(state)
    if record is not None and isinstance(reply.payload, dict):
        record["job"] = reply.payload


def _task_record(state: dict[str, Any]) -> dict[str, Any] | None:
    task_id = state.get("media_task")
    if not isinstance(task_id, str) or not task_id:
        return None
    record = state["media_tasks"].get(task_id)
    if isinstance(record, dict):
        return record
    return None


def _discard_output(state: dict[str, Any], payload: Any) -> None:
    if not isinstance(payload, dict):
        return
    from pathlib import Path

    from favlo.images import image_path, source_path
    from favlo.music import music_dir
    from favlo.voice import voice_dir
    from favlo.videos import source_path as video_source
    from favlo.videos import video_path

    data = payload.get("data")
    if isinstance(data, list):
        home = state.get("image_home")
        root = home if isinstance(home, Path) else None
        if root is not None:
            for item in data:
                if not isinstance(item, dict) or not isinstance(item.get("id"), str):
                    continue
                for path_of in (image_path, source_path):
                    try:
                        path_of(root, item["id"]).unlink(missing_ok=True)
                    except (OSError, ValueError):
                        continue
    item_id = payload.get("id")
    if not isinstance(item_id, str):
        return
    if item_id.startswith("video_"):
        home = state.get("video_home")
        root = home if isinstance(home, Path) else None
        if root is None:
            return
        try:
            video_source(root, item_id).unlink(missing_ok=True)
            video_path(root, item_id).unlink(missing_ok=True)
        except (OSError, ValueError):
            return
    if item_id.startswith("music_"):
        home = state.get("music_home")
        root = home if isinstance(home, Path) else None
        if root is None:
            return
        folder = music_dir(root)
        for suffix in ("wav", "mp3", "flac"):
            try:
                (folder / f"{item_id}.{suffix}").unlink(missing_ok=True)
            except OSError:
                continue
    if item_id.startswith("voice_"):
        home = state.get("voice_home")
        root = home if isinstance(home, Path) else None
        if root is None:
            return
        folder = voice_dir(root)
        for suffix in ("wav", "mp3", "flac"):
            try:
                (folder / f"{item_id}.{suffix}").unlink(missing_ok=True)
            except OSError:
                continue


def _mcp_dispatch(
    args: Any,
    state: dict[str, Any],
    kind: str,
    body: dict[str, Any],
    task_id: str = "",
) -> Any:
    if kind == "list":
        media = body.get("media") if isinstance(body, dict) else None
        mode = body.get("mode") if isinstance(body, dict) else None
        if media not in {"image", "video", "music", "voice"}:
            return []
        if media == "video" and mode not in {"i2v", "t2v"}:
            return []
        return _media_rows(media, mode if isinstance(mode, str) else None)
    _prepare_listener(state)
    if task_id:
        record = state["media_tasks"].get(task_id)
        if isinstance(record, dict):
            if record.get("dropped"):
                return {"status": "cancelled"}
            record["started"] = True
            record["paused"] = False
        state["cancel"].clear()
        state["media_task"] = task_id
    reply = LiveReply(threading.Event())
    try:
        _run_locked(args, state, kind, "application/json", json.dumps(body).encode(), reply)
    finally:
        if task_id and state.get("media_task") == task_id:
            state["media_task"] = None
    if state["cancel"].is_set():
        state["cancel"].clear()
        return {"status": "cancelled"}
    payload = reply.payload
    if not isinstance(payload, dict):
        message = "the pipeline wrote no reply"
        raise RuntimeError(message)
    if reply.status >= 400:
        raise RuntimeError(_error_message(payload))
    if payload.get("status") == "failed":
        raise RuntimeError(_error_message(payload))
    return payload


def _error_message(payload: dict[str, Any]) -> str:
    error = payload.get("error")
    if isinstance(error, dict) and isinstance(error.get("message"), str):
        return error["message"]
    return "the pipeline failed"


def _each_pipeline() -> list[tuple[str, str, Any]]:
    from favlo.cli import load_pipeline

    data = load_registry()
    found: list[tuple[str, str, Any]] = []
    seen: set[str] = set()
    for folder, entry in data["models"].items():
        pipeline = entry.get("pipeline")
        if not isinstance(pipeline, str) or pipeline in seen:
            continue
        runtime = entry.get("runtime")
        if not isinstance(runtime, str) or not runtime:
            message = f"{folder} has no runtime"
            raise LookupError(message)
        try:
            module = load_pipeline(pipeline, runtime=runtime, custom=bool(entry.get("custom")))
        except (FileNotFoundError, ImportError, OSError):
            continue
        seen.add(pipeline)
        found.append((folder, pipeline, module))
    return found


def _hidden_pipelines() -> set[str]:
    names: set[str] = set()
    for _folder, pipeline, module in _each_pipeline():
        if media_kinds(module) or callable(getattr(module, "handle_embed", None)):
            names.add(pipeline)
    return names


def _media_rows(media: str, mode: str | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for folder, pipeline, module in _each_pipeline():
        kinds = media_kinds(module)
        if media not in kinds:
            continue
        if media == "video":
            modes = getattr(module, "video_modes", None)
            offered = modes() if callable(modes) else ()
            if mode not in offered:
                continue
        row: dict[str, Any] = {"id": folder, "pipeline": pipeline, "media": [media]}
        if media == "voice":
            speakers = getattr(module, "list_voices", None)
            if callable(speakers):
                row["voices"] = speakers()
            emotions = getattr(module, "list_emotions", None)
            if callable(emotions):
                row["emotions"] = emotions()
        if media == "video":
            listed = getattr(module, "list_resolutions", None)
            if callable(listed):
                row["resolutions"] = listed()
        rows.append(row)
    return rows


async def _pid(request: Request) -> tuple[int, Response | None]:
    try:
        body = await request.json()
    except (json.JSONDecodeError, ValueError):
        return 0, _error(400, "pid is required", "bad_request")
    pid = body.get("pid") if isinstance(body, dict) else None
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
        return 0, _error(400, "pid is required", "bad_request")
    return pid, None


async def _read_body(request: Request) -> tuple[bytes, Response | None]:
    raw = await request.body()
    if len(raw) > _MAX_BODY:
        return b"", _error(413, "request body is too large", "invalid_request_error")
    return raw, None


def _json_response(payload: dict[str, Any], status: int = 200) -> Response:
    blob = json.dumps(payload).encode()
    return Response(blob, status_code=status, media_type="application/json")


def _error(
    status: int,
    message: str,
    kind: str = "invalid_request_error",
    param: str | None = None,
) -> Response:
    error: dict[str, Any] = {"message": message, "type": kind}
    if param is not None:
        error["param"] = param
    return _json_response({"error": error}, status)
