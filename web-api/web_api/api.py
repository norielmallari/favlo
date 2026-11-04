"""FastAPI application for favlo web."""

from __future__ import annotations

import asyncio
import os
import threading
from collections.abc import Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from web_api.bus import Bus
from web_api.chat import stream_reply
from web_api.heartbeat_clock import watch
from web_api.mcp_tasks import bind_followup, resume
from web_api import agui
from web_api.routes import (
    agents,
    channels,
    events,
    heartbeats,
    mcp,
    media,
    providers,
    sessions,
    settings,
    workspaces,
)
from web_api.runs import perform
from web_api.infrastructure.db.database import prepare
from web_api.store import Conflict, NotFound, Rejected, Store

BROWSER_ORIGIN = "http://127.0.0.1:8012"


def web_home() -> Path:
    override = os.environ.get("FAVLO_WEB_HOME")
    if override:
        return Path(override)
    return Path.home() / ".favlo" / "web"


def create_app(home: Path | None = None, reply: Callable[..., Any] | None = None) -> FastAPI:
    root = home or web_home()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        database = root / "app.db"
        await asyncio.to_thread(prepare, database)
        app.state.store = Store(database)
        app.state.bus = Bus()
        app.state.cancels = {}
        app.state.tasks = set()
        app.state.reply = reply or stream_reply
        app.state.seeded_providers = set()

        for run_id in await asyncio.to_thread(app.state.store.abandoned_runs):
            for event in await asyncio.to_thread(
                app.state.store.finish_run,
                run_id,
                "failed",
                "The server stopped during that run.",
            ):
                await app.state.bus.publish(event)
        stop = asyncio.Event()
        app.state.heartbeat_stop = stop
        app.state.gallery_index_running = {}
        app.state.gallery_index_lock = threading.Lock()
        clock = asyncio.create_task(watch(app))
        app.state.tasks.add(clock)
        clock.add_done_callback(app.state.tasks.discard)
        from web_api.gallery_embed import watch as watch_gallery

        gallery_clock = asyncio.create_task(watch_gallery(app))
        app.state.tasks.add(gallery_clock)
        gallery_clock.add_done_callback(app.state.tasks.discard)

        async def follow(run_id: str) -> None:
            cancel = asyncio.Event()
            app.state.cancels[run_id] = cancel
            job = asyncio.create_task(perform(app, cancel, run_id))
            app.state.tasks.add(job)
            job.add_done_callback(app.state.tasks.discard)

        bind_followup(follow)
        resume(app.state.store, app.state.bus, app.state.tasks)
        try:
            yield
        finally:
            bind_followup(None)
            stop.set()
        for task in list(app.state.tasks):
            task.cancel()
        app.state.store.close()

    app = FastAPI(lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[BROWSER_ORIGIN],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(agents.router)
    app.include_router(channels.router)
    app.include_router(sessions.router)
    app.include_router(providers.router)
    app.include_router(settings.router)
    app.include_router(heartbeats.router)
    app.include_router(mcp.router)
    app.include_router(media.router)
    app.include_router(workspaces.router)
    app.include_router(events.router)
    agui.install(app)

    @app.get("/health")
    async def health() -> dict[str, bool]:
        return {"ok": True}

    @app.get("/runs/{run_id}")
    async def read_run(run_id: str, request: Request):
        try:
            row = request.app.state.store.cancel_run(run_id)
        except NotFound as exc:
            return JSONResponse(
                status_code=404,
                content={"error": "not_found", "message": str(exc)},
            )
        return JSONResponse(status_code=200, content=row)

    @app.post("/runs/{run_id}/cancel")
    async def cancel_run(run_id: str, request: Request):
        try:
            row = request.app.state.store.cancel_run(run_id)
        except NotFound as exc:
            return JSONResponse(
                status_code=404,
                content={"error": "not_found", "message": str(exc)},
            )
        pending = request.app.state.cancels.get(run_id)
        if pending is not None:
            pending.set()
        # SSE may already be gone after a client abort; still close the run row.
        for event in request.app.state.store.finish_run(run_id, "cancelled", None):
            await request.app.state.bus.publish(event)
        return JSONResponse(status_code=202, content={"status": "accepted", "run_id": row["id"]})

    @app.exception_handler(NotFound)
    async def missing(_request: Request, exc: NotFound):
        return JSONResponse(status_code=404, content={"error": "not_found", "message": str(exc)})

    @app.exception_handler(Conflict)
    async def conflict(_request: Request, exc: Conflict):
        return JSONResponse(status_code=409, content={"error": "conflict", "message": str(exc)})

    @app.exception_handler(Rejected)
    async def rejected(_request: Request, exc: Rejected):
        return JSONResponse(status_code=400, content={"error": "invalid", "message": str(exc)})

    return app
