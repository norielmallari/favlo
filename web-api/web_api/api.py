"""FastAPI application for favlo web."""

from __future__ import annotations

import os
from collections.abc import Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from web_api.bus import Bus
from web_api.chat import stream_reply
from web_api.mcp_tasks import resume
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
)
from web_api.store import Conflict, NotFound, Rejected, Store

BROWSER_ORIGIN = "http://127.0.0.1:8081"


def web_home() -> Path:
    override = os.environ.get("FAVLO_WEB_HOME")
    if override:
        return Path(override)
    return Path.home() / ".favlo" / "web"


def create_app(home: Path | None = None, reply: Callable[..., Any] | None = None) -> FastAPI:
    root = home or web_home()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.store = Store(root / "app.db")
        app.state.bus = Bus()
        app.state.cancels = {}
        app.state.tasks = set()
        app.state.reply = reply or stream_reply
        app.state.seeded_providers = set()
        resume(app.state.store, app.state.bus, app.state.tasks)
        yield
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
    app.include_router(events.router)

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
