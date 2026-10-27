"""ASGI entry. uvicorn loads app:app with this directory on the path."""

from __future__ import annotations

from web_api.api import create_app

app = create_app()
