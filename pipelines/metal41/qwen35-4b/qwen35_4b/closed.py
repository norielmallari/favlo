"""Disconnect signal for one chat completion."""

from __future__ import annotations

from favlo.pipeline import ClientClosed


class RequestClosed(ClientClosed):
    """The client went away. ``LanguagePipeline`` catches this."""
