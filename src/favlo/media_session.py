"""Registry media defaults and one in-memory session of model choices.

``models/registry.json`` key ``media`` stores a folder id for ``image``,
``i2v``, ``t2v``, ``music``, and ``voice``. A generate call with no session
uses that id. ``set_*`` copies those defaults into a new session, then
replaces one kind. A later generate call that passes the session id uses
the session.
"""

from __future__ import annotations

import secrets
from typing import Any

from favlo.registry import media_default

KINDS = ("image", "i2v", "t2v", "music", "voice")


def default_model(kind: str, root: Any = None) -> str:
    """The folder id stored for ``kind``. An empty string means unset."""
    return media_default(kind, root)


def choose(state: dict[str, Any], kind: str, session: str) -> str:
    """The model id for one generate call."""
    token = session.strip()
    if token:
        record = _table(state).get(token)
        if not isinstance(record, dict):
            message = "unknown media session"
            raise LookupError(message)
        model = record.get(kind)
        if not isinstance(model, str) or not model:
            message = f"media session has no {kind} model"
            raise LookupError(message)
        return model
    model = default_model(kind)
    if not model:
        message = f"no default {kind} model"
        raise LookupError(message)
    return model


def assign(state: dict[str, Any], kind: str, model: str, session: str) -> str:
    """Store ``model`` for ``kind``. An empty session starts a new id."""
    token = session.strip()
    table = _table(state)
    if token:
        record = table.get(token)
        if not isinstance(record, dict):
            message = "unknown media session"
            raise LookupError(message)
    else:
        token = "media_" + secrets.token_hex(8)
        record = {key: default_model(key) for key in KINDS if default_model(key)}
        table[token] = record
    record[kind] = model
    return token


def _table(state: dict[str, Any]) -> dict[str, Any]:
    found = state.setdefault("media_sessions", {})
    if not isinstance(found, dict):
        message = "media sessions are not an object"
        raise LookupError(message)
    return found
