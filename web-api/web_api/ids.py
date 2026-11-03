"""Identifiers for application records. Every id is a canonical UUID string."""

from __future__ import annotations

import uuid

# Stable ids for rows the app seeds once.
FAVLO_AGENT_ID = str(uuid.uuid5(uuid.NAMESPACE_DNS, "favlo.agent"))
FAVLO_CHANNEL_ID = str(uuid.uuid5(uuid.NAMESPACE_DNS, "favlo.channel"))
FAVLO_MCP_ID = str(uuid.uuid5(uuid.NAMESPACE_DNS, "favlo.mcp"))
LOCAL_PROVIDER_ID = str(uuid.uuid5(uuid.NAMESPACE_DNS, "favlo.provider.local"))


def guid() -> str:
    """A new canonical UUID string."""
    return str(uuid.uuid4())


def is_guid(value: object) -> bool:
    """True when value is a canonical UUID string."""
    if not isinstance(value, str):
        return False
    try:
        return str(uuid.UUID(value)) == value
    except ValueError:
        return False
