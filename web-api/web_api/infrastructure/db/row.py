"""Turn a mapped row into the dict the rest of the web API already returns."""

from __future__ import annotations

from typing import Any

from sqlalchemy import inspect


def as_dict(row: Any) -> dict[str, Any]:
    mapper = inspect(row).mapper
    return {attr.key: getattr(row, attr.key) for attr in mapper.column_attrs}
