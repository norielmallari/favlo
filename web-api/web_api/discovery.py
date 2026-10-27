"""Read model ids from an OpenAI-compatible models list."""

from __future__ import annotations

from typing import Any

import httpx


class DiscoveryFailed(Exception):
    pass


def ids_from_payload(body: object) -> list[str]:
    if not isinstance(body, dict):
        return []
    data = body.get("data")
    if not isinstance(data, list):
        return []
    found: list[str] = []
    seen: set[str] = set()
    for row in data:
        if not isinstance(row, dict):
            continue
        model_id = row.get("id")
        if isinstance(model_id, str) and model_id and model_id not in seen:
            seen.add(model_id)
            found.append(model_id)
    return found


async def fetch_model_ids(base_url: str) -> list[str]:
    url = base_url.rstrip("/") + "/models"
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(url)
    except httpx.HTTPError as exc:
        raise DiscoveryFailed("Connection failed") from exc
    if response.status_code != 200:
        raise DiscoveryFailed("Connection failed")
    try:
        payload: Any = response.json()
    except ValueError as exc:
        raise DiscoveryFailed("Connection failed") from exc
    return ids_from_payload(payload)
